#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/trainer/train_two_stream.py
Project: /workspace/skeleton/project/trainer
Created Date: Friday June 7th 2024
Author: Kaixu Chen
-----
Comment:
This file implements the training process for cnn lstm method.
Here, saving the results and calculating the metrics are done in separate functions.

Have a good code time :)
-----
Last Modified: Friday June 7th 2024 7:50:12 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from torchvision.utils import save_image, flow_to_image

from models.make_model import CNNLSTM
from utils.helper import save_helper
from utils.metrics import ClassificationMetrics


def _expand_video_names(batch) -> list:
    """把 batch 里每条视频的名字按其 gait 段数展开,与逐段预测一一对应。

    collate_fn 把一条视频的所有段沿 batch 维拼接,所以段级预测的归属只能从
    info 里的 num_chunks 还原。存下来是为了能算患者级指标 —— 有效样本量是
    患者(每折 test 17 人),不是段(约 2800)。
    """
    names = []
    for item in batch.get("info", []):
        names.extend([item["video_name"]] * int(item["num_chunks"]))
    return names


class CNNLstmModule(LightningModule):

    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.model_type = hparams.model.model
        # 与 concept/clip 共用 loss.lr,避免对比实验被不同学习率混淆
        self.lr = hparams.loss.lr
        # 之前漏了 weight_decay,基线不带正则、主方法带,对比不公平
        self.weight_decay = float(getattr(hparams.loss, "weight_decay", 0.001))
        self.num_classes = hparams.model.model_class_num
        self.save_root = hparams.log_path

        # model define

        self.model = CNNLSTM(hparams)

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        self.metrics = ClassificationMetrics(self.num_classes)

        # 测试期把预测存下来,交给 analysis/compare_concept_runs.py 与主方法同口径汇总
        self.test_video_names: list = []
        self.test_pred_list = []
        self.test_label_list = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        train steop when trainer.fit called

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_

        Returns:
            loss: the calc loss
        """

        video = batch["video"].detach() # b, c, t, h, w
        label = batch["label"].detach()  # b, c, t, h, w
        # CNNLSTM 每段输出一个预测,标签不展开到帧

        loss = self.single_logic(label, video, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        """
        val step when trainer.fit called.

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_

        Returns:
            loss: the calc loss
            accuract: selected accuracy result.
        """

        # input and model define
        video = batch["video"].detach()  # b, c, t, h, w
        label = batch["label"].detach()  # b

        # CNNLSTM 每段输出一个预测,标签不展开到帧
        loss = self.single_logic(label, video, "val")

    def test_step(self, batch, batch_idx):
        """
        test step when trainer.test called

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_
        """
         # input and model define
        video = batch["video"].detach()  # b, c, t, h, w
        label = batch["label"].detach()  # b

        # not use the last frame
        # CNNLSTM 每段输出一个预测,标签不展开到帧
        self.test_video_names.extend(_expand_video_names(batch))
        loss = self.single_logic(label, video, "test")

    def on_test_epoch_end(self) -> None:
        # 与 concept 分支存同样的东西,汇总脚本才能把基线和主方法放进一张表
        save_helper(
            all_pred=self.test_pred_list,
            all_label=self.test_label_list,
            fold=self._fold_name(),
            save_path=self.save_root,
            num_class=self.num_classes,
            all_video_name=self.test_video_names,
        )

    def _fold_name(self) -> str:
        """从 logger 的 root_dir 取折号;fast_dev_run 下 logger 被禁用,root_dir 为 None。"""
        root_dir = getattr(self.logger, "root_dir", None) if self.logger else None
        return root_dir.split("/")[-1] if root_dir else "fold"

    def configure_optimizers(self):
        """
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        """

        optimzier = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        return {
            "optimizer": optimzier,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimzier),
                "monitor": "val/loss",
            },
        }
        # return torch.optim.SGD(self.parameters(), lr=self.lr)

    def _get_name(self):
        return self.model_type

    def single_logic(self, label: torch.Tensor, video: torch.Tensor, stage: str):

        b, c, t, h, w = video.shape

        # CNNLSTM 直接吃 5D,内部自己按时间维展开,不能像 2dcnn 那样先 reshape 成 b*t
        # eval model, feed data here
        if self.training:
            preds = self.model(video)

        else:
            with torch.no_grad():
                preds = self.model(video)

        # squeeze(dim=-1) to keep the torch.Size([1]), not null.

        loss = F.cross_entropy(
            preds.squeeze(dim=-1), label.long()
        )

        self.save_log(preds, label, loss, stage)

        return loss

    def save_log(self, pred: torch.Tensor, label: torch.Tensor, loss, stage: str):
        """记录 loss 与分类指标。

        阶段必须显式传进来。原先靠 `self.training` 分两支,test 阶段
        `self.training` 同样是 False,于是测试结果被写到了 `val/` 前缀下 ——
        B1_2dcnn / B2_cnn_lstm 的 test_metrics.txt 里只有 val 键就是这个原因。
        """

        preds = pred
        if preds.size()[0] != 1 or len(preds.size()) != 1:
            preds = preds.squeeze(dim=-1)
        # 多分类一律走 softmax。旧代码在非训练分支用了 sigmoid,虽然不改 argmax、
        # 指标不受影响,但存下来的"概率"不是概率,汇总脚本按概率处理会失真。
        pred_softmax = torch.softmax(preds, dim=-1)

        self.log(
            f"{stage}/loss",
            loss,
            on_epoch=True,
            on_step=(stage == "train"),
            batch_size=label.size()[0],
        )
        self.metrics.log(self, stage, pred_softmax, label, batch_size=label.size()[0])

        if stage == "test":
            self.test_pred_list.append(pred_softmax.detach().cpu())
            self.test_label_list.append(label.detach().long().cpu())

'''
File: train.py
Project: project
Created Date: 2023-10-19 02:29:47
Author: chenkaixu
-----
Comment:
 This file is the train/val/test process for the project.
 

Have a good code time!
-----
Last Modified: Sunday June 9th 2024 6:04:51 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

16-05-2024	Kaixu Chen	do not output metrics in test_step to log file.

22-03-2024	Kaixu Chen	add different class number mapping, now the class number is a hyperparameter.

14-12-2023	Kaixu Chen refactor the code, now it a simple code to train video frame from dataloader.

'''

from typing import Any, List, Optional, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from pytorch_lightning import LightningModule

from models.make_model import MakeVideoModule
from models.clinical_concept import _count_aux
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

class SingleModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.img_size = hparams.data.img_size
        # 与 concept/clip 共用 loss.lr,避免对比实验被不同学习率混淆
        self.lr = hparams.loss.lr
        # 之前这里直接 Adam(params, lr),把配置里的 weight_decay 丢了,而 concept/clip/pose
        # 都用上了 —— 基线不带正则、主方法带,对比不公平
        self.weight_decay = float(getattr(hparams.loss, "weight_decay", 0.001))

        self.num_classes = hparams.model.model_class_num
        self.save_root = hparams.log_path

        # define model
        self.video_cnn = MakeVideoModule(hparams)()

        # 角色二(VLM 教师):额外回归 VLM 给出的临床属性分数。把 slow_r50 的分类投影换成
        # Identity 拿到 2048 维特征,分类头与辅助头并列;推理时辅助头不参与判决
        self.w_aux = float(getattr(hparams.loss, "aux_weight", 0.0))
        # 特征级融合:把辅助量(几何属性, 已 z 标准化)直接拼到分类头前的特征上。推理时也要有 aux,
        # 所以只能配可从数据直接量出的目标(骨架几何量),不能配 VLM 猜的
        self.aux_concat = bool(getattr(hparams.model, "aux_concat", False))
        n_aux = _count_aux(getattr(hparams, "data", None))
        self.aux_dim = n_aux if (self.w_aux > 0 or self.aux_concat) else 0
        if self.aux_dim > 0:
            self.video_cnn.blocks[-1].proj = nn.Identity()
            self.cls_head = nn.Linear(2048 + (self.aux_dim if self.aux_concat else 0), self.num_classes)
            self.aux_head = nn.Linear(2048, self.aux_dim) if self.w_aux > 0 else None

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        self.metrics = ClassificationMetrics(self.num_classes)

        # 测试期把预测存下来,交给 analysis/compare_concept_runs.py 与主方法同口径汇总
        self.test_video_names: list = []
        self.test_pred_list: List[torch.Tensor] = []
        self.test_label_list: List[torch.Tensor] = []

    def forward(self, x, aux=None):
        return self._forward(x, aux)[0]

    def _forward(self, x, aux=None):
        """(logits, aux_pred 或 None)。aux (B, A) 为特征级拼接的几何量。"""
        out = self.video_cnn(x)
        if self.aux_dim > 0:
            feat = torch.cat([out, aux.float()], dim=-1) if self.aux_concat else out
            return self.cls_head(feat), (self.aux_head(out) if self.aux_head is not None else None)
        return out, None

    def _aux_loss(self, aux_pred, batch, logits):
        aux_target = batch.get("aux")
        if aux_pred is None or aux_target is None or self.w_aux <= 0:
            return logits.new_zeros(())
        return F.mse_loss(aux_pred, aux_target.detach().float())

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        
        # prepare the input and label
        video = batch["video"].detach()  # b, c, t, h, w
        label = batch["label"].detach().float().squeeze()  # b
        # sample_info = batch["info"] # b is the video instance number

        b, c, t, h, w = video.shape

        video_preds, aux_pred = self._forward(video, batch.get("aux"))
        video_preds_softmax = torch.softmax(video_preds, dim=1)

        # check shape 
        if b == 1:
            label = label.unsqueeze(0)
            
        assert label.shape[0] == video_preds.shape[0]

        cls_loss = F.cross_entropy(video_preds, label.long())
        aux_loss = self._aux_loss(aux_pred, batch, video_preds)
        loss = cls_loss + self.w_aux * aux_loss

        self.log("train/loss", loss, on_epoch=True, on_step=True)
        self.log("train/loss_cls", cls_loss, on_epoch=True, on_step=False, batch_size=b)
        self.log("train/loss_aux", aux_loss, on_epoch=True, on_step=False, batch_size=b)
        self.metrics.log(self, "train", video_preds_softmax, label, batch_size=b)

        return loss


    def validation_step(self, batch: torch.Tensor, batch_idx: int):

        # input and model define
        video = batch["video"].detach()  # b, c, t, h, w
        label = batch["label"].detach().float().squeeze()  # b

        b, c, t, h, w = video.shape

        video_preds, aux_pred = self._forward(video, batch.get("aux"))
        video_preds_softmax = torch.softmax(video_preds, dim=1)

        if b == 1:
            label = label.unsqueeze(0)

        # check shape 
        assert label.shape[0] == b

        cls_loss = F.cross_entropy(video_preds, label.long())
        aux_loss = self._aux_loss(aux_pred, batch, video_preds)
        loss = cls_loss + self.w_aux * aux_loss

        self.log("val/loss", loss, on_epoch=True, on_step=True)
        self.log("val/loss_aux", aux_loss, on_epoch=True, on_step=False, batch_size=b)
        self.metrics.log(self, "val", video_preds_softmax, label, batch_size=b)

    def test_step(self, batch: torch.Tensor, batch_idx: int):

        # input and model define
        video = batch["video"].detach()  # b, c, t, h, w
        label = batch["label"].detach().float().squeeze()  # b

        b, c, t, h, w = video.shape

        video_preds, aux_pred = self._forward(video, batch.get("aux"))
        video_preds_softmax = torch.softmax(video_preds, dim=1)

        if b == 1:
            label = label.unsqueeze(0)

        # check shape 
        assert label.shape[0] == b

        loss = F.cross_entropy(video_preds, label.long())

        self.log("test/loss", loss, on_epoch=True, on_step=True)
        self.metrics.log(self, "test", video_preds_softmax, label, batch_size=b)

        self.test_video_names.extend(_expand_video_names(batch))
        self.test_pred_list.append(video_preds_softmax.detach().cpu())
        self.test_label_list.append(label.detach().long().cpu())

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

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                # verbose 在 torch 2.7 已被移除,带上它在超算环境会直接 TypeError
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.trainer.estimated_stepping_batches,
                ),
                "monitor": "train/loss",
            },
        }

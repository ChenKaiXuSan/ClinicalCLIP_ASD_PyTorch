#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: train_pose.py
Project: trainer
Author: Kaixu Chen
-----
Comment:
纯姿态基线的 Lightning 模块。只吃骨架关键点,不看像素。

用途是回答"临床先验来自骨架,那只用骨架够不够"——如果这个基线就能接近
concept 架构的精度,视频分支的必要性需要重新论证。
"""

import logging
from typing import Dict

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from models.pose_stgcn import PoseSTGCN
from utils.helper import save_helper

logger = logging.getLogger(__name__)


class PoseModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()

        loss_cfg = getattr(hparams, "loss", {})
        self.lr = float(getattr(loss_cfg, "lr", 1e-4))
        self.weight_decay = float(getattr(loss_cfg, "weight_decay", 0.001))
        self.num_classes = int(getattr(hparams.model, "model_class_num", 3))

        self.model = PoseSTGCN(hparams)
        # 与 concept 一致:torchmetrics 默认 macro,即平衡准确率
        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)

        self.save_root = hparams.log_path

    def forward(self, pose: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(pose)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        pose = batch["pose"].detach()
        label = batch["label"].detach().long()

        logits = self.model(pose)["logits"]
        loss = F.cross_entropy(logits, label)
        probs = torch.softmax(logits, dim=1)

        on_step = stage == "train"
        bs = label.size(0)
        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=on_step,
                 batch_size=bs, prog_bar=True)
        self.log_dict(
            {
                f"{stage}/video_acc": self._accuracy(probs, label),
                f"{stage}/video_f1_score": self._f1_score(probs, label),
            },
            on_epoch=True, on_step=on_step, batch_size=bs,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def on_test_start(self) -> None:
        self.test_pred_list: list[torch.Tensor] = []
        self.test_label_list: list[torch.Tensor] = []

    def test_step(self, batch, batch_idx):
        pose = batch["pose"].detach()
        label = batch["label"].detach().long()

        logits = self.model(pose)["logits"]
        probs = torch.softmax(logits, dim=1)

        self.log("test/loss", F.cross_entropy(logits, label),
                 on_epoch=True, on_step=False, batch_size=label.size(0))
        self.log_dict(
            {
                "test/video_acc": self._accuracy(probs, label),
                "test/video_f1_score": self._f1_score(probs, label),
            },
            on_epoch=True, on_step=False, batch_size=label.size(0),
        )
        self.test_pred_list.append(probs.detach().cpu())
        self.test_label_list.append(label.detach().cpu())

    def _fold_name(self) -> str:
        root_dir = getattr(self.logger, "root_dir", None) if self.logger else None
        return root_dir.split("/")[-1] if root_dir else "fold"

    def on_test_epoch_end(self) -> None:
        save_helper(
            all_pred=self.test_pred_list,
            all_label=self.test_label_list,
            fold=self._fold_name(),
            save_path=self.save_root,
            num_class=self.num_classes,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train/loss"},
        }

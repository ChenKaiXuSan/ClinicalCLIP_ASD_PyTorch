#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: train_clip_align.py
Project: trainer
Created Date: 2026-02-03
Author: OpenAI Assistant
-----
Comment:
Lightning module for CLIP-style video-attention alignment.
"""

import logging
from typing import Dict

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)

from project.models.clip_align import VideoAttentionCLIP, clip_contrastive_loss

logger = logging.getLogger(__name__)


class CLIPAlignModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()

        self.lr = float(getattr(hparams.optimizer, "lr", 1e-3))
        self.num_classes = int(getattr(hparams.model, "model_class_num", 3))
        self.clip_loss_weight = float(getattr(hparams.loss, "clip_weight", 1.0))
        self.temperature = float(getattr(hparams.loss, "clip_temperature", 0.07))
        self.classifier_source = getattr(hparams.model, "clip_classifier_source", "video")

        self.model = VideoAttentionCLIP(hparams)

        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._precision = MulticlassPrecision(num_classes=self.num_classes)
        self._recall = MulticlassRecall(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)

    def forward(self, video: torch.Tensor, attn_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(video, attn_map, classifier_source=self.classifier_source)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        video = batch["video"].detach()
        attn_map = batch["attn_map"].detach()
        label = batch["label"].detach().long()

        outputs = self(video, attn_map)
        logits = outputs["logits"]

        cls_loss = F.cross_entropy(logits, label)
        align_loss = clip_contrastive_loss(
            outputs["video_embed"],
            outputs["attn_embed"],
            temperature=self.temperature,
        )

        loss = cls_loss + self.clip_loss_weight * align_loss

        probs = torch.softmax(logits, dim=1)
        metrics = {
            f"{stage}/video_acc": self._accuracy(probs, label),
            f"{stage}/video_precision": self._precision(probs, label),
            f"{stage}/video_recall": self._recall(probs, label),
            f"{stage}/video_f1_score": self._f1_score(probs, label),
        }

        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=stage == "train", batch_size=label.size(0))
        self.log(f"{stage}/loss_cls", cls_loss, on_epoch=True, on_step=stage == "train", batch_size=label.size(0))
        self.log(f"{stage}/loss_clip", align_loss, on_epoch=True, on_step=stage == "train", batch_size=label.size(0))
        self.log_dict(metrics, on_epoch=True, on_step=stage == "train", batch_size=label.size(0))

        logger.info(f"{stage} loss: {loss.item():.4f}")
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        self._shared_step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.trainer.estimated_stepping_batches
                ),
                "monitor": "train/loss",
            },
        }

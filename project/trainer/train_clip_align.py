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
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from project.models.clip_align import (
    VideoAttentionCLIP,
    clip_contrastive_loss_with_scale,
)
from project.utils.helper import save_helper

logger = logging.getLogger(__name__)


class CLIPAlignModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()

        self.lr = float(getattr(hparams.optimizer, "lr", 1e-3))
        self.num_classes = int(getattr(hparams.model, "model_class_num", 3))
        self.clip_loss_weight = float(getattr(hparams.loss, "clip_weight", 1.0))
        self.token_loss_weight = float(getattr(hparams.model, "lambda_token", 0.0))
        self.temperature = float(getattr(hparams.loss, "clip_temperature", 0.07))
        self.classifier_source = getattr(
            hparams.model, "clip_classifier_source", "video"
        )

        self.model = VideoAttentionCLIP(hparams)

        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._precision = MulticlassPrecision(num_classes=self.num_classes)
        self._recall = MulticlassRecall(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)

        self.save_root = hparams.log_path

    def forward(
        self, video: torch.Tensor, attn_map: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.model(video, attn_map, classifier_source=self.classifier_source)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        video = batch["video"].detach()
        attn_map = batch["attn_map"].detach()
        label = batch["label"].detach().long()

        outputs = self(video, attn_map)
        logits = outputs["logits"]

        cls_loss = F.cross_entropy(logits, label)
        align_loss = clip_contrastive_loss_with_scale(
            outputs["video_embed"],
            outputs["attn_embed"],
            logit_scale=self.model.logit_scale
        )
        token_align_loss = outputs.get("token_align_loss")
        if token_align_loss is None:
            token_align_loss = torch.tensor(0.0, device=logits.device)

        loss = (
            cls_loss
            + self.clip_loss_weight * align_loss
            + self.token_loss_weight * token_align_loss
        )

        probs = torch.softmax(logits, dim=1)
        metrics = {
            f"{stage}/video_acc": self._accuracy(probs, label),
            f"{stage}/video_precision": self._precision(probs, label),
            f"{stage}/video_recall": self._recall(probs, label),
            f"{stage}/video_f1_score": self._f1_score(probs, label),
        }

        self.log(
            f"{stage}/loss",
            loss,
            on_epoch=True,
            on_step=stage == "train",
            batch_size=label.size(0),
        )
        self.log(
            f"{stage}/loss_cls",
            cls_loss,
            on_epoch=True,
            on_step=stage == "train",
            batch_size=label.size(0),
        )
        self.log(
            f"{stage}/loss_clip",
            align_loss,
            on_epoch=True,
            on_step=stage == "train",
            batch_size=label.size(0),
        )
        self.log(
            f"{stage}/loss_token",
            token_align_loss,
            on_epoch=True,
            on_step=stage == "train",
            batch_size=label.size(0),
        )
        self.log_dict(
            metrics, on_epoch=True, on_step=stage == "train", batch_size=label.size(0)
        )

        logger.info(f"{stage} loss: {loss.item():.4f}")
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        self._shared_step(batch, "val")

    ##############
    # test step
    ##############
    # the order of the hook function is:
    # on_test_start -> test_step -> on_test_batch_end -> on_test_epoch_end -> on_test_end

    def on_test_start(self) -> None:
        """hook function for test start"""

        self.test_pred_list: list[torch.Tensor] = []
        self.test_label_list: list[torch.Tensor] = []
        self.test_video_embed_list: list[torch.Tensor] = []
        self.test_attn_embed_list: list[torch.Tensor] = []

        logger.info("test start")

    def on_test_end(self) -> None:
        """hook function for test end"""
        logger.info("test end")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        video = batch["video"].detach()
        attn_map = batch["attn_map"].detach()
        label = batch["label"].detach().long()

        outputs = self(video, attn_map)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1)

        loss = F.cross_entropy(logits, label)

        self.log(
            "test/loss", loss, on_epoch=True, on_step=False, batch_size=label.size(0)
        )

        metric_dict = {
            "test/video_acc": self._accuracy(probs, label),
            "test/video_precision": self._precision(probs, label),
            "test/video_recall": self._recall(probs, label),
            "test/video_f1_score": self._f1_score(probs, label),
        }
        self.log_dict(
            metric_dict, on_epoch=True, on_step=False, batch_size=label.size(0)
        )

        self.test_pred_list.append(probs.detach().cpu())
        self.test_label_list.append(label.detach().cpu())
        self.test_video_embed_list.append(outputs["video_embed"].detach().cpu())
        self.test_attn_embed_list.append(outputs["attn_embed"].detach().cpu())

        return probs, logits

    def on_test_epoch_end(self) -> None:
        """hook function for test epoch end"""

        if self.test_video_embed_list and self.test_attn_embed_list:
            video_embed = torch.cat(self.test_video_embed_list, dim=0)
            attn_embed = torch.cat(self.test_attn_embed_list, dim=0)
            sims = video_embed @ attn_embed.t()

            k = min(5, sims.size(1))
            targets = torch.arange(sims.size(0))

            topk_v2a = sims.topk(k, dim=1).indices
            r1_v2a = (topk_v2a[:, :1] == targets[:, None]).any(dim=1).float().mean()
            r5_v2a = (topk_v2a == targets[:, None]).any(dim=1).float().mean()

            topk_a2v = sims.topk(k, dim=0).indices
            r1_a2v = (topk_a2v[:1, :] == targets[None, :]).any(dim=0).float().mean()
            r5_a2v = (topk_a2v == targets[None, :]).any(dim=0).float().mean()

            self.log("test/retrieval_r1_v2a", r1_v2a, on_epoch=True, on_step=False)
            self.log("test/retrieval_r5_v2a", r5_v2a, on_epoch=True, on_step=False)
            self.log("test/retrieval_r1_a2v", r1_a2v, on_epoch=True, on_step=False)
            self.log("test/retrieval_r5_a2v", r5_a2v, on_epoch=True, on_step=False)

            if self.test_pred_list and self.test_label_list:
                probs = torch.cat(self.test_pred_list, dim=0)
                labels = torch.cat(self.test_label_list, dim=0)
                preds = probs.argmax(dim=1)
                correct = preds.eq(labels).float()
                sim_diag = sims.diag()

                if correct.bool().any():
                    sim_correct = sim_diag[correct.bool()].mean()
                else:
                    sim_correct = torch.tensor(0.0)

                if (~correct.bool()).any():
                    sim_incorrect = sim_diag[~correct.bool()].mean()
                else:
                    sim_incorrect = torch.tensor(0.0)

                sim_gap = sim_correct - sim_incorrect
                self.log("test/align_sim_correct", sim_correct, on_epoch=True, on_step=False)
                self.log("test/align_sim_incorrect", sim_incorrect, on_epoch=True, on_step=False)
                self.log("test/align_sim_gap", sim_gap, on_epoch=True, on_step=False)

                if sim_diag.numel() > 1 and correct.std() > 0:
                    corr = torch.corrcoef(
                        torch.stack([sim_diag.float(), correct.float()])
                    )[0, 1]
                    self.log("test/align_sim_corr", corr, on_epoch=True, on_step=False)

        # save the metrics to file
        save_helper(
            all_pred=self.test_pred_list,
            all_label=self.test_label_list,
            fold=self.logger.root_dir.split("/")[-1] if self.logger else "fold",
            save_path=self.save_root,
            num_class=self.num_classes,
        )

        logger.info("test epoch end")

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

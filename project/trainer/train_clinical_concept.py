#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: train_clinical_concept.py
Project: trainer
Author: Kaixu Chen
-----
Comment:
Clinical Concept Grounding 的 Lightning 模块。

推理只吃视频,医生标注仅参与训练期的两项监督(区域存在性 + 空间 grounding)。
因此"可解释性"是模型的预测量而非输入,可以直接与留出的医生标注对分:
  test/region_f1      预测关注区域与医生标注的一致度
  test/attn_iou       模型注意力与医生区域图的软 IoU
  test/attn_acc_gap   注意力对齐得好的样本是否分类也更准
"""

import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from models.clinical_concept import (
    REGIONS,
    ClinicalConceptNet,
    concept_contrastive_loss,
    grounding_loss,
    presence_loss,
)
from utils.helper import save_helper

logger = logging.getLogger(__name__)


class ClinicalConceptModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()

        loss_cfg = getattr(hparams, "loss", {})
        self.lr = float(getattr(loss_cfg, "lr", 0.01))
        self.weight_decay = float(getattr(loss_cfg, "weight_decay", 0.001))
        self.w_presence = float(getattr(loss_cfg, "presence_weight", 1.0))
        self.w_ground = float(getattr(loss_cfg, "grounding_weight", 1.0))
        self.w_concept = float(getattr(loss_cfg, "concept_weight", 0.1))

        self.num_classes = int(getattr(hparams.model, "model_class_num", 3))
        # 消融用:把医生区域换成随机区域,检验增益是否真来自临床知识
        self.shuffle_region = bool(getattr(hparams.model, "shuffle_region", False))

        self.model = ClinicalConceptNet(hparams)

        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)

        self.save_root = hparams.log_path

    def forward(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(video)

    def _maybe_shuffle(self, region_map, region_target):
        """消融:沿区域维度逐样本置换,等价于"同一副骨架、换一个区域"。

        不能在 batch 维度打乱——batch_size=1 时一个 batch 全是同一条视频的
        gait 段,区域标签本来就相同,打乱不起作用。

        若置换后分类精度不掉,说明增益来自姿态渲染而非医生选的区域,论文的
        核心主张就不成立。
        """
        if not self.shuffle_region or region_target is None:
            return region_map, region_target

        b, r = region_target.shape
        perm = torch.argsort(
            torch.rand(b, r, device=region_target.device), dim=1
        )  # (B, R) 每个样本一个独立置换
        target = region_target.gather(1, perm)
        index = perm.view(b, r, *([1] * (region_map.dim() - 2))).expand_as(region_map)
        return region_map.gather(1, index), target

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        video = batch["video"].detach()
        label = batch["label"].detach().long()
        region_map = batch.get("region_map")
        region_target = batch.get("region_target")
        if region_map is not None:
            region_map = region_map.detach()
            region_target = region_target.detach().float()
        region_map, region_target = self._maybe_shuffle(region_map, region_target)

        outputs = self.model(video)
        logits = outputs["logits"]

        cls_loss = F.cross_entropy(logits, label)
        zero = logits.new_zeros(())

        if region_target is not None:
            pres_loss = presence_loss(outputs["region_logits"], region_target)
            ground = grounding_loss(outputs["attn"], region_map, region_target)
        else:
            pres_loss = ground = zero

        concept = concept_contrastive_loss(outputs["region_feat"], outputs["concepts"])

        loss = (
            cls_loss
            + self.w_presence * pres_loss
            + self.w_ground * ground
            + self.w_concept * concept
        )

        probs = torch.softmax(logits, dim=1)
        on_step = stage == "train"
        bs = label.size(0)
        for name, value in [
            ("loss", loss),
            ("loss_cls", cls_loss),
            ("loss_presence", pres_loss),
            ("loss_grounding", ground),
            ("loss_concept", concept),
        ]:
            self.log(
                f"{stage}/{name}", value, on_epoch=True, on_step=on_step,
                batch_size=bs, prog_bar=name in ("loss", "loss_cls"),
            )
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

    ##############
    # test step
    ##############

    def on_test_start(self) -> None:
        self.test_pred_list: list[torch.Tensor] = []
        self.test_label_list: list[torch.Tensor] = []
        self.test_region_pred: list[torch.Tensor] = []
        self.test_region_true: list[torch.Tensor] = []
        self.test_attn_iou: list[torch.Tensor] = []

    @staticmethod
    def _soft_iou(attn: torch.Tensor, region_map: torch.Tensor, target: torch.Tensor):
        """模型注意力与医生区域图的软 IoU,只在医生标注过的区域上计算。"""
        b, r = attn.shape[:2]
        if region_map.shape[2:] != attn.shape[2:]:
            region_map = F.interpolate(
                region_map.reshape(b * r, 1, *region_map.shape[2:]),
                size=attn.shape[2:], mode="trilinear", align_corners=False,
            ).reshape(b, r, *attn.shape[2:])

        a = attn.reshape(b, r, -1)
        m = region_map.reshape(b, r, -1)
        a = a / a.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        m = m / m.amax(dim=-1, keepdim=True).clamp_min(1e-6)

        inter = torch.minimum(a, m).sum(-1)
        union = torch.maximum(a, m).sum(-1).clamp_min(1e-6)
        iou = inter / union

        weight = target * (m.sum(-1) > 1e-6)
        if weight.sum() <= 0:
            return None
        return (iou * weight).sum() / weight.sum()

    def test_step(self, batch, batch_idx):
        video = batch["video"].detach()
        label = batch["label"].detach().long()
        region_map = batch.get("region_map")
        region_target = batch.get("region_target")

        outputs = self.model(video)
        logits = outputs["logits"]
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

        if region_target is not None:
            region_target = region_target.detach().float()
            self.test_region_pred.append(
                outputs["region_logits"].sigmoid().detach().cpu()
            )
            self.test_region_true.append(region_target.cpu())
            iou = self._soft_iou(outputs["attn"].detach(), region_map.detach(), region_target)
            if iou is not None:
                self.test_attn_iou.append(iou.detach().cpu())

    def on_test_epoch_end(self) -> None:
        if self.test_region_pred:
            pred = torch.cat(self.test_region_pred) > 0.5
            # 软标签 >= 0.5 表示至少一位医生提到该区域
            true = torch.cat(self.test_region_true) >= 0.5
            tp = (pred & true).sum().float()
            precision = tp / pred.sum().clamp_min(1)
            recall = tp / true.sum().clamp_min(1)
            f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
            self.log("test/region_precision", precision, on_epoch=True)
            self.log("test/region_recall", recall, on_epoch=True)
            self.log("test/region_f1", f1, on_epoch=True)

        if self.test_attn_iou:
            self.log("test/attn_iou", torch.stack(self.test_attn_iou).mean(), on_epoch=True)

        fold_name = self.logger.root_dir.split("/")[-1] if self.logger else "fold"
        save_helper(
            all_pred=self.test_pred_list,
            all_label=self.test_label_list,
            fold=fold_name,
            save_path=self.save_root,
            num_class=self.num_classes,
        )

        if self.test_region_pred:
            out_path = Path(self.save_root) / "region"
            out_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "region_pred": torch.cat(self.test_region_pred),
                    "region_true": torch.cat(self.test_region_true),
                    "regions": REGIONS,
                },
                out_path / f"{fold_name}_region.pt",
            )
            logger.info("saved region predictions to %s", out_path)

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

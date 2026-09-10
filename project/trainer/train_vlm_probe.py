#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: train_vlm_probe.py
Project: trainer
Author: Kaixu Chen
-----
Comment:
V0_vlm_probe:冻结 VLM 视觉塔 + 线性探针,**不带任何临床先验**。

它回答两个问题,决定 VLM 这条线值不值得往下做:
  1. 通用 VLM 特征对侧视步态里的 ASD 信号敏感吗?(与 B0_3dcnn 患者级对比)
  2. findings.md 确认的瓶颈是训练方差(跨种子 0.125)。只训一个线性头,方差是否明显更小?

数据侧与 concept 系列完全一致(同一采样计划、同一划分),测试期同样存段级预测
供 analysis/patient_level_stats.py 做患者级汇总。region_map 只用来算一个
"无监督注意力"的对齐参照(attn 池化时才有),不参与训练。
"""
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from models.vlm_encoder import VLMTokenEncoder
from utils.helper import save_helper
from utils.metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


def _expand_video_names(batch) -> list:
    names = []
    for item in batch.get("info", []):
        names.extend([item["video_name"]] * int(item["num_chunks"]))
    return names


class VLMProbe(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        cfg = hparams.model
        if str(getattr(cfg, "token_backbone", "vlm")) != "vlm":
            raise ValueError("vlm_probe 只配 model.token_backbone=vlm")
        self.encoder = VLMTokenEncoder(
            backend=str(getattr(cfg, "vlm_backend", "siglip2")),
            model_name=str(getattr(cfg, "vlm_name", "google/siglip2-so400m-patch16-384")),
            hidden_dim=int(getattr(cfg, "concept_embed_dim", 256)),
            img_size=getattr(cfg, "vlm_img_size", None),
            trainable_blocks=int(getattr(cfg, "vlm_trainable_blocks", 0)),
            forward_chunk=int(getattr(cfg, "vlm_forward_chunk", 64)),
        )
        d = self.encoder.token_dim
        self.pool = str(getattr(cfg, "probe_pool", "mean"))
        if self.pool == "attn":
            self.query = nn.Parameter(torch.randn(1, d) * 0.02)
            self.key = nn.Linear(d, d, bias=False)
        elif self.pool != "mean":
            raise ValueError(f"probe_pool 只能是 mean/attn,收到 {self.pool}")
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, int(getattr(cfg, "model_class_num", 2))),
        )

    def forward(
        self,
        video: Optional[torch.Tensor] = None,
        raw_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # 探针直接吃投影前的 token(token_dim 维),token_proj 不用
        if raw_tokens is None:
            raw_tokens = self.encoder.encode_raw(video)
        raw_tokens = raw_tokens.float()
        b, d = raw_tokens.shape[:2]
        flat = raw_tokens.flatten(2).transpose(1, 2)  # (B, N, d)
        attn = None
        if self.pool == "mean":
            feat = flat.mean(dim=1)
        else:
            q = F.normalize(self.query, dim=-1)  # (1, d)
            k = F.normalize(self.key(flat), dim=-1)  # (B, N, d)
            attn = torch.einsum("qd,bnd->bn", q, k).mul(20.0).softmax(dim=-1)
            feat = torch.einsum("bn,bnd->bd", attn, flat)
            attn = attn.view(b, 1, *raw_tokens.shape[2:])  # (B,1,T',h,w)
        return {"logits": self.head(feat), "attn": attn}


class VLMProbeModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        loss_cfg = getattr(hparams, "loss", {})
        self.lr = float(getattr(loss_cfg, "lr", 1e-4))
        self.weight_decay = float(getattr(loss_cfg, "weight_decay", 0.001))
        self.num_classes = int(getattr(hparams.model, "model_class_num", 2))
        self.model = VLMProbe(hparams)
        self.metrics = ClassificationMetrics(self.num_classes)
        self.save_root = hparams.log_path

    def forward(self, video=None, raw_tokens=None):
        return self.model(video, raw_tokens=raw_tokens)

    @staticmethod
    def _inputs(batch) -> dict:
        video = batch.get("video")
        tokens = batch.get("tokens")
        return {
            "video": video.detach() if video is not None else None,
            "raw_tokens": tokens.detach() if tokens is not None else None,
        }

    def _shared_step(self, batch, stage: str):
        label = batch["label"].detach().long()
        logits = self.model(**self._inputs(batch))["logits"]
        loss = F.cross_entropy(logits, label)
        bs = label.size(0)
        self.log(
            f"{stage}/loss", loss, on_epoch=True, on_step=stage == "train",
            batch_size=bs, prog_bar=True,
        )
        self.metrics.log(self, stage, torch.softmax(logits, dim=1), label, batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def on_test_start(self) -> None:
        self.test_video_names: list = []
        self.test_pred_list: list = []
        self.test_label_list: list = []

    def test_step(self, batch, batch_idx):
        label = batch["label"].detach().long()
        logits = self.model(**self._inputs(batch))["logits"]
        probs = torch.softmax(logits, dim=1)
        self.log("test/loss", F.cross_entropy(logits, label),
                 on_epoch=True, on_step=False, batch_size=label.size(0))
        self.metrics.log(self, "test", probs, label, batch_size=label.size(0))
        self.test_video_names.extend(_expand_video_names(batch))
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
            all_video_name=self.test_video_names,
        )

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, int(self.trainer.max_epochs or 1))
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

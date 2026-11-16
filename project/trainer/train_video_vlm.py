#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: train_video_vlm.py
Project: trainer
Author: Kaixu Chen
-----
Comment:
视频 + VLM 双分支(model.backbone=video_vlm)的训练 / 验证 / 测试流程。结构见 models/video_vlm_fusion.py。

优化器、学习率、调度、指标与预测落盘格式同 SingleModule(B0)/ RegionBranchModule,所以
analysis/seed_summary.py、seed_metrics_table.py、uncertainty_fusion.py、region_branch_summary.py 可以直接读:
融合预测存 best_preds/<fold>_pred.pt,逐分支概率存 <fold>_branch_pred.pt + <fold>_branch_names.json。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from models.video_vlm_fusion import VideoVLMFusionNet
from trainer.train_res_3dcnn import _expand_video_names
from utils.helper import save_helper
from utils.metrics import ClassificationMetrics


class VideoVLMModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.lr = hparams.loss.lr
        self.weight_decay = float(getattr(hparams.loss, "weight_decay", 0.001))
        self.num_classes = hparams.model.model_class_num
        self.save_root = hparams.log_path

        self.net = VideoVLMFusionNet(hparams)
        self.branch_names = list(self.net.branch_names)

        self.save_hyperparameters()

        self.metrics = ClassificationMetrics(self.num_classes)

        self.test_video_names: list = []
        self.test_pred_list: List[torch.Tensor] = []
        self.test_label_list: List[torch.Tensor] = []
        self.test_branch_list: List[torch.Tensor] = []

    def forward(self, video, tokens):
        return self.net(video, tokens)["fused_prob"]

    def _shared_step(self, batch, stage: str):
        video = batch["video"].detach() if self.net.use_video else None
        tokens = batch["tokens"].detach() if self.net.use_vlm else None
        label = batch["label"].detach().long().view(-1)
        b = label.shape[0]

        out = self.net(video, tokens)
        branch_logits = out["branch_logits"]  # (B, n, C)
        assert branch_logits.shape[0] == b, f"段数与标签数不一致: {branch_logits.shape[0]} vs {b}"

        branch_ce = torch.stack(
            [F.cross_entropy(branch_logits[:, i].float(), label) for i in range(branch_logits.shape[1])]
        )
        if self.net.fusion == "feature":
            fuse_ce = F.cross_entropy(out["fused_logits"].float(), label)
            # 探针损失只训练各分支的线性头(特征已 detach),不改变融合模型
            loss = fuse_ce + branch_ce.mean()
        else:
            fuse_ce = branch_ce.new_zeros(())
            loss = branch_ce.mean()

        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=True, batch_size=b)
        if self.net.fusion == "feature":
            self.log(f"{stage}/loss_fuse", fuse_ce, on_epoch=True, on_step=False, batch_size=b)
        for name, ce in zip(self.branch_names, branch_ce):
            self.log(f"{stage}/loss_{name}", ce, on_epoch=True, on_step=False, batch_size=b)
        self.metrics.log(self, stage, out["fused_prob"], label, batch_size=b)
        return loss, out, label

    def training_step(self, batch, batch_idx: int):
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def on_train_epoch_end(self) -> None:
        if torch.cuda.is_available():
            self.log("train/gpu_mem_gb", torch.cuda.max_memory_allocated() / 2**30, on_epoch=True, on_step=False)

    def validation_step(self, batch, batch_idx: int):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int):
        _, out, label = self._shared_step(batch, "test")
        self.test_video_names.extend(_expand_video_names(batch))
        self.test_pred_list.append(out["fused_prob"].detach().float().cpu())
        self.test_label_list.append(label.detach().cpu())
        self.test_branch_list.append(torch.softmax(out["branch_logits"].detach().float(), dim=-1).cpu())

    def on_test_epoch_end(self) -> None:
        fold = self._fold_name()
        save_helper(
            all_pred=self.test_pred_list,
            all_label=self.test_label_list,
            fold=fold,
            save_path=self.save_root,
            num_class=self.num_classes,
            all_video_name=self.test_video_names,
        )
        # 逐分支概率 (N, n_branch, C),顺序同 branch_names;与 <fold>_pred.pt 逐行对应
        out_dir = Path(self.save_root) / "best_preds"
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.cat(self.test_branch_list, dim=0), out_dir / f"{fold}_branch_pred.pt")
        with open(out_dir / f"{fold}_branch_names.json", "w") as f:
            json.dump(self.branch_names, f)

    def _fold_name(self) -> str:
        """从 logger 的 root_dir 取折号;fast_dev_run 下 logger 被禁用,root_dir 为 None。"""
        root_dir = getattr(self.logger, "root_dir", None) if self.logger else None
        return root_dir.split("/")[-1] if root_dir else "fold"

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.trainer.estimated_stepping_batches,
                ),
                "monitor": "train/loss",
            },
        }

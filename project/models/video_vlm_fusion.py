#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: video_vlm_fusion.py
Project: models
Author: Kaixu Chen
-----
Comment:
视频 + VLM 双分支(model.backbone=video_vlm)。

  视频分支  原始视频 -> slow_r50(Kinetics 预训练,与 B0 相同) -> 2048 维
  VLM 分支  冻结 VLM 的离线 token 缓存(视频与医生知识指令一起输入 VLM 得到,data.feature_cache_dir)
            -> 时空均值 -> LayerNorm -> Linear -> GELU -> vvf_vlm_dim 维
  融合      feature: 两路特征各自 LayerNorm 后拼接 -> 线性分类头(主损失);
                     每路另接一个作用在 **detach 后特征**上的线性探针,只为读出各分支单独的成绩
            score:   两路各自一个线性头,损失 = 两路交叉熵均值,预测 = 概率平均

vvf_branches = video / vlm 时只保留一条分支(同一套数据与训练流程下的单分支对照)。
VLM 不参与训练,也不在显存里;token 维度从缓存目录的 manifest.json 读。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from models.region_branch import FEATURE_DIM, _make_backbone


class VideoVLMFusionNet(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        cfg = hparams.model
        self.num_classes = int(cfg.model_class_num)
        self.fusion = str(getattr(cfg, "vvf_fusion", "feature"))
        which = str(getattr(cfg, "vvf_branches", "both"))
        if self.fusion not in ("feature", "score") or which not in ("both", "video", "vlm"):
            raise ValueError(f"vvf_fusion={self.fusion} / vvf_branches={which} 不支持")
        self.use_video, self.use_vlm = which in ("both", "video"), which in ("both", "vlm")
        self.branch_names: List[str] = (["video"] if self.use_video else []) + (["vlm"] if self.use_vlm else [])

        dims = []
        if self.use_video:
            self.video_backbone = _make_backbone(hparams)
            self.video_norm = nn.LayerNorm(FEATURE_DIM)
            dims.append(FEATURE_DIM)
        if self.use_vlm:
            cache = str(getattr(hparams.data, "feature_cache_dir", "") or "")
            if not cache:
                raise ValueError("video_vlm 的 VLM 分支需要 data.feature_cache_dir(离线特征缓存)")
            token_dim = int(json.loads((Path(cache) / "manifest.json").read_text())["token_dim"])
            d = int(getattr(cfg, "vvf_vlm_dim", 256))
            self.vlm_proj = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, d), nn.GELU())
            self.vlm_norm = nn.LayerNorm(d)
            dims.append(d)
        self.branch_heads = nn.ModuleList([nn.Linear(d, self.num_classes) for d in dims])
        self.fuse_head = nn.Linear(sum(dims), self.num_classes) if self.fusion == "feature" else None

    def forward(self, video: Optional[torch.Tensor], tokens: Optional[torch.Tensor]) -> Dict[str, Optional[torch.Tensor]]:
        feats = []
        if self.use_video:
            feats.append(self.video_norm(self.video_backbone(video).flatten(1)))
        if self.use_vlm:
            pooled = tokens.float().mean(dim=(2, 3, 4))          # (B, token_dim)
            feats.append(self.vlm_norm(self.vlm_proj(pooled)))
        if self.fusion == "feature":
            fused_logits = self.fuse_head(torch.cat(feats, dim=-1))
            branch_logits = torch.stack([h(f.detach()) for h, f in zip(self.branch_heads, feats)], dim=1)
            fused_prob = torch.softmax(fused_logits.float(), dim=-1)
        else:
            fused_logits = None
            branch_logits = torch.stack([h(f) for h, f in zip(self.branch_heads, feats)], dim=1)
            fused_prob = torch.softmax(branch_logits.float(), dim=-1).mean(dim=1)
        return {"branch_logits": branch_logits, "fused_logits": fused_logits, "fused_prob": fused_prob}

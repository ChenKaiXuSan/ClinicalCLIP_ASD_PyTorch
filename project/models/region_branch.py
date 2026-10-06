#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: region_branch.py
Project: models
Author: Kaixu Chen
-----
Comment:
多分支区域模型(model.backbone=region)。

每条分支只看一个按骨架裁出的"身体段"小视频(dataloader/region_crops.py),另有一条看整帧的全身分支;
各分支最后在**分数层**(默认)或**特征层**融合。医生注意力以群体级先验的形式进入:它决定开哪几段
(model.region_set),推理时不需要任何医生标注。

设计依据(docs/vlm_backbone.md):
  * 这个项目里所有可学习的"软"注入都被绕开或退化(grounding 监督精度不变、sigmoid 门控学成常数、
    几何量辅助回归无效),而决策级融合 9/9 有效。所以默认融合放在分数层、权重固定:每条分支各自被
    监督成一个分类器,融合是概率的固定加权平均,没有可以退化的参数。
  * 信号在区域之间的关系里(躯干前倾单量 0.707,头前伸单量 0.409 但去掉它掉得最多)。所以默认划分
    是按"测量"切的重叠身体段(measure),并保留一条全身分支;按医生区域字面切(parts)作为对照。
  * 几十名患者的规模下参数量是负担:各分支默认**共享同一个骨干**,只是各有一个线性头。参数量与 B0
    相同,而骨干每步看到的样本变成 (1 + R) 倍。

两种融合:
  score    每条分支独立的线性头,损失 = 各分支交叉熵的均值;预测 = softmax 概率的固定加权平均。
  feature  各分支特征拼接后过一个线性头(损失只有这一项回传到骨干)。各分支另有一个接在
           **detach 后特征**上的线性探针,只为读出"这一段单独有多少证据",不影响骨干和融合头。

只用全身分支(region_set=none)时结构与 B0_3dcnn 等价(AdaptiveAvgPool3d(1) 在 224 输入上与
slow_r50 原来的 AvgPool3d((8,7,7)) 输出相同),可用来确认新链路没有引入回归。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from dataloader.region_crops import resolve_region_set
from models.make_model import MakeVideoModule

FEATURE_DIM = 2048


def _make_backbone(hparams) -> nn.Module:
    """slow_r50(Kinetics 预训练,与 B0 相同),输出 2048 维池化特征。

    头部的固定核池化换成自适应池化:区域小视频是 112×112,最后一层特征图只有 4×4,
    原来的 AvgPool3d((8,7,7)) 会因核大于输入而报错;224 输入下两者输出完全一致。
    """
    model = MakeVideoModule(hparams)()
    head = model.blocks[-1]
    head.pool = nn.AdaptiveAvgPool3d(1)
    head.proj = nn.Identity()
    return model


class RegionBranchNet(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        cfg = hparams.model
        self.num_classes = int(cfg.model_class_num)
        self.region_names: List[str] = resolve_region_set(getattr(cfg, "region_set", "measure"))
        self.use_global = bool(getattr(cfg, "region_use_global", True))
        self.fusion = str(getattr(cfg, "region_fusion", "score"))
        self.share = bool(getattr(cfg, "region_share_backbone", True))
        if self.fusion not in ("score", "feature"):
            raise ValueError(f"model.region_fusion 只支持 score / feature,收到 {self.fusion}")

        self.branch_names: List[str] = (["global"] if self.use_global else []) + list(self.region_names)
        n = len(self.branch_names)
        if n == 0:
            raise ValueError("没有任何分支:region_use_global=false 且 region_set 为空")

        self.backbones = nn.ModuleList([_make_backbone(hparams) for _ in range(1 if self.share else n)])
        self.heads = nn.ModuleList([nn.Linear(FEATURE_DIM, self.num_classes) for _ in range(n)])
        self.fuse_head = nn.Linear(FEATURE_DIM * n, self.num_classes) if self.fusion == "feature" else None

        # 分数层融合的固定权重(不可学习);缺省为均匀
        weights = getattr(cfg, "region_branch_weights", None)
        if weights is None:
            w = torch.ones(n)
        else:
            w = torch.tensor([float(x) for x in weights])
            if w.numel() != n:
                raise ValueError(f"region_branch_weights 需要 {n} 个值(分支顺序 {self.branch_names}),收到 {w.numel()}")
            if bool((w < 0).any()) or float(w.sum()) <= 0:
                raise ValueError("region_branch_weights 必须非负且不全为 0")
        self.register_buffer("branch_weights", w / w.sum())

    def _encode(self, branch: int, clip: torch.Tensor) -> torch.Tensor:
        return self.backbones[0 if self.share else branch](clip)

    def forward(self, video: Optional[torch.Tensor], region_clips: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """video (B, 3, T, H, W) 或 None;region_clips (B, R, 3, T, S, S) 或 None。

        Returns:
            branch_logits (B, n, C)  各分支自己的 logits(feature 融合时来自 detach 特征上的探针)
            fused_prob    (B, C)     融合后的类别概率,指标与预测都用它
            fused_logits  (B, C)     仅 feature 融合;score 融合时为 None
        """
        feats: List[torch.Tensor] = []
        if self.use_global:
            if video is None:
                raise ValueError("region_use_global=true 但 batch 里没有 video")
            feats.append(self._encode(0, video))

        r = len(self.region_names)
        if r > 0:
            if region_clips is None or region_clips.shape[1] != r:
                got = None if region_clips is None else tuple(region_clips.shape)
                raise ValueError(f"需要 {r} 段区域小视频 {self.region_names},收到 {got}")
            b = region_clips.shape[0]
            offset = 1 if self.use_global else 0
            if self.share:
                # 所有区域段并成一个大 batch 过同一个骨干
                f = self._encode(0, region_clips.flatten(0, 1)).view(b, r, FEATURE_DIM)
                feats.extend(f.unbind(dim=1))
            else:
                feats.extend(self._encode(offset + i, region_clips[:, i]) for i in range(r))

        if self.fusion == "feature":
            fused_logits = self.fuse_head(torch.cat(feats, dim=-1))
            # 探针接在 detach 后的特征上:读出每段单独的证据强度,但不改变骨干和融合头学到的东西
            branch_logits = torch.stack([h(f.detach()) for h, f in zip(self.heads, feats)], dim=1)
            return {
                "branch_logits": branch_logits,
                "fused_logits": fused_logits,
                "fused_prob": torch.softmax(fused_logits.float(), dim=-1),
            }

        branch_logits = torch.stack([h(f) for h, f in zip(self.heads, feats)], dim=1)
        probs = torch.softmax(branch_logits.float(), dim=-1)
        fused_prob = (probs * self.branch_weights.view(1, -1, 1)).sum(dim=1)
        return {"branch_logits": branch_logits, "fused_logits": None, "fused_prob": fused_prob}

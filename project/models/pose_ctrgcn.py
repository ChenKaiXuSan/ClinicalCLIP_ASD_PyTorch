#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: pose_ctrgcn.py
Project: models
Author: Kaixu Chen
-----
Comment:
第二个骨架基线:CTR-GCN(Chen et al., ICCV 2021, "Channel-wise Topology Refinement Graph Convolution")。
ST-GCN 用固定的骨架邻接;CTR-GCN 在共享拓扑之上为每个通道学一份精化的拓扑(输入相关的 V×V 关联),
时间维用多尺度卷积(膨胀 1 / 2 + 最大池化 + 1×1)。它在 NTU 上比 ST-GCN 高约 5 个点,是"骨架端到端"里
更强的代表;若它也在随机水平,"表示学习吃全部骨架学不到"这个结论才站得住。

深度与 pose_stgcn.py 对齐(4 个块,64/64/128/128),同一输入 (B, T, 17, C)、同一 data_bn、同一分类头,
这样两者之间只差图卷积算子本身。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.pose_stgcn import COCO_EDGES, NUM_JOINTS

# COCO 17 点以鼻子(0)为根的有向树:parent -> child
_PARENT = {1: 0, 2: 0, 3: 1, 4: 2, 5: 0, 6: 0, 7: 5, 9: 7, 8: 6, 10: 8, 11: 5, 12: 6, 13: 11, 15: 13, 14: 12, 16: 14}


def _normalize_digraph(a: torch.Tensor) -> torch.Tensor:
    d = a.sum(0)
    dn = torch.where(d > 0, d.pow(-1), torch.zeros_like(d))
    return a @ torch.diag(dn)


def build_subsets(num_joints: int = NUM_JOINTS) -> torch.Tensor:
    """(3, V, V):自连接、向心(child -> parent)、离心。与 CTR-GCN 官方 graph 同一构造。"""
    eye = torch.eye(num_joints)
    inward = torch.zeros(num_joints, num_joints)
    for child, parent in _PARENT.items():
        inward[parent, child] = 1.0  # a[j, i] = 1 表示 i -> j
    outward = inward.t().clone()
    return torch.stack([eye, _normalize_digraph(inward), _normalize_digraph(outward)])


class CTRGC(nn.Module):
    """单个子集的通道拓扑精化图卷积。"""

    def __init__(self, in_ch: int, out_ch: int, rel_reduction: int = 8) -> None:
        super().__init__()
        rel = 8 if in_ch <= 4 else max(in_ch // rel_reduction, 8)
        self.conv1 = nn.Conv2d(in_ch, rel, 1)
        self.conv2 = nn.Conv2d(in_ch, rel, 1)
        self.conv3 = nn.Conv2d(in_ch, out_ch, 1)
        self.conv4 = nn.Conv2d(rel, out_ch, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, a: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # x (N, C, T, V)
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))          # (N, R, V, V)
        x1 = self.conv4(x1) * alpha + a.unsqueeze(0).unsqueeze(0)     # (N, C_out, V, V)
        return torch.einsum("ncuv,nctv->nctu", x1, x3)


class UnitGCN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, a: torch.Tensor) -> None:
        super().__init__()
        self.num_subset = a.shape[0]
        self.register_buffer("a", a.clone())
        self.convs = nn.ModuleList([CTRGC(in_ch, out_ch) for _ in range(self.num_subset)])
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_ch)
        self.down = nn.Identity() if in_ch == out_ch else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = None
        for i in range(self.num_subset):
            z = self.convs[i](x, self.a[i], self.alpha)
            y = z if y is None else y + z
        return self.relu(self.bn(y) + self.down(x))


class TemporalConv(nn.Module):
    def __init__(self, ch: int, kernel: int = 3, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        pad = (kernel + (kernel - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(ch, ch, (kernel, 1), (stride, 1), (pad, 0), dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class MultiScaleTCN(nn.Module):
    """4 支:膨胀 1、膨胀 2、最大池化、1×1;各 out/4 通道后拼接。"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        b = out_ch // 4
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, b, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True), TemporalConv(b, 3, stride, 1)),
            nn.Sequential(nn.Conv2d(in_ch, b, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True), TemporalConv(b, 3, stride, 2)),
            nn.Sequential(nn.Conv2d(in_ch, b, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                          nn.MaxPool2d((3, 1), (stride, 1), (1, 0)), nn.BatchNorm2d(b)),
            nn.Sequential(nn.Conv2d(in_ch, out_ch - 3 * b, 1, (stride, 1)), nn.BatchNorm2d(out_ch - 3 * b)),
        ])
        self.residual = nn.Identity() if (in_ch == out_ch and stride == 1) else \
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, (stride, 1)), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], 1) + self.residual(x)


class CTRBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, a: torch.Tensor, stride: int = 1) -> None:
        super().__init__()
        self.gcn = UnitGCN(in_ch, out_ch, a)
        self.tcn = MultiScaleTCN(out_ch, out_ch, stride)
        self.residual = nn.Identity() if (in_ch == out_ch and stride == 1) else \
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, (stride, 1)), nn.BatchNorm2d(out_ch))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


class PoseCTRGCN(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        cfg = hparams.model
        num_classes = int(getattr(cfg, "model_class_num", 3))
        hidden = int(getattr(cfg, "pose_hidden_dim", 64))
        in_ch = int(getattr(cfg, "pose_in_channels", 3))
        a = build_subsets()
        self.data_bn = nn.BatchNorm1d(in_ch * NUM_JOINTS)
        self.blocks = nn.ModuleList([
            CTRBlock(in_ch, hidden, a), CTRBlock(hidden, hidden, a),
            CTRBlock(hidden, hidden * 2, a), CTRBlock(hidden * 2, hidden * 2, a),
        ])
        self.drop = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden * 2, num_classes)
        nn.init.normal_(self.classifier.weight, 0, (2.0 / num_classes) ** 0.5)

    def forward(self, pose: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t, v, c = pose.shape
        x = pose.permute(0, 3, 1, 2).contiguous()                      # (B, C, T, V)
        x = x.permute(0, 1, 3, 2).reshape(b, c * v, t)
        x = self.data_bn(x)
        x = x.reshape(b, c, v, t).permute(0, 1, 3, 2).contiguous()
        for blk in self.blocks:
            x = blk(x)
        feat = x.mean(dim=(2, 3))
        return {"logits": self.classifier(self.drop(feat)), "feat": feat}


__all__ = ["PoseCTRGCN", "build_subsets", "COCO_EDGES"]

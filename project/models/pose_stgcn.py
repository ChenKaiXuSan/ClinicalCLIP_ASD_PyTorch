#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: pose_stgcn.py
Project: models
Author: Kaixu Chen
-----
Comment:
纯姿态基线(ST-GCN 精简版)。

存在的意义:医生注意力图是从骨架关键点渲染出来的,所以审稿人一定会问——
既然先验来自姿态,那只用姿态能不能达到同样的精度?如果能,整个视频分支的
必要性就要重新论证。这个基线必须做得公平(用标准的时空图卷积而不是随手一个
GRU),否则"视频有必要"的结论不成立。

输入 (B, T, 17, 3) = 归一化坐标 xy + 关键点置信度,与视频分支共用同一批帧下标。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# COCO 17 关键点的骨架连接
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # 头部
    (0, 5), (0, 6), (5, 6),                   # 肩
    (5, 7), (7, 9), (6, 8), (8, 10),          # 手臂
    (5, 11), (6, 12), (11, 12),               # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16),   # 腿
]
NUM_JOINTS = 17


def build_adjacency(num_joints: int = NUM_JOINTS) -> torch.Tensor:
    """对称归一化邻接矩阵 D^-1/2 (A+I) D^-1/2。"""
    adj = torch.eye(num_joints)
    for i, j in COCO_EDGES:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    degree = adj.sum(dim=1)
    d_inv_sqrt = degree.pow(-0.5)
    return d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]


class STGCNBlock(nn.Module):
    """空间图卷积 + 时间卷积。输入输出均为 (B, C, T, V)。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0),
            ),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        # 沿关节维做图卷积
        x = torch.einsum("bctv,vw->bctw", x, adj)
        x = self.spatial(x)
        x = self.temporal(x)
        return self.relu(x + res)


class PoseSTGCN(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        cfg = hparams.model
        num_classes = int(getattr(cfg, "model_class_num", 3))
        hidden = int(getattr(cfg, "pose_hidden_dim", 64))

        self.register_buffer("adj", build_adjacency())
        self.data_bn = nn.BatchNorm1d(3 * NUM_JOINTS)

        self.blocks = nn.ModuleList([
            STGCNBlock(3, hidden),
            STGCNBlock(hidden, hidden),
            STGCNBlock(hidden, hidden * 2),
            STGCNBlock(hidden * 2, hidden * 2),
        ])
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def forward(self, pose: torch.Tensor) -> dict[str, torch.Tensor]:
        # pose: (B, T, V, C) -> (B, C, T, V)
        b, t, v, c = pose.shape
        x = pose.permute(0, 3, 1, 2).contiguous()

        # 逐关节通道做归一化,消除不同视频的尺度/位置差异
        x = x.permute(0, 1, 3, 2).reshape(b, c * v, t)
        x = self.data_bn(x)
        x = x.reshape(b, c, v, t).permute(0, 1, 3, 2).contiguous()

        for block in self.blocks:
            x = block(x, self.adj)

        feat = x.mean(dim=(2, 3))  # 时空全局平均
        return {"logits": self.classifier(feat), "feat": feat}

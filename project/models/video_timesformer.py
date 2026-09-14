#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: video_timesformer.py
Project: models
Author: Kaixu Chen
-----
Comment:
视频 Transformer 基线(TimeSformer-base,Kinetics-400 微调权重,8 帧 × 224,divided space-time attention)。

存在的意义:论文主张"5 个临床几何量高于所有端到端视频模型",审稿人会问是不是因为 slow_r50 弱。
TimeSformer 的预训练配置正好是 8 帧 224,与本仓库"每 1 秒段均匀取 8 帧"的采样完全一致,不需要改数据链路。
接口与 slow_r50 一样:输入 (B, C, T, H, W),输出 logits;走 trainer/train_res_3dcnn.py 的 SingleModule,
学习率 / 权重衰减 / 轮数 / 精度与 B0 完全相同。

权重先在登录节点下载到 HF_HOME(计算节点无外网):
    python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/timesformer-base-finetuned-k400')"
"""
from __future__ import annotations

import torch
import torch.nn as nn

K400_MEAN, K400_STD = 0.45, 0.225


class TimeSformerVideo(nn.Module):
    def __init__(self, num_classes: int, name: str = "facebook/timesformer-base-finetuned-k400",
                 gradient_checkpointing: bool = True) -> None:
        super().__init__()
        from transformers import TimesformerForVideoClassification

        self.model = TimesformerForVideoClassification.from_pretrained(
            name, num_labels=num_classes, ignore_mismatched_sizes=True, local_files_only=True,
        )
        self.num_frames = int(self.model.config.num_frames)
        # batch 是"一条视频的全部 gait 段"(最多 28 段 × 8 帧 × 196 patch),12 层 divided attention 的激活
        # 会逼近 80GB;开梯度检查点换算力保内存,不改变结果
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T, H, W) -> (B, T, C, H, W)。数据链路给的是 Div255 后的 [0,1],TimeSformer 要 Kinetics 归一化
        if x.min() >= 0 and x.max() <= 1:
            x = (x - K400_MEAN) / K400_STD
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        if x.shape[1] != self.num_frames:
            idx = torch.linspace(0, x.shape[1] - 1, self.num_frames).round().long().to(x.device)
            x = x[:, idx]
        return self.model(pixel_values=x).logits

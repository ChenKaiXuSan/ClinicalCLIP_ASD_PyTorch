#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: clip_align.py
Project: models
Created Date: 2026-02-03
Author: OpenAI Assistant
-----
Comment:
CLIP-style video-attention alignment model for clinician-guided gait analysis.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple3DEncoder(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.fc = nn.Linear(64, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x)


class FrameEncoder(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x)


class Simple2DEncoder(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int) -> None:
        super().__init__()
        self.frame_encoder = FrameEncoder(in_channels, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        frame_features = self.frame_encoder(frames).reshape(b, t, -1)
        return frame_features.mean(dim=1)


class SimpleCNNLSTMEncoder(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.frame_encoder = FrameEncoder(in_channels, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        frame_features = self.frame_encoder(frames).reshape(b, t, -1)
        out, _ = self.lstm(frame_features)
        return self.fc(out[:, -1, :])


class ClipProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        return F.normalize(x, dim=-1)


def clip_contrastive_loss(
    video_embed: torch.Tensor,
    attn_embed: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    logits = video_embed @ attn_embed.t()
    logits = logits / temperature
    labels = torch.arange(video_embed.size(0), device=video_embed.device)
    loss_video = F.cross_entropy(logits, labels)
    loss_attn = F.cross_entropy(logits.t(), labels)
    return (loss_video + loss_attn) / 2


class VideoAttentionCLIP(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()

        self.feature_dim = int(getattr(hparams.model, "clip_feature_dim", 512))
        self.embed_dim = int(getattr(hparams.model, "clip_embed_dim", 256))
        self.num_classes = int(getattr(hparams.model, "model_class_num", 3))
        self.attn_in_channels = int(getattr(hparams.model, "attn_in_channels", 1))
        self.clip_backbone = getattr(hparams.model, "clip_backbone", "3dcnn")

        self.video_encoder = self._build_encoder(self.clip_backbone, in_channels=3)
        self.attn_encoder = Simple3DEncoder(self.attn_in_channels, self.feature_dim)

        self.video_projection = ClipProjectionHead(self.feature_dim, self.embed_dim)
        self.attn_projection = ClipProjectionHead(self.feature_dim, self.embed_dim)

        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

    def _build_encoder(self, backbone: str, in_channels: int) -> nn.Module:
        if backbone == "2dcnn":
            return Simple2DEncoder(in_channels, self.feature_dim)
        if backbone == "cnn_lstm":
            return SimpleCNNLSTMEncoder(in_channels, self.feature_dim)
        return Simple3DEncoder(in_channels, self.feature_dim)

    def forward(
        self,
        video: torch.Tensor,
        attn_map: torch.Tensor,
        classifier_source: Literal["video", "attn", "fusion"] = "video",
    ) -> dict[str, torch.Tensor]:
        video_feat = self.video_encoder(video)
        attn_feat = self.attn_encoder(attn_map)

        video_embed = self.video_projection(video_feat)
        attn_embed = self.attn_projection(attn_feat)

        if classifier_source == "attn":
            logits = self.classifier(attn_feat)
        elif classifier_source == "fusion":
            logits = self.classifier((video_feat + attn_feat) / 2)
        else:
            logits = self.classifier(video_feat)

        return {
            "logits": logits,
            "video_embed": video_embed,
            "attn_embed": attn_embed,
            "video_feat": video_feat,
            "attn_feat": attn_feat,
        }

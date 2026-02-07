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


class Simple3DTokenEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MapGuidedVideoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_dim: int,
        attn_channels: int = 1,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = Simple3DTokenEncoder(in_channels, hidden_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(attn_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(
        self, video: torch.Tensor, attn_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.backbone(video)
        attn_summary = summarize_attention(attn_map)
        gate = self.gate_mlp(attn_summary)
        gated_tokens = tokens * gate[:, :, None, None, None]
        pooled = self.pool(gated_tokens).flatten(1)
        return self.fc(pooled), gated_tokens, gate


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


def summarize_attention(attn_map: torch.Tensor) -> torch.Tensor:
    """Compute the spatiotemporal average of attention maps (B, C, T, H, W -> B, C)."""
    return attn_map.mean(dim=(2, 3, 4))


class VideoAttentionCLIP(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()

        self.feature_dim = int(getattr(hparams.model, "clip_feature_dim", 512))
        self.embed_dim = int(getattr(hparams.model, "clip_embed_dim", 256))
        self.num_classes = int(getattr(hparams.model, "model_class_num", 3))
        self.attn_in_channels = int(getattr(hparams.model, "attn_in_channels", 1))
        self.clip_backbone = getattr(hparams.model, "clip_backbone", "3dcnn")
        self.map_guided = bool(getattr(hparams.model, "map_guided", False))
        self.map_guided_type = getattr(hparams.model, "map_guided_type", "channel")
        self.map_guided_hidden_dim = int(
            getattr(hparams.model, "map_guided_hidden_dim", 64)
        )

        if self.map_guided and self.clip_backbone == "3dcnn":
            self.video_encoder = MapGuidedVideoEncoder(
                in_channels=3,
                feature_dim=self.feature_dim,
                attn_channels=self.attn_in_channels,
                hidden_dim=self.map_guided_hidden_dim,
            )
        else:
            self.video_encoder = self._build_encoder(self.clip_backbone, in_channels=3)

        self.map_gate = None  # gating branch for non-3dcnn backbones when map_guided is enabled
        if self.map_guided and self.clip_backbone != "3dcnn":
            self.map_gate = nn.Sequential(
                nn.Linear(self.attn_in_channels, self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.Sigmoid(),
            )
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
        video_tokens = None
        video_gate = None
        if self.map_guided and self.clip_backbone == "3dcnn":
            video_feat, video_tokens, video_gate = self.video_encoder(video, attn_map)
        else:
            video_feat = self.video_encoder(video)
            if self.map_gate is not None:
                attn_summary = summarize_attention(attn_map)
                video_gate = self.map_gate(attn_summary)
                video_feat = video_feat * video_gate
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
            "video_tokens": video_tokens,
            "video_gate": video_gate,
        }

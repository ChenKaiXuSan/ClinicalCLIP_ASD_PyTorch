#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: clip_align.py
Project: models
Created Date: 2026-02-12
Author: OpenAI Assistant
-----
Comment:
CLIP-style video-attention alignment model for clinician-guided gait analysis.
Map-guided version (spatiotemporal gating + attention pooling) where:
  video.shape    = (B, 3, T, H, W)
  attn_map.shape = (B, 1, T, H, W)

Key upgrades vs. naive gating:
- Downsample attn_map to token resolution (T',H',W') and use it for:
  (1) spatiotemporal gating of tokens
  (2) attention-weighted pooling to form video feature
- Learnable logit_scale (CLIP temperature)
- Optional token-level alignment regularizer (energy-attention correlation)

This file is self-contained and drop-in if your training step consumes:
  outputs["logits"], ["video_embed"], ["attn_embed"], optional ["token_align_loss"].
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utility
# -----------------------------
def summarize_attention(attn_map: torch.Tensor) -> torch.Tensor:
    """Spatiotemporal average: (B, C, T, H, W) -> (B, C)."""
    return attn_map.mean(dim=(2, 3, 4))


def downsample_attn_to_tokens(attn_map: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """
    Downsample and normalize attention map to match token resolution.

    Args:
        attn_map: (B, 1, T, H, W)
        tokens:   (B, D, T', H', W')

    Returns:
        attn_ds:  (B, 1, T', H', W') normalized ~[0,1] per sample
    """
    attn_ds = F.interpolate(
        attn_map,
        size=tokens.shape[-3:],
        mode="trilinear",
        align_corners=False,
    )

    # Per-sample min-max normalization for stability across annotators / intensity scales
    b = attn_ds.shape[0]
    flat = attn_ds.view(b, -1)
    amin = flat.min(dim=1, keepdim=True).values
    amax = flat.max(dim=1, keepdim=True).values
    attn_norm = (flat - amin) / (amax - amin + 1e-6)
    return attn_norm.view_as(attn_ds)


# -----------------------------
# Encoders
# -----------------------------
class Simple3DEncoder(nn.Module):
    """Global 3D encoder -> (B, feature_dim)."""

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
    """3D encoder that returns token volume (B, D, T', H', W')."""

    def __init__(self, in_channels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # -> (T/2,H/2,W/2)
            nn.Conv3d(32, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrameEncoder(nn.Module):
    """2D frame encoder -> (B, feature_dim)."""

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
    """Encode video by averaging per-frame 2D features."""

    def __init__(self, in_channels: int, feature_dim: int) -> None:
        super().__init__()
        self.frame_encoder = FrameEncoder(in_channels, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        frame_features = self.frame_encoder(frames).reshape(b, t, -1)
        return frame_features.mean(dim=1)


class SimpleCNNLSTMEncoder(nn.Module):
    """Encode video by CNN->LSTM."""

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


# -----------------------------
# Map-guided Video Encoder (NEW)
# -----------------------------
class SpatiotemporalMapGuidedVideoEncoder(nn.Module):
    """
    True spatiotemporal map guidance.

    Steps:
      tokens = backbone(video) -> (B, D, T', H', W')
      attn_ds = downsample(attn_map) -> (B, 1, T', H', W') normalized ~[0,1]
      gated_tokens = tokens * (1 + alpha * attn_ds)   OR   tokens * (1 + alpha * sigmoid(conv(attn_ds)))
      video_feat = attention-weighted pooling over gated_tokens using attn_ds/gate

    Returns:
      video_feat:   (B, feature_dim)
      video_tokens: (B, D, T', H', W') gated tokens (for visualization / regularization)
      attn_weight:  (B, 1, T', H', W') weight used for pooling (for visualization)
    """

    def __init__(
        self,
        in_channels: int,
        feature_dim: int,
        hidden_dim: int = 64,
        init_alpha: float = 1.0,
        use_sigmoid_gate: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = Simple3DTokenEncoder(in_channels, hidden_dim)

        # Learnable strength of map-guidance
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

        self.use_sigmoid_gate = bool(use_sigmoid_gate)
        if self.use_sigmoid_gate:
            self.map_gate = nn.Sequential(
                nn.Conv3d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(8, 1, kernel_size=1),
            )

        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(
        self, video: torch.Tensor, attn_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.backbone(video)  # (B, D, T', H', W')
        attn_ds = downsample_attn_to_tokens(attn_map, tokens)  # (B,1,T',H',W')

        if self.use_sigmoid_gate:
            gate = torch.sigmoid(self.map_gate(attn_ds))  # (B,1,T',H',W')
            attn_weight = gate
            gated_tokens = tokens * (1.0 + self.alpha * gate)
        else:
            attn_weight = attn_ds
            gated_tokens = tokens * (1.0 + self.alpha * attn_ds)

        # Attention-weighted pooling
        w = attn_weight
        denom = w.sum(dim=(2, 3, 4), keepdim=True) + 1e-6  # (B,1,1,1,1)
        pooled = (gated_tokens * w).sum(dim=(2, 3, 4), keepdim=False) / denom.squeeze(-1).squeeze(-1).squeeze(-1)
        # pooled: (B, D)

        feat = self.fc(pooled)  # (B, feature_dim)
        return feat, gated_tokens, attn_weight


class ChannelMapGuidedVideoEncoder(nn.Module):
    """
    Channel-only guidance.

    Steps:
      tokens = backbone(video) -> (B, D, T', H', W')
      attn_summary = mean(attn_map) -> (B, 1)
      gate = MLP(attn_summary) -> (B, D)
      gated_tokens = tokens * gate
      video_feat = global average of gated_tokens
    """

    def __init__(
        self,
        in_channels: int,
        feature_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = Simple3DTokenEncoder(in_channels, hidden_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(
        self, video: torch.Tensor, attn_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.backbone(video)  # (B, D, T', H', W')
        attn_summary = summarize_attention(attn_map)  # (B, 1)
        gate = self.gate_mlp(attn_summary)  # (B, D)
        gated_tokens = tokens * gate[:, :, None, None, None]

        pooled = gated_tokens.mean(dim=(2, 3, 4))  # (B, D)
        feat = self.fc(pooled)

        attn_weight = downsample_attn_to_tokens(attn_map, tokens)
        return feat, gated_tokens, attn_weight


class WeightedPoolVideoEncoder(nn.Module):
    """
    Attention-weighted pooling without gating.

    Steps:
      tokens = backbone(video) -> (B, D, T', H', W')
      attn_ds = downsample(attn_map) -> (B, 1, T', H', W')
      video_feat = sum(tokens * attn_ds) / sum(attn_ds)
    """

    def __init__(
        self,
        in_channels: int,
        feature_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = Simple3DTokenEncoder(in_channels, hidden_dim)
        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(
        self, video: torch.Tensor, attn_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.backbone(video)  # (B, D, T', H', W')
        attn_ds = downsample_attn_to_tokens(attn_map, tokens)  # (B,1,T',H',W')

        denom = attn_ds.sum(dim=(2, 3, 4), keepdim=True) + 1e-6
        pooled = (tokens * attn_ds).sum(dim=(2, 3, 4)) / denom.squeeze(-1).squeeze(-1).squeeze(-1)
        feat = self.fc(pooled)

        return feat, tokens, attn_ds


# -----------------------------
# Projection head (CLIP)
# -----------------------------
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


def clip_contrastive_loss_with_scale(
    video_embed: torch.Tensor,
    attn_embed: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    CLIP InfoNCE with learnable temperature.
    logit_scale should be a scalar Parameter storing log(1/temperature).
    """
    video_embed = F.normalize(video_embed, dim=-1)
    attn_embed = F.normalize(attn_embed, dim=-1)

    logits = (video_embed @ attn_embed.t()) * logit_scale.exp().clamp(max=100.0)
    labels = torch.arange(video_embed.size(0), device=video_embed.device)
    loss_v = F.cross_entropy(logits, labels)
    loss_a = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_v + loss_a)


# -----------------------------
# Optional: token-level alignment regularizer
# -----------------------------
def token_energy_alignment_loss(tokens: torch.Tensor, attn_weight: torch.Tensor) -> torch.Tensor:
    """
    Encourage correlation between token energy and attention weight.

    tokens:      (B, D, T', H', W')
    attn_weight: (B, 1, T', H', W') normalized

    Returns:
        loss scalar (lower is better): 1 - corr(energy, attn)
    """
    energy = tokens.pow(2).mean(dim=1, keepdim=True)  # (B,1,T',H',W')

    b = energy.shape[0]
    e = energy.view(b, -1)
    w = attn_weight.view(b, -1)

    e = (e - e.mean(dim=1, keepdim=True)) / (e.std(dim=1, keepdim=True) + 1e-6)
    w = (w - w.mean(dim=1, keepdim=True)) / (w.std(dim=1, keepdim=True) + 1e-6)

    corr = (e * w).mean(dim=1)  # (B,)
    return (1.0 - corr).mean()


# -----------------------------
# Main Model
# -----------------------------
class VideoAttentionCLIP(nn.Module):
    """
    Video + clinician attention-map CLIP.

    Forward returns dict:
      logits, video_embed, attn_embed, video_feat, attn_feat, video_tokens, attn_weight, logit_scale,
      optional token_align_loss

    Training recipe (example):
      out = model(video, attn_map)
      loss_cls = CE(out["logits"], y)
      loss_clip = clip_contrastive_loss_with_scale(out["video_embed"], out["attn_embed"], model.logit_scale)
      loss = loss_cls + lambda_clip * loss_clip + lambda_token * out.get("token_align_loss", 0)
    """

    def __init__(self, hparams) -> None:
        super().__init__()

        self.feature_dim = int(getattr(hparams.model, "clip_feature_dim", 512))
        self.embed_dim = int(getattr(hparams.model, "clip_embed_dim", 256))
        self.num_classes = int(getattr(hparams.model, "model_class_num", 3))

        self.attn_in_channels = int(getattr(hparams.model, "attn_in_channels", 1))
        self.clip_backbone = getattr(hparams.model, "clip_backbone", "3dcnn")

        # Map-guided toggles
        self.map_guided = bool(getattr(hparams.model, "map_guided", True))
        self.map_guided_type = getattr(hparams.model, "map_guided_type", "spatiotemporal")
        self.map_guided_hidden_dim = int(getattr(hparams.model, "map_guided_hidden_dim", 64))
        self.map_guided_alpha = float(getattr(hparams.model, "map_guided_alpha", 1.0))
        self.map_guided_sigmoid_gate = bool(
            getattr(hparams.model, "map_guided_sigmoid_gate", False)
        )

        # Token align weight (optional)
        self.lambda_token = float(getattr(hparams.model, "lambda_token", 0.0))

        # Build video encoder
        if self.map_guided:
            if self.map_guided_type == "channel":
                self.video_encoder = ChannelMapGuidedVideoEncoder(
                    in_channels=3,
                    feature_dim=self.feature_dim,
                    hidden_dim=self.map_guided_hidden_dim,
                )
            elif self.map_guided_type == "weighted_pool":
                self.video_encoder = WeightedPoolVideoEncoder(
                    in_channels=3,
                    feature_dim=self.feature_dim,
                    hidden_dim=self.map_guided_hidden_dim,
                )
            else:
                self.video_encoder = SpatiotemporalMapGuidedVideoEncoder(
                    in_channels=3,
                    feature_dim=self.feature_dim,
                    hidden_dim=self.map_guided_hidden_dim,
                    init_alpha=self.map_guided_alpha,
                    use_sigmoid_gate=self.map_guided_sigmoid_gate,
                )
        else:
            self.video_encoder = self._build_encoder(self.clip_backbone, in_channels=3)

        # Attention-map encoder (treat map as a 3D volume)
        self.attn_encoder = Simple3DEncoder(self.attn_in_channels, self.feature_dim)

        # Projection heads
        self.video_projection = ClipProjectionHead(self.feature_dim, self.embed_dim)
        self.attn_projection = ClipProjectionHead(self.feature_dim, self.embed_dim)

        # Classifier head (on feature_dim, not embed_dim)
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

        # Learnable CLIP temperature
        init_temp = float(getattr(hparams.model, "clip_init_temperature", 0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / init_temp))

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
        attn_weight = None

        if self.map_guided:
            video_feat, video_tokens, attn_weight = self.video_encoder(video, attn_map)
        else:
            video_feat = self.video_encoder(video)

        attn_feat = self.attn_encoder(attn_map)

        video_embed = self.video_projection(video_feat)
        attn_embed = self.attn_projection(attn_feat)

        if classifier_source == "attn":
            logits = self.classifier(attn_feat)
        elif classifier_source == "fusion":
            logits = self.classifier(0.5 * (video_feat + attn_feat))
        else:
            logits = self.classifier(video_feat)

        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "video_embed": video_embed,
            "attn_embed": attn_embed,
            "video_feat": video_feat,
            "attn_feat": attn_feat,
            "video_tokens": video_tokens,
            "attn_weight": attn_weight,
            "logit_scale": self.logit_scale.exp(),
        }

        if self.lambda_token > 0 and (video_tokens is not None) and (attn_weight is not None):
            out["token_align_loss"] = token_energy_alignment_loss(video_tokens, attn_weight)

        return out


# -----------------------------
# Minimal sanity test (optional)
# -----------------------------
if __name__ == "__main__":
    class _M:  # minimal hparams mock
        pass

    class _H:
        pass

    hparams = _H()
    hparams.model = _M()
    hparams.model.clip_feature_dim = 512
    hparams.model.clip_embed_dim = 256
    hparams.model.model_class_num = 3
    hparams.model.attn_in_channels = 1
    hparams.model.clip_backbone = "3dcnn"
    hparams.model.map_guided = True
    hparams.model.map_guided_hidden_dim = 64
    hparams.model.map_guided_alpha = 1.0
    hparams.model.map_guided_sigmoid_gate = False
    hparams.model.lambda_token = 0.1
    hparams.model.clip_init_temperature = 0.07

    model = VideoAttentionCLIP(hparams)
    video = torch.randn(8, 3, 8, 224, 224)
    attn = torch.randn(8, 1, 8, 224, 224)

    out = model(video, attn, classifier_source="video")
    print({k: (v.shape if isinstance(v, torch.Tensor) else None) for k, v in out.items() if v is not None})

    loss_clip = clip_contrastive_loss_with_scale(out["video_embed"], out["attn_embed"], model.logit_scale)
    print("clip_loss:", float(loss_clip))
    if "token_align_loss" in out:
        print("token_align_loss:", float(out["token_align_loss"]))

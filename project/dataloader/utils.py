#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: utils.py
Project: dataloader
Created Date: 2026-02-03
Author: OpenAI Assistant
-----
Comment:
Utilities for video preprocessing used by the data module.
"""

from typing import Callable, Dict

import torch
from torch import Tensor


class UniformTemporalSubsample:
    """
    Uniform temporal subsampling for (T, C, H, W) or (B, T, C, H, W) tensors.
    If the clip is shorter than the target length, nearest frames are duplicated.
    """

    def __init__(self, num_samples: int):
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        self.num_samples = num_samples

    def _compute_indices(self, t: int, device: torch.device) -> Tensor:
        idx_float = torch.linspace(0, max(t - 1, 0), self.num_samples, device=device)
        return torch.round(idx_float).long()

    def __call__(self, video: Tensor) -> Tensor:
        is_batched = video.ndim == 5
        if not is_batched and video.ndim != 4:
            raise ValueError("Input must be (T, C, H, W) or (B, T, C, H, W)")

        t = video.shape[-4]
        idx = self._compute_indices(t, video.device)
        return torch.index_select(video, -4, idx)


class ApplyTransformToKey:
    """
    Applies a transform to a key of a dictionary input.
    """

    def __init__(self, key: str, transform: Callable):
        self._key = key
        self._transform = transform

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x[self._key] = self._transform(x[self._key])
        return x


class Div255(torch.nn.Module):
    """
    Scale clip frames from [0, 255] to [0, 1].
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / 255.0

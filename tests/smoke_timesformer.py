#!/usr/bin/env python3
"""TimeSformer 视频基线的形状级冒烟测试(CPU,需先把权重下到 HF_HOME):
SingleModule(backbone=timesformer) 前向 + 反向,输入 (B, 3, 8, 224, 224)。

    source pegasus/setup_env.sh && python tests/smoke_timesformer.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))

from trainer.train_res_3dcnn import SingleModule  # noqa: E402


def main() -> None:
    cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
    cfg.model.backbone = "timesformer"
    cfg.model.model_class_num = 2
    cfg.train.experiment = "smoke_timesformer"
    cfg.log_path = tempfile.mkdtemp()
    m = SingleModule(cfg)
    n = sum(p.numel() for p in m.parameters())
    x = torch.rand(2, 3, 8, 224, 224)
    logits = m(x)
    assert logits.shape == (2, 2), logits.shape
    logits.sum().backward()
    print(f"timesformer: 参数 {n/1e6:.1f}M, logits {tuple(logits.shape)}, num_frames {m.video_cnn.num_frames}  ok")
    print("smoke_timesformer: 通过")


if __name__ == "__main__":
    main()

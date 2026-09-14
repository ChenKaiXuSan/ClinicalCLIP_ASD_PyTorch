#!/usr/bin/env python3
"""CTR-GCN 骨架基线的形状级冒烟测试(CPU):2D (x,y,score) 与 3D (x,y,z,score) 两种输入,前向 + 反向。

    python tests/smoke_ctrgcn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))

from models.pose_ctrgcn import PoseCTRGCN, build_subsets  # noqa: E402
from models.pose_stgcn import PoseSTGCN  # noqa: E402


def main() -> None:
    a = build_subsets()
    assert a.shape == (3, 17, 17) and torch.allclose(a[0], torch.eye(17))
    assert (a[1].sum(0)[1:] > 0.99).all(), "向心子集每个非根关节都应有一个父节点"
    for in_ch in (3, 4):
        cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
        cfg.model.model_class_num = 2
        cfg.model.pose_in_channels = in_ch
        m = PoseCTRGCN(cfg)
        x = torch.randn(5, 8, 17, in_ch)
        out = m(x)
        assert out["logits"].shape == (5, 2) and out["feat"].shape == (5, 128), out["logits"].shape
        out["logits"].sum().backward()
        n_ctr = sum(p.numel() for p in m.parameters())
        n_st = sum(p.numel() for p in PoseSTGCN(cfg).parameters())
        print(f"in_ch={in_ch}: logits {tuple(out['logits'].shape)}  参数 CTR-GCN {n_ctr/1e3:.0f}k vs ST-GCN {n_st/1e3:.0f}k  ok")
    print("smoke_ctrgcn: 通过")


if __name__ == "__main__":
    main()

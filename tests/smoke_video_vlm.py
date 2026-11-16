#!/usr/bin/env python3
"""视频 + VLM 双分支(backbone=video_vlm)的冒烟测试(CPU):真实数据的一个 batch 同时带 video 与 tokens,
两种融合与单分支对照的前向 / 反向,探针损失不回传到骨干。

    python tests/smoke_video_vlm.py --root-path $DATA --cache $DATA/vlm_cache/qwen3vl_32b_448_clinical
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))

from models.video_vlm_fusion import VideoVLMFusionNet  # noqa: E402


def make_cfg(root_path: str, cache: str, **over):
    cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
    cfg.model.backbone = "video_vlm"
    cfg.model.model_class_num = 2
    cfg.data.num_workers = 0
    cfg.data.feature_cache_dir = cache
    cfg.paths.root_path = root_path
    cfg.train.experiment = "smoke_video_vlm"
    cfg.log_path = tempfile.mkdtemp()
    for k, v in over.items():
        cfg.model[k] = v
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-path", required=True)
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()
    from dataloader.data_loader import WalkDataModule

    info = Path(args.root_path) / "clinical_CLIP_dataset"
    index = json.load(open(info / "index_mapping/2/index.json"))
    paths = [str(info / "json_mix" / p.split("json_mix/")[-1]) for p in index["0"]["test"][:1]]
    cfg = make_cfg(args.root_path, args.cache)
    dm = WalkDataModule(cfg, {"train": paths, "val": paths, "test": paths}); dm.setup()
    batch = next(iter(dm.val_dataloader()))
    v, t, y = batch["video"], batch["tokens"], batch["label"].long().view(-1)
    print(f"[data] video {tuple(v.shape)} tokens {tuple(t.shape)} {t.dtype} label {tuple(y.shape)}")
    assert v.shape[0] == t.shape[0] == y.shape[0]
    v, t, y = v[:2], t[:2], y[:2]
    for over in ({"vvf_fusion": "feature"}, {"vvf_fusion": "score"}, {"vvf_branches": "video"}, {"vvf_branches": "vlm"}):
        net = VideoVLMFusionNet(make_cfg(args.root_path, args.cache, **over))
        out = net(v if net.use_video else None, t if net.use_vlm else None)
        n = len(net.branch_names)
        assert out["branch_logits"].shape == (2, n, 2) and out["fused_prob"].shape == (2, 2)
        assert torch.allclose(out["fused_prob"].sum(1), torch.ones(2), atol=1e-5)
        if net.fusion == "feature":
            # 探针损失不应有梯度流回骨干 / 投影
            probe = sum(F.cross_entropy(out["branch_logits"][:, i], y) for i in range(n))
            probe.backward(retain_graph=True)
            body = [p for name, p in net.named_parameters() if not name.startswith("branch_heads") and p.grad is not None]
            assert all(float(p.grad.abs().sum()) == 0 for p in body), "探针损失漏进了骨干"
            net.zero_grad()
            F.cross_entropy(out["fused_logits"], y).backward()
            assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for name, p in net.named_parameters() if not name.startswith("branch_heads"))
        else:
            F.cross_entropy(out["branch_logits"][:, 0], y).backward()
        print(f"[model] {over}: 分支 {net.branch_names}, 参数 {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M ok")
    print("all ok")


if __name__ == "__main__":
    main()

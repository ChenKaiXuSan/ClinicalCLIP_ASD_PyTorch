#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""3D 骨架姿态基线(B3_pose_3d)的形状级冒烟:paths.skeleton_path 指向 seg_skeleton_pkl_3d,
model.pose_in_channels=4,取两条视频跑一个 train step。CPU 可跑。

    python tests/smoke_pose3d.py --root-path $DATA
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))
from dataloader.data_loader import WalkDataModule  # noqa: E402
from trainer.train_pose import PoseModule  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-path", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    info = Path(args.root_path) / "clinical_CLIP_dataset"
    index = json.load(open(info / "index_mapping/2/index.json"))
    paths = [str(info / "json_mix" / p.split("json_mix/")[-1]) for p in index["0"]["test"][:2]]

    for skel, ch in (("seg_skeleton_pkl", 3), ("seg_skeleton_pkl_3d", 4)):
        cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
        cfg.paths.root_path = args.root_path
        cfg.paths.skeleton_path = str(info / skel)
        cfg.model.backbone = "pose"
        cfg.model.model_class_num = 2
        cfg.model.pose_in_channels = ch
        cfg.data.num_workers = 0
        cfg.train.experiment = "smoke_pose3d"
        cfg.log_path = tempfile.mkdtemp()
        dm = WalkDataModule(cfg, {"train": paths, "val": paths, "test": paths})
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        pose = batch["pose"]
        assert pose.shape[-1] == ch, (pose.shape, ch)
        m = PoseModule(cfg)
        loss = m.training_step(batch, 0)
        loss.backward()
        nz = (pose[..., -1] > 0).float().mean().item()
        print(f"[{skel}] pose {tuple(pose.shape)}, 坐标范围 [{pose[..., :ch - 1].min():.2f}, {pose[..., :ch - 1].max():.2f}], "
              f"有效帧占比 {nz:.2f}, loss {loss.item():.3f}, data_bn {m.model.data_bn.num_features}")
    print("ok")


if __name__ == "__main__":
    main()

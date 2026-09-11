#!/usr/bin/env python3
"""角色二(VLM 属性辅助回归)的冒烟测试,登录节点 CPU 可跑。

    python tests/smoke_aux.py --root-path $DATA --aux-dir logs/qwen_attributes/qwen3-vl-8b-instruct_448

覆盖:辅助目标加载与 z 标准化 / batch 带 aux / concept 与 3dcnn 各一步反传且辅助头有梯度 /
aux_weight=0 时基线行为不变(3dcnn 仍是原始 proj 头)。
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
from trainer.train_clinical_concept import ClinicalConceptModule  # noqa: E402
from trainer.train_res_3dcnn import SingleModule  # noqa: E402


def make_cfg(root, aux_dir, backbone, w):
    cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
    cfg.paths.root_path = root
    cfg.model.backbone = backbone
    cfg.model.model_class_num = 2
    cfg.data.aux_targets_dir = aux_dir
    cfg.data.num_workers = 0
    cfg.loss.aux_weight = w
    cfg.train.experiment = "smoke_aux"
    cfg.log_path = tempfile.mkdtemp()
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-path", required=True)
    ap.add_argument("--aux-dir", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    info = Path(args.root_path) / "clinical_CLIP_dataset"
    index = json.load(open(info / "index_mapping/2/index.json"))
    paths = [str(info / "json_mix" / p.split("json_mix/")[-1]) for p in index["0"]["test"][:2]]

    # 1. 数据:aux 形状与标准化
    cfg = make_cfg(args.root_path, args.aux_dir, "concept", 1.0)
    dm = WalkDataModule(cfg, {"train": paths, "val": paths, "test": paths}); dm.setup()
    ds = dm.train_gait_dataset
    names = ds.aux_names
    allv = torch.cat(list(ds._aux.values()))
    assert allv.mean(0).abs().max() < 1e-3 and (allv.std(0) - 1).abs().max() < 1e-2, "z 标准化不对"
    batch = next(iter(dm.train_dataloader()))
    assert batch["aux"].shape == (batch["video"].shape[0], len(names)), batch["aux"].shape
    print(f"[1] aux ok: {len(names)} 个属性 {names}; batch aux {tuple(batch['aux'].shape)} video {tuple(batch['video'].shape)}")

    # 2. concept + aux:一步反传,辅助头有梯度,loss_aux 非零
    m = ClinicalConceptModule(cfg)
    assert m.model.aux_head is not None
    loss = m._shared_step(batch, "train"); loss.backward()
    g = sum(p.grad.abs().sum().item() for p in m.model.aux_head.parameters())
    assert g > 0, "concept 辅助头没有梯度"
    print(f"[2] concept+aux ok: loss {loss.item():.3f}, aux_head grad {g:.3f}")

    # 3. 3dcnn + aux(CPU 上 slow_r50 前向约 1 分钟)
    cfg3 = make_cfg(args.root_path, args.aux_dir, "3dcnn", 1.0)
    m3 = SingleModule(cfg3)
    assert m3.aux_dim == len(names) and isinstance(m3.video_cnn.blocks[-1].proj, torch.nn.Identity)
    logits, aux = m3._forward(batch["video"][:2])
    assert logits.shape == (2, 2) and aux.shape == (2, len(names)), (logits.shape, aux.shape)
    loss3 = m3.training_step({k: (v[:2] if torch.is_tensor(v) and v.shape[0] == batch["video"].shape[0] else v) for k, v in batch.items()}, 0)
    loss3.backward()
    g3 = sum(p.grad.abs().sum().item() for p in m3.aux_head.parameters())
    assert g3 > 0
    print(f"[3] 3dcnn+aux ok: loss {loss3.item():.3f}, aux_head grad {g3:.3f}")

    # 4. aux_weight=0:3dcnn 保持原始头,前向只返回 logits
    cfg0 = make_cfg(args.root_path, args.aux_dir, "3dcnn", 0.0)
    m0 = SingleModule(cfg0)
    assert m0.aux_dim == 0 and isinstance(m0.video_cnn.blocks[-1].proj, torch.nn.Linear)
    assert m0._forward(batch["video"][:1])[1] is None
    print("[4] aux_weight=0 基线行为不变")
    print("ALL OK")


if __name__ == "__main__":
    main()

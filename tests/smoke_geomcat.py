#!/usr/bin/env python3
"""几何量特征级拼接(model.aux_concat)的冒烟测试,CPU 可跑。

    python tests/smoke_geomcat.py --root-path $DATA --aux-dir logs/geom_attributes_doctor
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


def make_cfg(root, aux_dir, backbone, w_aux, concat):
    cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
    cfg.paths.root_path = root
    cfg.model.backbone = backbone
    cfg.model.model_class_num = 2
    cfg.model.aux_concat = concat
    cfg.data.aux_targets_dir = aux_dir
    cfg.data.num_workers = 0
    cfg.loss.aux_weight = w_aux
    cfg.train.experiment = "smoke_geomcat"
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

    cfg = make_cfg(args.root_path, args.aux_dir, "concept", 0.0, True)
    dm = WalkDataModule(cfg, {"train": paths, "val": paths, "test": paths}); dm.setup()
    batch = next(iter(dm.train_dataloader()))
    A = batch["aux"].shape[1]
    print(f"[1] aux {tuple(batch['aux'].shape)}, {dm.train_gait_dataset.aux_names}")

    # concept + concat, no aux loss
    m = ClinicalConceptModule(cfg)
    assert m.model.aux_concat and m.model.classifier[1].in_features == 256 + 256 + A
    loss = m._shared_step(batch, "train"); loss.backward()
    g = sum(p.grad.abs().sum().item() for p in m.model.classifier.parameters())
    print(f"[2] concept+concat ok: cls_in {m.model.classifier[1].in_features}, loss {loss.item():.3f}, grad {g:.2f}")
    # 拼接确实影响输出:把 aux 置零后 logits 不同
    with torch.no_grad():
        o1 = m.model(batch["video"][:2], aux=batch["aux"][:2])["logits"]
        o2 = m.model(batch["video"][:2], aux=torch.zeros_like(batch["aux"][:2]))["logits"]
    assert (o1 - o2).abs().max() > 1e-4, "aux 没进分类头"
    print("[3] concept: aux 改变 logits")

    # 3dcnn + concat (+ aux loss both on)
    cfg3 = make_cfg(args.root_path, args.aux_dir, "3dcnn", 1.0, True)
    m3 = SingleModule(cfg3)
    assert m3.aux_concat and m3.cls_head.in_features == 2048 + A and m3.aux_head is not None
    sub = {k: (v[:2] if torch.is_tensor(v) and v.shape[0] == batch["video"].shape[0] else v) for k, v in batch.items()}
    loss3 = m3.training_step(sub, 0); loss3.backward()
    print(f"[4] 3dcnn+concat+aux ok: cls_in {m3.cls_head.in_features}, loss {loss3.item():.3f}")
    # 3dcnn concat only
    cfg4 = make_cfg(args.root_path, args.aux_dir, "3dcnn", 0.0, True)
    m4 = SingleModule(cfg4)
    assert m4.aux_concat and m4.aux_head is None and m4.cls_head.in_features == 2048 + A
    with torch.no_grad():
        l1, _ = m4._forward(batch["video"][:1], batch["aux"][:1])
        l2, _ = m4._forward(batch["video"][:1], torch.zeros_like(batch["aux"][:1]))
    assert (l1 - l2).abs().max() > 1e-4
    print("[5] 3dcnn concat-only ok, aux 改变 logits")
    print("ALL OK")


if __name__ == "__main__":
    main()

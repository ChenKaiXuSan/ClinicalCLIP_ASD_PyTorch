#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: patient_metrics.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
患者级概率 json(attribute_classifier / prior_sparse_lr / p0_controls --out 格式)的完整指标:
平衡准确率、AUC、敏感度(ASD 召回)、特异度(non-ASD 召回)、各自的 bootstrap 95% CI。
也接受 logs/train 里的实验名(--tags,需 --data-root 与 --seed),从 best_preds 聚合成患者级。

    python analysis/patient_metrics.py --files logs/geom_probs/geom3d_doctor_lr_probs.json ... --tags B0_3dcnn_e50 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from late_fusion import patient_probs  # noqa: E402
from patient_level_stats import build_patient_map  # noqa: E402


def load_json(path, drop=()):
    d = json.load(open(path))
    pids = [p for p in sorted(d) if p not in drop]
    P = np.array([d[p]["prob"][0] for p in pids])  # P(ASD) (索引 0 = ASD)
    y = np.array([int(d[p]["label"] == 0) for p in pids])  # ASD -> 1
    return P, y


def load_tag(root, tag, pmap, seed, drop=()):
    d = patient_probs(Path(root), tag, pmap, seed)
    pids = [p for p in sorted(d) if p not in drop]
    P = np.array([d[p][0][0] for p in pids])
    y = np.array([int(d[p][1] == 0) for p in pids])
    return P, y


def metrics(P, y):
    pred = (P > 0.5).astype(int)
    sens = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
    spec = ((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
    return {"macro": (sens + spec) / 2, "auc": roc_auc_score(y, P), "sens": sens, "spec": spec}


def boot(P, y, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in ("macro", "auc", "sens", "spec")}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        m = metrics(P[i], y[i])
        for k in vals:
            vals[k].append(m[k])
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in vals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--root", default="logs/train")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-missing", default=None, help="bank.pkl 路径: 去掉无 3D 结果的患者后再算")
    args = ap.parse_args()
    drop = set()
    if args.exclude_missing:
        import pickle
        drop = {Path(v["json"]).stem.split("-")[0] for v in pickle.load(open(args.exclude_missing, "rb"))["videos"].values() if v["frac_ok"] == 0}
    runs = {Path(f).stem.replace("_probs", ""): load_json(f, drop) for f in args.files}
    if args.tags:
        pmap = build_patient_map(args.data_root)
        for t in args.tags:
            runs[f"{t}_s{args.seed}"] = load_tag(args.root, t, pmap, args.seed, drop)
    print(f"{'配置':44s} {'n':>3s} {'平衡准确率':>20s} {'AUC':>20s} {'敏感度(ASD)':>20s} {'特异度(non-ASD)':>20s}")
    for name, (P, y) in runs.items():
        m, ci = metrics(P, y), boot(P, y)
        cell = lambda k: f"{m[k]:.3f} [{ci[k][0]:.2f},{ci[k][1]:.2f}]"
        print(f"{name:44s} {len(y):3d} {cell('macro'):>20s} {cell('auc'):>20s} {cell('sens'):>20s} {cell('spec'):>20s}")


if __name__ == "__main__":
    main()

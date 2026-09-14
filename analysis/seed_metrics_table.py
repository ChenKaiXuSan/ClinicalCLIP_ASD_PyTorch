#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: seed_metrics_table.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
主表用:若干训练配置 × 多种子的患者级指标(平衡准确率 / AUC / 敏感度 / 特异度)均值 ± 标准差,
可选去掉无 3D 结果的患者(67 人口径)。指标口径同 patient_metrics.py(段概率按患者取均值)。

    python analysis/seed_metrics_table.py --data-root $DATA --exclude-missing logs/measure_bank_3d/bank.pkl \
        --tags B0_3dcnn_e50 B1_2dcnn_e50 ... --seeds 42 1337 2024
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patient_level_stats import build_patient_map  # noqa: E402
from patient_metrics import load_tag, metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2024])
    ap.add_argument("--exclude-missing", default=None, help="bank.pkl 路径: 去掉无 3D 结果的患者")
    args = ap.parse_args()
    drop = set()
    if args.exclude_missing:
        drop = {Path(v["json"]).stem.split("-")[0] for v in pickle.load(open(args.exclude_missing, "rb"))["videos"].values() if v["frac_ok"] == 0}
    pmap = build_patient_map(args.data_root)
    keys = ["macro", "auc", "sens", "spec"]
    print(f"{'配置':30s} {'n':>3s} {'种子数':>4s} " + " ".join(f"{k:>14s}" for k in ("平衡准确率", "AUC", "敏感度", "特异度")) + "   逐种子 macro")
    for t in args.tags:
        vals = {k: [] for k in keys}
        n = 0
        for s in args.seeds:
            try:
                P, y = load_tag(args.root, t, pmap, s, drop)
            except Exception as exc:  # noqa: BLE001
                print(f"{t:30s} seed {s}: 缺 ({exc})")
                continue
            if len(y) == 0:
                continue
            m = metrics(P, y)
            n = len(y)
            for k in keys:
                vals[k].append(m[k])
        if not vals["macro"]:
            continue
        cells = " ".join(f"{np.mean(vals[k]):.3f}±{np.std(vals[k]):.3f}" for k in keys)
        print(f"{t:30s} {n:3d} {len(vals['macro']):4d} {cells}   " + " / ".join(f"{v:.3f}" for v in vals["macro"]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: seed_summary.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
多种子的患者级汇总:每个配置 × 每个种子给出患者级平衡准确率(5 折 test 拼成 79 人,段概率按患者取均值),
再给种子均值 ± 标准差,以及每个种子上对参照配置的 McNemar(改错 : 改对, p)。
patient_level_stats.py 只看 s42;这里补种子维度,口径完全一致。

    python analysis/seed_summary.py --data-root $DATA --ref B0_3dcnn_e50 \
        --tags G0_3dcnn_geomaux_e50 G0b_3dcnn_geomaux0.3_e50 G3_3dcnn_geomcat_e50 --seeds 42 1337 2024
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from late_fusion import macro, mcnemar, patient_probs  # noqa: E402
from patient_level_stats import build_patient_map, macro_ci  # noqa: E402


def n_folds(root: Path, tag: str, seed: int) -> int:
    return sum(1 for f in range(5) if any((root / f"{tag}__f{f}_s{seed}").rglob("best_preds/*_pred.pt")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ref", required=True, help="参照配置(McNemar 的 A)")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2024])
    ap.add_argument("--allow-partial", action="store_true", help="折不齐也算(只在交集患者上比)")
    args = ap.parse_args()

    pmap = build_patient_map(args.data_root)
    root = Path(args.root)

    def load(tag, seed):
        k = n_folds(root, tag, seed)
        if k == 0 or (k < 5 and not args.allow_partial):
            return None, k
        return patient_probs(root, tag, pmap, seed), k

    refs = {s: load(args.ref, s) for s in args.seeds}
    hdr = f"{'配置':30s} " + " ".join(f"{'s' + str(s):>22s}" for s in args.seeds) + f" {'均值±std':>12s}"
    print("患者级平衡准确率 [95% CI];每格后面是对参照的 McNemar 参照独对:本配置独对 p")
    print(hdr)
    for tag in [args.ref] + args.tags:
        line = f"{tag:30s} "
        vals = []
        for s in args.seeds:
            R, kr = refs[s]
            B, kb = load(tag, s)
            if B is None:
                line += f"{'(' + str(kb) + '/5 折)':>22s} "
                continue
            pids = sorted(B)
            if tag != args.ref and R is not None:
                pids = sorted(set(pids) & set(R))
            label = np.array([B[p][1] for p in pids])
            pb = np.stack([B[p][0] for p in pids]).argmax(1)
            m = macro(pb, label)
            vals.append(m)
            lo, hi = macro_ci(pb, label, n_boot=2000)
            cell = f"{m:.3f} [{lo:.2f},{hi:.2f}]"
            if tag != args.ref and R is not None:
                pr = np.stack([R[p][0] for p in pids]).argmax(1)
                a_only, b_only, p = mcnemar(pr == label, pb == label)
                cell += f" {a_only}:{b_only} p{p:.2f}"
            line += f"{cell:>22s} "
        if vals:
            line += f"{np.mean(vals):.3f}±{np.std(vals):.3f}"
        print(line)


if __name__ == "__main__":
    main()

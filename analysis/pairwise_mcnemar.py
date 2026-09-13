#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: pairwise_mcnemar.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
对若干患者级概率 json(attribute_classifier.py / prior_sparse_lr.py --out 的格式)两两做 McNemar,
并给每个文件的患者级平衡准确率 [bootstrap CI]。

    python analysis/pairwise_mcnemar.py logs/geom_probs/bank_hard_doctor_segment_min_a1.0_probs.json logs/geom_probs/geom3d_doctor_lr_probs.json ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patient_level_stats import macro_ci  # noqa: E402


def load(path):
    d = json.load(open(path))
    pids = sorted(d)
    pred = np.array([int(np.argmax(d[p]["prob"])) for p in pids])
    label = np.array([int(d[p]["label"]) for p in pids])
    return pids, pred, label


def macro(pred, y):
    return float(np.mean([((pred == c) & (y == c)).sum() / max((y == c).sum(), 1) for c in np.unique(y)]))


def main():
    files = sys.argv[1:]
    runs = {Path(f).stem.replace("_probs", ""): load(f) for f in files}
    names = list(runs)
    print(f"{'文件':44s} {'macro [95% CI]':>22s}")
    for n in names:
        _, pred, y = runs[n]
        lo, hi = macro_ci(pred, y, n_boot=2000)
        print(f"{n:44s} {macro(pred, y):.3f} [{lo:.3f},{hi:.3f}]")
    print("\nMcNemar(行独对 : 列独对, p):")
    print(" " * 44 + " ".join(f"{n[:14]:>16s}" for n in names))
    for a in names:
        pa, preda, ya = runs[a]
        row = f"{a:44s}"
        for b in names:
            if a == b:
                row += f"{'-':>16s} "
                continue
            pb, predb, yb = runs[b]
            common = sorted(set(pa) & set(pb))
            ia = [pa.index(p) for p in common]
            ib = [pb.index(p) for p in common]
            oka = preda[ia] == ya[ia]
            okb = predb[ib] == yb[ib]
            n01, n10 = int((oka & ~okb).sum()), int((~oka & okb).sum())
            p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            row += f"{f'{n01}:{n10} p{p:.2f}':>16s} "
        print(row)


if __name__ == "__main__":
    main()

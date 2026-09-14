#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: gate_feature_diag.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
方法 2 的诊断:可靠性特征到底能不能预测"哪条分支判对"。对每个 (视频分支, 种子, 几何分支),
把患者分成 视频对/几何错、几何对/视频错、都对、都错 四类,报每个特征对 "视频判对" 与 "几何判对"
的患者级 AUC(不训练,纯统计),以及"只有一条分支对"的患者数——这是门控能改动的全部空间。

    python analysis/gate_feature_diag.py --data-root $DATA --video B0_3dcnn_e50 ... --geom logs/geom_probs/geom3d_doctor_lr_probs.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patient_level_stats import build_patient_map  # noqa: E402
from uncertainty_fusion import FEATS, geom_reliability, load_geom, load_video  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--video", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2024])
    ap.add_argument("--geom", nargs="+", required=True)
    ap.add_argument("--attr-dir", default="logs/geom_attributes_3d_doctor")
    args = ap.parse_args()
    pmap = build_patient_map(args.data_root)
    R = geom_reliability(Path(args.attr_dir), pmap)
    for gpath in args.geom:
        G = load_geom(gpath)
        print(f"\n===== 几何分支 {Path(gpath).stem}")
        print(f"{'视频分支':24s} {'seed':>5s} {'都对':>4s} {'仅视频对':>6s} {'仅几何对':>6s} {'都错':>4s} | "
              + "AUC(特征 -> 视频判对): " + " ".join(f"{f:>7s}" for f in FEATS)
              + " | AUC(-> 几何判对): " + " ".join(f"{f:>7s}" for f in FEATS))
        acc = {f: [] for f in FEATS}; accg = {f: [] for f in FEATS}
        for vtag in args.video:
            for seed in args.seeds:
                V = load_video(Path(args.root), vtag, seed, pmap)
                pids = sorted(set(V) & set(G) & set(R))
                y = np.array([V[p]["label"] for p in pids])
                ok_v = np.array([V[p]["p"].argmax() == V[p]["label"] for p in pids])
                ok_g = np.array([G[p]["p"].argmax() == V[p]["label"] for p in pids])
                U = np.array([[abs(V[p]["p"][1] - 0.5) * 2, V[p]["std"], np.log(V[p]["n"]),
                               abs(G[p]["p"][1] - 0.5) * 2, R[p][0], R[p][1]] for p in pids])
                def auc(t, j):
                    return roc_auc_score(t, U[:, j]) if 0 < t.sum() < len(t) else float("nan")
                a_v = [auc(ok_v, j) for j in range(len(FEATS))]
                a_g = [auc(ok_g, j) for j in range(len(FEATS))]
                for j, f in enumerate(FEATS):
                    acc[f].append(a_v[j]); accg[f].append(a_g[j])
                print(f"{vtag:24s} {seed:5d} {int((ok_v & ok_g).sum()):4d} {int((ok_v & ~ok_g).sum()):6d} "
                      f"{int((~ok_v & ok_g).sum()):6d} {int((~ok_v & ~ok_g).sum()):4d} | "
                      + " " * 22 + " ".join(f"{a:7.2f}" for a in a_v) + " | " + " " * 18 + " ".join(f"{a:7.2f}" for a in a_g))
        print(f"{'均值':24s} {'':5s} {'':4s} {'':6s} {'':6s} {'':4s} | " + " " * 22
              + " ".join(f"{np.nanmean(acc[f]):7.2f}" for f in FEATS) + " | " + " " * 18
              + " ".join(f"{np.nanmean(accg[f]):7.2f}" for f in FEATS))


if __name__ == "__main__":
    main()

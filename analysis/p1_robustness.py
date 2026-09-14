#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: p1_robustness.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
方法 1 的 P1 鲁棒性对照(同一测量库、同一固定患者划分、段级训练、患者级判类):

  nonlinear        随机森林 / 直方图梯度提升吃全部 1272 个测量(及只吃医生区域子集),对比"医生边界 + 稀疏线性"
  exclude-missing  去掉有视频无 3D 结果的患者后重训 手挑 5 量 / 硬过滤 / 非医生区域;并把已有概率 json 限制到剩余患者重算

    python analysis/p1_robustness.py nonlinear --data-root $DATA --out-dir logs/geom_probs/p1
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p0_controls import keep_touching, report, save_probs  # noqa: E402
from patient_level_stats import macro_ci  # noqa: E402
from prior_sparse_lr import DOCTOR_REGIONS, HAND5, Runner, load_bank, macro  # noqa: E402


def missing_patients(bank_path: str) -> set:
    bank = pickle.load(open(bank_path, "rb"))
    return {Path(v["json"]).stem.split("-")[0] for v in bank["videos"].values() if v["frac_ok"] == 0}


def fit_nonlinear(kind, Xs, seg_pat, y, splits, cols, seed=0):
    n = len(y)
    pred, prob = np.zeros(n, int), np.zeros(n)
    for tr, te in splits:
        mtr, mte = np.isin(seg_pat, tr), np.isin(seg_pat, te)
        Xtr, Xte = Xs[mtr][:, cols], Xs[mte][:, cols]
        med = np.nanmedian(Xtr, axis=0)
        Xtr, Xte = np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)
        ytr = y[seg_pat[mtr]]
        if kind == "rf":
            clf = RandomForestClassifier(500, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=8, random_state=seed)
        else:
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, class_weight="balanced", random_state=seed)
        clf.fit(Xtr, ytr)
        pr = clf.predict_proba(Xte)[:, 1]
        rp = seg_pat[mte]
        for i in te:
            prob[i] = pr[rp == i].mean()
        pred[te] = (prob[te] > 0.5).astype(int)
    return pred, prob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["nonlinear", "exclude-missing"])
    ap.add_argument("--bank", default="logs/measure_bank_3d/bank.pkl")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--Cs", default="0.01,0.1")
    ap.add_argument("--probs", nargs="*", default=[], help="exclude-missing: 限制到剩余患者重算的已有概率 json")
    ap.add_argument("--out-dir", default="logs/geom_probs/p1")
    args = ap.parse_args()

    Xp, Xs, seg_pat, y, pids, feats, regions, splits = load_bank(args.bank, args.data_root, 2)
    Cs = [float(c) for c in args.Cs.split(",")]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doctor_keep = keep_touching(regions, DOCTOR_REGIONS)
    hand_keep = np.array([f in HAND5 for f in feats])
    ones = np.ones(Xp.shape[1])

    if args.cmd == "nonlinear":
        print(f"{'模型 / 候选集':40s} {'macro [95% CI]':>22s}")
        for kind in ("rf", "hgb"):
            for cname, keep in (("全库 1272", np.ones(Xp.shape[1], bool)), ("医生区域", doctor_keep), ("手挑 5 量", hand_keep)):
                pred, prob = fit_nonlinear(kind, Xs, seg_pat, y, splits, np.where(keep)[0])
                report(f"{kind} / {cname}", pred, [], y)
                save_probs(out / f"{kind}_{cname.split()[0]}_probs.json", pids, prob, y)

    elif args.cmd == "exclude-missing":
        miss = missing_patients(args.bank)
        keep_p = np.array([p not in miss for p in pids])
        idx_keep = set(np.where(keep_p)[0].tolist())
        print(f"有视频无 3D 结果的患者 {len(miss)} 人 (ASD {sum(y[i] for i, p in enumerate(pids) if p in miss)}), 剩余 {int(keep_p.sum())} 人")
        sp2 = [(np.array([i for i in tr if i in idx_keep]), np.array([i for i in te if i in idx_keep])) for tr, te in splits]
        runner = Runner(Xp, Xs, seg_pat, y, sp2, Cs, "segment", 0)
        print(f"\n重训(去掉这些患者), {'候选集':32s} {'macro [95% CI]':>22s}")
        y_sub = y[keep_p]
        for name, keep in (("hand5", hand_keep), ("hard_doctor", doctor_keep), ("hard_other", ~doctor_keep)):
            pred, prob, sel, _ = runner.run(ones, keep)
            report(f"重训 {name}", pred[keep_p], sel, y_sub)
        if args.probs:
            print(f"\n已有概率限制到剩余 {int(keep_p.sum())} 人(不重训):")
            for f in args.probs:
                d = json.load(open(f))
                ps = [p for p in sorted(d) if p not in miss]
                pred = np.array([int(np.argmax(d[p]["prob"])) for p in ps])
                lab = np.array([int(d[p]["label"]) for p in ps])
                pm = [p for p in sorted(d) if p in miss]
                acc_m = np.mean([int(np.argmax(d[p]["prob"])) == int(d[p]["label"]) for p in pm]) if pm else float("nan")
                lo, hi = macro_ci(pred, lab, n_boot=2000)
                print(f"  {Path(f).stem.replace('_probs', ''):40s} {macro(pred, lab):.3f} [{lo:.3f},{hi:.3f}]   被去掉患者的准确率 {acc_m:.2f}")


if __name__ == "__main__":
    main()

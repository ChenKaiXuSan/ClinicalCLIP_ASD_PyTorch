#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: eval_measure.py
Project: analysis/carepd
Author: Kaixu Chen
-----
Comment:
CARE-PD 上"测量 + 逻辑回归"的评测,协议与官方一致:逐队列留一受试者(LOSO,用数据集自带的折文件),
按行走预测 UPDRS-gait(0-3),各折预测拼起来算宏平均 F1:F1_all(出现过的全部类别)与 F1_0-2(只在 0/1/2 上取宏平均,
3 分仍参加训练)。分类器:标准化 + 多项逻辑回归(class_weight=balanced),超参固定不调。

官方深度基线(冻结运动编码器 + 线性探针,LOSO,论文正文):PD-GaM 最好 0.73、BMCLab 最好 0.68;
七个编码器的均值 PD-GaM 0.620、BMCLab 0.559、T-SDU-PD 0.417、3DGait 0.271(附录表 D.2)。

    python analysis/carepd/eval_measure.py --bank logs/carepd/bank.pkl
    python analysis/carepd/eval_measure.py --sets-json logs/carepd/llm/<model>/<cond>.json     # LLM 提出的测量集合
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path("/work/1/SKIING/chenkaixu/data/care_pd")
REPORTED = {"PD-GaM": (0.73, 0.620), "BMCLab": (0.68, 0.559), "T-SDU-PD": (None, 0.417), "3DGait": (None, 0.271)}
# 先验手工集合:按 MDS-UPDRS 步态条目的评分要点(步幅、步速、抬脚高度、摆臂、姿势)各取一两个量,写在看任何结果之前
CLASSIC = ["dz|ank_L-ank_R|range", "speed|pelvis|mean", "dz|wr_L-hip_mid|range", "dz|wr_R-hip_mid|range",
           "incl|sh_mid-hip_mid|mean", "dz|head-hip_mid|mean", "dy|hip_mid-foot_L|range", "dy|hip_mid-foot_R|range",
           "ang|knee_L|range", "ang|knee_R|range", "ang|hip_L|range", "ang|hip_R|range"]


def loso_folds(cohort: str, subjects: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(set(subjects))
    f = next((ROOT / "folds" / "UPDRS_Datasets").glob(f"{cohort}_{n}fold_participants.pkl"), None)
    folds = []
    if f is not None:
        for _, sp in sorted(pickle.load(open(f, "rb")).items()):
            ev = {str(s) for s in sp["eval"]}; tr = {str(s) for s in sp["train"]}
            te = np.isin(subjects, list(ev)); trn = np.isin(subjects, list(tr))
            if te.any() and trn.any():
                folds.append((np.where(trn)[0], np.where(te)[0]))
    if not folds:      # 折文件与受试者编号对不上时退回自建 LOSO
        for s in sorted(set(subjects)):
            folds.append((np.where(subjects != s)[0], np.where(subjects == s)[0]))
    return folds


def predict(X, y, folds, model: str, seed: int = 0) -> np.ndarray:
    pred = np.full(len(y), -1)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        if model == "lr":
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
        elif model == "lr_small_c":
            clf = LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000)
        elif model == "l1":
            clf = LogisticRegression(C=0.05, penalty="l1", solver="saga", class_weight="balanced", max_iter=3000)
        else:
            clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=8, random_state=seed)
        clf.fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    return pred


def scores(y, pred) -> tuple[float, float]:
    ok = pred >= 0
    labs = sorted(set(y[ok]))
    return (f1_score(y[ok], pred[ok], labels=labs, average="macro", zero_division=0),
            f1_score(y[ok], pred[ok], labels=[0, 1, 2], average="macro", zero_division=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="logs/carepd/bank.pkl")
    ap.add_argument("--random", type=int, default=30)
    ap.add_argument("--sets-json", nargs="*", default=[], help="LLM 提案文件(llm_measure_propose 的输出格式)")
    ap.add_argument("--skip-bank", action="store_true")
    args = ap.parse_args()
    bank = pickle.load(open(args.bank, "rb"))
    feats = bank["features"]; col = {f: i for i, f in enumerate(feats)}
    rng = np.random.default_rng(0)
    print(f"{'队列':10s} {'方法':34s} {'F1_all':>7s} {'F1_0-2':>7s}   官方深度基线(最好 / 七个编码器均值)")
    summary = {}
    for c, d in bank["cohorts"].items():
        X, y, sub = d["X"], d["y"], d["subject"]
        folds = loso_folds(c, sub)
        best, mean = REPORTED[c]
        ref = f"{best if best else '—'} / {mean}"
        rows = [("手工 12 量 + LR", predict(X[:, [col[f] for f in CLASSIC]], y, folds, "lr"))]
        if not args.skip_bank:
            rows += [("全库 1573 量 + LR(C=0.01)", predict(X, y, folds, "lr_small_c")),
                     ("全库 1573 量 + 随机森林", predict(X, y, folds, "rf"))]
        for name, p in rows:
            a, b = scores(y, p); summary[(c, name)] = (a, b)
            print(f"{c:10s} {name:34s} {a:7.3f} {b:7.3f}   {ref}", flush=True)
        if args.random:
            rs = np.array([scores(y, predict(X[:, rng.choice(len(feats), len(CLASSIC), replace=False)], y, folds, "lr")) for _ in range(args.random)])
            print(f"{c:10s} {'随机 12 量 + LR (' + str(args.random) + ' 次)':34s} {rs[:, 0].mean():7.3f} {rs[:, 1].mean():7.3f}   ± {rs[:, 1].std():.3f}, 95 分位 {np.percentile(rs[:, 1], 95):.3f}", flush=True)
        for sj in args.sets_json:
            S = json.load(open(sj))
            vals = []
            for s in S["samples"]:
                cols = [col[x] for x in s["columns"] if x in col]
                if s["greedy"] or not cols:
                    continue
                vals.append(scores(y, predict(X[:, cols], y, folds, "lr")))
            if vals:
                v = np.array(vals)
                print(f"{c:10s} {'LLM ' + Path(sj).parent.name[:14] + ' ' + Path(sj).stem:34s} {v[:, 0].mean():7.3f} {v[:, 1].mean():7.3f}   ± {v[:, 1].std():.3f} (n={len(v)})", flush=True)
        print(f"{'':10s} 受试者 {len(set(sub))}, 行走 {len(y)}, 折 {len(folds)}, 多数类基线 F1_0-2 {f1_score(y, np.full(len(y), np.bincount(y).argmax()), labels=[0, 1, 2], average='macro', zero_division=0):.3f}")


if __name__ == "__main__":
    main()

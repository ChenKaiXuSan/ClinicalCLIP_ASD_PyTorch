#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: fusion_gate.py
Project: analysis/carepd
Author: Kaixu Chen
-----
Comment:
证据门控在 CARE-PD 上是否也成立:测量分支(先验 12 量 + 多项逻辑回归)与端到端分支(ST-GCN,train_deep.py 存下的
行走级概率)做决策级融合。与 ASD 的 uncertainty_fusion.py 同一套嵌套:6 折受试者划分,每个外层折的融合参数只在
其余折的折外预测上选,测试折只前向。

  deep / measure  两个分支单独
  fixed0.5        概率平均
  nested_w        全局权重 w(训练折上按宏平均 F1 选)
  gate            阶梯证据门控:偏离度 d(该次行走 12 个量的平均 |z|,z 用训练折统计量)低于训练折的 q 分位 -> w_lo,否则 w_hi
  gate_perm       把 d 在行走间随机置换后的门控(对照)
指标:各折预测拼起来的宏平均 F1(0/1/2),对 ST-GCN 的各次重复取均值 ± 标准差。

    python analysis/carepd/fusion_gate.py --bank logs/carepd/bank.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_measure import CLASSIC  # noqa: E402
from size_curve import six_folds  # noqa: E402

GRID_W = np.round(np.arange(0, 1.0001, 0.05), 2)
GRID_Q, GRID_LO, GRID_HI = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5), (0.0, 0.25, 0.5), (0.75, 1.0)


def f1(y, p):
    return f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)


def measure_branch(X, y, sub, folds, n_class):
    """折外的测量分支概率与偏离度。"""
    P = np.zeros((len(y), n_class)); d = np.zeros(len(y)); fold = np.full(len(y), -1)
    for fi, (tr_s, te_s) in enumerate(folds):
        tr, te = np.isin(sub, tr_s), np.isin(sub, te_s)
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(sc.transform(X[tr]), y[tr])
        P[np.ix_(np.where(te)[0], clf.classes_)] = clf.predict_proba(sc.transform(X[te]))
        d[te] = np.abs(sc.transform(X[te])).mean(1)
        fold[te] = fi
    return P, d, fold


def fuse(Pd, Pm, w):
    return (1 - np.asarray(w)[:, None]) * Pd + np.asarray(w)[:, None] * Pm


def run(Pd, Pm, d, y, fold, rng=None):
    if rng is not None:
        d = rng.permutation(d)
    out = {k: np.zeros(len(y), int) for k in ("deep", "measure", "fixed0.5", "nested_w", "gate")}
    for f in np.unique(fold):
        te, tr = fold == f, fold != f
        out["deep"][te] = Pd[te].argmax(1); out["measure"][te] = Pm[te].argmax(1)
        out["fixed0.5"][te] = fuse(Pd[te], Pm[te], np.full(te.sum(), 0.5)).argmax(1)
        best = max(GRID_W, key=lambda w: f1(y[tr], fuse(Pd[tr], Pm[tr], np.full(tr.sum(), w)).argmax(1)))
        out["nested_w"][te] = fuse(Pd[te], Pm[te], np.full(te.sum(), best)).argmax(1)
        bestg = None
        for q in GRID_Q:
            thr = np.quantile(d[tr], q)
            for lo in GRID_LO:
                for hi in GRID_HI:
                    s = f1(y[tr], fuse(Pd[tr], Pm[tr], np.where(d[tr] < thr, lo, hi)).argmax(1))
                    if bestg is None or s > bestg[0]:
                        bestg = (s, thr, lo, hi)
        _, thr, lo, hi = bestg
        out["gate"][te] = fuse(Pd[te], Pm[te], np.where(d[te] < thr, lo, hi)).argmax(1)
    return {k: f1(y, v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="logs/carepd/bank.pkl")
    ap.add_argument("--probs-dir", default="logs/carepd")
    ap.add_argument("--perm", type=int, default=20)
    args = ap.parse_args()
    bank = pickle.load(open(args.bank, "rb"))
    col = {f: i for i, f in enumerate(bank["features"])}
    cols = [col[f] for f in CLASSIC]
    keys = ["deep", "measure", "fixed0.5", "nested_w", "gate"]
    print(f"{'队列':10s} " + " ".join(f"{k:>14s}" for k in keys) + f" {'gate_perm':>14s}   门控 - max(单分支)")
    for c, dct in bank["cohorts"].items():
        f = Path(args.probs_dir) / f"deep_probs_{c}.pkl"
        if not f.exists():
            print(f"{c:10s} 缺 {f}"); continue
        D = pickle.load(open(f, "rb"))
        y, sub = dct["y"], dct["subject"]
        assert len(D["y"]) == len(y) and (D["y"] == y).all() and (D["subject"] == sub).all(), "两个分支的行走顺序不一致"
        n_class = D["probs"].shape[2]
        Pm, d, fold = measure_branch(dct["X"][:, cols], y, sub, six_folds(c), n_class)
        res = [run(Pd, Pm, d, y, fold) for Pd in D["probs"]]
        rng = np.random.default_rng(0)
        perm = [run(D["probs"][i % len(D["probs"])], Pm, d, y, fold, rng)["gate"] for i in range(args.perm)]
        m = {k: np.array([r[k] for r in res]) for k in keys}
        gain = m["gate"] - np.maximum(m["deep"], m["measure"])
        print(f"{c:10s} " + " ".join(f"{m[k].mean():.3f} ± {m[k].std():.3f}" for k in keys) + f" {np.mean(perm):.3f} ± {np.std(perm):.3f}   {gain.mean():+.3f} ({(gain > 0).sum()}/{len(gain)})")


if __name__ == "__main__":
    main()

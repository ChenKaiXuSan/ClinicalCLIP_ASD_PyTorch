#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: eval_gate.py
Project: analysis/koa
Author: Kaixu Chen
-----
Comment:
KOA-PD-NM 上的测量分支与证据门控,做法逐项对应 ASD 的 analysis/uncertainty_fusion.py:

  测量分支   先验 12 量(common.PRIOR,受试者级 = 该受试者各段视频的均值)-> 中位数填补 -> 标准化 -> 多项逻辑回归(C=1, balanced),
             在每折的开发集(train + val)上拟合,test 受试者只前向
  偏离度     该受试者 12 个量的平均 |z|,z 用开发集统计量
  阶梯门控   偏离度低于训练折的 q 分位 -> w_lo,否则 w_hi;(q, w_lo, w_hi) 在其余折的折外预测上按(平衡准确率, -NLL)选,
             网格与 ASD 完全相同
  对照       固定 0.5、嵌套全局 w、偏离度在受试者间置换后的门控

第二个分支从 logs/koa/video/<tag>_s<seed>.pkl 读(tag = rgb / masked / sil / skel),每个 tag 报各种子的均值 ± 标准差。
另报:测量分支在 10 份随机划分上的成绩、随机 12 量对照、只用年龄 / 年龄 + 性别的逻辑回归(混杂检查)。
指标:受试者级平衡准确率(三类 NM / KOA / PD);--binary 时把 KOA 与 PD 合并成"异常"。

    python analysis/koa/eval_gate.py --tags masked sil rgb skel
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PRIOR, PRIOR_KOA, PRIOR_PD, load_gait, splits, subject_table  # noqa: E402

GRID_W = np.round(np.arange(0, 1.0001, 0.05), 2)
GRID_Q, GRID_LO, GRID_HI = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5), (0.0, 0.25, 0.5), (0.75, 1.0)
NC = 3


def bacc(y, p) -> float:
    return float(np.mean([(p[y == c] == c).mean() for c in range(NC) if (y == c).any()]))


def nll(P, y) -> float:
    return float(-np.log(P[np.arange(len(y)), y] + 1e-9).mean())


def fuse(Pv, Pm, w):
    w = np.asarray(w, dtype=float).reshape(-1, 1)
    return (1 - w) * Pv + w * Pm


def fit_lr(Xtr, ytr, Xte):
    """返回 (test 概率, test 偏离度)。"""
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    Ztr, Zte = sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xte))
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(Ztr, ytr)
    P = np.zeros((len(Xte), NC)); P[:, clf.classes_] = clf.predict_proba(Zte)
    return P, np.abs(Zte).mean(1)


def oof(X, y, folds):
    P = np.zeros((len(y), NC)); d = np.zeros(len(y))
    for tr, te in folds:
        P[te], d[te] = fit_lr(X[tr], y[tr], X[te])
    return P, d


def gate_run(Pv, Pm, d, y, fold):
    pred = {k: np.zeros(len(y), int) for k in ("second", "measure", "fixed0.5", "nested_w", "gate")}
    for k in np.unique(fold):
        te, tr = fold == k, fold != k
        pred["second"][te], pred["measure"][te] = Pv[te].argmax(1), Pm[te].argmax(1)
        pred["fixed0.5"][te] = fuse(Pv[te], Pm[te], np.full(te.sum(), 0.5)).argmax(1)
        best = None
        for w in GRID_W:
            pf = fuse(Pv[tr], Pm[tr], np.full(tr.sum(), w))
            key = (bacc(y[tr], pf.argmax(1)), -nll(pf, y[tr]))
            if best is None or key > best[0]:
                best = (key, w)
        pred["nested_w"][te] = fuse(Pv[te], Pm[te], np.full(te.sum(), best[1])).argmax(1)
        best = None
        for q in GRID_Q:
            thr = float(np.quantile(d[tr], q))
            for lo in GRID_LO:
                for hi in GRID_HI:
                    pf = fuse(Pv[tr], Pm[tr], np.where(d[tr] < thr, lo, hi))
                    key = (bacc(y[tr], pf.argmax(1)), -nll(pf, y[tr]))
                    if best is None or key > best[0]:
                        best = (key, thr, lo, hi)
        _, thr, lo, hi = best
        pred["gate"][te] = fuse(Pv[te], Pm[te], np.where(d[te] < thr, lo, hi)).argmax(1)
    out = {k: bacc(y, v) for k, v in pred.items()}
    out["ceiling"] = float(np.mean([((pred["second"] == y) | (pred["measure"] == y))[y == c].mean() for c in range(NC)]))
    return out


def collapse(P):
    return np.stack([P[:, 0], P[:, 1:].sum(1)], 1)


def main() -> None:
    global NC
    ap = argparse.ArgumentParser()
    ap.add_argument("--gait", default="logs/koa/gait.pkl")
    ap.add_argument("--video-dir", default="logs/koa/video")
    ap.add_argument("--tags", nargs="+", default=["masked", "sil", "rgb", "skel"])
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--random", type=int, default=200)
    ap.add_argument("--binary", action="store_true")
    args = ap.parse_args()
    G = load_gait(args.gait)
    subs, y3, X = subject_table(G)
    y = (y3 > 0).astype(int) if args.binary else y3
    NC = 2 if args.binary else 3
    col = {f: i for i, f in enumerate(G["features"])}
    cols = [col[f] for f in PRIOR]
    sp = splits(subs, y3)
    folds = [(np.concatenate([s["train"], s["val"]]), s["test"]) for s in sp]
    fold = np.zeros(len(y), int)
    for k, (_, te) in enumerate(folds):
        fold[te] = k
    print(f"受试者 {len(y)}, 类别 {np.bincount(y).tolist()}, 任务 {'异常 vs 正常' if args.binary else 'NM / KOA / PD'}")

    # ---- 测量分支单独
    def cv10(Xs):
        v = []
        for s in range(10):
            f = list(StratifiedKFold(5, shuffle=True, random_state=1000 + s).split(Xs, y3))
            v.append(bacc(y, oof(Xs, y, f)[0].argmax(1)))
        return float(np.mean(v)), float(np.std(v))
    sets = {"先验 12 量": cols, "只用帕金森的 8 量": [col[f] for f in PRIOR_PD], "只用膝关节炎的 4 量": [col[f] for f in PRIOR_KOA],
            "全部 30 量": list(range(X.shape[1]))}
    print("== 测量分支单独(10 份随机 5 折划分的均值 ± 标准差 | 与视频同一划分)")
    for n, c in sets.items():
        m, s = cv10(X[:, c])
        print(f"{n:16s} {m:.3f} ± {s:.3f} | {bacc(y, oof(X[:, c], y, folds)[0].argmax(1)):.3f}")
    rng = np.random.default_rng(0)
    rnd = np.array([bacc(y, oof(X[:, rng.choice(X.shape[1], len(cols), replace=False)], y, folds)[0].argmax(1)) for _ in range(args.random)])
    ref = bacc(y, oof(X[:, cols], y, folds)[0].argmax(1))
    print(f"随机 12 量({args.random} 次, 同一划分): {rnd.mean():.3f} ± {rnd.std():.3f}, 先验 12 量处于第 {(rnd < ref).mean() * 100:.0f} 百分位")
    demo = G["demo"]
    A = np.array([[demo[s][1]] for s in subs], float); AS = np.c_[A, [[demo[s][0] == "M"] for s in subs], [[demo[s][2]] for s in subs]].astype(float)
    print(f"混杂检查: 只用年龄 {cv10(A)[0]:.3f}, 年龄 + 性别 + 身高 {cv10(AS)[0]:.3f}, 先验 12 量 + 年龄性别身高 {cv10(np.c_[X[:, cols], AS])[0]:.3f}")

    # ---- 门控
    Pm, d = oof(X[:, cols], y, folds)
    keys = ["second", "measure", "fixed0.5", "nested_w", "gate"]
    print("\n== 证据门控(第二个分支逐种子,均值 ± 标准差)")
    print(f"{'第二个分支':10s} {'种子数':>4s} " + " ".join(f"{k:>14s}" for k in keys) + f" {'gate_perm':>14s} {'上限':>6s}   门控 - max(单分支)  门控 > 置换的比例")
    for tag in args.tags:
        fs = sorted(Path(args.video_dir).glob(f"{tag}_s*.pkl"))
        if not fs:
            print(f"{tag:10s} 没有结果"); continue
        res, perms = [], []
        for f in fs:
            V = pickle.load(open(f, "rb"))
            assert (V["subjects"] == subs).all() and (V["fold"] == fold).all(), "视频分支与测量分支的受试者或划分不一致"
            Pv = collapse(V["prob"]) if args.binary else V["prob"]
            res.append(gate_run(Pv, Pm, d, y, fold))
            pr = np.random.default_rng(1)
            perms.append([gate_run(Pv, Pm, pr.permutation(d), y, fold)["gate"] for _ in range(args.perm // len(fs))])
        m = {k: np.array([r[k] for r in res]) for k in keys + ["ceiling"]}
        gain = m["gate"] - np.maximum(m["second"], m["measure"])
        allp = np.concatenate(perms)
        print(f"{tag:10s} {len(fs):4d} " + " ".join(f"{m[k].mean():.3f} ± {m[k].std():.3f}" for k in keys) +
              f" {allp.mean():.3f} ± {allp.std():.3f} {m['ceiling'].mean():6.3f}   {gain.mean():+.3f} ({(gain > 0).sum()}/{len(gain)})"
              f"        {np.mean([(np.array(p) < g).mean() for p, g in zip(perms, m['gate'])]):.2f}")


if __name__ == "__main__":
    main()

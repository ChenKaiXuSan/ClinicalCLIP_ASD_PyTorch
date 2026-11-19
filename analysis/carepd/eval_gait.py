#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: eval_gait.py
Project: analysis/carepd
Author: Kaixu Chen
-----
Comment:
CARE-PD 第二版评测:步态周期测量(gait_params.py)对第一版静态几何测量(build_bank.py),三种协议。

  loso   队列内留一受试者(官方主协议)
  lodo   留一队列:在另外三个队列上训练,目标队列一个样本也不用
  mida   队列内留一受试者 + 把另外三个队列全部并入训练(多队列域内适配)
  后缀 _cn = 特征按队列各自标准化(目标队列只用它的训练受试者;lodo_cn 用到目标队列的无标签统计量,属直推式,单列)

测量集合:v1_prior12(第一版手写 12 量)、rubric8(只取 UPDRS 3.10 点名的 8 个周期量)、gait30(全部周期量)、gait30+v1(合并)。
分类器:lr(C=1 不调)、lr_cv(C 在训练折内按受试者分组的 5 折选)、rf。缺失值用训练折中位数填补。
指标:各折预测拼起来的宏平均 F1(0/1/2)。所有选择只发生在训练折内。

    python analysis/carepd/eval_gait.py --protocols loso --models lr lr_cv rf
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_measure import CLASSIC, loso_folds  # noqa: E402
from gait_params import RUBRIC  # noqa: E402

warnings.filterwarnings("ignore")
CS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def f1(y, p):
    return f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)


def prep(Xtr, Xte):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    return sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xte))


def lr(C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=3000)


def fit_predict(Xtr, ytr, gtr, Xte, model: str, seed: int = 0, proba: bool = False, n_class: int = 4):
    """Xtr / Xte 已标准化。lr_cv 的 C 只在训练折内选。"""
    if model == "rf":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=8, random_state=seed)
    elif model == "lr_cv":
        n_g = len(set(gtr))
        best, best_c = -1, 1.0
        if n_g >= 4:
            sp = list(GroupKFold(min(5, n_g)).split(Xtr, ytr, gtr))
            for C in CS:
                p = np.full(len(ytr), -1)
                for a, b in sp:
                    if len(set(ytr[a])) > 1:
                        p[b] = lr(C).fit(Xtr[a], ytr[a]).predict(Xtr[b])
                s = f1(ytr[p >= 0], p[p >= 0])
                if s > best + 1e-9:
                    best, best_c = s, C
        clf = lr(best_c)
    else:
        clf = lr(1.0)
    clf.fit(Xtr, ytr)
    if not proba:
        return clf.predict(Xte)
    P = np.zeros((len(Xte), n_class)); P[:, clf.classes_] = clf.predict_proba(Xte)
    return P


def cohort_norm(X, cohort, fit_mask):
    """按队列各自标准化:每个队列用自己在 fit_mask 内的行估计中位数填补与均值 / 标准差。"""
    Z = np.zeros_like(X, dtype=np.float64)
    for c in np.unique(cohort):
        m = cohort == c
        ref = X[m & fit_mask] if (m & fit_mask).any() else X[m]
        _, Z[m] = prep(ref, X[m])
    return Z


def run(D, cols_by_set, protocol: str, model: str, target: str, proba: bool = False):
    X, y, sub, coh = D
    tgt = coh == target
    cn = protocol.endswith("_cn"); base = protocol.replace("_cn", "")
    idx_t = np.where(tgt)[0]
    if base == "lodo":
        folds = [(np.where(~tgt)[0], idx_t)]
    else:
        folds = []
        for tr, te in loso_folds(target, sub[tgt]):
            tr_idx = idx_t[tr]
            if base == "mida":
                tr_idx = np.concatenate([tr_idx, np.where(~tgt)[0]])
            folds.append((tr_idx, idx_t[te]))
    out = {}
    for name, cols in cols_by_set.items():
        Xs = X[:, cols]
        pred = np.full((len(y), 4), np.nan) if proba else np.full(len(y), -1)
        for tr, te in folds:
            if cn:
                fit = np.zeros(len(y), bool); fit[tr] = True
                if base == "lodo":
                    fit[te] = True                      # 直推式:目标队列的无标签统计量
                Z = cohort_norm(Xs, coh, fit); Xtr, Xte = Z[tr], Z[te]
            else:
                Xtr, Xte = prep(Xs[tr], Xs[te])
            g = np.array([f"{c}/{s}" for c, s in zip(coh[tr], sub[tr])])
            pred[te] = fit_predict(Xtr, y[tr], g, Xte, model, proba=proba)
        out[name] = pred[tgt]
    return out, y[tgt]


def load(gait_path: str, bank_path: str):
    """特征矩阵 = [30 个步态周期量 | 第一版的 1573 个静态几何量];列名 gait|<name> 与第一版的原名。"""
    G = pickle.load(open(gait_path, "rb")); B = pickle.load(open(bank_path, "rb"))
    X, y, sub, coh = [], [], [], []
    for c in G["cohorts"]:
        g, b = G["cohorts"][c], B["cohorts"][c]
        assert (g["walk"] == b["walk"]).all()
        X.append(np.concatenate([g["X"], b["X"]], 1)); y.append(g["y"]); sub.append(g["subject"].astype(str)); coh += [c] * len(g["y"])
    names = [f"gait|{f}" for f in G["features"]] + list(B["features"])
    return (np.concatenate(X).astype(np.float64), np.concatenate(y), np.concatenate(sub), np.array(coh)), names, list(G["cohorts"])


def f1_all(y, p):
    """官方的 0-3 口径:真值与预测中出现过的全部类别取宏平均。"""
    return f1_score(y, p, average="macro", zero_division=0)


def boot_ci(y, p, sub, n: int = 2000, seed: int = 0):
    """按受试者重抽样的 95% 置信区间。"""
    rng = np.random.default_rng(seed)
    idx = {u: np.where(sub == u)[0] for u in np.unique(sub)}
    us = list(idx)
    vals = []
    for _ in range(n):
        ii = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        vals.append(f1(y[ii], p[ii]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gait", default="logs/carepd/gait.pkl")
    ap.add_argument("--bank", default="logs/carepd/bank.pkl")
    ap.add_argument("--protocols", nargs="+", default=["loso"])
    ap.add_argument("--models", nargs="+", default=["lr", "lr_cv", "rf"])
    ap.add_argument("--random", type=int, default=0, help="随机对照的次数(只在 loso + lr 下做)")
    ap.add_argument("--sets-json", nargs="*", default=[], help="LLM 提案文件(llm_propose.py 的输出)")
    ap.add_argument("--ci", action="store_true", help="rubric8 的受试者自助法置信区间与 0-3 口径")
    ap.add_argument("--out", default="logs/carepd/eval_gait.json")
    args = ap.parse_args()
    D, names, cohorts = load(args.gait, args.bank)
    col = {n: i for i, n in enumerate(names)}
    ng = sum(n.startswith("gait|") for n in names)
    sets = {"v1_prior12": [col[f] for f in CLASSIC], "rubric8": [col[f"gait|{f}"] for f in RUBRIC], "gait30": list(range(ng))}
    res = {}
    for proto in args.protocols:
        for model in args.models:
            print(f"== 协议 {proto}  分类器 {model}")
            print(f"{'队列':10s} " + " ".join(f"{s:>11s}" for s in sets))
            for c in cohorts:
                out, yt = run(D, sets, proto, model, c)
                row = {s: float(f1(yt, p)) for s, p in out.items()}
                res[f"{proto}|{model}|{c}"] = row
                extra = ""
                if args.ci:
                    lo, hi = boot_ci(yt, out["rubric8"], D[2][D[3] == c])
                    extra = f"   rubric8: 0-3 口径 {f1_all(yt, out['rubric8']):.3f}, 95% CI [{lo:.3f}, {hi:.3f}]"
                    res[f"{proto}|{model}|{c}"].update({"rubric8_f1_all": float(f1_all(yt, out["rubric8"])), "rubric8_ci": [lo, hi]})
                print(f"{c:10s} " + " ".join(f"{row[s]:11.3f}" for s in sets) + extra, flush=True)
    if args.random:
        rng = np.random.default_rng(0)
        k = len(RUBRIC)
        r_g = {f"g{i}": list(rng.choice(ng, k, replace=False)) for i in range(args.random)}
        r_a = {f"a{i}": list(rng.choice(len(names), k, replace=False)) for i in range(args.random)}
        print(f"== 随机对照(loso, lr, 每次 {k} 个量, {args.random} 次): 先验 rubric8 所在的百分位")
        for c in cohorts:
            out, yt = run(D, {"rubric8": sets["rubric8"], **r_g, **r_a}, "loso", "lr", c)
            ref = f1(yt, out["rubric8"])
            vg = np.array([f1(yt, out[n]) for n in r_g]); va = np.array([f1(yt, out[n]) for n in r_a])
            res[f"random|{c}"] = {"rubric8": float(ref), "gait30_pool": vg.tolist(), "all_pool": va.tolist()}
            print(f"{c:10s} rubric8 {ref:.3f} | 从 30 个周期量里随机: {vg.mean():.3f} ± {vg.std():.3f}, 百分位 {(vg < ref).mean() * 100:.0f} | "
                  f"从全部 {len(names)} 个量里随机: {va.mean():.3f} ± {va.std():.3f}, 百分位 {(va < ref).mean() * 100:.0f}", flush=True)
    for sj in args.sets_json:
        S = json.load(open(sj))
        ls = {f"s{s['sample']}": [col[x] for x in s["columns"] if x in col] for s in S["samples"] if not s["greedy"]}
        ls = {n: v for n, v in ls.items() if v}
        tag = f"{Path(sj).parent.name[:12]} {Path(sj).stem}"
        frac = np.mean([np.mean([x.startswith('gait|') for x in s["columns"]]) for s in S["samples"] if s["columns"]])
        row = []
        for c in cohorts:
            out, yt = run(D, ls, "loso", "lr", c)
            v = np.array([f1(yt, p) for p in out.values()])
            row.append(f"{v.mean():.3f} ± {v.std():.3f}"); res[f"llm|{tag}|{c}"] = v.tolist()
        print(f"LLM {tag:28s} " + "  ".join(row) + f"   (n={len(ls)}, 周期量占比 {frac:.2f})", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

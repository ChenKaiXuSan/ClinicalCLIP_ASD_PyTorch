#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: size_curve.py
Project: analysis/carepd
Author: Kaixu Chen
-----
Comment:
训练样本量曲线(测量侧):逐队列用数据集自带的 6 折受试者划分;每折从训练受试者里随机抽 k 人训练,在该折的
测试受试者上预测;各折预测拼起来算宏平均 F1(0/1/2),重复 R 次取均值 ± 标准差。k 取 2、4、8、12、16、全部。
深度模型的同口径曲线由 train_deep.py 给出(同样的折、同样的 k、同样的抽样种子)。

    python analysis/carepd/size_curve.py --bank logs/carepd/bank.pkl --out logs/carepd/size_curve_measure.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_measure import CLASSIC, ROOT  # noqa: E402

KS = [2, 4, 8, 12, 16, None]


def six_folds(cohort: str) -> list[tuple[list[str], list[str]]]:
    f = ROOT / "folds" / "UPDRS_Datasets" / f"{cohort}_6fold_participants.pkl"
    return [([str(s) for s in sp["train"]], [str(s) for s in sp["eval"]]) for _, sp in sorted(pickle.load(open(f, "rb")).items())]


def subsample(train_subjects: list[str], k: int | None, rep: int, fold: int) -> list[str]:
    """与 train_deep.py 共用:同一 (fold, k, rep) 抽到同一批训练受试者。"""
    if k is None or k >= len(train_subjects):
        return list(train_subjects)
    rng = np.random.default_rng(10007 * rep + 101 * fold + k)
    return list(rng.choice(sorted(train_subjects), k, replace=False))


def fit_predict(Xtr, ytr, Xte, model: str, seed: int):
    if len(set(ytr)) < 2:
        return np.full(len(Xte), ytr[0])
    sc = StandardScaler().fit(Xtr)
    if model == "rf":
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=8, random_state=seed)
    else:
        clf = LogisticRegression(C=1.0 if model == "lr" else 0.01, class_weight="balanced", max_iter=3000)
    return clf.fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="logs/carepd/bank.pkl")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default="logs/carepd/size_curve_measure.json")
    ap.add_argument("--sets-json", default=None, help="LLM 提案文件:用其中每个采样集合再画一条曲线(各集合取均值)")
    args = ap.parse_args()
    bank = pickle.load(open(args.bank, "rb"))
    col = {f: i for i, f in enumerate(bank["features"])}
    methods = {"prior12_lr": ("lr", [col[f] for f in CLASSIC]), "bank_lr": ("lr_c", None), "bank_rf": ("rf", None)}
    llm_sets = []
    if args.sets_json:
        S = json.load(open(args.sets_json))
        llm_sets = [[col[x] for x in s["columns"] if x in col] for s in S["samples"] if not s["greedy"] and s["columns"]]
    out = {}
    for c, d in bank["cohorts"].items():
        X, y, sub = d["X"], d["y"], d["subject"]
        folds = six_folds(c)
        print(f"== {c}: 受试者 {len(set(sub))}, 每折训练受试者 {[len(t) for t, _ in folds]}")
        print(f"{'训练受试者数 k':>14s} " + " ".join(f"{m:>18s}" for m in list(methods) + (["llm_lr"] if llm_sets else [])))
        out[c] = {}
        for k in KS:
            row = {}
            names = list(methods) + (["llm_lr"] if llm_sets else [])
            for m in names:
                vals = []
                for rep in range(args.reps):
                    pred = np.full(len(y), -1)
                    for fi, (tr_s, te_s) in enumerate(folds):
                        tr = np.isin(sub, subsample(tr_s, k, rep, fi)); te = np.isin(sub, te_s)
                        if m == "llm_lr":
                            cols = llm_sets[rep % len(llm_sets)]; model = "lr"
                        else:
                            model, cols = methods[m]
                        Xc = X if cols is None else X[:, cols]
                        pred[te] = fit_predict(Xc[tr], y[tr], Xc[te], model, rep)
                    ok = pred >= 0
                    vals.append(f1_score(y[ok], pred[ok], labels=[0, 1, 2], average="macro", zero_division=0))
                    if k is None and m != "llm_lr" and m != "bank_rf":
                        break                      # 全量且确定性的方法不需要重复
                row[m] = (float(np.mean(vals)), float(np.std(vals)))
            out[c]["all" if k is None else str(k)] = row
            print(f"{'全部' if k is None else k:>14} " + " ".join(f"{row[m][0]:.3f} ± {row[m][1]:.3f}".rjust(18) for m in names), flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

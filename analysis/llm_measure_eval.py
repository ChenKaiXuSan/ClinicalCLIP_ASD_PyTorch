#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: llm_measure_eval.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
评测 llm_measure_propose.py 的产出:LLM 提出的测量 -> 测量库的列 -> 患者级逻辑回归。
口径与 5 量 LR 的 0.717 相同:67 人(去掉有视频无 3D 的患者),患者值 = 全部段的均值,
StandardScaler + LR(C=1, balanced),10 份 StratifiedKFold(5, shuffle, random_state=1000+s)。

对照(同一口径内重算,逐份可比):
  hand5      作者手工定的 5 个量(库内 7 列)
  random-k   全库随机 k 列(k = LLM 集合的列数),R 次,给出各条件均值所在的百分位
  l1-all     全库 L1 逻辑回归(看数据的选择)

    python analysis/llm_measure_eval.py --root logs/llm_measure
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HAND5 = ["incl|sh_mid-hip_mid|mean", "dx|sh_mid-hip_mid|mean", "incl|ear_mid-sh_mid|mean",
         "ang|sh_mid-hip_mid-kn_L|min", "ang|sh_mid-hip_mid-kn_R|min",
         "ang|sh_mid-hip_mid-kn_L|range", "ang|sh_mid-hip_mid-kn_R|range"]
CONDS = ["marked_task", "marked_blind", "unmarked_task", "unmarked_blind", "noregion_task"]


def load_patients(bank_path: str):
    bank = pickle.load(open(bank_path, "rb"))
    feats = list(bank["features"])
    miss = {Path(v["json"]).stem.split("-")[0] for v in bank["videos"].values() if v["frac_ok"] == 0}
    rows: dict[str, list] = {}
    lab: dict[str, int] = {}
    for v in bank["videos"].values():
        p = Path(v["json"]).stem.split("-")[0]
        if p in miss:
            continue
        rows.setdefault(p, []).append(np.asarray(v["vals"], dtype=np.float64))
        lab[p] = int(v["label"] == 0)
    pids = sorted(rows)
    X = np.array([np.nanmean(np.concatenate(rows[p], 0), 0) for p in pids])
    X = np.where(np.isnan(X), np.nanmedian(X, 0), X)
    y = np.array([lab[p] for p in pids])
    return feats, X, y


def macro(pred, y):
    return float(np.mean([((pred == c) & (y == c)).sum() / max((y == c).sum(), 1) for c in (0, 1)]))


def evaluate(X, y, n=10, C=1.0, penalty="l2"):
    ms, aucs = [], []
    for s in range(n):
        pred, prob = np.zeros(len(y), int), np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=1000 + s).split(X, y):
            sc = StandardScaler().fit(X[tr])
            kw = {"solver": "liblinear"} if penalty == "l1" else {}
            clf = LogisticRegression(C=C, penalty=penalty, class_weight="balanced", max_iter=2000, **kw).fit(sc.transform(X[tr]), y[tr])
            pred[te] = clf.predict(sc.transform(X[te])); prob[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        ms.append(macro(pred, y)); aucs.append(roc_auc_score(y, prob))
    return float(np.mean(ms)), float(np.mean(aucs)), float(np.std(ms))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/llm_measure")
    ap.add_argument("--bank", default="logs/measure_bank_3d/bank.pkl")
    ap.add_argument("--random", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    feats, X, y = load_patients(args.bank)
    col = {f: i for i, f in enumerate(feats)}
    print(f"患者 {len(y)} (ASD {y.sum()}),测量库 {len(feats)} 列")

    def score(cols):
        idx = [col[c] for c in cols if c in col]
        return evaluate(X[:, idx], y) if idx else (float("nan"), float("nan"), 0.0)

    ref = score(HAND5)
    print(f"hand5(作者手工, 7 列): macro {ref[0]:.3f}  AUC {ref[1]:.3f}")
    l1 = evaluate(X, y, C=0.1, penalty="l1")
    print(f"l1-all(全库 L1, C=0.1): macro {l1[0]:.3f}  AUC {l1[1]:.3f}")

    results = {}
    sizes = Counter()
    for mdir in sorted(p for p in Path(args.root).iterdir() if p.is_dir()):
        print(f"\n===== {mdir.name}")
        print(f"{'条件':16s} {'样本':>4s} {'合法率':>6s} {'列数':>5s} {'macro 均值±sd':>15s} {'AUC':>6s} {'贪心':>6s} {'共识集':>6s} {'含躯干倾角':>9s}")
        for cond in CONDS:
            f = mdir / f"{cond}.json"
            if not f.exists():
                continue
            d = json.load(open(f))
            S = [s for s in d["samples"] if not s["greedy"]]
            G = [s for s in d["samples"] if s["greedy"]]
            sc = [score(s["columns"]) for s in S if s["columns"]]
            for s in S:
                sizes[len(s["columns"])] += 1
            valid = np.mean([len(s["items"]) / max(len(s["items"]) + s["n_invalid"], 1) for s in S])
            # 共识集:所有采样里出现频率最高的 5 个条目(按条目而非列计数)
            cnt = Counter()
            item_cols = {}
            for s in S:
                seen = set()
                for it in s["items"]:
                    key = json.dumps({k: it[k] for k in ("type", "upper", "lower", "quantity", "angle", "stat") if k in it}, sort_keys=True)
                    if key in seen:
                        continue
                    seen.add(key); cnt[key] += 1
            from llm_measure_propose import parse  # noqa: E402
            top = [k for k, _ in cnt.most_common(5)]
            cons_cols = parse(json.dumps([json.loads(k) for k in top]))[1]
            cons = score(cons_cols)
            g = score(G[0]["columns"]) if G and G[0]["columns"] else (float("nan"),) * 3
            has_lean = np.mean([any(c in s["columns"] for c in ("incl|sh_mid-hip_mid|mean", "dx|sh_mid-hip_mid|mean")) for s in S])
            m = np.array([x[0] for x in sc]); a = np.array([x[1] for x in sc])
            results[(mdir.name, cond)] = {"macro": m.tolist(), "auc": a.tolist(), "greedy": g[0], "consensus": cons[0],
                                           "consensus_cols": cons_cols, "top_items": cnt.most_common(8), "n_cols": [len(s["columns"]) for s in S]}
            print(f"{cond:16s} {len(sc):4d} {valid:6.2f} {np.mean([len(s['columns']) for s in S]):5.1f} "
                  f"{m.mean():.3f} ± {m.std():.3f} {a.mean():6.3f} {g[0]:6.3f} {cons[0]:6.3f} {has_lean:9.2f}")
        for cond in ("marked_task", "unmarked_task"):
            r = results.get((mdir.name, cond))
            if r:
                print(f"  [{cond}] 最常提出的测量:")
                for k, c in r["top_items"][:6]:
                    it = json.loads(k)
                    desc = f"{it['quantity']}({it['upper']}, {it['lower']})" if it["type"] == "pair" else f"angle {it['angle']}"
                    print(f"      {c:2d} 次  {desc} · {it['stat']}")

    # 随机 k 列对照
    rng = np.random.default_rng(0)
    print("\n随机 k 列(全库):")
    rand = {}
    for k in sorted({k for k in sizes if k > 0} | {7}):
        if sizes.get(k, 0) < 3 and k != 7:
            continue
        ms = np.array([evaluate(X[:, rng.choice(X.shape[1], k, replace=False)], y, n=5)[0] for _ in range(args.random)])
        rand[k] = ms
        print(f"  k={k}: {ms.mean():.3f} ± {ms.std():.3f}  [5%, 95%] = [{np.percentile(ms, 5):.3f}, {np.percentile(ms, 95):.3f}]  hand5 所在百分位 {(ms < ref[0]).mean() * 100:.0f}")
    allr = np.concatenate(list(rand.values())) if rand else np.array([])
    print("\n各条件均值在随机对照中的百分位:")
    for (mn, cond), r in results.items():
        if len(r["macro"]) and len(allr):
            mm = float(np.mean(r["macro"]))
            print(f"  {mn:24s} {cond:16s} {mm:.3f} -> 第 {(allr < mm).mean() * 100:.0f} 百分位;采样中超过随机 95 分位的比例 {np.mean(np.array(r['macro']) > np.percentile(allr, 95)):.2f}")
    if args.out:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in results.items()}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()

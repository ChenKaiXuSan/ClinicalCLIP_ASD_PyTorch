#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: prior_sparse_lr.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
方法 1:**医生注意力先验引导的稀疏测量选择**。
在 measure_bank_3d.py 的测量库上做 L1 逻辑回归,每个测量的惩罚强度由它涉及的医生区域的标注先验决定:
  p_j = (pi_bar / pi_j) ** alpha,  pi_j = 该测量涉及区域的先验按 --prior-agg 聚合(max / min / gmean), other 取最低区域值
  实现: 标准化后把第 j 列除以 p_j,再做普通 L1 —— 等价于对原系数施加 p_j |beta_j| 的惩罚。
alpha = 0 即均匀 L1(无先验)。

两种训练层级(--level):
  patient  段 -> 患者取均值后 79 行(与 attribute_classifier.py 同口径)
  segment  14k 段直接训练,按患者分组做内外层 CV,测试患者的段概率取均值再判类
5 折按 index.json 的患者划分;C 在每个外折的训练患者上做 4 折内层交叉验证选(患者级平衡准确率)。

变体(--variants):
  doctor    医生先验            uniform  均匀 L1            inverted  反先验(高标注区域惩罚更重)
  random    区域组 -> 先验的映射随机置换(--n-random 次)      hard_doctor  只用涉及 lumbar/head/shoulder 的测量 + 均匀 L1
  hard_other 只用不涉及这三区的测量 + 均匀 L1                hand5   只用手挑 5 个量在库里的等价列(检查库里有没有那份信息)
报告: 患者级平衡准确率 [bootstrap CI]、每折选中测量数、跨折选择稳定性(Jaccard)、选中测量的区域分布、高频测量;
doctor 对 uniform 的 McNemar;doctor 在随机先验分布中的百分位。

    python analysis/prior_sparse_lr.py --bank logs/measure_bank_3d/bank.pkl --data-root $DATA --alpha 1.0 --prior-agg min --level patient
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patient_level_stats import macro_ci  # noqa: E402

warnings.filterwarnings("ignore")
DOCTOR_REGIONS = ("lumbar_pelvis", "head", "shoulder")
DEFAULT_PRIOR = {"lumbar_pelvis": 0.715, "head": 0.468, "shoulder": 0.234, "wrist": 0.025, "foot": 0.013}
# 手挑 5 个量在测量库里的等价列
HAND5 = ["incl|sh_mid-hip_mid|mean", "dx|sh_mid-hip_mid|mean", "incl|ear_mid-sh_mid|mean",
         "ang|sh_mid-hip_mid-kn_L|min", "ang|sh_mid-hip_mid-kn_R|min",
         "ang|sh_mid-hip_mid-kn_L|range", "ang|sh_mid-hip_mid-kn_R|range"]


def macro(pred, y):
    return float(np.mean([((pred == c) & (y == c)).sum() / max((y == c).sum(), 1) for c in np.unique(y)]))


def load_bank(path: str, data_root: str, class_num: int):
    bank = pickle.load(open(path, "rb"))
    feats, regions = bank["features"], bank["regions"]
    per_pat = defaultdict(list)
    label = {}
    for name, v in bank["videos"].items():
        pid = Path(v["json"]).stem.split("-")[0]
        per_pat[pid].append(v["vals"])
        label[pid] = int(v["label"] == 0)  # ASD -> 1
    pids = sorted(per_pat)
    Xp = np.array([np.nanmean(np.concatenate(per_pat[p]), axis=0) for p in pids])
    y = np.array([label[p] for p in pids])
    Xs = np.concatenate([np.concatenate(per_pat[p]) for p in pids])
    seg_pat = np.concatenate([np.full(sum(len(a) for a in per_pat[p]), i) for i, p in enumerate(pids)])
    folds = json.load(open(Path(data_root) / "clinical_CLIP_dataset" / "index_mapping" / str(class_num) / "index.json"))
    pat_index = {p: i for i, p in enumerate(pids)}
    splits = []
    for k in sorted(folds, key=lambda s: int(s)):
        s = folds[k]
        test_p = {Path(x).stem.split("-")[0] for x in s["test"]}
        train_p = {Path(x).stem.split("-")[0] for x in s["train"] + s["val"]}
        splits.append((np.array([pat_index[p] for p in pids if p in train_p]),
                       np.array([pat_index[p] for p in pids if p in test_p])))
    return Xp, Xs, seg_pat, y, pids, feats, regions, splits


def feature_prior(regions, prior: dict, other: float, agg: str) -> np.ndarray:
    out = []
    for rs in regions:
        v = np.array([prior.get(r, other) for r in rs])
        out.append(v.max() if agg == "max" else v.min() if agg == "min" else float(np.exp(np.log(v).mean())))
    return np.array(out)


def penalty_from_prior(pi: np.ndarray, alpha: float) -> np.ndarray:
    p = (pi.mean() / pi) ** alpha
    return p / p.mean()


class Runner:
    def __init__(self, Xp, Xs, seg_pat, y, splits, Cs, level, seed):
        self.Xp, self.Xs, self.seg_pat, self.y, self.splits, self.Cs, self.level, self.seed = Xp, Xs, seg_pat, y, splits, Cs, level, seed

    def _rows(self, pats):
        """患者下标 -> (特征行, 每行所属患者下标)。"""
        if self.level == "patient":
            return self.Xp[pats], pats
        m = np.isin(self.seg_pat, pats)
        return self.Xs[m], self.seg_pat[m]

    @staticmethod
    def _patient_pred(prob_rows, row_pat, pats):
        p = np.array([prob_rows[row_pat == i].mean() for i in pats])
        return (p > 0.5).astype(int), p

    def _fit(self, Z, yz, C):
        clf = LogisticRegression(penalty="l1", solver="liblinear", C=C, class_weight="balanced", max_iter=2000)
        return clf.fit(Z, yz)

    def run(self, penalty: np.ndarray, keep: np.ndarray):
        y = self.y
        n = len(y)
        pred, prob = np.zeros(n, int), np.zeros(n)
        selected, chosen_C = [], []
        cols = np.where(keep)[0]
        for tr, te in self.splits:
            Xtr, rtr = self._rows(tr)
            Xte, rte = self._rows(te)
            Xtr, Xte = Xtr[:, cols], Xte[:, cols]
            med = np.nanmedian(Xtr, axis=0)
            Xtr = np.where(np.isnan(Xtr), med, Xtr)
            Xte = np.where(np.isnan(Xte), med, Xte)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
            Ztr = (Xtr - mu) / sd / penalty[cols]
            Zte = (Xte - mu) / sd / penalty[cols]
            ytr_rows = y[rtr]
            # 内层 CV(按患者)选 C
            best_C, best_s = self.Cs[0], -1
            if self.level == "patient":
                inner = list(StratifiedKFold(4, shuffle=True, random_state=self.seed).split(Ztr, ytr_rows))
            else:
                inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=self.seed).split(Ztr, ytr_rows, groups=rtr))
            for C in self.Cs:
                s = []
                for itr, iva in inner:
                    clf = self._fit(Ztr[itr], ytr_rows[itr], C)
                    va_pats = np.unique(rtr[iva])
                    pv, _ = self._patient_pred(clf.predict_proba(Ztr[iva])[:, 1], rtr[iva], va_pats)
                    s.append(macro(pv, y[va_pats]))
                if np.mean(s) > best_s + 1e-9:
                    best_s, best_C = float(np.mean(s)), C
            clf = self._fit(Ztr, ytr_rows, best_C)
            pred[te], prob[te] = self._patient_pred(clf.predict_proba(Zte)[:, 1], rte, te)
            selected.append(set(cols[np.abs(clf.coef_[0]) > 1e-8].tolist()))
            chosen_C.append(best_C)
        return pred, prob, selected, chosen_C


def jaccard_mean(sets):
    vals = [len(a & b) / max(len(a | b), 1) for a, b in itertools.combinations(sets, 2)]
    return float(np.mean(vals)) if vals else float("nan")


def region_label(rs) -> str:
    return "+".join(sorted(rs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="logs/measure_bank_3d/bank.pkl")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--class-num", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--prior-agg", default="max", choices=["max", "min", "gmean"])
    ap.add_argument("--level", default="patient", choices=["patient", "segment"])
    ap.add_argument("--other-prior", type=float, default=None, help="无人标区域(肘/膝/胸段)的先验, 默认取最低区域值")
    ap.add_argument("--Cs", default="0.01,0.03,0.1,0.3,1,3,10")
    ap.add_argument("--variants", nargs="+", default=["doctor", "uniform", "inverted", "hard_doctor", "hard_other", "hand5", "random"])
    ap.add_argument("--n-random", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None, help="把各变体的患者级概率存成 late_fusion 可读的 json")
    args = ap.parse_args()

    Xp, Xs, seg_pat, y, pids, feats, regions, splits = load_bank(args.bank, args.data_root, args.class_num)
    Cs = [float(c) for c in args.Cs.split(",")]
    prior = dict(DEFAULT_PRIOR)
    other = args.other_prior if args.other_prior is not None else min(prior.values())
    pi = feature_prior(regions, prior, other, args.prior_agg)
    runner = Runner(Xp, Xs, seg_pat, y, splits, Cs, args.level, args.seed)
    tag = f"level={args.level} alpha={args.alpha} prior-agg={args.prior_agg}"
    print(f"患者 {len(pids)} (ASD {y.sum()} / non-ASD {len(y) - y.sum()}), 段 {Xs.shape[0]}, 测量 {Xp.shape[1]}, {tag}, other 先验 {other}")
    q = np.quantile(pi, [0, 0.25, 0.5, 0.75, 1])
    print("测量先验分位数:", np.round(q, 3), " 拿到最高先验的测量占比 %.2f" % np.mean(pi >= pi.max() - 1e-9))

    results = {}
    ones = np.ones(Xp.shape[1])
    all_keep = np.ones(Xp.shape[1], bool)
    doctor_keep = np.array([any(r in DOCTOR_REGIONS for r in rs) for rs in regions])
    hand_keep = np.array([f in HAND5 for f in feats])
    strict_keep = np.array([all(r in DOCTOR_REGIONS for r in rs) for rs in regions])
    group_of = np.array([region_label(r) for r in regions])
    groups = sorted(set(group_of))

    def run(name, penalty, keep):
        pred, prob, sel, Cc = runner.run(penalty, keep)
        results[name] = {"pred": pred, "prob": prob, "sel": sel, "C": Cc, "macro": macro(pred, y)}
        print(f"  {name}: {results[name]['macro']:.3f}", flush=True)

    for v in args.variants:
        if v == "doctor":
            run(v, penalty_from_prior(pi, args.alpha), all_keep)
        elif v == "uniform":
            run(v, ones, all_keep)
        elif v == "inverted":
            run(v, penalty_from_prior(1.0 / pi, args.alpha), all_keep)
        elif v == "hard_doctor":
            run(v, ones, doctor_keep)
        elif v == "hard_other":
            run(v, ones, ~doctor_keep)
        elif v == "hand5":
            run(v, ones, hand_keep)
        elif v == "hard_doctor_strict":
            run(v, ones, strict_keep)
        elif v == "hard_random":
            # 硬约束版的随机对照: 随机选一批区域组, 测量数与 hard_doctor 相当(±15%), 均匀 L1
            rng = np.random.default_rng(args.seed)
            target = int(doctor_keep.sum())
            ms = []
            while len(ms) < args.n_random:
                k = rng.integers(2, len(groups))
                pick = set(rng.choice(groups, size=k, replace=False).tolist())
                keep = np.array([g in pick for g in group_of])
                if not (0.85 * target <= keep.sum() <= 1.15 * target) or keep.sum() == 0:
                    continue
                pred, _, _, _ = runner.run(ones, keep)
                ms.append(macro(pred, y))
                print(f"  hard_random {len(ms)}/{args.n_random}: {ms[-1]:.3f} (测量 {int(keep.sum())}, 区域组 {k})", flush=True)
            results["hard_random"] = {"dist": np.array(ms)}
        elif v == "random":
            rng = np.random.default_rng(args.seed)
            labels = sorted({region_label(r) for r in regions})
            base = {lab: feature_prior([frozenset(lab.split("+"))], prior, other, args.prior_agg)[0] for lab in labels}
            vals = list(base.values())
            ms = []
            for t in range(args.n_random):
                pmap = dict(zip(labels, rng.permutation(vals)))
                pi_r = np.array([pmap[region_label(r)] for r in regions])
                pred, _, _, _ = runner.run(penalty_from_prior(pi_r, args.alpha), all_keep)
                ms.append(macro(pred, y))
                print(f"  random {t + 1}/{args.n_random}: {ms[-1]:.3f}", flush=True)
            results["random"] = {"dist": np.array(ms)}

    print(f"\n[{tag}]\n{'变体':12s} {'患者级 macro [95% CI]':>26s} {'选中数/折':>10s} {'Jaccard':>8s} {'C':>26s}  选中测量的区域分布")
    for name, r in results.items():
        if "dist" in r:
            d = r["dist"]
            print(f"{name:12s} {d.mean():.3f} ± {d.std():.3f}  (min {d.min():.3f}, max {d.max():.3f}, n={len(d)})")
            continue
        lo, hi = macro_ci(r["pred"], y, n_boot=2000)
        nsel = [len(s) for s in r["sel"]]
        cnt = Counter(region_label(regions[i]) for s in r["sel"] for i in s)
        top = ", ".join(f"{k}:{v}" for k, v in cnt.most_common(5))
        print(f"{name:12s} {r['macro']:.3f} [{lo:.3f},{hi:.3f}]{'':>6s} {np.mean(nsel):6.1f}     {jaccard_mean(r['sel']):.2f}   {str(r['C']):>26s}  {top}")

    if "random" in results and "doctor" in results:
        d = results["random"]["dist"]
        print(f"\n医生先验 {results['doctor']['macro']:.3f} 在随机先验分布中的百分位: {np.mean(d < results['doctor']['macro']):.2f}")
    if "hard_random" in results and "hard_doctor" in results:
        d = results["hard_random"]["dist"]
        print(f"硬约束医生区域 {results['hard_doctor']['macro']:.3f} 在随机区域子集分布中的百分位: {np.mean(d < results['hard_doctor']['macro']):.2f}")
    from scipy import stats
    for a_name, b_name in [("doctor", "uniform"), ("hard_doctor", "uniform"), ("hard_doctor", "hand5"),
                           ("hard_doctor", "hard_other"), ("hard_doctor_strict", "uniform"), ("hard_doctor", "hard_doctor_strict")]:
        if a_name in results and b_name in results:
            a, b = results[a_name]["pred"] == y, results[b_name]["pred"] == y
            n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
            p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            print(f"McNemar {a_name} vs {b_name}: {a_name} 独对 {n01} : {b_name} 独对 {n10}, p={p:.2f}")

    for name in ("doctor", "uniform", "hard_doctor", "hard_doctor_strict"):
        if name in results:
            freq = Counter(i for s in results[name]["sel"] for i in s)
            print(f"\n{name} 在 ≥3 折被选中的测量:")
            for i, c in sorted(freq.items(), key=lambda kv: -kv[1]):
                if c >= 3:
                    print(f"  {c}/5  {feats[i]:36s} [{region_label(regions[i])}]")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, r in results.items():
            if "dist" in r:
                continue
            json.dump({p: {"prob": [float(r["prob"][i]), float(1 - r["prob"][i])], "label": int(1 - y[i])}
                       for i, p in enumerate(pids)}, open(out / f"bank_{name}_{args.level}_{args.prior_agg}_a{args.alpha}_probs.json", "w"), indent=1)
        print(f"\n患者级概率已存 {out}")


if __name__ == "__main__":
    main()

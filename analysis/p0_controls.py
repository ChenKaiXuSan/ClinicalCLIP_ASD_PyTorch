#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: p0_controls.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
方法 1(医生区域硬过滤的稀疏测量选择)的 P0 对照,全部复用 prior_sparse_lr.py 的数据与嵌套流程(段级训练、按患者分组、
C 内层 4 折选、患者级判类)。子命令:

  region-ablation   候选集 = 医生三区(lumbar_pelvis / head / shoulder)去掉一个 / 只留一个,共 7 个候选集
  random5           随机抽 5 个测量 × N 次:全库 / 医生区域内 / 非医生区域内;报分布与手挑 5 量的分位
  random-regions    6 个区域标签任取 3 个做候选集(20 种),真实的医生三区排第几
  repeated-splits   换 N 份随机患者划分,重跑 手挑 5 量 / 硬过滤 / 非医生区域,报均值 ± 标准差

每个子命令都把患者级概率存到 --out-dir(late_fusion / pairwise_mcnemar / patient_metrics 可读)。

    python analysis/p0_controls.py region-ablation --data-root $DATA --out-dir logs/geom_probs/p0
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patient_level_stats import macro_ci  # noqa: E402
from prior_sparse_lr import DOCTOR_REGIONS, HAND5, Runner, load_bank, macro, region_label  # noqa: E402

ALL_REGIONS = ("lumbar_pelvis", "head", "shoulder", "wrist", "foot", "other")


def keep_touching(regions, combo) -> np.ndarray:
    """候选 = 至少一个地标落在 combo 里的测量(宽松)。"""
    return np.array([any(r in combo for r in rs) for rs in regions])


def keep_within(regions, combo) -> np.ndarray:
    """候选 = 全部地标都落在 combo 里的测量(严格)。"""
    return np.array([all(r in combo for r in rs) for rs in regions])


def save_probs(path: Path, pids, prob, y):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({p: {"prob": [float(prob[i]), float(1 - prob[i])], "label": int(1 - y[i])} for i, p in enumerate(pids)},
              open(path, "w"), indent=1)


def report(name, pred, sel, y, extra=""):
    lo, hi = macro_ci(pred, y, n_boot=2000)
    nsel = np.mean([len(s) for s in sel]) if sel else float("nan")
    print(f"{name:40s} {macro(pred, y):.3f} [{lo:.3f},{hi:.3f}]  选中 {nsel:6.1f}  {extra}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["region-ablation", "random5", "random-regions", "repeated-splits"])
    ap.add_argument("--bank", default="logs/measure_bank_3d/bank.pkl")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--class-num", type=int, default=2)
    ap.add_argument("--level", default="segment", choices=["patient", "segment"])
    ap.add_argument("--Cs", default="0.01,0.1")
    ap.add_argument("--n", type=int, default=100, help="random5 的次数 / repeated-splits 的划分数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="logs/geom_probs/p0")
    ap.add_argument("--strict", action="store_true", help="候选集用严格版(全部地标都在区域集合内), 默认宽松版(至少一个)")
    ap.add_argument("--exclude-missing", action="store_true", help="repeated-splits: 去掉有视频无 3D 结果的患者(中位数填充行)")
    args = ap.parse_args()
    keep_fn = keep_within if args.strict else keep_touching
    sfx = "_strict" if args.strict else ""

    Xp, Xs, seg_pat, y, pids, feats, regions, splits = load_bank(args.bank, args.data_root, args.class_num)
    Cs = [float(c) for c in args.Cs.split(",")]
    runner = Runner(Xp, Xs, seg_pat, y, splits, Cs, args.level, args.seed)
    ones = np.ones(Xp.shape[1])
    out = Path(args.out_dir)
    doctor_keep = keep_fn(regions, DOCTOR_REGIONS)
    hand_keep = np.array([f in HAND5 for f in feats])
    print(f"患者 {len(pids)}, 测量 {Xp.shape[1]}, level={args.level}, Cs={Cs}, cmd={args.cmd}", flush=True)

    if args.cmd == "region-ablation":
        sets = {"doctor3(lumbar+head+shoulder)": DOCTOR_REGIONS}
        for r in DOCTOR_REGIONS:
            sets[f"minus_{r}"] = tuple(x for x in DOCTOR_REGIONS if x != r)
        for r in DOCTOR_REGIONS:
            sets[f"only_{r}"] = (r,)
        print(f"{'候选集':40s} {'macro [95% CI]':>22s}  {'选中数':>8s}  候选测量数")
        for name, combo in sets.items():
            keep = keep_fn(regions, combo)
            pred, prob, sel, _ = runner.run(ones, keep)
            report(name, pred, sel, y, f"候选 {int(keep.sum())}")
            save_probs(out / f"ablate_{name.split('(')[0]}{sfx}_{args.level}_probs.json", pids, prob, y)

    elif args.cmd == "random5":
        rng = np.random.default_rng(args.seed)
        pred, prob, sel, _ = runner.run(ones, hand_keep)
        ref = macro(pred, y)
        print(f"手挑 5 量(库内 {int(hand_keep.sum())} 列) 参照: {ref:.3f}")
        pools = {"全库": np.arange(Xp.shape[1]), "医生区域内": np.where(doctor_keep)[0], "非医生区域内": np.where(~doctor_keep)[0]}
        for pname, pool in pools.items():
            ms = []
            for t in range(args.n):
                keep = np.zeros(Xp.shape[1], bool)
                keep[rng.choice(pool, 5, replace=False)] = True
                p_, _, _, _ = runner.run(ones, keep)
                ms.append(macro(p_, y))
                if (t + 1) % 20 == 0:
                    print(f"  {pname} {t + 1}/{args.n}: 当前均值 {np.mean(ms):.3f}", flush=True)
            ms = np.array(ms)
            print(f"随机 5 量 [{pname}] (n={args.n}): {ms.mean():.3f} ± {ms.std():.3f}  [{ms.min():.3f}, {ms.max():.3f}]  "
                  f"手挑 5 量的分位 {(ms < ref).mean():.2f}  ≥手挑的次数 {(ms >= ref).sum()}", flush=True)
            np.save(out / f"random5_{pname}_{args.level}.npy", ms)

    elif args.cmd == "random-regions":
        rows = []
        for combo in itertools.combinations(ALL_REGIONS, 3):
            keep = keep_fn(regions, combo)
            if keep.sum() == 0:
                continue
            pred, prob, sel, _ = runner.run(ones, keep)
            m = macro(pred, y)
            rows.append((m, combo, int(keep.sum()), np.mean([len(s) for s in sel])))
            tag = "+".join(combo)
            print(f"  {tag:36s} {m:.3f}  候选 {int(keep.sum()):4d}  选中 {np.mean([len(s) for s in sel]):6.1f}"
                  + ("   <== 医生三区" if set(combo) == set(DOCTOR_REGIONS) else ""), flush=True)
            save_probs(out / f"regions3_{tag}{sfx}_{args.level}_probs.json", pids, prob, y)
        rows.sort(key=lambda r: -r[0])
        print("\n20 个三区域组合按 macro 排序:")
        for i, (m, combo, nc, ns) in enumerate(rows):
            mark = "  <== 医生三区" if set(combo) == set(DOCTOR_REGIONS) else ""
            print(f"  {i + 1:2d}. {m:.3f}  {'+'.join(combo):36s} 候选 {nc:4d}{mark}")
        ms = np.array([r[0] for r in rows])
        doc = next(r[0] for r in rows if set(r[1]) == set(DOCTOR_REGIONS))
        print(f"医生三区 {doc:.3f}; 其余 19 个组合 {ms[ms != doc].mean():.3f} ± {ms[ms != doc].std():.3f}, 分位 {(ms < doc).mean():.2f}")

    elif args.cmd == "repeated-splits":
        variants = {"hand5": hand_keep, "hard_doctor": keep_touching(regions, DOCTOR_REGIONS),
                    "strict_doctor": keep_within(regions, DOCTOR_REGIONS), "hard_other": ~keep_touching(regions, DOCTOR_REGIONS),
                    "uniform": np.ones(Xp.shape[1], bool)}
        sub = np.arange(len(y))
        if args.exclude_missing:
            import pickle
            miss = {Path(v["json"]).stem.split("-")[0] for v in pickle.load(open(args.bank, "rb"))["videos"].values() if v["frac_ok"] == 0}
            sub = np.array([i for i, p in enumerate(pids) if p not in miss])
            print(f"去掉无 3D 结果的患者 {len(miss)} 人, 剩余 {len(sub)}")
        res = {k: [] for k in variants}
        res["hand5_patient"] = []
        for s in range(args.n):
            skf = StratifiedKFold(5, shuffle=True, random_state=1000 + s)
            sp = [(sub[tr], sub[te]) for tr, te in skf.split(np.zeros(len(sub)), y[sub])]
            r_seg = Runner(Xp, Xs, seg_pat, y, sp, Cs, "segment", args.seed)
            r_pat = Runner(Xp, Xs, seg_pat, y, sp, [0.01, 0.03, 0.1, 0.3, 1, 3, 10], "patient", args.seed)
            for k, keep in variants.items():
                pred, _, _, _ = r_seg.run(ones, keep)
                res[k].append(macro(pred[sub], y[sub]))
            pred, _, _, _ = r_pat.run(ones, hand_keep)
            res["hand5_patient"].append(macro(pred[sub], y[sub]))
            print(f"  划分 {s + 1}/{args.n}: " + "  ".join(f"{k} {v[-1]:.3f}" for k, v in res.items()), flush=True)
        print(f"\n{args.n} 份随机患者划分(固定划分上的值: hand5 0.786, hard_doctor 0.803, hard_other 0.697, hand5_patient 0.732):")
        for k, v in res.items():
            v = np.array(v)
            print(f"  {k:14s} {v.mean():.3f} ± {v.std():.3f}  [{v.min():.3f}, {v.max():.3f}]")
        json.dump({k: [float(x) for x in v] for k, v in res.items()}, open(out / f"repeated_splits{'_nomiss' if args.exclude_missing else ''}.json", "w"), indent=1)
        d = np.array(res["hard_doctor"]) - np.array(res["hard_other"])
        ds = np.array(res["strict_doctor"]) - np.array(res["hard_other"])
        print(f"  逐划分差: 宽松医生 - 非医生 {d.mean():+.3f} ± {d.std():.3f} (正 {int((d > 0).sum())}/{len(d)});"
              f" 严格医生 - 非医生 {ds.mean():+.3f} ± {ds.std():.3f} (正 {int((ds > 0).sum())}/{len(ds)})")


if __name__ == "__main__":
    main()

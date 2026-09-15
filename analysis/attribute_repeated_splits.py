#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: attribute_repeated_splits.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
几何属性(geom_attributes*/ 下的 json)的患者级逻辑回归,换 N 份随机患者划分,报平衡准确率与 AUC 的均值 ± std。
与 attribute_classifier.py 同一模型(段分数按患者取均值 -> 标准化 -> class_weight=balanced 的 LR, C=1),
只是把固定的 index.json 划分换成 StratifiedKFold(5, shuffle, random_state=1000+s),与 p0_controls.py
repeated-splits 用的是同一批划分,所以与测量库变体逐份可比。

    python analysis/attribute_repeated_splits.py --attr-root logs/geom_attributes_3d --n 10 --exclude-missing logs/measure_bank_3d/bank.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DOCTOR5 = ["trunk_lean", "shoulder_offset", "head_forward", "hip_flexion_max", "hip_range"]
OTHER4 = ["knee_flexion_max", "step_length", "gait_speed", "arm_swing"]
EXTRA3D = ["lower_trunk_lean", "spine_curve", "pelvic_obliquity", "trunk_lateral", "lateral_sway"]
SETS = {"医生部位 5 量": DOCTOR5, "未标部位 4 量": OTHER4, "全部 9 量": DOCTOR5 + OTHER4, "医生 5 + 3D 独有 5": DOCTOR5 + EXTRA3D}
# 按医生区域整组消融(区域归属见 geom_attributes.py 注释):腰椎骨盆 = 躯干前倾 + 髋屈曲 + 髋幅度;头 = 头前伸;肩 = 肩偏移
LUMBAR, HEAD, SHOULDER = ["trunk_lean", "hip_flexion_max", "hip_range"], ["head_forward"], ["shoulder_offset"]
SETS.update({
    "去腰椎骨盆组(剩 2)": HEAD + SHOULDER, "去头(剩 4)": LUMBAR + SHOULDER, "去肩(剩 4)": LUMBAR + HEAD,
    "只腰椎骨盆组(3)": LUMBAR, "只头前伸(1)": HEAD, "只肩偏移(1)": SHOULDER, "只躯干前倾(1)": ["trunk_lean"],
    "只髋屈曲(1)": ["hip_flexion_max"], "只髋幅度(1)": ["hip_range"],
})
DEFAULT_SETS = ["医生部位 5 量", "未标部位 4 量", "全部 9 量", "医生 5 + 3D 独有 5"]


def load(attr_root: Path, attrs: list[str]):
    per: dict[str, dict] = {}
    for a in attrs:
        f = attr_root / f"{a}.json"
        if not f.exists():
            return None
        for v, rec in json.load(open(f)).items():
            p = Path(rec["json"]).stem.split("-")[0]
            per.setdefault(p, {"y": int(rec["label"] == 0)}).setdefault(a, []).extend(rec["scores"])
    pids = sorted(per)
    X = np.array([[np.mean(per[p][a]) for a in attrs] for p in pids])
    y = np.array([per[p]["y"] for p in pids])
    return pids, X, y


def macro(pred, y):
    return float(np.mean([((pred == c) & (y == c)).sum() / max((y == c).sum(), 1) for c in np.unique(y)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr-root", default="logs/geom_attributes_3d")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--exclude-missing", default=None, help="bank.pkl 路径: 去掉无 3D 结果的患者")
    ap.add_argument("--sets", nargs="*", default=DEFAULT_SETS, help="可选: " + ", ".join(SETS) + "; all = 全部")
    args_pre, _ = ap.parse_known_args()
    if args_pre.sets == ["all"]:
        ap.set_defaults(sets=list(SETS))
    args = ap.parse_args()

    miss = set()
    if args.exclude_missing:
        bank = pickle.load(open(args.exclude_missing, "rb"))
        miss = {Path(v["json"]).stem.split("-")[0] for v in bank["videos"].values() if v["frac_ok"] == 0}
    print(f"{'量的集合':18s} {'人数':>4s} {'macro 均值±std [范围]':>28s} {'AUC 均值±std':>16s}")
    for name in args.sets:
        got = load(Path(args.attr_root), SETS[name])
        if got is None:
            print(f"{name:18s} (缺属性文件, 跳过)")
            continue
        pids, X, y = got
        keep = np.array([p not in miss for p in pids])
        Xk, yk = X[keep], y[keep]
        ms, aucs = [], []
        for s in range(args.n):
            pred, prob = np.zeros(len(yk), int), np.zeros(len(yk))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=1000 + s).split(Xk, yk):
                sc = StandardScaler().fit(Xk[tr])
                clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=1000).fit(sc.transform(Xk[tr]), yk[tr])
                pred[te] = clf.predict(sc.transform(Xk[te]))
                prob[te] = clf.predict_proba(sc.transform(Xk[te]))[:, 1]
            ms.append(macro(pred, yk)); aucs.append(roc_auc_score(yk, prob))
        ms, aucs = np.array(ms), np.array(aucs)
        print(f"{name:18s} {len(yk):4d} {ms.mean():.3f} ± {ms.std():.3f} [{ms.min():.3f}, {ms.max():.3f}]   {aucs.mean():.3f} ± {aucs.std():.3f}")


if __name__ == "__main__":
    main()

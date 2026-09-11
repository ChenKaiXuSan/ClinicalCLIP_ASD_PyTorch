#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: attribute_classifier.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
方案 2 的分类与检验:把 eval_qwen_attributes.py 的属性分数(每段 8 个 yes/no logit 差)
聚合到患者级(段均值),按主线同一套 5 折患者划分训练逻辑回归,test 患者的预测拼成 79 人,
报患者级平衡准确率 + bootstrap 区间。另报:
  - 每个属性单独对 ASD 的患者级 AUC(VLM 看到的属性有没有诊断信息)
  - 每个属性与医生标注的对应关系:属性分数在"医生标了腰椎骨盆"的患者上是否更高(点二列相关)
  - 逻辑回归系数(哪些属性在起作用)

    python analysis/attribute_classifier.py --attr-dir logs/qwen_attributes --data-root $DATA
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "project"))
from patient_level_stats import macro_ci  # noqa: E402


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def macro(pred, label):
    return float(np.mean([((pred == c) & (label == c)).sum() / max((label == c).sum(), 1) for c in np.unique(label)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--class-num", type=int, default=2)
    ap.add_argument("--C", type=float, default=1.0, help="逻辑回归正则强度的倒数")
    ap.add_argument("--agg", default="mean", choices=["mean", "max"], help="段 -> 患者的聚合")
    ap.add_argument("--out", default=None, help="把患者级 P(ASD) 存成 json,供 late_fusion.py --branch-json 融合")
    args = ap.parse_args()

    attr_files = sorted(Path(args.attr_dir).glob("*.json"))
    attrs = [f.stem for f in attr_files]
    if not attrs:
        sys.exit("没有属性结果")
    data = {a: json.load(open(f)) for a, f in zip(attrs, attr_files)}
    videos = set.intersection(*(set(d) for d in data.values()))
    print(f"属性 {len(attrs)} 个: {attrs}\n视频 {len(videos)} 条(各属性交集)")

    # video -> patient(与 patient_level_stats.build_patient_map 同规则:json 文件名 '-' 前)
    root = Path(args.data_root)
    pat_of = {v: Path(data[attrs[0]][v]["json"]).stem.split("-")[0] for v in videos}
    # 二分类标签:原始 label 0 = ASD -> 1;其余 0
    y_video = {v: int(data[attrs[0]][v]["label"] == 0) for v in videos}

    # 患者级属性向量
    per_pat = defaultdict(lambda: defaultdict(list))
    for v in videos:
        for a in attrs:
            per_pat[pat_of[v]][a].extend(data[a][v]["scores"])
    pids = sorted(per_pat)
    agg = np.mean if args.agg == "mean" else np.max
    X = np.array([[agg(per_pat[p][a]) for a in attrs] for p in pids])
    y = np.array([max(y_video[v] for v in videos if pat_of[v] == p) for p in pids])
    print(f"患者 {len(pids)} (ASD {y.sum()} / non-ASD {len(y) - y.sum()})\n")

    print("每个属性单独 -> ASD 患者级 AUC(>0.5 表示 ASD 患者该属性分数更高):")
    for j, a in enumerate(attrs):
        print(f"  {a:24s} AUC {auc(X[:, j], y):.3f}   yes 率(段) {np.mean([s > 0 for v in videos for s in data[a][v]['scores']]):.3f}")

    # 与医生标注的对应:医生是否标了 lumbar_pelvis / head / shoulder(至少一位)
    try:
        from dataloader.med_attn_map import MedAttnMap, REGIONS
        info = root / "clinical_CLIP_dataset"
        med = MedAttnMap(str(info / "doctor_result"), str(info / "seg_skeleton_pkl"))
        # presence_for 按视频名匹配医生 csv 里的 "video file name"(患者级);取该患者任一视频
        pres = {}
        for p in pids:
            v = next(v for v in videos if pat_of[v] == p)
            pres[p] = med.presence_for(v).numpy()
        P = np.array([pres[p] for p in pids])  # (n, R)
        has_doc = (P.sum(1) > 0)
        print(f"\n有医生标注的患者 {int(has_doc.sum())} 人。属性分数 vs 医生标注区域的点二列相关 r(p):")
        for j, a in enumerate(attrs):
            row = []
            for k, r in enumerate(REGIONS):
                if P[has_doc, k].std() == 0:
                    row.append(f"{r}: n/a")
                    continue
                rr, pp = stats.pointbiserialr((P[has_doc, k] > 0).astype(int), X[has_doc, j])
                row.append(f"{r}: {rr:+.2f}({pp:.2f})")
            print(f"  {a:24s} " + "  ".join(row))
    except Exception as exc:  # noqa: BLE001
        print(f"\n(跳过医生标注对应: {exc})")

    # 5 折逻辑回归:沿用 index.json 的患者划分(test 患者集互不相交)
    folds = json.load(open(root / "clinical_CLIP_dataset" / "index_mapping" / str(args.class_num) / "index.json"))
    pat_index = {p: i for i, p in enumerate(pids)}
    pred = np.full(len(pids), -1)
    prob = np.full(len(pids), np.nan)
    coefs = []
    for k, split in folds.items():
        test_p = {Path(x).stem.split("-")[0] for x in split["test"]}
        train_p = {Path(x).stem.split("-")[0] for x in split["train"] + split["val"]}
        tr = [pat_index[p] for p in pids if p in train_p]
        te = [pat_index[p] for p in pids if p in test_p]
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
        prob[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        coefs.append(clf.coef_[0])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        # 与 late_fusion 的患者概率向量约定一致:[P(ASD 类=索引0), P(non-ASD)],主线 label 0 = ASD
        json.dump({p: {"prob": [float(prob[i]), float(1 - prob[i])], "label": int(1 - y[i])}
                   for i, p in enumerate(pids) if not np.isnan(prob[i])}, open(args.out, "w"), indent=1)
        print(f"患者级概率已存 {args.out}")
    ok = pred >= 0
    lo, hi = macro_ci(pred[ok], y[ok])
    print(f"\n逻辑回归(5 折患者划分, class_weight=balanced, C={args.C}):")
    print(f"  患者级平衡准确率 {macro(pred[ok], y[ok]):.3f} [95% CI {lo:.3f}, {hi:.3f}]   n={int(ok.sum())}   随机 0.5 / 视频基线 B0 0.720")
    print("  平均系数(标准化后, 正 = 指向 ASD):")
    for a, c in sorted(zip(attrs, np.mean(coefs, 0)), key=lambda t: -abs(t[1])):
        print(f"    {a:24s} {c:+.3f}")


if __name__ == "__main__":
    main()

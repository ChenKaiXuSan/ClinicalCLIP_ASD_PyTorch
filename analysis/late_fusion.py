#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: late_fusion.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
决策级晚期融合:两条分支(比如 slow_r50 端到端 与 Qwen 特征分支)各自的患者级概率取加权平均,
再判类。零训练成本,用来回答"VLM 分支有没有与视频分支互补的信息" —— 若晚期融合都不比
单分支好,特征级融合基本也不会好;若好,再去做特征级融合。

口径同 patient_level_stats.py:5 折 test 拼成 79 个患者,段级概率按患者取均值,
macro = 平衡准确率,bootstrap 区间,McNemar 对单分支。

    python analysis/late_fusion.py --root logs/train --data-root $DATA --pairs B0_3dcnn:Q1_qwen_concept_clinical ...
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patient_level_stats import build_patient_map, macro_ci  # noqa: E402


def patient_probs(exp_root: Path, tag: str, pmap: dict[str, str]) -> dict[str, tuple[np.ndarray, int]]:
    """{患者: (概率向量, 真值)},段级概率按患者均值。"""
    out = {}
    for fold in range(5):
        runs = sorted(
            {p.parent.parent for p in (exp_root / f"{tag}__f{fold}_s42").rglob("best_preds/*_pred.pt")},
            key=lambda p: p.stat().st_mtime,
        )
        if not runs:
            continue
        best = runs[-1] / "best_preds"
        for pf in sorted(best.glob("*_pred.pt")):
            lf = pf.with_name(pf.name.replace("_pred.pt", "_label.pt"))
            nf = pf.with_name(pf.name.replace("_pred.pt", "_video_name.json"))
            prob = torch.load(pf, map_location="cpu", weights_only=False).float().numpy()
            label = torch.load(lf, map_location="cpu", weights_only=False).long().numpy()
            names = json.loads(nf.read_text())
            by = defaultdict(list)
            for i, v in enumerate(names):
                by[pmap.get(v, v)].append(i)
            for pid, idx in by.items():
                out[pid] = (prob[idx].mean(0), int(np.bincount(label[idx]).argmax()))
    return out


def macro(pred, label):
    return float(np.mean([((pred == c) & (label == c)).mean() / max((label == c).mean(), 1e-9) * 1.0
                          if False else ((pred == c) & (label == c)).sum() / max((label == c).sum(), 1)
                          for c in np.unique(label)]))


def mcnemar(a_ok, b_ok):
    a_only = int((a_ok & ~b_ok).sum())
    b_only = int((~a_ok & b_ok).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(a_only, b_only), n, 0.5)))
    return a_only, b_only, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="A:B,A 是视频分支,B 是 VLM 分支")
    ap.add_argument("--weights", default="0.5", help="B 分支权重,逗号分隔,如 0.3,0.5,0.7")
    args = ap.parse_args()

    pmap = build_patient_map(args.data_root)
    root = Path(args.root)
    ws = [float(w) for w in args.weights.split(",")]

    def load_branch(tag: str):
        """实验名 -> 从 logs/train 读;以 .json 结尾 -> 读 attribute_classifier.py --out 的患者级概率。"""
        if tag.endswith(".json"):
            d = json.load(open(tag))
            return {p: (np.asarray(v["prob"], dtype=float), int(v["label"])) for p, v in d.items()}
        return patient_probs(root, tag, pmap)
    print(f"{'A + B':58s} {'A':>6s} {'B':>6s}  " + "  ".join(f"融合w={w:.1f} [95% CI]        A独:融独 p" for w in ws))
    for pair in args.pairs:
        a_tag, b_tag = pair.split(":")
        A, B = load_branch(a_tag), load_branch(b_tag)
        b_tag = Path(b_tag).stem if b_tag.endswith(".json") else b_tag
        pids = sorted(set(A) & set(B))
        if len(pids) < len(A) or len(pids) < len(B):
            print(f"  警告: {a_tag} {len(A)} 人, {b_tag} {len(B)} 人, 交集 {len(pids)} 人")
        label = np.array([A[p][1] for p in pids])
        pa = np.stack([A[p][0] for p in pids])
        pb = np.stack([B[p][0] for p in pids])
        pred_a, pred_b = pa.argmax(1), pb.argmax(1)
        line = f"{a_tag} + {b_tag}"[:58].ljust(58) + f" {macro(pred_a, label):.3f} {macro(pred_b, label):.3f}  "
        for w in ws:
            pf = (1 - w) * pa + w * pb
            pred_f = pf.argmax(1)
            lo, hi = macro_ci(pred_f, label)
            a_only, f_only, p = mcnemar(pred_a == label, pred_f == label)
            line += f"{macro(pred_f, label):.3f} [{lo:.3f},{hi:.3f}]  {a_only:2d}:{f_only:<2d} {p:.2f}   "
        print(line)


if __name__ == "__main__":
    main()

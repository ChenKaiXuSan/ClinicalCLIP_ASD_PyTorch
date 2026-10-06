#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: region_branch_summary.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
多分支区域模型(backbone=region)的逐分支汇总:每条分支(全身 / 各身体段)单独的患者级平衡准确率与 AUC,
以及融合后的结果;多种子给均值 ± std。回答"每一段身体的视频证据有多强",与测量侧的单量结果
(躯干前倾 0.707、头前伸 0.409 …)在另一种模态上对照。

读 best_preds/<fold>_branch_pred.pt (N, n_branch, C)、<fold>_branch_names.json、<fold>_label.pt、
<fold>_video_name.json;口径同 seed_summary.py(5 折 test 拼成全部患者,段概率按患者取均值)。

    python analysis/region_branch_summary.py --data-root $DATA --tags R1_region_measure_score_e50 \
        --seeds 42 1337 2024 --exclude-missing logs/measure_bank_3d/bank.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from late_fusion import macro  # noqa: E402
from patient_level_stats import build_patient_map  # noqa: E402


def load_branches(root: Path, tag: str, seed: int, pmap: dict, drop: set):
    """-> (分支名列表, {患者: (n_branch+1, C) 概率, 最后一行是融合}, {患者: 标签})。"""
    probs, labels, names = {}, {}, None
    for fold in range(5):
        runs = sorted({p.parent.parent for p in (root / f"{tag}__f{fold}_s{seed}").rglob("best_preds/*_branch_pred.pt")},
                      key=lambda p: p.stat().st_mtime)
        if not runs:
            raise FileNotFoundError(f"{tag} seed {seed} 缺第 {fold} 折的 branch_pred")
        best = runs[-1] / "best_preds"
        for bf in sorted(best.glob("*_branch_pred.pt")):
            stem = bf.name.replace("_branch_pred.pt", "")
            branch = torch.load(bf, map_location="cpu", weights_only=False).float().numpy()        # (N, n, C)
            fused = torch.load(best / f"{stem}_pred.pt", map_location="cpu", weights_only=False).float().numpy()
            label = torch.load(best / f"{stem}_label.pt", map_location="cpu", weights_only=False).long().numpy()
            vnames = json.loads((best / f"{stem}_video_name.json").read_text())
            names = json.loads((best / f"{stem}_branch_names.json").read_text())
            allp = np.concatenate([branch, fused[:, None]], axis=1)                                  # (N, n+1, C)
            by = defaultdict(list)
            for i, v in enumerate(vnames):
                by[pmap.get(v, v)].append(i)
            for pid, idx in by.items():
                if pid in drop:
                    continue
                probs[pid] = allp[idx].mean(0)
                labels[pid] = int(np.bincount(label[idx]).argmax())
    return names, probs, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2024])
    ap.add_argument("--exclude-missing", default=None, help="bank.pkl 路径: 去掉无 3D 结果的患者(67 人口径)")
    args = ap.parse_args()

    drop = set()
    if args.exclude_missing:
        drop = {Path(v["json"]).stem.split("-")[0] for v in pickle.load(open(args.exclude_missing, "rb"))["videos"].values() if v["frac_ok"] == 0}
    pmap = build_patient_map(args.data_root)
    root = Path(args.root)
    for tag in args.tags:
        acc, auc, names, n_pat = defaultdict(list), defaultdict(list), None, 0
        for seed in args.seeds:
            try:
                names, probs, labels = load_branches(root, tag, seed, pmap, drop)
            except FileNotFoundError as exc:
                print(f"{tag} seed {seed}: 跳过 ({exc})")
                continue
            pids = sorted(probs)
            n_pat = len(pids)
            y = np.array([labels[p] for p in pids])
            P = np.stack([probs[p] for p in pids])                # (n_pat, n+1, C)
            for i, name in enumerate(list(names) + ["FUSED"]):
                acc[name].append(macro(P[:, i].argmax(1), y))
                # 标签 0 = ASD;AUC 以 ASD 为正类
                auc[name].append(roc_auc_score((y == 0).astype(int), P[:, i, 0]))
        if not acc:
            continue
        print(f"\n== {tag}  (患者 {n_pat}, 种子 {len(next(iter(acc.values())))})")
        print(f"{'分支':16s} {'平衡准确率':>16s} {'AUC':>16s}   逐种子准确率")
        for name in list(names) + ["FUSED"]:
            a, u = np.array(acc[name]), np.array(auc[name])
            print(f"{name:16s} {a.mean():.3f}±{a.std():.3f}   {u.mean():.3f}±{u.std():.3f}   " + " / ".join(f"{x:.3f}" for x in a))


if __name__ == "__main__":
    main()

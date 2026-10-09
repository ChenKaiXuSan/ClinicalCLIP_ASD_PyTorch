#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: bout_pooling.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
"疲劳感知的聚合":ASD 在一次行走的后段姿态更差(docs/bout.md),所以患者级聚合不必对所有片段一视同仁。
片段编号 = 经过的时间顺序,归一化秩 r ∈ [0, 1]。对比几种聚合:
  all     全部片段均值(现状)
  early   r < 0.5          late    r ≥ 0.5          last_q  r ≥ 0.75
  ramp    权重 0.25 + r(线性偏向后段)
片段少于 MIN_CLIPS 的患者一律用 all。

1) 测量:5 个医生区域量按各聚合得到患者向量,10 份划分逻辑回归 + 单量 AUC。
2) 视频:B0 / M0 / A1 三种子的段级预测,先段 -> 片段均值,再按各聚合得到患者概率,报患者级平衡准确率与 AUC。

    python analysis/bout_pooling.py --data-root $DATA --exclude-missing logs/measure_bank_3d/bank.pkl
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
from gait_phase import macro, repeated_lr  # noqa: E402

DOCTOR5 = ["trunk_lean", "shoulder_offset", "head_forward", "hip_flexion_max", "hip_range"]
MIN_CLIPS = 4
POOLS = ["all", "early", "late", "last_q", "ramp"]


def pool(idxs: list[int], vals: np.ndarray, how: str) -> float:
    order = np.argsort(idxs)
    v = np.asarray(vals, float)[order]
    n = len(v)
    if n < MIN_CLIPS or how == "all":
        return float(np.nanmean(v))
    r = np.arange(n) / (n - 1)
    if how == "early":
        return float(np.nanmean(v[r < 0.5]))
    if how == "late":
        return float(np.nanmean(v[r >= 0.5]))
    if how == "last_q":
        return float(np.nanmean(v[r >= 0.75]))
    if how == "ramp":
        w = 0.25 + r
        return float(np.nansum(w * v) / np.sum(w[np.isfinite(v)]))
    raise ValueError(how)


def stem_map(root_path: str) -> dict[str, str]:
    """video_name -> json 文件名(<患者>-NNNN)。"""
    m = {}
    for jf in Path(root_path, "clinical_CLIP_dataset", "json_mix").rglob("*.json"):
        try:
            m[json.loads(jf.read_text())["video_name"]] = jf.stem
        except (ValueError, KeyError, OSError):
            continue
    return m


def measure_part(attr_root: Path, miss: set[str], n: int) -> None:
    clips: dict[str, dict] = {}
    for a in DOCTOR5:
        for v, rec in json.load(open(attr_root / f"{a}.json")).items():
            stem = Path(rec["json"]).stem
            pid, idx = stem.split("-")[0], int(stem.split("-")[1])
            d = clips.setdefault(pid, {"y": int(rec["label"] == 0), "c": defaultdict(dict)})
            d["c"][idx][a] = float(np.mean(rec["scores"]))
    pids = sorted(p for p in clips if p not in miss)
    y = np.array([clips[p]["y"] for p in pids])
    print(f"== 测量:患者 {len(pids)} (ASD {y.sum()}),片段 ≥ {MIN_CLIPS} 的 {sum(len(clips[p]['c']) >= MIN_CLIPS for p in pids)} 人")
    print(f"{'聚合':8s} {'LR macro':>16s} {'AUC':>6s}   单量 AUC: " + " / ".join(DOCTOR5))
    for how in POOLS:
        X = np.array([[pool(list(clips[p]["c"]), [clips[p]["c"][i].get(a, np.nan) for i in clips[p]["c"]], how) for a in DOCTOR5] for p in pids])
        X = np.where(np.isnan(X), np.nanmedian(X, 0), X)
        m, auc = repeated_lr(X, y, n, 1.0)
        uni = [roc_auc_score(y, X[:, j]) for j in range(len(DOCTOR5))]
        print(f"{how:8s} {m.mean():.3f} ± {m.std():.3f}   {auc.mean():.3f}   " + "  ".join(f"{max(u, 1 - u):.3f}" for u in uni))


def video_part(train_root: Path, tags: list[str], seeds: list[int], smap: dict[str, str], miss: set[str]) -> None:
    print(f"\n== 视频:患者级(段 -> 片段均值 -> 按聚合),67 人口径")
    print(f"{'配置':26s} " + " ".join(f"{h:>13s}" for h in POOLS) + "   (平衡准确率 / AUC,三种子均值)")
    for tag in tags:
        res = {h: ([], []) for h in POOLS}
        for s in seeds:
            per_pat: dict[str, dict] = {}
            for k in range(5):
                runs = sorted({p.parent.parent for p in (train_root / f"{tag}__f{k}_s{s}").rglob("best_preds/*_pred.pt")}, key=lambda p: str(p))
                if not runs:
                    break
                best = runs[-1] / "best_preds"
                for pf in sorted(best.glob("*_pred.pt")):
                    if pf.name.endswith("_branch_pred.pt"):
                        continue
                    prob = torch.load(pf, map_location="cpu", weights_only=False).float().numpy()
                    lab = torch.load(pf.with_name(pf.name.replace("_pred.pt", "_label.pt")), map_location="cpu", weights_only=False).long().numpy()
                    names = json.loads(pf.with_name(pf.name.replace("_pred.pt", "_video_name.json")).read_text())
                    for pr, lb, nm in zip(prob, lab, names):
                        stem = smap.get(nm)
                        if stem is None:
                            continue
                        pid, idx = stem.split("-")[0], int(stem.split("-")[1])
                        if pid in miss:
                            continue
                        d = per_pat.setdefault(pid, {"y": int(lb == 0), "c": defaultdict(list)})
                        d["c"][idx].append(float(pr[0]))          # P(ASD)
            else:
                pids = sorted(per_pat)
                y = np.array([per_pat[p]["y"] for p in pids])
                for h in POOLS:
                    p_asd = np.array([pool(list(per_pat[p]["c"]), [np.mean(per_pat[p]["c"][i]) for i in per_pat[p]["c"]], h) for p in pids])
                    res[h][0].append(macro((p_asd > 0.5).astype(int), y))
                    res[h][1].append(roc_auc_score(y, p_asd))
        line = f"{tag:26s} " + " ".join(f"{np.mean(res[h][0]):.3f} / {np.mean(res[h][1]):.3f}" if res[h][0] else f"{'-':>13s}" for h in POOLS)
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--attr-root", default="logs/geom_attributes_3d_doctor")
    ap.add_argument("--train-root", default="logs/train")
    ap.add_argument("--tags", nargs="+", default=["B0_3dcnn_e50", "M0_concept_learned_e50", "A1_no_grounding_e50", "B5b_timesformer_lr2e5_e50"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2024])
    ap.add_argument("--exclude-missing", default=None)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    miss = set()
    if args.exclude_missing:
        bank = pickle.load(open(args.exclude_missing, "rb"))
        miss = {Path(v["json"]).stem.split("-")[0] for v in bank["videos"].values() if v["frac_ok"] == 0}
    measure_part(Path(args.attr_root), miss, args.n)
    video_part(Path(args.train_root), args.tags, args.seeds, stem_map(args.data_root), miss)


if __name__ == "__main__":
    main()

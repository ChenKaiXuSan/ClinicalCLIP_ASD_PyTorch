#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: measure_bank_3d.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
方法 1 的第一步:从 SAM-3D-Body 3D 关节自动生成**测量库**,替代手挑的 5 个几何量。
每个测量都带"涉及哪些医生区域"的标签,供 prior_sparse_lr.py 按医生注意力先验分组惩罚。

地标 18 个(mhr70 关键点 + MHR127 脊柱关节): 鼻、双耳中点、左右肩、肩中点、胸段脊柱(37)、左右肘、
腰段脊柱(35)、左右腕、左右髋、髋中点、左右膝、左右踝。地标 -> 医生区域:
  head: 鼻/耳; shoulder: 肩; lumbar_pelvis: 髋/髋中点/腰段脊柱; wrist: 腕; foot: 踝; other(无人标): 肘/膝/胸段脊柱。

逐帧量:
  每对地标 (上, 下): 矢状面倾角 incl(度, +向行进方向)、前后偏移 dx / 躯干长、高度差 dy / 躯干长、左右距离 dz / 躯干长
  12 个运动链三元组: 中间关节处的夹角(度)
段级统计(与训练同一 1 秒段、8 帧计划): 对地标对取 mean / range,对三元组角取 mean / min / max / range。
共 153×4×2 + 12×4 = 1272 个测量。缺帧规则同 geom_attributes_3d.py。

输出 logs/measure_bank_3d/bank.pkl:
  {"features": [名字], "regions": [每个测量涉及的区域集合], "landmarks": [...],
   "videos": {video_name: {"vals": (n_chunks, F) float32(缺失 nan), "label", "json", "frac_ok": 有 3D 的段占比}}}

    python analysis/measure_bank_3d.py --root-path $DATA --sam3d-root $SAM3D --out logs/measure_bank_3d/bank.pkl
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataloader.whole_video_dataset import _plan_frame_index, _probe  # noqa: E402
from geom_attributes_3d import MAX_GAP, load_frames  # noqa: E402

# (名字, 来源, 索引) 来源 k = pred_keypoints_3d(70), j = pred_joint_coords(127), mid = 两个地标的中点
LANDMARKS = [
    ("nose", "k", 0), ("ear_mid", "mid", (("k", 3), ("k", 4))),
    ("sh_L", "k", 5), ("sh_R", "k", 6), ("sh_mid", "mid", (("k", 5), ("k", 6))),
    ("spine37", "j", 37),
    ("el_L", "k", 7), ("el_R", "k", 8),
    ("spine35", "j", 35),
    ("wr_L", "k", 41), ("wr_R", "k", 62),
    ("hip_L", "k", 9), ("hip_R", "k", 10), ("hip_mid", "mid", (("k", 9), ("k", 10))),
    ("kn_L", "k", 11), ("kn_R", "k", 12),
    ("ank_L", "k", 13), ("ank_R", "k", 14),
]
NAMES = [n for n, _, _ in LANDMARKS]
IDX = {n: i for i, n in enumerate(NAMES)}
REGION_OF = {
    "nose": "head", "ear_mid": "head",
    "sh_L": "shoulder", "sh_R": "shoulder", "sh_mid": "shoulder",
    "spine37": "other", "el_L": "other", "el_R": "other",
    "spine35": "lumbar_pelvis", "hip_L": "lumbar_pelvis", "hip_R": "lumbar_pelvis", "hip_mid": "lumbar_pelvis",
    "wr_L": "wrist", "wr_R": "wrist",
    "kn_L": "other", "kn_R": "other",
    "ank_L": "foot", "ank_R": "foot",
}
TRIPLETS = [
    ("ear_mid", "sh_mid", "hip_mid"), ("sh_mid", "hip_mid", "kn_L"), ("sh_mid", "hip_mid", "kn_R"),
    ("hip_L", "kn_L", "ank_L"), ("hip_R", "kn_R", "ank_R"),
    ("sh_L", "el_L", "wr_L"), ("sh_R", "el_R", "wr_R"),
    ("spine37", "spine35", "hip_mid"), ("sh_mid", "spine35", "hip_mid"),
    ("nose", "ear_mid", "sh_mid"), ("hip_mid", "sh_mid", "el_L"), ("hip_mid", "sh_mid", "el_R"),
]
PAIR_Q = ["incl", "dx", "dy", "dz"]
PAIR_STATS = ["mean", "range"]
TRI_STATS = ["mean", "min", "max", "range"]
PAIRS = list(itertools.combinations(range(len(NAMES)), 2))  # (上, 下) 按 LANDMARKS 的自上而下顺序


def feature_table():
    feats, regions = [], []
    for i, j in PAIRS:
        for q in PAIR_Q:
            for s in PAIR_STATS:
                feats.append(f"{q}|{NAMES[i]}-{NAMES[j]}|{s}")
                regions.append(frozenset({REGION_OF[NAMES[i]], REGION_OF[NAMES[j]]}))
    for a, b, c in TRIPLETS:
        for s in TRI_STATS:
            feats.append(f"ang|{a}-{b}-{c}|{s}")
            regions.append(frozenset({REGION_OF[a], REGION_OF[b], REGION_OF[c]}))
    return feats, regions


def landmarks_of(fr) -> np.ndarray:
    k, j = fr["kp"], fr["j"]
    out = np.zeros((len(LANDMARKS), 3))
    for n, (name, src, idx) in enumerate(LANDMARKS):
        if src == "k":
            out[n] = k[idx]
        elif src == "j":
            out[n] = j[idx]
        else:
            (s1, i1), (s2, i2) = idx
            out[n] = ((k if s1 == "k" else j)[i1] + (k if s2 == "k" else j)[i2]) / 2
    return out


def _angle(a, b, c) -> float:
    v1, v2 = a - b, c - b
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def frame_quantities(L: np.ndarray, fwd: float) -> tuple[np.ndarray, np.ndarray]:
    """L (18,3) -> 地标对量 (n_pairs, 4), 三元组角 (n_tri,)。"""
    torso = float(np.linalg.norm(L[IDX["sh_mid"]] - L[IDX["hip_mid"]])) + 1e-6
    pq = np.zeros((len(PAIRS), 4))
    for p, (i, j) in enumerate(PAIRS):
        v = L[i] - L[j]  # 下 -> 上
        pq[p, 0] = math.degrees(math.atan2(v[0] * fwd, -v[1]))
        pq[p, 1] = v[0] * fwd / torso
        pq[p, 2] = -v[1] / torso
        pq[p, 3] = abs(v[2]) / torso
    tq = np.array([_angle(L[IDX[a]], L[IDX[b]], L[IDX[c]]) for a, b, c in TRIPLETS])
    return pq, tq


def video_bank(frames: dict, plan, fps: float) -> np.ndarray:
    n_feat = len(PAIRS) * 4 * len(PAIR_STATS) + len(TRIPLETS) * len(TRI_STATS)
    out = np.full((plan.shape[0], n_feat), np.nan, dtype=np.float32)
    if not frames:
        return out
    avail = np.array(sorted(frames))
    first, last = frames[avail[0]], frames[avail[-1]]
    dx_total = last["cam_t"][0] - first["cam_t"][0]
    if abs(dx_total) > 0.1:
        fwd = 1.0 if dx_total > 0 else -1.0
    else:
        d = np.mean([fr["kp"][0, 0] - fr["kp"][[5, 6], 0].mean() for fr in frames.values()])
        fwd = 1.0 if d > 0 else -1.0
    for ci, idx in enumerate(plan.tolist()):
        pqs, tqs = [], []
        for f in idx:
            j = int(np.abs(avail - f).argmin())
            if abs(int(avail[j]) - f) > MAX_GAP:
                continue
            pq, tq = frame_quantities(landmarks_of(frames[int(avail[j])]), fwd)
            pqs.append(pq)
            tqs.append(tq)
        if len(pqs) < 4:
            continue
        P, T = np.stack(pqs), np.stack(tqs)  # (t, n_pairs, 4), (t, n_tri)
        pair_feats = np.stack([P.mean(0), P.max(0) - P.min(0)], axis=-1).reshape(-1)  # (n_pairs,4,2)->flat 与 feature_table 顺序一致
        tri_feats = np.stack([T.mean(0), T.min(0), T.max(0), T.max(0) - T.min(0)], axis=-1).reshape(-1)
        out[ci] = np.concatenate([pair_feats, tri_feats])
    return out


def _work(item):
    name, video_path, sam_dir, num_samples = item
    total, fps = _probe(video_path)
    plan = _plan_frame_index(total, max(int(fps), 1), num_samples)
    frames = load_frames(sam_dir) if sam_dir else {}
    vals = video_bank(frames, plan, fps)
    return name, vals, float(1 - np.isnan(vals).any(1).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-path", required=True)
    ap.add_argument("--sam3d-root", required=True)
    ap.add_argument("--class-num", type=int, default=2)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root_path)
    info = root / "clinical_CLIP_dataset"
    folds = json.load(open(info / "index_mapping" / str(args.class_num) / "index.json"))
    first = next(iter(folds.values()))
    paths = sorted({p for s in first.values() for p in s})
    paths = [info / "json_mix" / p.split("json_mix/")[-1] for p in paths]
    sam_dirs = {Path(d).parent.name: d for d in glob.glob(f"{args.sam3d_root}/*/*/video")}

    items, meta_by = [], {}
    for p in paths:
        meta = json.load(open(p))
        name = meta["video_name"]
        if name in meta_by:
            continue
        video_path = str(p).split("json_mix/")[0] + "video/" + "/".join(meta["video_path"].split("/")[-3:])
        items.append((name, video_path, sam_dirs.get(name), args.num_samples))
        meta_by[name] = {"label": int(meta["label"]), "json": str(p).split("json_mix/")[-1]}

    feats, regions = feature_table()
    videos = {}
    with Pool(args.workers) as pool:
        for i, (name, vals, frac_ok) in enumerate(pool.imap_unordered(_work, items, chunksize=4)):
            videos[name] = {"vals": vals, "frac_ok": frac_ok, **meta_by[name]}
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(items)}", flush=True)
    all_vals = np.concatenate([v["vals"] for v in videos.values()])
    print(f"视频 {len(videos)}, 段 {all_vals.shape[0]}, 测量 {len(feats)}, 段级缺失 {np.isnan(all_vals[:, 0]).mean():.1%}")
    from collections import Counter
    print("测量的区域标签分布:", Counter("+".join(sorted(r)) for r in regions).most_common(12))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"features": feats, "regions": regions, "landmarks": NAMES, "videos": videos}, open(out, "wb"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

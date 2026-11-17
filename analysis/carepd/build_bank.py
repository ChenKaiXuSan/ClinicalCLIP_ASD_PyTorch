#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: build_bank.py
Project: analysis/carepd
Author: Kaixu Chen
-----
Comment:
CARE-PD 四个带 UPDRS-gait 标签的队列(PD-GaM / BMCLab / T-SDU-PD / 3DGait)的测量库,语法与 ASD 的
measure_bank_3d.py 相同,便于同一套"先验 -> 测量 -> 逻辑回归"流程直接迁移。

逐帧(30 fps,身体坐标系:x 左右, y 向上, z 前进,原点骨盆;见 smpl_fk.body_frame):
  每对地标 (上, 下):incl 矢状倾角(度,+ 向前)、dz 前后偏移 / 躯干长、dy 高度差 / 躯干长、dx 左右距离 / 躯干长
  13 个关节角(度)
  speed 骨盆的水平移动速度(腿长 / 秒)
窗口统计(1 秒不重叠窗口):地标对取 mean / range,关节角取 mean / min / max / range,speed 取 mean;
一次行走 = 各窗口统计的均值。共 190 × 4 × 2 + 13 × 4 + 1 = 1573 个测量。

    python analysis/carepd/build_bank.py --root /work/1/SKIING/chenkaixu/data/care_pd --out logs/carepd/bank.pkl
"""
from __future__ import annotations

import argparse
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smpl_fk import JOINT_NAMES, body_frame, forward_kinematics  # noqa: E402

J = {n: i for i, n in enumerate(JOINT_NAMES)}
COHORTS = ["PD-GaM", "BMCLab", "T-SDU-PD", "3DGait"]
# 地标(自上而下);mid 是两个关节的中点
LANDMARKS = [("head", "head"), ("neck", "neck"), ("sh_L", "sh_L"), ("sh_R", "sh_R"), ("sh_mid", ("sh_L", "sh_R")),
             ("spine3", "spine3"), ("el_L", "el_L"), ("el_R", "el_R"), ("spine1", "spine1"), ("wr_L", "wr_L"), ("wr_R", "wr_R"),
             ("hip_L", "hip_L"), ("hip_R", "hip_R"), ("hip_mid", ("hip_L", "hip_R")), ("kn_L", "kn_L"), ("kn_R", "kn_R"),
             ("ank_L", "ank_L"), ("ank_R", "ank_R"), ("foot_L", "foot_L"), ("foot_R", "foot_R")]
NAMES = [n for n, _ in LANDMARKS]
IDX = {n: i for i, n in enumerate(NAMES)}
ANGLES = {
    "head_trunk": ("head", "neck", "hip_mid"), "trunk_bend": ("sh_mid", "spine1", "hip_mid"), "spine_curve": ("neck", "spine3", "spine1"),
    "hip_L": ("sh_mid", "hip_L", "kn_L"), "hip_R": ("sh_mid", "hip_R", "kn_R"),
    "knee_L": ("hip_L", "kn_L", "ank_L"), "knee_R": ("hip_R", "kn_R", "ank_R"),
    "ankle_L": ("kn_L", "ank_L", "foot_L"), "ankle_R": ("kn_R", "ank_R", "foot_R"),
    "elbow_L": ("sh_L", "el_L", "wr_L"), "elbow_R": ("sh_R", "el_R", "wr_R"),
    "shoulder_arm_L": ("hip_mid", "sh_L", "el_L"), "shoulder_arm_R": ("hip_mid", "sh_R", "el_R"),
}
PAIR_Q, PAIR_STATS, TRI_STATS = ["incl", "dz", "dy", "dx"], ["mean", "range"], ["mean", "min", "max", "range"]
PAIRS = list(itertools.combinations(range(len(NAMES)), 2))
FPS = 30


def feature_names() -> list[str]:
    f = [f"{q}|{NAMES[i]}-{NAMES[j]}|{s}" for i, j in PAIRS for q in PAIR_Q for s in PAIR_STATS]
    f += [f"ang|{a}|{s}" for a in ANGLES for s in TRI_STATS]
    return f + ["speed|pelvis|mean"]


def landmarks(B: np.ndarray) -> np.ndarray:
    out = []
    for _, src in LANDMARKS:
        out.append(B[:, J[src]] if isinstance(src, str) else 0.5 * (B[:, J[src[0]]] + B[:, J[src[1]]]))
    return np.stack(out, 1)                       # (T, L, 3)


def walk_features(pose: np.ndarray, trans: np.ndarray, fps: int) -> np.ndarray:
    step = max(1, int(round(fps / FPS)))
    pose, trans = np.asarray(pose)[::step], np.asarray(trans)[::step]
    T = pose.shape[0]
    if T < FPS:
        return np.full(len(feature_names()), np.nan)
    P = forward_kinematics(pose, trans)
    L = landmarks(body_frame(P))
    trunk = float(np.linalg.norm(L[:, IDX["sh_mid"]] - L[:, IDX["hip_mid"]], axis=1).mean())
    leg = float((np.linalg.norm(P[:, J["hip_L"]] - P[:, J["kn_L"]], axis=1) + np.linalg.norm(P[:, J["kn_L"]] - P[:, J["ank_L"]], axis=1)).mean())
    i, j = np.array([p[0] for p in PAIRS]), np.array([p[1] for p in PAIRS])
    d = L[:, i] - L[:, j]                          # 上 - 下, (T, P, 3)
    frame = {
        "incl": np.degrees(np.arctan2(d[..., 2], d[..., 1])), "dz": d[..., 2] / trunk, "dy": d[..., 1] / trunk, "dx": np.abs(d[..., 0]) / trunk,
    }
    ang = []
    for a, b, c in ANGLES.values():
        v1, v2 = L[:, IDX[a]] - L[:, IDX[b]], L[:, IDX[c]] - L[:, IDX[b]]
        cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
        ang.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    ang = np.stack(ang, 1)                         # (T, A)
    hv = np.linalg.norm(np.diff(P[:, 0, [0, 2]], axis=0), axis=1) * FPS / leg     # 腿长 / 秒
    nw = T // FPS
    W = lambda x: x[: nw * FPS].reshape(nw, FPS, *x.shape[1:])                   # noqa: E731
    feats = []
    pair_stats = {q: W(frame[q]) for q in PAIR_Q}
    cols = []
    for q in PAIR_Q:
        w = pair_stats[q]
        cols.append(np.stack([w.mean(1).mean(0), (w.max(1) - w.min(1)).mean(0)], 1))        # (P, 2)
    feats.append(np.stack(cols, 1).reshape(-1))    # 顺序: pair -> q -> stat
    wa = W(ang)
    feats.append(np.stack([wa.mean(1).mean(0), wa.min(1).mean(0), wa.max(1).mean(0), (wa.max(1) - wa.min(1)).mean(0)], 1).reshape(-1))
    feats.append(np.array([hv.mean()]))
    return np.concatenate(feats).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/work/1/SKIING/chenkaixu/data/care_pd")
    ap.add_argument("--out", default="logs/carepd/bank.pkl")
    args = ap.parse_args()
    names = feature_names()
    bank = {"features": names, "cohorts": {}}
    for c in COHORTS:
        d = pickle.load(open(Path(args.root) / "Canonicalized_SMPL_pickles" / f"{c}_canonical.pkl", "rb"))
        X, y, sub, walk, med = [], [], [], [], []
        for s, wd in d.items():
            for w, r in wd.items():
                if r.get("UPDRS_GAIT") is None:
                    continue
                X.append(walk_features(r["pose"], r["trans"], int(r["fps"])))
                y.append(int(r["UPDRS_GAIT"])); sub.append(str(s)); walk.append(str(w)); med.append(r.get("medication"))
        X = np.stack(X); y = np.array(y)
        ok = ~np.isnan(X).any(1)
        bank["cohorts"][c] = {"X": X[ok], "y": y[ok], "subject": np.array(sub)[ok], "walk": np.array(walk)[ok], "medication": np.array(med, dtype=object)[ok]}
        print(f"{c}: 行走 {ok.sum()} (丢弃过短 {(~ok).sum()}), 受试者 {len(set(np.array(sub)[ok]))}, 类别 {np.bincount(y[ok]).tolist()}, 特征 {X.shape[1]}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(bank, open(args.out, "wb"))
    print("写入", args.out)


if __name__ == "__main__":
    main()

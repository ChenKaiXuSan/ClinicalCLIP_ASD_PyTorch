#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: build_measures.py
Project: analysis/koa
Author: Kaixu Chen
-----
Comment:
KOA-PD-NM 的第三步:SAM-3D-Body 关键点 -> 步态周期测量(与 CARE-PD 完全相同的 30 个量,analysis/carepd/gait_params.py),
以及骨架网络对照用的关节序列。

坐标:SAM-3D-Body 输出相机系(x 右, y 下, z 远离相机),全局位置 = pred_keypoints_3d + pred_cam_t。侧面机位下
行进方向 = ±x_cam(由 cam_t 的首尾位移定号 s),换到行走系(x 左, y 上, z 前) = (s·z_cam, −y_cam, s·x_cam),右手系。
只用行走者完整在画面内、且 3D 估计成功的最长连续一段;骨盆高度的线性趋势去掉(相机俯仰造成的假坡度)。
时间一律按 25 fps(detect.py 已按时间重采样),所以与类别绑定的原始帧率不会进到任何一个量里。

输出 logs/koa/gait.pkl:
  {"features", "videos": [名], "X": (n_video, 30), "subject", "label"(0 NM / 1 KOA / 2 PD), "severity", "direction",
   "n_frames", "joints": {video: (T, 24, 4)}(SMPL 关节顺序的身体系坐标 + 行走速度,给 ST-GCN), "demo": {subject: (sex, age, height)}}

    python analysis/koa/build_measures.py --proc $KOA/proc --out logs/koa/gait.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "carepd"))
from gait_params import FEATURES, params_from_joints, smooth  # noqa: E402
from smpl_fk import JOINT_NAMES, body_frame  # noqa: E402

J = {n: i for i, n in enumerate(JOINT_NAMES)}
FPS = 25.0
LABEL = {"NM": 0, "KOA": 1, "PD": 2}
# mhr70 关键点序号(与 analysis/measure_bank_3d.py 一致);趾根用大脚趾
K = {"nose": 0, "ear_L": 3, "ear_R": 4, "sh_L": 5, "sh_R": 6, "el_L": 7, "el_R": 8, "hip_L": 9, "hip_R": 10, "kn_L": 11, "kn_R": 12,
     "ank_L": 13, "ank_R": 14, "foot_L": 15, "foot_R": 18, "wr_L": 41, "wr_R": 62}


def parse_name(stem: str) -> tuple[str, str, str, int]:
    """001_KOA_01_EL -> (subject 001_KOA_EL, KOA, EL, 次序 1);001_NM_02 -> (001_NM, NM, '', 2)。"""
    p = stem.split("_")
    sev = p[3] if len(p) > 3 else ""
    return "_".join([p[0], p[1]] + ([sev] if sev else [])), p[1], sev, int(p[2])


def to_smpl24(kp: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
    """(T,70,3) 相机系 + 平移 -> (T,24,3) 行走系下 SMPL 关节顺序;SMPL 有而这里没有的关节用相邻关节插值。"""
    G = kp + cam_t[:, None]
    s = 1.0 if (cam_t[-1, 0] - cam_t[0, 0]) >= 0 else -1.0
    W = np.stack([s * G[..., 2], -G[..., 1], s * G[..., 0]], -1)
    g = lambda n: W[:, K[n]]                                  # noqa: E731
    P = np.full((len(W), 24, 3), np.nan)
    for n in ("hip_L", "hip_R", "kn_L", "kn_R", "ank_L", "ank_R", "foot_L", "foot_R", "sh_L", "sh_R", "el_L", "el_R", "wr_L", "wr_R"):
        P[:, J[n]] = g(n)
    pel, neck = 0.5 * (g("hip_L") + g("hip_R")), 0.5 * (g("sh_L") + g("sh_R"))
    P[:, J["pelvis"]], P[:, J["neck"]] = pel, neck
    P[:, J["head"]] = 0.5 * (g("ear_L") + g("ear_R"))
    for n, w in (("spine1", 0.25), ("spine2", 0.5), ("spine3", 0.75)):
        P[:, J[n]] = pel + w * (neck - pel)
    P[:, J["collar_L"]], P[:, J["collar_R"]] = 0.5 * (neck + g("sh_L")), 0.5 * (neck + g("sh_R"))
    P[:, J["hand_L"]], P[:, J["hand_R"]] = g("wr_L"), g("wr_R")
    return P


def valid_run(det, ok: np.ndarray) -> slice:
    """行走者完整在画面内且 3D 成功的最长连续段。"""
    H, W = det["size"]
    b = det["boxes"]
    inside = (b[:, 0] > 0.01 * W) & (b[:, 2] < 0.99 * W) & ok
    best, cur, start = (0, 0), 0, 0
    for i, v in enumerate(np.append(inside, False)):
        if v:
            if cur == 0:
                start = i
            cur += 1
        else:
            if cur > best[1] - best[0]:
                best = (start, start + cur)
            cur = 0
    return slice(*best)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="/work/1/SKIING/chenkaixu/data/koa_pd_nm/proc")
    ap.add_argument("--out", default="logs/koa/gait.pkl")
    args = ap.parse_args()
    proc = Path(args.proc)
    demo = {}
    for line in open(Path(__file__).with_name("demographics.tsv")):
        if line.startswith("#") or line.startswith("subject"):
            continue
        s, sex, age, h = line.split()
        demo[s] = (sex, int(age), float(h))
    out = {"features": FEATURES, "videos": [], "X": [], "subject": [], "label": [], "severity": [], "direction": [], "n_frames": [],
           "joints": {}, "valid": {}, "demo": demo}
    for f in sorted((proc / "sam3d").glob("*.npz")):
        S = np.load(f); D = np.load(proc / "det" / f.name)
        sl = valid_run(D, S["ok"])
        n = sl.stop - sl.start
        sub, dis, sev, seq = parse_name(f.stem)
        assert sub in demo, sub
        if n < FPS:
            print(f"[skip] {f.stem}: 有效帧只有 {n}")
            continue
        P = to_smpl24(S["kp3d"][sl].astype(np.float64), S["cam_t"][sl].astype(np.float64))
        x = params_from_joints(P, FPS, detrend=True)
        Ps = smooth(P, 3)
        B = body_frame(Ps).astype(np.float32)
        v = np.linalg.norm(np.diff(smooth(Ps[:, 0][:, [0, 2]], 13), axis=0, prepend=Ps[:1, 0][:, [0, 2]]), axis=1).astype(np.float32) * FPS
        out["joints"][f.stem] = np.concatenate([B, np.repeat(v[:, None, None], 24, 1)], -1)
        out["valid"][f.stem] = (sl.start, sl.stop)
        out["videos"].append(f.stem); out["X"].append(x); out["subject"].append(sub); out["label"].append(LABEL[dis]); out["severity"].append(sev)
        out["direction"].append(seq); out["n_frames"].append(n)
    for k in ("X", "label", "n_frames", "direction"):
        out[k] = np.array(out[k])
    out["subject"] = np.array(out["subject"]); out["severity"] = np.array(out["severity"])
    X, y = out["X"], out["label"]
    print(f"视频 {len(X)}, 受试者 {len(set(out['subject']))}, 类别(NM/KOA/PD)视频数 {np.bincount(y).tolist()}, 有效帧中位 {np.median(out['n_frames']):.0f}")
    print("缺失率 > 0 的量:", {FEATURES[i]: round(float(m), 2) for i, m in enumerate(np.isnan(X).mean(0)) if m > 0})
    print(f"{'量':18s} " + " ".join(f"{g:>9s}" for g in ("NM", "KOA", "PD")))
    for i, n in enumerate(FEATURES):
        print(f"{n:18s} " + " ".join(f"{np.nanmedian(X[y == c, i]):9.2f}" for c in range(3)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(args.out, "wb"))
    print("写入", args.out)


if __name__ == "__main__":
    main()

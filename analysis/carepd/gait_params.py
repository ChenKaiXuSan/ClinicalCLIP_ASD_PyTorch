#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: gait_params.py
Project: analysis/carepd
Author: Kaixu Chen
-----
Comment:
CARE-PD 的步态周期测量(第二版测量库)。第一版 build_bank.py 沿用 ASD 的语法(地标对的静态几何 + 1 秒窗口极差),
但 MDS-UPDRS 3.10 步态条目评的是"步幅、步速、抬脚高度、足跟着地、摆臂",这些是按步定义的量,1 秒窗口会把慢步截断。
这里按步态事件直接算临床上的时空参数,单位是物理量(米、秒、度;CARE-PD 的 beta 置零,所有人同一副骨架),
所以不同队列之间可以直接合并。同时修正第一版把 PD-GaM 的 25 fps 当成 30 fps 的问题(时间一律用真实帧率)。

事件:a(t) = 左踝 - 右踝的前后距离(身体坐标系),平滑后的极大值 = 左足着地,极小值 = 右足着地。
行走列表与顺序取自第一版的 bank.pkl,保证与测量库、ST-GCN 概率逐行对齐;算不出的量记 NaN,评测时用训练折中位数填补。

    python analysis/carepd/gait_params.py --bank logs/carepd/bank.pkl --out logs/carepd/gait.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smpl_fk import JOINT_NAMES, body_frame, forward_kinematics  # noqa: E402

J = {n: i for i, n in enumerate(JOINT_NAMES)}
FEATURES = [
    "speed", "cadence", "step_time_cv", "step_len", "step_len_cv", "step_len_asym", "walk_ratio", "step_width",
    "ank_clear", "ank_clear_min", "toe_clear", "foot_pitch_hs", "foot_pitch_to", "foot_pitch_range",
    "arm_swing", "arm_swing_min", "arm_swing_asym", "sh_flex_range", "sh_flex_range_min", "elbow_mean",
    "knee_rom", "knee_rom_asym", "hip_rom", "hip_rom_asym", "ankle_rom",
    "trunk_lean", "head_fwd", "trunk_rot_range", "pelvis_vert_osc", "double_support",
]
# 先验集合:只取 MDS-UPDRS 3.10 评分要点点名的量(步幅、步速、抬脚高度、足跟着地、摆臂),写在看任何结果之前
RUBRIC = ["step_len", "speed", "cadence", "ank_clear", "toe_clear", "foot_pitch_hs", "arm_swing", "arm_swing_min"]


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    c = np.cumsum(np.insert(xp, 0, 0, axis=0), axis=0)
    return (c[k:] - c[:-k]) / k


def rrange(x: np.ndarray) -> float:
    """稳健极差:2.5–97.5 分位。"""
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(hi - lo)


def angle(a, b, c) -> np.ndarray:
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def asym(l: float, r: float) -> float:
    return abs(l - r) / (abs(l) + abs(r) + 1e-9)


def walk_params(pose: np.ndarray, trans: np.ndarray, fps: int) -> np.ndarray:
    step = max(1, int(round(fps / 30)))
    f = fps / step                                           # 真实的有效帧率
    return params_from_joints(forward_kinematics(np.asarray(pose)[::step], np.asarray(trans)[::step]), f)


def params_from_joints(P: np.ndarray, f: float, detrend: bool = False) -> np.ndarray:
    """P (T, 24, 3):SMPL 关节顺序的世界坐标(x 左, y 上, z 前);只用到骨盆、髋、膝、踝、趾根、肩、肘、腕、头,
    其余关节可以是 NaN。f = 帧率。detrend=True 时去掉骨盆高度随时间的线性趋势(单目重建里相机俯仰造成的假坡度)。"""
    k = max(1, int(round(0.1 * f)) | 1)
    P = smooth(P, k)
    if detrend and len(P) >= 3:
        t = np.arange(len(P))
        a = np.polyfit(t, P[:, 0, 1], 1)[0]
        P = P.copy(); P[..., 1] -= (a * (t - t.mean()))[:, None]
    B = body_frame(P)
    T = len(P)
    out = dict.fromkeys(FEATURES, np.nan)
    if T < f:
        return np.array([out[n] for n in FEATURES], dtype=np.float32)
    g = lambda n: B[:, J[n]]                                 # noqa: E731
    dur = (T - 1) / f
    pel = smooth(P[:, 0][:, [0, 2]], max(1, int(round(0.5 * f)) | 1))     # 0.5 秒平滑:去掉逐帧抖动对路径长度的抬高
    out["speed"] = float(np.linalg.norm(np.diff(pel, axis=0), axis=1).sum() / dur)
    # ---- 事件
    a = g("ank_L")[:, 2] - g("ank_R")[:, 2]
    dist, prom = max(1, int(0.3 * f)), 0.05
    hsL, _ = find_peaks(a, distance=dist, prominence=prom)
    hsR, _ = find_peaks(-a, distance=dist, prominence=prom)
    ev = np.sort(np.concatenate([hsL, hsR]))
    side = np.where(np.isin(ev, hsL), 0, 1)
    if len(ev) >= 2:
        st = np.diff(ev) / f
        st = st[(st > 0.25) & (st < 2.0)]
        if len(st):
            out["cadence"] = float(60.0 / np.median(st))
            if len(st) >= 3:
                out["step_time_cv"] = float(st.std() / st.mean())
    if len(ev) >= 1:
        sl = np.abs(a[ev])
        out["step_len"] = float(sl.mean())
        if len(sl) >= 3:
            out["step_len_cv"] = float(sl.std() / sl.mean())
        if (side == 0).any() and (side == 1).any():
            out["step_len_asym"] = asym(sl[side == 0].mean(), sl[side == 1].mean())
        if np.isfinite(out["cadence"]):
            out["walk_ratio"] = float(out["step_len"] / out["cadence"] * 100)
        out["step_width"] = float(np.abs(g("ank_L")[ev, 0] - g("ank_R")[ev, 0]).mean())
    # ---- 抬脚与足跟着地
    cl = [rrange(P[:, J[f"ank_{s}"], 1]) for s in "LR"]
    out["ank_clear"], out["ank_clear_min"] = float(np.mean(cl)), float(np.min(cl))
    out["toe_clear"] = float(np.mean([rrange(P[:, J[f"foot_{s}"], 1]) for s in "LR"]))
    pitch = {s: np.degrees(np.arctan2(g(f"foot_{s}")[:, 1] - g(f"ank_{s}")[:, 1], g(f"foot_{s}")[:, 2] - g(f"ank_{s}")[:, 2])) for s in "LR"}
    base = {s: np.median(pitch[s]) for s in "LR"}             # 足底放平时的基线(踝 -> 趾根本身就向下倾)
    hs = [pitch[s][e] - base[s] for s, e in (("L", hsL), ("R", hsR)) if len(e)]
    to = [pitch[s][e] - base[s] for s, e in (("L", hsR), ("R", hsL)) if len(e)]   # 对侧着地时本侧正要离地
    if hs:
        out["foot_pitch_hs"] = float(np.mean(np.concatenate(hs)))
    if to:
        out["foot_pitch_to"] = float(np.mean(np.concatenate(to)))
    out["foot_pitch_range"] = float(np.mean([rrange(pitch[s]) for s in "LR"]))
    # ---- 摆臂
    sw = [rrange(g(f"wr_{s}")[:, 2] - g(f"sh_{s}")[:, 2]) for s in "LR"]
    out["arm_swing"], out["arm_swing_min"], out["arm_swing_asym"] = float(np.mean(sw)), float(np.min(sw)), asym(*sw)
    fl = [rrange(np.degrees(np.arctan2(g(f"el_{s}")[:, 2] - g(f"sh_{s}")[:, 2], -(g(f"el_{s}")[:, 1] - g(f"sh_{s}")[:, 1])))) for s in "LR"]
    out["sh_flex_range"], out["sh_flex_range_min"] = float(np.mean(fl)), float(np.min(fl))
    out["elbow_mean"] = float(np.mean([angle(g(f"sh_{s}"), g(f"el_{s}"), g(f"wr_{s}")).mean() for s in "LR"]))
    # ---- 下肢关节活动度
    kn = [rrange(angle(g(f"hip_{s}"), g(f"kn_{s}"), g(f"ank_{s}"))) for s in "LR"]
    hp = [rrange(np.degrees(np.arctan2(g(f"kn_{s}")[:, 2] - g(f"hip_{s}")[:, 2], -(g(f"kn_{s}")[:, 1] - g(f"hip_{s}")[:, 1])))) for s in "LR"]
    out["knee_rom"], out["knee_rom_asym"] = float(np.mean(kn)), asym(*kn)
    out["hip_rom"], out["hip_rom_asym"] = float(np.mean(hp)), asym(*hp)
    out["ankle_rom"] = float(np.mean([rrange(angle(g(f"kn_{s}"), g(f"ank_{s}"), g(f"foot_{s}"))) for s in "LR"]))
    # ---- 躯干
    shm, hipm = 0.5 * (g("sh_L") + g("sh_R")), 0.5 * (g("hip_L") + g("hip_R"))
    d = shm - hipm
    out["trunk_lean"] = float(np.degrees(np.arctan2(d[:, 2], d[:, 1])).mean())
    dh = g("head") - shm
    out["head_fwd"] = float(np.degrees(np.arctan2(dh[:, 2], dh[:, 1])).mean())
    sl_ = g("sh_L") - g("sh_R")
    out["trunk_rot_range"] = rrange(np.degrees(np.arctan2(sl_[:, 2], sl_[:, 0])))
    out["pelvis_vert_osc"] = rrange(P[:, 0, 1])
    # ---- 双支撑:踝的水平速度低于行走速度一半视为支撑
    v = {s: np.linalg.norm(np.diff(P[:, J[f"ank_{s}"]][:, [0, 2]], axis=0), axis=1) * f for s in "LR"}
    thr = 0.5 * max(out["speed"], 1e-3)
    out["double_support"] = float(((v["L"] < thr) & (v["R"] < thr)).mean())
    return np.array([out[n] for n in FEATURES], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/work/1/SKIING/chenkaixu/data/care_pd")
    ap.add_argument("--bank", default="logs/carepd/bank.pkl")
    ap.add_argument("--out", default="logs/carepd/gait.pkl")
    args = ap.parse_args()
    bank = pickle.load(open(args.bank, "rb"))
    res = {"features": FEATURES, "cohorts": {}}
    for c, bd in bank["cohorts"].items():
        d = pickle.load(open(Path(args.root) / "Canonicalized_SMPL_pickles" / f"{c}_canonical.pkl", "rb"))
        recs = {(str(s), str(w)): r for s, wd in d.items() for w, r in wd.items()}      # 有的队列编号是整数
        X = np.stack([walk_params(r["pose"], r["trans"], int(r["fps"])) for r in (recs[(s, w)] for s, w in zip(bd["subject"], bd["walk"]))])
        res["cohorts"][c] = {**{k: bd[k] for k in ("y", "subject", "walk", "medication")}, "X": X}
        nan = np.isnan(X).mean(0)
        print(f"== {c}: 行走 {len(X)}, 缺失率最高的量 " + ", ".join(f"{FEATURES[i]} {nan[i]:.2f}" for i in np.argsort(-nan)[:3]))
        med = np.nanmedian(X, 0)
        print("   中位数: " + ", ".join(f"{n} {m:.2f}" for n, m in zip(FEATURES, med)), flush=True)
    pickle.dump(res, open(args.out, "wb"))
    print("写入", args.out)


if __name__ == "__main__":
    main()

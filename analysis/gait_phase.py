#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: gait_phase.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
"注意力区域 × 步态相位":医生区域决定量哪里,步态相位决定量什么时候。

现有 5 量按 1 秒窗口取均值 / 极值,把周期内的时间结构平均掉了。这里用逐帧 3D 关节点
(SAM-3D-Body, 30 fps, 根关节相对的相机坐标, x 为行进方向)先检测步态事件,再把每个量
按相位分辨成 "区域 × 相位" 的 token。

事件检测(坐标法, Zeni et al. 2008):踝相对髋中点沿行进方向的位移,极大 = 触地 HS,极小 = 离地 TO。
周期 = 同一条腿相邻两次 HS,时长限制在 [0.6, 2.0] s。周期内按
    HS -> 对侧 TO -> 对侧 HS -> 本侧 TO -> 下次 HS
切成 4 个相位:承重 / 单支撑 / 推离 / 摆动;对侧事件缺失或乱序时退化为四等分(记 event_based=False)。

每个周期一行,字段分四类(角度都在矢状面 x-y 内量,定义与 geom_attributes_3d.py 一致):
  cyc_*          周期级:现有 5 量改在周期窗口上算,外加周期时长。对照"换成按周期聚合"本身的效应。
  ph{k}_<q>      相位级:12 个逐帧量在第 k 相位的均值。医生区域的 4 个是 lean / head / shoff / hip,
                 其余 8 个(knee / step / arm / low / curve / obl / lat / sway)供未标部位对照与随机抽取对照。
  tim_*          时间量:躯干前倾峰值所在相位、周期内调制幅度等 —— "什么时候"本身。
  shuf{s}_ph{k}_<q>  相位打乱对照:帧随机分到 4 个相位(各相位帧数不变)后的均值,s = 0..4。

    python analysis/gait_phase.py extract --root-path $DATA --sam3d-root $SAM3D --out-dir logs/gait_phase_3d
    python analysis/gait_phase.py eval --out-dir logs/gait_phase_3d --attr-root logs/geom_attributes_3d_doctor \
        --exclude-missing logs/measure_bank_3d/bank.pkl
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))
sys.path.insert(0, str(ROOT / "analysis"))

from dataloader.whole_video_dataset import _probe  # noqa: E402
from geom_attributes_3d import (  # noqa: E402
    L_ANK, L_EAR, L_HIP, L_KNEE, L_SH, L_WR, MAX_GAP, NOSE, R_ANK, R_EAR, R_HIP, R_KNEE, R_SH, R_WR,
    SPINE_HIGH, SPINE_LOW, _joint_angle, _sag, load_frames,
)

# 逐帧量的名字与归属。前 4 个是医生区域的量(头 / 肩 / 腰椎骨盆 / 腰椎骨盆)
FRAME_Q = ["lean", "head", "shoff", "hip", "knee", "step", "arm", "low", "curve", "obl", "lat", "sway"]
DOCTOR_Q = ["lean", "head", "shoff", "hip"]
UNMARKED_Q = ["knee", "step"]
PHASES = ["loading", "single_support", "pre_swing", "swing"]
N_PHASE = 4
N_SHUF = 5
# 秒。上限 3.0 而不是 2.0:26 名患者的部分录像里同腿相邻触地间隔集中在 2.1–2.5 s(慢行或时间基准不同),
# 2.0 会把这些视频整条丢掉;相位只用周期内的相对位置,不受绝对时长影响。漏检一次 HS 造成的
# 双倍周期由"周期内恰好一次 TO"的规则挡住。
MIN_CYCLE, MAX_CYCLE = 0.6, 3.0
PROM_FRAC = 0.25                     # 峰的显著度阈值 = 位移幅度(5–95 分位极差)的这个比例


# ----------------------------------------------------------------------------- 逐帧量
def frame_series(frames: dict[int, dict]) -> tuple[np.ndarray, dict[str, np.ndarray], float, int]:
    """把逐帧关节点整理成等间隔的时间序列(缺帧为 nan)。返回 (有效掩码, {量: 序列}, fwd, 总帧数)。"""
    avail = np.array(sorted(frames))
    n = int(avail[-1]) + 1
    first, last = frames[int(avail[0])], frames[int(avail[-1])]
    dx_total = last["cam_t"][0] - first["cam_t"][0]
    if abs(dx_total) > 0.1:
        fwd = 1.0 if dx_total > 0 else -1.0
    else:
        d = np.mean([fr["kp"][NOSE, 0] - fr["kp"][[L_SH, R_SH], 0].mean() for fr in frames.values()])
        fwd = 1.0 if d > 0 else -1.0

    cols = ["ank_l", "ank_r", "hip_l", "hip_r", "knee_l", "knee_r", "leg",
            "lean", "head", "shoff", "step", "arm", "low", "curve", "obl", "lat", "sway", "torso", "dz_hip"]
    S = {c: np.full(n, np.nan) for c in cols}
    for f in avail:
        fr = frames[int(f)]
        k, jc = fr["kp"], fr["j"]
        S["dz_hip"][f] = k[L_HIP, 2] - k[R_HIP, 2]
        sh = k[[L_SH, R_SH]].mean(0)
        hip = k[[L_HIP, R_HIP]].mean(0)
        trunk = sh - hip
        torso = float(np.linalg.norm(trunk)) + 1e-6
        S["torso"][f] = torso
        S["ank_l"][f] = (k[L_ANK, 0] - hip[0]) * fwd
        S["ank_r"][f] = (k[R_ANK, 0] - hip[0]) * fwd
        S["lean"][f] = _sag(trunk[0] * fwd, trunk[1])
        S["shoff"][f] = trunk[0] * fwd / torso
        ear = k[[L_EAR, R_EAR]].mean(0)
        S["head"][f] = _sag((ear[0] - sh[0]) * fwd, ear[1] - sh[1])
        legs = []
        for side, (hp, kn, an) in (("l", (L_HIP, L_KNEE, L_ANK)), ("r", (R_HIP, R_KNEE, R_ANK))):
            S[f"hip_{side}"][f] = 180.0 - _joint_angle(sh, k[hp], k[kn])      # 髋屈角,+屈
            S[f"knee_{side}"][f] = 180.0 - _joint_angle(k[hp], k[kn], k[an])  # 膝屈角
            legs.append(float(np.linalg.norm(k[hp] - k[kn]) + np.linalg.norm(k[kn] - k[an])))
        S["leg"][f] = float(np.mean(legs))
        S["step"][f] = abs(k[L_ANK, 0] - k[R_ANK, 0]) / S["leg"][f]
        S["arm"][f] = float(np.mean([(k[L_WR, 0] - hip[0]) * fwd / torso, (k[R_WR, 0] - hip[0]) * fwd / torso]))
        low = jc[SPINE_LOW] - hip
        up = jc[SPINE_HIGH] - jc[SPINE_LOW]
        a_low = _sag(low[0] * fwd, low[1])
        S["low"][f] = a_low
        S["curve"][f] = _sag(up[0] * fwd, up[1]) - a_low
        hl, hr = k[L_HIP], k[R_HIP]
        S["obl"][f] = abs(math.degrees(math.atan2(hl[1] - hr[1], abs(hl[2] - hr[2]) + 1e-6)))
        S["lat"][f] = abs(math.degrees(math.atan2(trunk[2], -trunk[1])))
        S["sway"][f] = trunk[2] / torso
    valid = ~np.isnan(S["lean"])
    # 侧视行走时左腿要么一直离相机近要么一直远:两髋深度差的符号在整条视频里应恒定,
    # 符号反了的帧(全库约 0.1%)是单目重建的镜像翻转,按缺帧处理。
    dz = S["dz_hip"]
    flipped = valid & (np.sign(dz) != np.sign(np.nanmedian(dz)))
    for c in cols:
        S[c][flipped] = np.nan
    valid &= ~flipped
    S["n_flipped"] = np.array([int(flipped.sum())])
    return valid, S, fwd, n


def _despike(x: np.ndarray, k: int = 5) -> np.ndarray:
    """逐段中值滤波去孤立尖峰(两腿交叉、远腿被遮挡时踝关节会单帧跳 0.3–0.9 m;3D 结果两帧重复,
    尖峰通常持续 2 帧,窗口 5 能盖住)。只在连续的有效段上做,nan 原样保留。"""
    from scipy.signal import medfilt

    y = x.copy()
    ok = ~np.isnan(x)
    start = None
    for i in range(len(x) + 1):
        if i < len(x) and ok[i]:
            start = i if start is None else start
        elif start is not None:
            if i - start >= k:
                y[start:i] = medfilt(x[start:i], k)
            start = None
    return y


def _fill_short_gaps(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """线性插补 ≤ MAX_GAP 的缺帧;更长的缺口保持 nan(含缺口的周期会被丢弃)。"""
    y = x.copy()
    idx = np.arange(len(x))
    if valid.sum() < 2:
        return y
    y[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    # 把长缺口还原成 nan
    start = None
    for i in range(len(x) + 1):
        if i < len(x) and not valid[i]:
            start = i if start is None else start
        elif start is not None:
            if i - start > MAX_GAP:
                y[start:i] = np.nan
            start = None
    return y


# ----------------------------------------------------------------------------- 事件检测
def detect_events(sig: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """一条腿:返回 (HS 帧号, TO 帧号)。sig 已插补,nan 段不出事件。"""
    from scipy.signal import butter, filtfilt, find_peaks

    x = sig.copy()
    ok = ~np.isnan(x)
    if ok.sum() < int(1.0 * fps):
        return np.array([], int), np.array([], int)
    # 低通 6 Hz;nan 处临时用均值填,事件落在 nan 区的后面剔除
    fill = float(np.nanmean(x))
    xf = np.where(ok, x, fill)
    b, a = butter(2, 6.0 / (fps / 2.0))
    if len(xf) > 3 * max(len(a), len(b)):
        xf = filtfilt(b, a, xf)
    rng = float(np.nanpercentile(x, 95) - np.nanpercentile(x, 5))
    if rng < 0.05:    # 幅度不到 5 cm:没在走
        return np.array([], int), np.array([], int)
    dist = max(int(MIN_CYCLE * fps), 1)
    hs, _ = find_peaks(xf, distance=dist, prominence=PROM_FRAC * rng)
    to, _ = find_peaks(-xf, distance=dist, prominence=PROM_FRAC * rng)
    hs = np.array([h for h in hs if ok[h]], int)
    to = np.array([t for t in to if ok[t]], int)
    return hs, to


def build_cycles(ev: dict[str, tuple[np.ndarray, np.ndarray]], fps: float, valid_filled: np.ndarray) -> list[dict]:
    """同一条腿相邻两次 HS 为一个周期;相位边界按事件,缺失时四等分。"""
    cycles = []
    for leg, other in (("l", "r"), ("r", "l")):
        hs, to = ev[leg]
        ohs, oto = ev[other]
        for h0, h1 in zip(hs[:-1], hs[1:]):
            dur = (h1 - h0) / fps
            if not (MIN_CYCLE <= dur <= MAX_CYCLE):
                continue
            if not valid_filled[h0:h1 + 1].all():      # 周期内有长缺口
                continue
            t_in = [t for t in to if h0 < t < h1]
            if len(t_in) != 1:
                continue
            t_own = int(t_in[0])
            c_to = [t for t in oto if h0 < t < h1]
            c_hs = [h for h in ohs if h0 < h < h1]
            event_based = len(c_to) == 1 and len(c_hs) == 1 and h0 < c_to[0] < c_hs[0] < t_own < h1
            if event_based:
                bounds = [int(h0), int(c_to[0]), int(c_hs[0]), t_own, int(h1)]
            else:
                bounds = [int(round(v)) for v in np.linspace(h0, h1, N_PHASE + 1)]
            cycles.append({"leg": leg, "hs0": int(h0), "hs1": int(h1), "to": t_own,
                           "cto": int(c_to[0]) if len(c_to) == 1 else None,
                           "chs": int(c_hs[0]) if len(c_hs) == 1 else None,
                           "event_based": bool(event_based), "bounds": bounds, "dur": float(dur)})
    cycles.sort(key=lambda c: c["hs0"])
    return cycles


# ----------------------------------------------------------------------------- 周期特征
def cycle_features(c: dict, S: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, float]:
    h0, h1 = c["hs0"], c["hs1"]
    leg = c["leg"]
    idx = np.arange(h0, h1)              # 不含下一次 HS
    q = {k: S[k] for k in ("lean", "head", "shoff", "step", "arm", "low", "curve", "obl", "lat", "sway")}
    q["hip"] = S[f"hip_{leg}"]
    q["knee"] = S[f"knee_{leg}"]
    out: dict[str, float] = {}

    # 周期级:现有 5 量在周期窗口上的版本(髋屈最大 / 髋幅度按原定义合并两腿)
    hip_both = np.concatenate([S["hip_l"][idx], S["hip_r"][idx]])
    out["cyc_trunk_lean"] = float(np.nanmean(q["lean"][idx]))
    out["cyc_head_forward"] = float(np.nanmean(q["head"][idx]))
    out["cyc_shoulder_offset"] = float(np.nanmean(q["shoff"][idx]))
    out["cyc_hip_flexion_max"] = float(np.nanmax(hip_both))
    out["cyc_hip_range"] = float(np.nanmax(hip_both) - np.nanmin(hip_both))
    out["cyc_duration"] = c["dur"]

    # 相位级
    b = c["bounds"]
    phase_idx = [np.arange(b[k], max(b[k + 1], b[k] + 1)) for k in range(N_PHASE)]
    for k, pi in enumerate(phase_idx):
        for name in FRAME_Q:
            out[f"ph{k}_{name}"] = float(np.nanmean(q[name][pi]))

    # 时间量:峰值相位(0–1)与周期内调制
    rel = (idx - h0) / max(h1 - h0, 1)
    for name in ("lean", "head", "shoff", "hip"):
        v = q[name][idx]
        out[f"tim_{name}_peak_phase"] = float(rel[int(np.nanargmax(v))])
        out[f"tim_{name}_mod"] = float(np.nanmax(v) - np.nanmin(v))
    # 单支撑期相对摆动期的躯干前倾差:代偿是否随支撑状态变化
    out["tim_lean_ss_minus_swing"] = out["ph1_lean"] - out["ph3_lean"]

    # 相位打乱对照:帧随机分到 4 个相位,每相位帧数与真实相位相同
    sizes = [len(pi) for pi in phase_idx]
    for s in range(N_SHUF):
        perm = rng.permutation(idx)
        start = 0
        for k, sz in enumerate(sizes):
            pi = perm[start:start + sz]
            start += sz
            for name in DOCTOR_Q + UNMARKED_Q:
                out[f"shuf{s}_ph{k}_{name}"] = float(np.nanmean(q[name][pi])) if len(pi) else float("nan")
    return out


def _work(item):
    name, video_path, sam_dir = item
    total, fps = _probe(video_path)
    fps = float(fps) if fps else 30.0
    frames = load_frames(sam_dir) if sam_dir else {}
    rec = {"n_video_frames": int(total), "fps": fps, "n_3d_frames": len(frames), "cycles": [],
           "n_hs": 0, "event_based": 0}
    if len(frames) < int(1.0 * fps):
        return name, rec
    valid, S, fwd, n = frame_series(frames)
    n_flipped = int(S.pop("n_flipped")[0])
    filled = {k: _despike(_fill_short_gaps(v, valid)) for k, v in S.items()}
    valid_filled = ~np.isnan(filled["lean"])
    ev = {leg: detect_events(filled[f"ank_{leg}"], fps) for leg in ("l", "r")}
    rec["n_hs"] = int(len(ev["l"][0]) + len(ev["r"][0]))
    rec["n_flipped"] = n_flipped
    rec["events"] = {leg: {"hs": ev[leg][0].tolist(), "to": ev[leg][1].tolist()} for leg in ("l", "r")}
    rec["ank"] = {leg: [None if np.isnan(v) else round(float(v), 3) for v in filled[f"ank_{leg}"]] for leg in ("l", "r")}
    cycles = build_cycles(ev, fps, valid_filled)
    rng = np.random.default_rng(abs(hash(name)) % (2 ** 32))
    for c in cycles:
        c["feats"] = cycle_features(c, filled, rng)
    rec["cycles"] = cycles
    rec["event_based"] = int(sum(c["event_based"] for c in cycles))
    return name, rec


def cmd_extract(args) -> None:
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
        items.append((name, video_path, sam_dirs.get(name)))
        meta_by[name] = {"label": int(meta["label"]), "json": str(p).split("json_mix/")[-1]}
    if args.limit:
        items = items[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_video: dict[str, dict] = {}
    with Pool(args.workers) as pool:
        for i, (name, rec) in enumerate(pool.imap_unordered(_work, items, chunksize=4)):
            per_video[name] = {**rec, **meta_by[name]}
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(items)}", flush=True)

    # 汇总统计
    n_vid = len(per_video)
    n_3d = sum(1 for v in per_video.values() if v["n_3d_frames"] > 0)
    n_cyc = sum(len(v["cycles"]) for v in per_video.values())
    n_with = sum(1 for v in per_video.values() if v["cycles"])
    n_eb = sum(v["event_based"] for v in per_video.values())
    durs = np.array([c["dur"] for v in per_video.values() for c in v["cycles"]])
    rel = {k: [] for k in ("cto", "chs", "to")}
    for v in per_video.values():
        for c in v["cycles"]:
            if c["event_based"]:
                span = c["hs1"] - c["hs0"]
                for k in rel:
                    rel[k].append((c[k] - c["hs0"]) / span)
    pat = {}
    for v in per_video.values():
        pid = Path(v["json"]).stem.split("-")[0]
        pat.setdefault(pid, {"cycles": 0, "has3d": False})
        pat[pid]["cycles"] += len(v["cycles"])
        pat[pid]["has3d"] |= v["n_3d_frames"] > 0
    pat3d = [p for p, d in pat.items() if d["has3d"]]
    pat_nocyc = [p for p in pat3d if pat[p]["cycles"] == 0]
    summary = {
        "videos": n_vid, "videos_with_3d": n_3d, "videos_with_cycles": n_with, "cycles": n_cyc,
        "cycles_event_based": n_eb, "cycles_per_video_with_3d": n_cyc / max(n_3d, 1),
        "cycle_duration_median": float(np.median(durs)) if len(durs) else None,
        "cycle_duration_p5_p95": [float(np.percentile(durs, 5)), float(np.percentile(durs, 95))] if len(durs) else None,
        "rel_event_median": {k: float(np.median(x)) for k, x in rel.items() if x},
        "patients_with_3d": len(pat3d), "patients_without_cycles": pat_nocyc,
        "cycles_per_patient_median": float(np.median([pat[p]["cycles"] for p in pat3d])) if pat3d else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    json.dump(summary, open(out_dir / "summary.json", "w"), ensure_ascii=False, indent=1)
    json.dump(per_video, open(out_dir / "features.json", "w"))
    print(f"写入 {out_dir / 'features.json'}")


# ----------------------------------------------------------------------------- 评测
def load_patients(out_dir: Path) -> tuple[dict[str, dict], list[str]]:
    """患者 -> {y, feats: {名: 均值}, n_cycles};周期先跨视频拼起来再取均值,与段到患者同一口径。"""
    per_video = json.load(open(out_dir / "features.json"))
    pat: dict[str, dict] = {}
    names: set[str] = set()
    for v in per_video.values():
        pid = Path(v["json"]).stem.split("-")[0]
        d = pat.setdefault(pid, {"y": int(v["label"] == 0), "rows": [], "has3d": False})
        d["has3d"] |= v["n_3d_frames"] > 0
        for c in v["cycles"]:
            d["rows"].append(c["feats"])
            names.update(c["feats"])
    names_l = sorted(names)
    for pid, d in pat.items():
        d["n_cycles"] = len(d["rows"])
        if d["rows"]:
            M = np.array([[r.get(n, np.nan) for n in names_l] for r in d["rows"]], dtype=float)
            d["feats"] = dict(zip(names_l, np.nanmean(M, axis=0)))
        else:
            d["feats"] = {}
        del d["rows"]
    return pat, names_l


def macro(pred, y):
    return float(np.mean([((pred == c) & (y == c)).sum() / max((y == c).sum(), 1) for c in np.unique(y)]))


def repeated_lr(X: np.ndarray, y: np.ndarray, n: int, C: float, penalty: str = "l2") -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    ms, aucs = [], []
    for s in range(n):
        pred, prob = np.zeros(len(y), int), np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=1000 + s).split(X, y):
            sc = StandardScaler().fit(X[tr])
            kw = {"solver": "liblinear"} if penalty == "l1" else {}
            clf = LogisticRegression(C=C, penalty=penalty, class_weight="balanced", max_iter=2000, **kw)
            clf.fit(sc.transform(X[tr]), y[tr])
            pred[te] = clf.predict(sc.transform(X[te]))
            prob[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        ms.append(macro(pred, y))
        aucs.append(roc_auc_score(y, prob))
    return np.array(ms), np.array(aucs)


def cmd_eval(args) -> None:
    from attribute_repeated_splits import DOCTOR5, load as load_attr

    out_dir = Path(args.out_dir)
    pat, names = load_patients(out_dir)
    miss = set()
    if args.exclude_missing:
        bank = pickle.load(open(args.exclude_missing, "rb"))
        miss = {Path(v["json"]).stem.split("-")[0] for v in bank["videos"].values() if v["frac_ok"] == 0}

    # 基线:现有按秒 5 量(同一批患者、同一批划分)
    pids, X5, y5 = load_attr(Path(args.attr_root), DOCTOR5)
    keep = [p not in miss for p in pids]
    pids = [p for p, k in zip(pids, keep) if k]
    X5, y5 = X5[np.array(keep)], y5[np.array(keep)]
    y = y5
    n_nocyc = sum(1 for p in pids if pat.get(p, {}).get("n_cycles", 0) == 0)
    print(f"患者 {len(pids)} 人(去掉无 3D 结果 {len(miss)} 人);其中没有检出任何周期的 {n_nocyc} 人,"
          f"相位特征用其余患者的中位数填充;每人周期数中位数 "
          f"{np.median([pat[p]['n_cycles'] for p in pids if p in pat]):.0f}")

    def matrix(cols: list[str]) -> np.ndarray:
        M = np.array([[pat.get(p, {}).get("feats", {}).get(c, np.nan) for c in cols] for p in pids], dtype=float)
        med = np.nanmedian(M, axis=0)
        return np.where(np.isnan(M), med, M)

    ph = lambda qs: [f"ph{k}_{q}" for q in qs for k in range(N_PHASE)]
    CYC5 = ["cyc_trunk_lean", "cyc_shoulder_offset", "cyc_head_forward", "cyc_hip_flexion_max", "cyc_hip_range"]
    TIM = ["tim_lean_peak_phase", "tim_lean_mod", "tim_head_mod", "tim_shoff_mod", "tim_hip_peak_phase",
           "tim_lean_ss_minus_swing"]
    sets: list[tuple[str, np.ndarray]] = [
        ("S0 现有 5 量, 按秒(基线)", X5),
        ("S1 同 5 量, 按周期", matrix(CYC5)),
        ("S2 医生区域 4 量 × 4 相位 (16 token)", matrix(ph(DOCTOR_Q))),
        ("S3 S2 + 时间量 (22)", matrix(ph(DOCTOR_Q) + TIM)),
        ("S4 S1 + S3 (27)", matrix(CYC5 + ph(DOCTOR_Q) + TIM)),
        ("S5 时间量单独 (6)", matrix(TIM)),
        ("C1 未标部位 2 量 × 4 相位 (8)", matrix(ph(UNMARKED_Q))),
        ("C2 S2 + 未标部位相位 (24)", matrix(ph(DOCTOR_Q) + ph(UNMARKED_Q))),
    ]
    print(f"\n{'集合':40s} {'macro 均值±std [范围]':>28s} {'AUC 均值±std':>16s}   L1 (C=0.5) macro / AUC")
    for name, X in sets:
        ms, aucs = repeated_lr(X, y, args.n, args.C)
        ms1, aucs1 = repeated_lr(X, y, args.n, 0.5, "l1")
        print(f"{name:40s} {ms.mean():.3f} ± {ms.std():.3f} [{ms.min():.3f}, {ms.max():.3f}]   "
              f"{aucs.mean():.3f} ± {aucs.std():.3f}   {ms1.mean():.3f} / {aucs1.mean():.3f}")

    # 相位打乱对照:S2 的 16 个 token 换成帧随机分相位后的均值,5 次打乱各跑 n 份划分
    print("\n相位打乱对照(帧随机分到 4 个相位,各相位帧数不变):")
    for label, qs in (("S2 医生 4 量 × 4 相位", DOCTOR_Q), ("C1 未标 2 量 × 4 相位", UNMARKED_Q)):
        vals, aucv = [], []
        for s in range(N_SHUF):
            cols = [f"shuf{s}_ph{k}_{q}" for q in qs for k in range(N_PHASE)]
            ms, aucs = repeated_lr(matrix(cols), y, args.n, args.C)
            vals.append(ms.mean()); aucv.append(aucs.mean())
        print(f"  {label:34s} 打乱后 macro {np.mean(vals):.3f} ± {np.std(vals):.3f}   AUC {np.mean(aucv):.3f}")

    # 随机抽取对照:12 个逐帧量里随机 4 个 × 4 相位,100 次,医生 4 个所在的百分位
    rng = np.random.default_rng(0)
    base_ms, _ = repeated_lr(matrix(ph(DOCTOR_Q)), y, args.n, args.C)
    draws = []
    for _ in range(args.random_draws):
        qs = list(rng.choice(FRAME_Q, 4, replace=False))
        ms, _ = repeated_lr(matrix(ph(qs)), y, args.n_random, args.C)
        draws.append(ms.mean())
    draws = np.array(draws)
    pct = float((draws < base_ms.mean()).mean() * 100)
    print(f"\n随机 4 量 × 4 相位({args.random_draws} 次, 每次 {args.n_random} 份划分): "
          f"{draws.mean():.3f} ± {draws.std():.3f} [{draws.min():.3f}, {draws.max():.3f}];"
          f"医生 4 量的 {base_ms.mean():.3f} 在第 {pct:.0f} 百分位")

    # 逐 token 的单变量 AUC:哪个区域的哪个相位带信号
    from sklearn.metrics import roc_auc_score
    print("\n单 token AUC(行 = 量, 列 = 相位 " + " / ".join(PHASES) + ";按秒 / 按周期的整体值在最后两列):")
    sec_auc = {a: roc_auc_score(y, X5[:, i]) for i, a in enumerate(DOCTOR5)}
    sec_map = {"lean": "trunk_lean", "head": "head_forward", "shoff": "shoulder_offset", "hip": "hip_flexion_max"}
    cyc_map = {"lean": "cyc_trunk_lean", "head": "cyc_head_forward", "shoff": "cyc_shoulder_offset", "hip": "cyc_hip_flexion_max"}
    for q in DOCTOR_Q + UNMARKED_Q:
        row = []
        for k in range(N_PHASE):
            a = roc_auc_score(y, matrix([f"ph{k}_{q}"])[:, 0])
            row.append(max(a, 1 - a))
        tail = ""
        if q in sec_map:
            a_s, a_c = sec_auc[sec_map[q]], roc_auc_score(y, matrix([cyc_map[q]])[:, 0])
            tail = f"   秒 {max(a_s, 1 - a_s):.3f}  周期 {max(a_c, 1 - a_c):.3f}"
        print(f"  {q:6s} " + "  ".join(f"{a:.3f}" for a in row) + tail)
    for t in TIM:
        a = roc_auc_score(y, matrix([t])[:, 0])
        print(f"  {t:26s} {max(a, 1 - a):.3f}")


FEATURE_SETS = {
    "cyc5": ["cyc_trunk_lean", "cyc_shoulder_offset", "cyc_head_forward", "cyc_hip_flexion_max", "cyc_hip_range"],
    "ph16": [f"ph{k}_{q}" for q in DOCTOR_Q for k in range(N_PHASE)],
    "tim": ["tim_lean_peak_phase", "tim_lean_mod", "tim_head_mod", "tim_shoff_mod", "tim_hip_peak_phase",
            "tim_lean_ss_minus_swing"],
    "unmarked8": [f"ph{k}_{q}" for q in UNMARKED_Q for k in range(N_PHASE)],
}
FEATURE_SETS["ph16_tim"] = FEATURE_SETS["ph16"] + FEATURE_SETS["tim"]
FEATURE_SETS["all27"] = FEATURE_SETS["cyc5"] + FEATURE_SETS["ph16_tim"]


def cmd_export(args) -> None:
    """把一组相位特征写成 geom_attributes*/ 同款的逐属性 json(每个周期一个分数),
    供 attribute_classifier.py 按训练折产出患者概率(证据门控的几何分支)。没有周期的视频不写。"""
    per_video = json.load(open(Path(args.out_dir) / "features.json"))
    cols = FEATURE_SETS[args.set]
    dst = Path(args.attr_out)
    dst.mkdir(parents=True, exist_ok=True)
    n_vid = 0
    for c in cols:
        d = {}
        for name, v in per_video.items():
            if not v["cycles"]:
                continue
            d[name] = {"scores": [float(cy["feats"][c]) for cy in v["cycles"]], "label": v["label"], "json": v["json"]}
        json.dump(d, open(dst / f"{c}.json", "w"))
        n_vid = len(d)
    print(f"写入 {len(cols)} 个属性 × {n_vid} 条视频到 {dst}")


# ----------------------------------------------------------------------------- 验证图
def cmd_plot(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    per_video = json.load(open(out_dir / "features.json"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    durs = {0: [], 1: []}
    for v in per_video.values():
        for c in v["cycles"]:
            durs[int(v["label"] == 0)].append(c["dur"])
    ax = axes[0, 0]
    ax.hist([durs[1], durs[0]], bins=np.arange(0.6, 2.05, 0.05), label=[f"ASD (n={len(durs[1])})", f"non-ASD (n={len(durs[0])})"], stacked=True)
    ax.set_xlabel("cycle duration (s)"); ax.set_title("detected gait cycles"); ax.legend()
    ax = axes[0, 1]
    rel = {k: [] for k in ("cto", "chs", "to")}
    for v in per_video.values():
        for c in v["cycles"]:
            if c["event_based"]:
                span = c["hs1"] - c["hs0"]
                for k in rel:
                    rel[k].append(100 * (c[k] - c["hs0"]) / span)
    ax.hist([rel["cto"], rel["chs"], rel["to"]], bins=np.arange(0, 101, 2.5),
            label=["contralateral toe-off", "contralateral heel strike", "toe-off"])
    ax.set_xlabel("% of gait cycle"); ax.set_title("event timing within cycle (normal: ~10 / 50 / 60)"); ax.legend(fontsize=8)
    ax = axes[0, 2]
    per_vid = [len(v["cycles"]) for v in per_video.values() if v["n_3d_frames"] > 0]
    ax.hist(per_vid, bins=np.arange(-0.5, 8.5, 1)); ax.set_xlabel("cycles per clip"); ax.set_title("cycles per clip (clips with 3D)")
    # 三条示例
    names = [n for n, v in per_video.items() if len(v["cycles"]) >= 2][:3]
    for ax, n in zip(axes[1], names):
        v = per_video[n]
        fps = v["fps"]
        for leg, col in (("l", "C0"), ("r", "C1")):
            s = np.array([np.nan if x is None else x for x in v["ank"][leg]])
            t = np.arange(len(s)) / fps
            ax.plot(t, s, col, label=f"ankle {leg.upper()}")
            ev = v["events"][leg]
            ax.plot(np.array(ev["hs"]) / fps, s[ev["hs"]], col, marker="^", ls="none", ms=8)
            ax.plot(np.array(ev["to"]) / fps, s[ev["to"]], col, marker="v", ls="none", ms=8)
        for c in v["cycles"]:
            for b in c["bounds"]:
                ax.axvline(b / fps, color="k", lw=0.4, alpha=0.5)
        ax.set_title(f"{n[:28]}  label={'ASD' if v['label'] == 0 else 'non-ASD'}", fontsize=9)
        ax.set_xlabel("s"); ax.set_ylabel("ankle - hip, forward (m)"); ax.legend(fontsize=7)
    fig.suptitle("gait event detection on per-frame 3D keypoints (triangle up = heel strike, down = toe-off)")
    fig.tight_layout()
    fig.savefig(out_dir / "validation.png", dpi=130)
    print(f"写入 {out_dir / 'validation.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("extract")
    a.add_argument("--root-path", required=True)
    a.add_argument("--sam3d-root", required=True)
    a.add_argument("--class-num", type=int, default=2)
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--out-dir", default="logs/gait_phase_3d")
    a.set_defaults(fn=cmd_extract)
    e = sub.add_parser("eval")
    e.add_argument("--out-dir", default="logs/gait_phase_3d")
    e.add_argument("--attr-root", default="logs/geom_attributes_3d_doctor")
    e.add_argument("--exclude-missing", default=None)
    e.add_argument("--n", type=int, default=10)
    e.add_argument("--C", type=float, default=1.0)
    e.add_argument("--random-draws", type=int, default=100)
    e.add_argument("--n-random", type=int, default=3)
    e.set_defaults(fn=cmd_eval)
    p = sub.add_parser("plot")
    p.add_argument("--out-dir", default="logs/gait_phase_3d")
    p.set_defaults(fn=cmd_plot)
    x = sub.add_parser("export")
    x.add_argument("--out-dir", default="logs/gait_phase_3d")
    x.add_argument("--set", default="ph16_tim", choices=sorted(FEATURE_SETS))
    x.add_argument("--attr-out", required=True)
    x.set_defaults(fn=cmd_export)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

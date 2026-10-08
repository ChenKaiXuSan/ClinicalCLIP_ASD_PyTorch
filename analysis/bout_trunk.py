#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: bout_trunk.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
设计 B 的第一步验证:在侧视**长录像**(raw_data/<类>/whole_video_unprocessed/<日期>/full_lat*.mp4,
固定机位,患者在视野里来回走过,一次录像几十次经过)上,用 YOLOv8-pose 的 2D 关键点逐帧量
医生区域的三个矢状面量,按"经过"聚合,再看它们沿录像时间轴怎么变。

每帧(只取画面里最大的那个人,避免陪同人员):
  trunk_lean      肩中点相对髋中点的矢状倾角,+ 前倾(按该次经过的行进方向定正负)
  head_forward    耳中点相对肩中点的矢状倾角,+ 前伸
  shoulder_offset 肩中点相对髋中点沿行进方向的偏移 / 躯干长
经过 = 连续检出同一人的帧段(容忍 0.5 s 缺检),短于 1.5 s 的丢弃;行进方向由髋中点 x 的净位移决定。

输出 <out>/<类>__<日期>__<文件名>.json:逐经过的均值 / 标准差 / 帧数 / 置信度,以及逐帧序列;
`--plot` 画出三个量随时间的散点、逐经过均值与线性拟合(斜率,度/分钟)。

    python analysis/bout_trunk.py --video <full_lat.mp4> --out logs/bout_trunk --weights $W/yolov8m-pose.pt --stride 2 --plot
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# COCO-17
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
CONF = 0.3
MIN_PASS_SEC = 1.5
GAP_SEC = 0.5


def frame_quantities(kp: np.ndarray) -> dict | None:
    """kp (17,3) 像素坐标 + 置信度。返回未定向的量(trunk 向量等),方向在经过级决定。"""
    def mid(a, b):
        pa, pb = kp[a], kp[b]
        ok = [p for p in (pa, pb) if p[2] >= CONF]
        return np.mean([p[:2] for p in ok], axis=0) if ok else None
    sh, hip, ear = mid(L_SH, R_SH), mid(L_HIP, R_HIP), mid(L_EAR, R_EAR)
    if sh is None or hip is None:
        return None
    trunk = sh - hip            # 图像坐标 y 向下
    torso = float(np.linalg.norm(trunk)) + 1e-6
    out = {"hip_x": float(hip[0]), "hip_y": float(hip[1]), "torso": torso,
           "trunk_dx": float(trunk[0]), "trunk_dy": float(trunk[1])}
    if ear is not None:
        h = ear - sh
        out["head_dx"], out["head_dy"] = float(h[0]), float(h[1])
    out["conf"] = float(np.mean([kp[i, 2] for i in (L_SH, R_SH, L_HIP, R_HIP)]))
    return out


def _sag(dx: float, dy_down: float) -> float:
    """矢状倾角(度):向量相对竖直向上的夹角,+ 为沿行进方向。dy_down 为图像坐标(向下为正)。"""
    return math.degrees(math.atan2(dx, -dy_down))


def run_pose(video: Path, weights: Path, stride: int, imgsz: int, device: str):
    from ultralytics import YOLO
    import av

    model = YOLO(str(weights))
    c = av.open(str(video))
    st = c.streams.video[0]
    fps = float(st.average_rate)
    rows = []
    for i, fr in enumerate(c.decode(video=0)):
        if i % stride:
            continue
        img = fr.to_ndarray(format="bgr24")
        res = model.predict(img, imgsz=imgsz, conf=0.25, verbose=False, device=device)[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            continue
        boxes = res.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        j = int(np.argmax(areas))
        kp = res.keypoints.data[j].cpu().numpy()      # (17,3)
        q = frame_quantities(kp)
        if q is None:
            continue
        q.update({"frame": i, "t": i / fps, "box_h": float(boxes[j, 3] - boxes[j, 1]), "n_person": int(len(boxes))})
        rows.append(q)
    c.close()
    return rows, fps, int(st.frames or 0)


def split_passes(rows: list[dict], fps: float, stride: int) -> list[dict]:
    passes, cur = [], []
    gap = GAP_SEC
    for r in rows:
        if cur and r["t"] - cur[-1]["t"] > gap:
            passes.append(cur); cur = []
        cur.append(r)
    if cur:
        passes.append(cur)
    out = []
    for p in passes:
        dur = p[-1]["t"] - p[0]["t"]
        if dur < MIN_PASS_SEC or len(p) < 5:
            continue
        # 逐帧行进方向:髋中点 x 的平滑速度(±0.3 s 窗口)。患者会在视野内掉头,整段一个符号会把
        # 一半帧算反(头前伸 +35° 变 -30° 的镜像翻转就是这么来的);速度太小的帧(掉头、站立)丢弃。
        t = np.array([r["t"] for r in p]); x = np.array([r["hip_x"] for r in p]); bh = np.array([r["box_h"] for r in p])
        win = max(1, int(round(0.3 / max(np.median(np.diff(t)), 1e-3))))
        v = np.full(len(p), np.nan)
        for i in range(len(p)):
            a, b = max(0, i - win), min(len(p) - 1, i + win)
            if t[b] > t[a]:
                v[i] = (x[b] - x[a]) / (t[b] - t[a])
        vmin = 0.15 * np.median(bh)            # 每秒至少走 0.15 个身高(像素)才算在走
        keep = np.abs(v) > vmin
        fwd = np.sign(v)
        lean = np.array([_sag(r["trunk_dx"] * f, r["trunk_dy"]) for r, f in zip(p, fwd)])
        head = np.array([_sag(r["head_dx"] * f, r["head_dy"]) if "head_dx" in r else np.nan for r, f in zip(p, fwd)])
        shoff = np.array([r["trunk_dx"] * f / r["torso"] for r, f in zip(p, fwd)])
        # 合理范围过滤:躯干倾角超过 ±60° 的帧是边缘半露 / 检错人
        keep &= np.abs(lean) < 60
        if keep.sum() < 5:
            continue
        out.append({
            "t_start": p[0]["t"], "t_end": p[-1]["t"], "dur": dur, "n": int(keep.sum()), "n_raw": len(p),
            "fwd": float(np.sign(np.nanmean(fwd[keep]))),
            "trunk_lean": float(np.median(lean[keep])), "trunk_lean_sd": float(np.std(lean[keep])),
            "head_forward": float(np.nanmedian(head[keep])) if np.isfinite(head[keep]).any() else None,
            "shoulder_offset": float(np.median(shoff[keep])),
            "conf": float(np.mean([r["conf"] for r, k in zip(p, keep) if k])),
            "box_h": float(np.mean(bh[keep])),
            "n_person_max": int(max(r["n_person"] for r in p)),
            "frames": [{"t": r["t"], "lean": float(l), "conf": r["conf"]} for r, l, k in zip(p, lean, keep) if k],
        })
    return out


def summarize(passes: list[dict]) -> dict:
    if len(passes) < 3:
        return {"n_pass": len(passes)}
    t0 = passes[0]["t_start"]
    t = np.array([(p["t_start"] - t0) / 60.0 for p in passes])      # 分钟,从第一次经过起算
    y = np.array([p["trunk_lean"] for p in passes])
    A = np.c_[t, np.ones_like(t)]
    slope, icpt = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - (slope * t + icpt)
    k = max(1, len(passes) // 4)
    return {
        "n_pass": len(passes), "span_min": float(t[-1]),
        "trunk_lean_mean": float(y.mean()), "between_pass_sd": float(y.std()),
        "within_pass_sd_median": float(np.median([p["trunk_lean_sd"] for p in passes])),
        "slope_deg_per_min": float(slope), "resid_sd": float(resid.std()),
        "first_quarter_mean": float(y[:k].mean()), "last_quarter_mean": float(y[-k:].mean()),
        "last_minus_first": float(y[-k:].mean() - y[:k].mean()),
        "conf_mean": float(np.mean([p["conf"] for p in passes])), "box_h_mean": float(np.mean([p["box_h"] for p in passes])),
    }


def plot(passes: list[dict], summ: dict, title: str, png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    t0 = passes[0]["t_start"]
    for ax, key, lab in zip(axes, ("trunk_lean", "head_forward", "shoulder_offset"), ("trunk lean (deg)", "head forward (deg)", "shoulder offset (/torso)")):
        tt = [(p["t_start"] - t0) / 60 for p in passes if p[key] is not None]
        yy = [p[key] for p in passes if p[key] is not None]
        ax.plot(tt, yy, "o-", ms=4)
        if key == "trunk_lean":
            for p in passes:
                ax.plot([(f["t"] - t0) / 60 for f in p["frames"]], [f["lean"] for f in p["frames"]], ".", color="gray", ms=2, alpha=0.4)
            if "slope_deg_per_min" in summ:
                tl = np.array([tt[0], tt[-1]]); ax.plot(tl, summ["slope_deg_per_min"] * tl + (summ["trunk_lean_mean"] - summ["slope_deg_per_min"] * np.mean(tt)), "r--",
                                                          label=f"slope {summ['slope_deg_per_min']:+.2f} deg/min, last-first {summ['last_minus_first']:+.1f}")
                ax.legend()
        ax.set_ylabel(lab)
    axes[-1].set_xlabel("minutes since first pass")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png, dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="logs/bout_trunk")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--stride", type=int, default=2, help="每隔几帧取一帧")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    video = Path(args.video)
    tag = args.tag or "__".join([video.parts[-4], video.parts[-2], video.stem])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows, fps, n_frames = run_pose(video, Path(args.weights), args.stride, args.imgsz, args.device)
    passes = split_passes(rows, fps, args.stride)
    summ = summarize(passes)
    summ.update({"video": str(video), "fps": fps, "n_frames": n_frames, "stride": args.stride, "detected_frames": len(rows)})
    json.dump({"summary": summ, "passes": passes}, open(out / f"{tag}.json", "w"))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items() if k != "video"}, ensure_ascii=False))
    if args.plot and len(passes) >= 3:
        plot(passes, summ, tag, out / f"{tag}.png")
        print("图:", out / f"{tag}.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: detect.py
Project: analysis/koa
Author: Kaixu Chen
-----
Comment:
KOA-PD-NM(Mendeley 44pfnysy89:膝骨关节炎 50 / 帕金森 16 / 正常 30,侧面 RGB,每人左右各走一次)的第一步:
YOLOv8-seg 逐帧检测 + 实例分割,把行走者(横向位移最大的那条轨迹)挑出来,输出

  det/<video>.npz    frame_idx(原始帧号), boxes (N,4), fps(原始), eff_fps(= 25), size(H,W), n_det(每帧人数)
  crops/<video>.npz  以人为中心的方形裁剪 crops (N,224,224,3) uint8 RGB 与同一裁剪下的人物掩码 mask (N,224,224) bool

两个与类别绑定的采集差异必须在这里抹平,否则视频分支可以不看步态就分类:
  帧率:KOA 98/100 段是 50 fps,正常组 51/59 段是 30 fps。统一按时间重采样到 25 fps(第 k 帧取原始帧 round(k·fps/25))。
  背景:三组的拍摄场地不同。掩码一并存下,训练时可以抠掉背景或只用轮廓。
每段最多取行走者轨迹的前 30 秒(三段重度帕金森的录像长 150–320 秒)。
裁剪框边长 = 1.3 × 该视频人框高度的中位数,中心逐帧跟随人框(平滑)。

    python analysis/koa/detect.py --video-root $KOA/video --out $KOA/proc [--chunk 0/4]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

EXT = {".mov", ".mp4", ".avi", ".m4v"}
TARGET_FPS = 25.0
MAX_FRAMES = 750


def list_videos(root: str, chunk: str = "0/1", limit: int = 0) -> list[Path]:
    vids = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in EXT and p.stat().st_size > 0)
    k, n = (int(x) for x in chunk.split("/"))
    vids = [v for i, v in enumerate(vids) if i % n == k]
    return vids[:limit] if limit else vids


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def link_tracks(dets: list[list[tuple]], max_gap: int = 8) -> list[dict]:
    """贪心 IoU 连接。dets[t] = [(box, conf, 实例序号), ...];返回轨迹 [{t: (box, 实例序号)}]。"""
    tracks: list[dict] = []
    last: list[int] = []
    for t, ds in enumerate(dets):
        used = set()
        for i in sorted(range(len(ds)), key=lambda i: -ds[i][1]):
            best, bj = 0.2, -1
            for j, tr in enumerate(tracks):
                if j in used or t - last[j] > max_gap:
                    continue
                s = iou(ds[i][0], tr[last[j]][0])
                if s > best:
                    best, bj = s, j
            if bj < 0:
                tracks.append({}); last.append(t); bj = len(tracks) - 1
            tracks[bj][t] = (ds[i][0], ds[i][2]); last[bj] = t; used.add(bj)
    return tracks


def pick_walker(tracks: list[dict], H: int) -> dict | None:
    """行走者 = 横向位移 × 持续帧数最大的轨迹(站着不动的陪同人员位移小)。"""
    best, bs = None, -1.0
    for tr in tracks:
        if len(tr) < 10:
            continue
        b = np.array([v[0] for v in tr.values()])
        cx = (b[:, 0] + b[:, 2]) / 2
        h = np.median(b[:, 3] - b[:, 1])
        s = (np.percentile(cx, 95) - np.percentile(cx, 5) + 0.05 * h) * np.sqrt(len(tr)) * (h / H)
        if s > bs:
            best, bs = tr, s
    return best


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    return np.convolve(np.pad(x, (pad, pad), mode="edge"), np.ones(k) / k, mode="valid")


def crop_square(img: np.ndarray, cx: float, cy: float, side: float, interp) -> np.ndarray:
    import cv2
    H, W = img.shape[:2]
    s = int(round(side)); x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    pad = np.zeros((s, s) + img.shape[2:], dtype=img.dtype)
    xs, ys, xe, ye = max(x0, 0), max(y0, 0), min(x0 + s, W), min(y0 + s, H)
    if xe > xs and ye > ys:
        pad[ys - y0: ye - y0, xs - x0: xe - x0] = img[ys:ye, xs:xe]
    return cv2.resize(pad, (224, 224), interpolation=interp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="/work/1/SKIING/chenkaixu/pydeps/yolo/yolov8m-seg.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import av
    import cv2
    from ultralytics import YOLO

    vids = list_videos(args.video_root, args.chunk, args.limit)
    out = Path(args.out); (out / "det").mkdir(parents=True, exist_ok=True); (out / "crops").mkdir(exist_ok=True)
    model = YOLO(args.weights)
    print(f"视频 {len(vids)} 个(分片 {args.chunk})", flush=True)
    for v in vids:
        name = v.stem
        if (out / "crops" / f"{name}.npz").exists():
            continue
        with av.open(str(v)) as c:
            st = c.streams.video[0]
            fps = float(st.average_rate)
            n_src = st.frames or int(float(st.duration * st.time_base) * fps)
            pick = np.unique(np.round(np.arange(0, n_src / fps * TARGET_FPS) * fps / TARGET_FPS).astype(int))   # 按时间重采样
            want = {int(f): j for j, f in enumerate(pick)}
            dets, masks, frames, H, W = [], [], [], 0, 0
            for i, fr in enumerate(c.decode(st)):
                if i not in want:
                    continue
                rgb = fr.to_ndarray(format="rgb24")
                H, W = rgb.shape[:2]
                r = model.predict(rgb[..., ::-1], imgsz=args.imgsz, conf=0.25, classes=[0], verbose=False, device=args.device)[0]
                ds, ms = [], []
                if r.boxes is not None and len(r.boxes) and r.masks is not None:
                    bx, cf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
                    ms = [cv2.resize(m, (W // 2, H // 2), interpolation=cv2.INTER_LINEAR) > 0.5 for m in r.masks.data.cpu().numpy()]
                    ds = [(bx[j], float(cf[j]), j) for j in range(len(bx))]
                dets.append(ds); masks.append(ms); frames.append(cv2.resize(rgb, (W // 2, H // 2), interpolation=cv2.INTER_AREA))
                if len(dets) >= 4 * MAX_FRAMES:                      # 极长录像的保护
                    break
        tr = pick_walker(link_tracks(dets), H)
        if tr is None:
            print(f"[skip] {name}: 没有找到行走者", flush=True)
            continue
        ts = np.array(sorted(tr))[:MAX_FRAMES]
        boxes = np.stack([tr[t][0] for t in ts])
        frame_idx = pick[ts]
        np.savez_compressed(out / "det" / f"{name}.npz", frame_idx=frame_idx, boxes=boxes, fps=fps, eff_fps=TARGET_FPS,
                            size=np.array([H, W]), n_det=np.array([len(d) for d in dets]), t_index=ts)
        side = 1.3 * float(np.median(boxes[:, 3] - boxes[:, 1]))
        kk = max(3, int(round(0.3 * TARGET_FPS)) | 1)
        cx, cy = smooth((boxes[:, 0] + boxes[:, 2]) / 2, kk), smooth((boxes[:, 1] + boxes[:, 3]) / 2, kk)
        crops = np.zeros((len(ts), 224, 224, 3), dtype=np.uint8); mk = np.zeros((len(ts), 224, 224), dtype=bool)
        for j, t in enumerate(ts):                                   # 半分辨率帧上裁,坐标相应减半
            crops[j] = crop_square(frames[t], cx[j] / 2, cy[j] / 2, side / 2, cv2.INTER_AREA)
            mk[j] = crop_square(masks[t][tr[t][1]].astype(np.uint8), cx[j] / 2, cy[j] / 2, side / 2, cv2.INTER_NEAREST) > 0
        np.savez_compressed(out / "crops" / f"{name}.npz", crops=crops, mask=mk, frame_idx=frame_idx, side=side)
        cxr = (np.percentile(cx, 95) - np.percentile(cx, 5)) / W
        print(f"{name}: fps {fps:.0f} -> 25, 采样 {len(dets)} 帧, 行走者 {len(ts)} 帧, 人高 {np.median(boxes[:, 3] - boxes[:, 1]):.0f}px, "
              f"横向位移 {cxr:.2f} 画面宽, 每帧人数中位 {np.median([len(d) for d in dets]):.0f}, 掩码占比 {mk.mean():.2f}", flush=True)
    print("[done]")


if __name__ == "__main__":
    main()

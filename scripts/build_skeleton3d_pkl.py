#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: build_skeleton3d_pkl.py
Project: scripts
Author: Kaixu Chen
-----
Comment:
把 SAM-3D-Body 的逐帧 npz 转成与 seg_skeleton_pkl/whole_annotations.pkl 相同结构的 3D 骨架 pkl,
给 ST-GCN 姿态基线(model.backbone=pose)吃 3D 关节:
  annotations[i] = {frame_dir, img_shape, original_shape, total_frames, label,
                    keypoint (1, F, 17, 3) 相机坐标系米(x 行进 / y 向下 / z 左右, 根相对),
                    keypoint_score (1, F, 17) 有 3D 结果的帧为 1, 否则 0}
关节顺序按 COCO-17(与 2D pkl 一致,ST-GCN 的邻接矩阵不用改):
  mhr70 -> COCO: nose 0, eyes 1/2, ears 3/4, shoulders 5/6, elbows 7/8, wrists 41/62, hips 9/10, knees 11/12, ankles 13/14。
缺帧取最近的可用帧(相差 >5 帧记 0 分);整条视频没有 3D 结果的从 pkl 里略去(pose_for 会给全零)。
frame_dir / total_frames / label 直接沿用 2D pkl 里的同名条目,保证 video_name 匹配与帧对齐规则完全相同。

    python scripts/build_skeleton3d_pkl.py --pkl2d $DATA/clinical_CLIP_dataset/seg_skeleton_pkl/whole_annotations.pkl \
        --sam3d-root $SAM3D --out $DATA/clinical_CLIP_dataset/seg_skeleton_pkl_3d/whole_annotations.pkl
"""
from __future__ import annotations

import argparse
import glob
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np

COCO_FROM_MHR70 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 41, 62, 9, 10, 11, 12, 13, 14]
MAX_GAP = 5


def convert(item):
    frame_dir, total, sam_dir = item
    files = sorted(glob.glob(f"{sam_dir}/*_sam3d_body.npz"))
    if not files:
        return frame_dir, None, None
    frames = {}
    for f in files:
        o = np.load(f, allow_pickle=True)["output"].item()
        fi = int(o.get("frame_idx", int(Path(f).name.split("_")[0])))
        frames[fi] = np.asarray(o["pred_keypoints_3d"], dtype=np.float32)[COCO_FROM_MHR70]
    avail = np.array(sorted(frames))
    kp = np.zeros((total, 17, 3), np.float32)
    sc = np.zeros((total, 17), np.float32)
    for t in range(total):
        j = int(np.abs(avail - t).argmin())
        if abs(int(avail[j]) - t) <= MAX_GAP:
            kp[t] = frames[int(avail[j])]
            sc[t] = 1.0
    return frame_dir, kp[None], sc[None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl2d", required=True)
    ap.add_argument("--sam3d-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    d2 = pickle.load(open(args.pkl2d, "rb"))
    sam_dirs = {Path(d).parent.name: d for d in glob.glob(f"{args.sam3d_root}/*/*/video")}
    items = []
    for a in d2["annotations"]:
        name = Path(a["frame_dir"]).stem
        items.append((a["frame_dir"], int(a["total_frames"]), sam_dirs.get(name)))
    by_dir = {a["frame_dir"]: a for a in d2["annotations"]}
    out_ann, n_missing, n_frames_zero = [], 0, 0
    with Pool(args.workers) as pool:
        for i, (frame_dir, kp, sc) in enumerate(pool.imap_unordered(convert, items, chunksize=8)):
            if kp is None:
                n_missing += 1
                continue
            a = by_dir[frame_dir]
            n_frames_zero += int((sc[0].sum(1) == 0).sum())
            out_ann.append({**{k: v for k, v in a.items() if k not in ("keypoint", "keypoint_score")},
                            "keypoint": kp, "keypoint_score": sc})
            if (i + 1) % 300 == 0:
                print(f"  {i + 1}/{len(items)}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"annotations": out_ann, "split": d2.get("split")}, open(out, "wb"))
    tot_frames = sum(int(a["total_frames"]) for a in out_ann)
    print(f"2D 条目 {len(d2['annotations'])}, 转出 {len(out_ann)}, 无 3D 结果略去 {n_missing}, "
          f"缺帧(记 0 分) {n_frames_zero}/{tot_frames} ({n_frames_zero / max(tot_frames, 1):.1%}) -> {out}")


if __name__ == "__main__":
    main()

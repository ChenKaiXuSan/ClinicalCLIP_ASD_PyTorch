#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: geom_attributes_3d.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
geom_attributes.py 的 3D 版:同一组临床几何量改从 SAM-3D-Body 的逐帧 3D 关节
(pred_keypoints_3d, 70 点, 相机坐标系: x 沿行进方向, y 向下, z 深度=左右; 根关节相对)算出,
再加只有 3D 才量得到的量。段划分、帧采样计划与 2D 版 / 训练完全一致,输出格式相同,所以
attribute_classifier.py / late_fusion.py / data.aux_targets_dir 直接可用,能与 2D 数字逐项对比。

与 2D 相同的 9 个量(定义见 geom_attributes.py; 角度在矢状面 x-y 内量):
  trunk_lean head_forward shoulder_offset hip_flexion_max hip_range knee_flexion_max
  step_length gait_speed arm_swing
  差别: head_forward 用双耳中点(3D 没有近侧/远侧之分); arm_swing 取双腕均值; 长度单位是米,
  归一化后无量纲; gait_speed 用根关节平移 pred_cam_t 的 x 位移(单目深度估计, 抖动大, 仅供参考)。
3D 独有的量:
  lower_trunk_lean   下躯干(髋中点 -> MHR 脊柱关节 35, 约躯干高 0.3)矢状倾角, +前倾            [腰椎骨盆]
  spine_curve        上段(35 -> 37, 约躯干高 0.9)倾角 - 下段倾角, +为上段更前倾(胸腰段前屈)   [腰椎骨盆]
  pelvic_obliquity   两髋连线相对水平面的倾角绝对值(冠状面), 段内均值                          [腰椎骨盆]
  trunk_lateral      躯干向量在冠状面(z-y)的侧倾绝对值, 段内均值                               [腰椎骨盆/肩]
  lateral_sway       肩中点相对髋中点的左右(z)偏移的段内极差 / 躯干长                          [肩]
骨盆朝向没有用 pred_global_rots: 根/骨盆旋转的符号在视频中途会翻转, 关节几何更稳。

缺帧: 取最近的可用帧(相差 > 5 帧视为缺失); 整条视频无 3D 结果 (44 条) 的段记缺失, 用中位数填充。

    python analysis/geom_attributes_3d.py --root-path $DATA --sam3d-root $SAM3D --out-dir logs/geom_attributes_3d
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))

from dataloader.whole_video_dataset import _plan_frame_index, _probe  # noqa: E402

# mhr70 关键点(前 17 个同 COCO 顺序; 41/62 为左右手腕)
NOSE, L_EAR, R_EAR = 0, 3, 4
L_SH, R_SH, L_EL, R_EL = 5, 6, 7, 8
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 9, 10, 11, 12, 13, 14
L_WR, R_WR = 41, 62
# MHR 127 关节里的脊柱链: 1 骨盆 -> 34 -> 35 -> 36 -> 37 -> 110/112 颈 -> 111/113 头
SPINE_LOW, SPINE_HIGH = 35, 37
MAX_GAP = 5

ATTRS_2D = ["trunk_lean", "head_forward", "shoulder_offset", "hip_flexion_max", "hip_range",
            "knee_flexion_max", "step_length", "gait_speed", "arm_swing"]
ATTRS_3D = ["lower_trunk_lean", "spine_curve", "pelvic_obliquity", "trunk_lateral", "lateral_sway"]
ATTRS = ATTRS_2D + ATTRS_3D


def _sag(dx: float, dy: float) -> float:
    """矢状面向量相对铅垂向上方向的夹角(度), y 向下, dx>0(行进方向)为正。"""
    return math.degrees(math.atan2(dx, -dy))


def _joint_angle(a, b, c) -> float:
    v1, v2 = a - b, c - b
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def load_frames(video_dir: str) -> dict[int, dict]:
    """{帧号: {'kp': (70,3), 'j': (127,3), 'cam_t': (3,)}}, 只保留需要的字段。"""
    out = {}
    for f in sorted(glob.glob(f"{video_dir}/*_sam3d_body.npz")):
        o = np.load(f, allow_pickle=True)["output"].item()
        fi = int(o.get("frame_idx", int(Path(f).name.split("_")[0])))
        out[fi] = {"kp": np.asarray(o["pred_keypoints_3d"], dtype=np.float64),
                   "j": np.asarray(o["pred_joint_coords"], dtype=np.float64),
                   "cam_t": np.asarray(o["pred_cam_t"], dtype=np.float64)}
    return out


def video_attributes(frames: dict[int, dict], plan: torch.Tensor, fps: float) -> np.ndarray:
    out = np.full((plan.shape[0], len(ATTRS)), np.nan, dtype=np.float32)
    if not frames:
        return out
    avail = np.array(sorted(frames))
    # 行进方向: 根平移 x 总位移; 太小时用鼻子相对肩中点
    first, last = frames[avail[0]], frames[avail[-1]]
    dx_total = last["cam_t"][0] - first["cam_t"][0]
    if abs(dx_total) > 0.1:
        fwd = 1.0 if dx_total > 0 else -1.0
    else:
        d = np.mean([fr["kp"][NOSE, 0] - fr["kp"][[L_SH, R_SH], 0].mean() for fr in frames.values()])
        fwd = 1.0 if d > 0 else -1.0

    for ci, idx in enumerate(plan.tolist()):
        rows = {a: [] for a in ["lean", "head", "shoff", "hip", "knee", "step", "arm_l", "arm_r", "hipx", "torso", "leg",
                                "low", "curve", "obl", "lat", "sway", "t"]}
        for f in idx:
            j = int(np.abs(avail - f).argmin())
            if abs(int(avail[j]) - f) > MAX_GAP:
                continue
            fr = frames[int(avail[j])]
            k, jc = fr["kp"], fr["j"]
            sh = k[[L_SH, R_SH]].mean(0)
            hip = k[[L_HIP, R_HIP]].mean(0)
            trunk = sh - hip
            torso = float(np.linalg.norm(trunk)) + 1e-6
            rows["torso"].append(torso)
            rows["t"].append(int(avail[j]) / fps)
            rows["hipx"].append((fr["cam_t"][0] + hip[0]) * fwd)
            rows["lean"].append(_sag(trunk[0] * fwd, trunk[1]))
            rows["shoff"].append(trunk[0] * fwd / torso)
            ear = k[[L_EAR, R_EAR]].mean(0)
            rows["head"].append(_sag((ear[0] - sh[0]) * fwd, ear[1] - sh[1]))
            for hp, kn, an in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)):
                rows["hip"].append(_joint_angle(sh, k[hp], k[kn]))
                rows["knee"].append(_joint_angle(k[hp], k[kn], k[an]))
                rows["leg"].append(float(np.linalg.norm(k[hp] - k[kn]) + np.linalg.norm(k[kn] - k[an])))
            rows["step"].append(abs(k[L_ANK, 0] - k[R_ANK, 0]))
            rows["arm_l"].append((k[L_WR, 0] - hip[0]) * fwd / torso)
            rows["arm_r"].append((k[R_WR, 0] - hip[0]) * fwd / torso)
            # 3D 独有
            low = jc[SPINE_LOW] - hip
            up = jc[SPINE_HIGH] - jc[SPINE_LOW]
            a_low = _sag(low[0] * fwd, low[1])
            rows["low"].append(a_low)
            rows["curve"].append(_sag(up[0] * fwd, up[1]) - a_low)
            hl, hr = k[L_HIP], k[R_HIP]
            rows["obl"].append(abs(math.degrees(math.atan2(hl[1] - hr[1], abs(hl[2] - hr[2]) + 1e-6))))
            rows["lat"].append(abs(math.degrees(math.atan2(trunk[2], -trunk[1]))))
            rows["sway"].append(trunk[2] / torso)
        if len(rows["lean"]) < 4:
            continue
        torso = float(np.mean(rows["torso"]))
        leg = float(np.mean(rows["leg"]))
        dur = rows["t"][-1] - rows["t"][0]
        arm = [float(np.max(r) - np.min(r)) for r in (rows["arm_l"], rows["arm_r"])]
        vals = {
            "trunk_lean": float(np.mean(rows["lean"])),
            "head_forward": float(np.mean(rows["head"])),
            "shoulder_offset": float(np.mean(rows["shoff"])),
            "hip_flexion_max": 180.0 - float(np.min(rows["hip"])),
            "hip_range": float(np.max(rows["hip"]) - np.min(rows["hip"])),
            "knee_flexion_max": 180.0 - float(np.min(rows["knee"])),
            "step_length": float(np.max(rows["step"])) / leg,
            "gait_speed": (rows["hipx"][-1] - rows["hipx"][0]) / dur / torso if dur > 0 else np.nan,
            "arm_swing": float(np.mean(arm)),
            "lower_trunk_lean": float(np.mean(rows["low"])),
            "spine_curve": float(np.mean(rows["curve"])),
            "pelvic_obliquity": float(np.mean(rows["obl"])),
            "trunk_lateral": float(np.mean(rows["lat"])),
            "lateral_sway": float(np.max(rows["sway"]) - np.min(rows["sway"])),
        }
        out[ci] = [vals[a] for a in ATTRS]
    return out


def _work(item):
    name, video_path, sam_dir, num_samples = item
    total, fps = _probe(video_path)
    plan = _plan_frame_index(total, max(int(fps), 1), num_samples)
    frames = load_frames(sam_dir) if sam_dir else {}
    n_frames = len(frames)
    max_idx = max(frames) + 1 if frames else 0
    return name, video_attributes(frames, plan, fps), total, n_frames, max_idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-path", required=True)
    ap.add_argument("--sam3d-root", required=True, help="sam3d_body_results 目录, 下面是 <类>/<视频名>/video/*.npz")
    ap.add_argument("--class-num", type=int, default=2)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
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

    per_video: dict[str, dict] = {}
    n_seg = n_missing = n_novid = n_short = 0
    with Pool(args.workers) as pool:
        for i, (name, vals, total, n_frames, max_idx) in enumerate(pool.imap_unordered(_work, items, chunksize=4)):
            n_seg += vals.shape[0]
            n_missing += int(np.isnan(vals).any(1).sum())
            if n_frames == 0:
                n_novid += 1
            elif max_idx < total - MAX_GAP:
                n_short += 1
            per_video[name] = {"vals": vals, **meta_by[name]}
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(items)}", flush=True)

    all_vals = np.concatenate([v["vals"] for v in per_video.values()])
    med_vals = np.nanmedian(all_vals, axis=0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"视频 {len(per_video)} 条 (无 3D 结果 {n_novid}, 3D 帧数明显少于视频 {n_short}), "
          f"段 {n_seg}, 含缺失的段 {n_missing} ({n_missing / max(n_seg, 1):.1%})")
    print(f"{'属性':18s} {'中位数':>9s} {'均值':>9s} {'标准差':>9s} {'缺失率':>7s}")
    for j, a in enumerate(ATTRS):
        col = all_vals[:, j]
        print(f"{a:18s} {np.nanmedian(col):9.2f} {np.nanmean(col):9.2f} {np.nanstd(col):9.2f} {np.isnan(col).mean():7.1%}")
        d = {}
        for name, v in per_video.items():
            scores = np.where(np.isnan(v["vals"][:, j]), med_vals[j], v["vals"][:, j])
            d[name] = {"scores": [float(x) for x in scores], "label": v["label"], "json": v["json"]}
        json.dump(d, open(out_dir / f"{a}.json", "w"))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()

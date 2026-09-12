#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: geom_attributes.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
从 2D 骨架(seg_skeleton_pkl, COCO-17, 侧视)计算**临床几何属性**,替代 VLM 猜出来的属性。
哪些量由医生标注的区域(群体级:腰椎骨盆 / 头部 / 肩部 + 文献代偿量)决定,值从骨架量出来;
所有患者算同一组量,推理不依赖任何医生标注。

每个 1 秒 gait 段(与训练同一帧采样计划,8 帧)一组值:
  trunk_lean        躯干前倾角(肩中点-髋中点 vs 铅垂线, +为向行进方向前倾), 段内均值   [腰椎骨盆]
  head_forward      头前伸角(近侧耳-肩中点 vs 铅垂线, +为前伸), 段内均值                  [头部]
  shoulder_offset   肩中点相对髋中点的前后偏移 / 躯干长, 段内均值                          [肩部]
  hip_flexion_max   髋屈曲最大值 = 180 - min 角(肩中点-髋-膝)                              [腰椎骨盆/骨盆后倾代偿]
  hip_range         髋角段内极差(髋伸展幅度不足)                                            [同上]
  knee_flexion_max  膝屈曲最大值 = 180 - min 角(髋-膝-踝)                                  [文献代偿, 医生未标]
  step_length       两踝最大水平距离 / 腿长                                                [脚]
  gait_speed        髋中点沿行进方向位移 / 秒 / 躯干长                                      [文献]
  arm_swing         近侧腕相对髋中点的水平位移极差 / 躯干长                                [手腕]
角度单位:度。行进方向由髋中点在整条视频上的位移决定,不足时用面部朝向。
置信度 < 0.3 的关节点所在帧不用;有效帧 < 4 的段记缺失,最后用该属性全库中位数填充并报缺失率。

输出与 eval_qwen_attributes.py 相同的格式(每属性一个 json),所以 attribute_classifier.py /
late_fusion.py / 辅助回归(data.aux_targets_dir)全部直接可用。

    python analysis/geom_attributes.py --root-path $DATA --out-dir logs/geom_attributes
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))

from dataloader.med_attn_map import MedAttnMap  # noqa: E402
from dataloader.whole_video_dataset import _plan_frame_index, _probe  # noqa: E402

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16
CONF = 0.3
ATTRS = ["trunk_lean", "head_forward", "shoulder_offset", "hip_flexion_max", "hip_range",
         "knee_flexion_max", "step_length", "gait_speed", "arm_swing"]


def _angle_vertical(dx: float, dy: float) -> float:
    """向量 (dx, dy)(图像坐标, y 向下)相对铅垂线向上方向的夹角, 度, dx>0 为正。"""
    return math.degrees(math.atan2(dx, -dy))


def _joint_angle(a, b, c) -> float:
    """b 处的夹角(度)。"""
    v1, v2 = a - b, c - b
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def video_attributes(kp: np.ndarray, sc: np.ndarray, plan: torch.Tensor, fps: float) -> np.ndarray:
    """kp (F,17,2) 归一化坐标, sc (F,17); plan (n_chunks, 8) 帧下标 -> (n_chunks, len(ATTRS)), 缺失为 nan。"""
    F_ = kp.shape[0]
    hip_mid_all = kp[:, [L_HIP, R_HIP]].mean(1)
    # 行进方向:整条视频髋中点 x 位移;太小时用面部朝向(鼻子相对肩中点)
    dx_total = hip_mid_all[-1, 0] - hip_mid_all[0, 0]
    if abs(dx_total) > 0.02:
        fwd = 1.0 if dx_total > 0 else -1.0
    else:
        sh_mid_all = kp[:, [L_SH, R_SH]].mean(1)
        fwd = 1.0 if (kp[:, NOSE, 0] - sh_mid_all[:, 0]).mean() > 0 else -1.0
    # 近侧耳 / 腕:整条视频置信度更高的一侧
    ear = L_EAR if sc[:, L_EAR].mean() >= sc[:, R_EAR].mean() else R_EAR
    wrist = L_WR if sc[:, L_WR].mean() >= sc[:, R_WR].mean() else R_WR

    out = np.full((plan.shape[0], len(ATTRS)), np.nan, dtype=np.float32)
    for ci, idx in enumerate(plan.tolist()):
        idx = [min(i, F_ - 1) for i in idx]
        rows = {a: [] for a in ["lean", "head", "shoff", "hip", "knee", "step", "arm", "hipx", "torso", "leg"]}
        for f in idx:
            k, s = kp[f], sc[f]
            core = [L_SH, R_SH, L_HIP, R_HIP]
            if (s[core] < CONF).any():
                continue
            sh = k[[L_SH, R_SH]].mean(0)
            hip = k[[L_HIP, R_HIP]].mean(0)
            torso = float(np.linalg.norm(sh - hip)) + 1e-6
            rows["torso"].append(torso)
            rows["hipx"].append(hip[0] * fwd)
            rows["lean"].append(_angle_vertical((sh[0] - hip[0]) * fwd, sh[1] - hip[1]))
            rows["shoff"].append((sh[0] - hip[0]) * fwd / torso)
            if s[ear] >= CONF:
                rows["head"].append(_angle_vertical((k[ear, 0] - sh[0]) * fwd, k[ear, 1] - sh[1]))
            for hp, kn, an in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)):
                if s[[hp, kn]].min() >= CONF:
                    rows["hip"].append(_joint_angle(sh, k[hp], k[kn]))
                if s[[hp, kn, an]].min() >= CONF:
                    rows["knee"].append(_joint_angle(k[hp], k[kn], k[an]))
                    rows["leg"].append(float(np.linalg.norm(k[hp] - k[kn]) + np.linalg.norm(k[kn] - k[an])))
            if s[[L_ANK, R_ANK]].min() >= CONF:
                rows["step"].append(abs(k[L_ANK, 0] - k[R_ANK, 0]))
            if s[wrist] >= CONF:
                rows["arm"].append((k[wrist, 0] - hip[0]) * fwd / torso)
        if len(rows["lean"]) < 4:
            continue
        torso = float(np.mean(rows["torso"]))
        leg = float(np.mean(rows["leg"])) if rows["leg"] else np.nan
        dur = (idx[-1] - idx[0]) / fps if idx[-1] > idx[0] else np.nan
        vals = {
            "trunk_lean": float(np.mean(rows["lean"])),
            "head_forward": float(np.mean(rows["head"])) if rows["head"] else np.nan,
            "shoulder_offset": float(np.mean(rows["shoff"])),
            "hip_flexion_max": 180.0 - float(np.min(rows["hip"])) if rows["hip"] else np.nan,
            "hip_range": float(np.max(rows["hip"]) - np.min(rows["hip"])) if len(rows["hip"]) > 1 else np.nan,
            "knee_flexion_max": 180.0 - float(np.min(rows["knee"])) if rows["knee"] else np.nan,
            "step_length": float(np.max(rows["step"])) / leg if rows["step"] and leg == leg else np.nan,
            "gait_speed": (rows["hipx"][-1] - rows["hipx"][0]) / dur / torso if dur == dur and len(rows["hipx"]) > 1 else np.nan,
            "arm_swing": float(np.max(rows["arm"]) - np.min(rows["arm"])) if len(rows["arm"]) > 1 else np.nan,
        }
        out[ci] = [vals[a] for a in ATTRS]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-path", required=True)
    ap.add_argument("--class-num", type=int, default=2)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    root = Path(args.root_path)
    info = root / "clinical_CLIP_dataset"
    folds = json.load(open(info / "index_mapping" / str(args.class_num) / "index.json"))
    first = next(iter(folds.values()))
    paths = sorted({p for s in first.values() for p in s})
    paths = [info / "json_mix" / p.split("json_mix/")[-1] for p in paths]
    med = MedAttnMap(str(info / "doctor_result"), str(info / "seg_skeleton_pkl"))

    per_video: dict[str, dict] = {}
    n_seg = n_missing = 0
    for p in paths:
        meta = json.load(open(p))
        name = meta["video_name"]
        if name in per_video:
            continue
        ann = med.skeleton_for(name)
        if ann is None:
            print(f"无骨架: {name}")
            continue
        video_path = str(p).split("json_mix/")[0] + "video/" + "/".join(meta["video_path"].split("/")[-3:])
        total, fps = _probe(video_path)
        plan = _plan_frame_index(total, max(int(fps), 1), args.num_samples)
        kp = np.asarray(ann["keypoint"], dtype=np.float32)[0]
        sc = np.asarray(ann["keypoint_score"], dtype=np.float32)[0]
        vals = video_attributes(kp, sc, plan, fps)
        n_seg += vals.shape[0]
        n_missing += int(np.isnan(vals).any(1).sum())
        per_video[name] = {"vals": vals, "label": int(meta["label"]), "json": str(p).split("json_mix/")[-1]}

    all_vals = np.concatenate([v["vals"] for v in per_video.values()])
    med_vals = np.nanmedian(all_vals, axis=0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"视频 {len(per_video)} 条, 段 {n_seg}, 含缺失的段 {n_missing} ({n_missing/max(n_seg,1):.1%})")
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

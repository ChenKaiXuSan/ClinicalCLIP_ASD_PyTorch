#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: region_crops.py
Project: dataloader
Author: Kaixu Chen
-----
Comment:
多分支区域模型(model.backbone=region)的数据端:按 2D 骨架把每个 1 秒 gait 段裁成若干"身体段"小视频,
每条分支只看一段。医生注意力在这里的作用是**决定开哪几段**(群体级先验,推理不需要任何医生标注)。

三套划分(model.region_set):
  measure   按"测量"划分的重叠身体段 —— 每段包含定义一个临床量所需的两端地标,段内的相对关系得以保留:
              head_shoulder  头相对肩(头前伸)
              trunk          肩相对髋(躯干前倾、肩偏移)
              hip_knee       骨盆与大腿(髋屈曲)
  parts     按医生区域字面划分,每条分支只看一个区域(head / shoulder / lumbar_pelvis)。
            跨区域的关系被切断,是"只输入划分"的直接实现,也是 measure 的对照
  unmarked  医生未标的下肢三段(knee / knee_ankle / foot),与 measure 段数相同,是区域特异性的对照。
            上肢不作对照:侧视下手臂与躯干在图像上重叠,裁手臂等于把腰椎骨盆一起裁进来

裁剪框的约束:
  1. **等比例**:框是正方形,边长 = scale × 该视频的躯干长(整条视频一个值),再等比例缩放到 crop_size。
     横纵不同比例会改掉躯干前倾角;边长不随帧变,尺度在时间上一致。
  2. **越界补零,不压缩**:框超出画面的部分补 0,不把框挤回画面里。
  3. **框的中心怎么随时间走**(data.region_track):
       follow(默认) 逐帧跟随该段的地标中心。实测(12 条视频 75 个段)同一秒内躯干中心在画面里来回摆动的
                    极差中位数是画面宽度的 6.2%(90 分位 11.8%)—— 源视频是按人体分割结果裁的,裁剪窗随步态
                    相位晃动;而关键点自身的噪声只有约 0.5%。逐帧跟随把这个晃动去掉,段内剩下的是该身体段
                    自己的姿态与运动。
       fixed        取该段 8 帧地标中心的中位数,段内不动。保留上面那个晃动(与整帧输入一致),作为对照。
     中心 = 该段各"地标组"中心的均值(例如躯干 = 肩中点与髋中点的中点),每组用其中置信度够的点。
     这样某一侧的点时有时无(侧视下远侧耳朵只有一半帧可见)不会把中心拉偏。

地标缺失时的回退:该帧缺 → 用该段有效帧的中位数;整段缺 → 用整条视频的中位数;整条视频都缺 → 用全库典型位置。
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

CONF = 0.3
# 全库 2D 骨架的躯干长(肩中点-髋中点, 归一化坐标)中位数;5%-95% 为 0.262-0.315
DEFAULT_TRUNK = 0.287

# 段名 -> (地标组列表(COCO-17 下标), 边长 = scale × 躯干长, 全库典型中心 (x, y) 归一化坐标)
# 典型中心与 scale 由 4114 帧骨架统计得到:各段地标跨度分别约为躯干长的
# head_shoulder 0.51 / trunk 1.03 / hip_knee 0.78 / knee_ankle 0.85 / head 0.31 倍,scale 取跨度加两侧余量
HEAD, SHOULDER, HIP, KNEE, ANKLE = [0, 1, 2, 3, 4], [5, 6], [11, 12], [13, 14], [15, 16]
SEGMENTS: Dict[str, Tuple[List[List[int]], float, Tuple[float, float]]] = {
    # ---- measure:按测量划分 ----
    "head_shoulder": ([HEAD, SHOULDER], 1.0, (0.52, 0.15)),
    "trunk": ([SHOULDER, HIP], 1.5, (0.50, 0.34)),
    "hip_knee": ([HIP, KNEE], 1.4, (0.50, 0.59)),
    # ---- parts:按医生区域字面划分 ----
    "head": ([HEAD], 0.7, (0.54, 0.10)),
    "shoulder": ([SHOULDER], 0.8, (0.50, 0.20)),
    "lumbar_pelvis": ([HIP], 1.0, (0.50, 0.485)),
    # ---- unmarked:未标部位(下肢) ----
    "knee": ([KNEE], 1.0, (0.50, 0.70)),
    "knee_ankle": ([KNEE, ANKLE], 1.4, (0.50, 0.80)),
    "foot": ([ANKLE], 1.0, (0.50, 0.90)),
}

REGION_SETS: Dict[str, List[str]] = {
    "measure": ["head_shoulder", "trunk", "hip_knee"],
    "parts": ["head", "shoulder", "lumbar_pelvis"],
    "unmarked": ["knee", "knee_ankle", "foot"],
    "none": [],
}
TRACK_MODES = ("follow", "fixed")


def resolve_region_set(spec) -> List[str]:
    """'measure' / 'parts' / 'unmarked' / 'none',或逗号分隔的段名,或段名列表 -> 段名列表。"""
    if spec is None:
        return []
    if isinstance(spec, str):
        if spec in REGION_SETS:
            return list(REGION_SETS[spec])
        names = [s.strip() for s in spec.split(",") if s.strip()]
    else:
        names = [str(s) for s in spec]
    unknown = [n for n in names if n not in SEGMENTS]
    if unknown:
        raise ValueError(f"未知的身体段 {unknown};可选 {sorted(SEGMENTS)} 或集合名 {sorted(REGION_SETS)}")
    return names


def trunk_length(pose: torch.Tensor) -> float:
    """整条视频的躯干长(肩中点-髋中点)中位数;pose (U, 17, 3) = 归一化 (x, y, score)。无有效帧时返回全库中位数。"""
    ok = (pose[:, SHOULDER + HIP, 2] >= CONF).all(dim=1)
    if not bool(ok.any()):
        return DEFAULT_TRUNK
    p = pose[ok]
    length = (p[:, SHOULDER, :2].mean(1) - p[:, HIP, :2].mean(1)).norm(dim=-1)
    val = float(length.median())
    # 极端值(检测失败)夹到合理范围,避免框小到没有内容或大到盖住全身
    return min(max(val, 0.18), 0.42)


def frame_centers(pose: torch.Tensor, groups: Sequence[Sequence[int]]) -> torch.Tensor:
    """pose (F, 17, 3) -> 每帧的段中心 (F, 2);任一地标组在该帧没有有效点时该行为 nan。"""
    centers = []
    for kps in groups:
        xy = pose[:, list(kps), :2]                        # (F, K, 2)
        valid = (pose[:, list(kps), 2] >= CONF).float()    # (F, K)
        cnt = valid.sum(dim=1, keepdim=True)               # (F, 1)
        c = (xy * valid.unsqueeze(-1)).sum(dim=1) / cnt.clamp_min(1.0)
        centers.append(torch.where(cnt > 0, c, torch.full_like(c, float("nan"))))
    return torch.stack(centers, dim=0).mean(dim=0)         # 含 nan 的组会把该帧传播成 nan


def _nanmedian_rows(x: torch.Tensor) -> torch.Tensor:
    """(F, 2) -> 有效行的逐列中位数 (2,);没有有效行时返回 nan。"""
    ok = ~torch.isnan(x).any(dim=1)
    if not bool(ok.any()):
        return torch.full((2,), float("nan"))
    return x[ok].median(dim=0).values


def segment_boxes(
    pose_chunk: torch.Tensor,
    names: Sequence[str],
    trunk: float,
    video_center: Dict[str, torch.Tensor],
    track: str = "follow",
) -> torch.Tensor:
    """一个 1 秒段各身体段的逐帧裁剪框 (R, T, 3) = (cx, cy, side),归一化坐标;side 以画面高度为单位。"""
    t = pose_chunk.shape[0]
    out = []
    for n in names:
        groups, scale, default = SEGMENTS[n]
        per_frame = frame_centers(pose_chunk, groups)       # (T, 2), 可能含 nan
        seg = _nanmedian_rows(per_frame)
        if bool(torch.isnan(seg).any()):
            seg = video_center[n]
        if bool(torch.isnan(seg).any()):
            seg = torch.tensor(default)
        if track == "fixed":
            centers = seg.unsqueeze(0).expand(t, 2)
        else:
            bad = torch.isnan(per_frame).any(dim=1, keepdim=True)
            centers = torch.where(bad, seg.unsqueeze(0).expand(t, 2), per_frame)
        out.append(torch.cat([centers, torch.full((t, 1), scale * trunk)], dim=1))
    return torch.stack(out) if out else torch.zeros((0, t, 3))


def crop_resize(frames: torch.Tensor, boxes: torch.Tensor, size: int) -> torch.Tensor:
    """frames (T, 3, H, W) uint8;boxes (T, 3) 或 (3,) = (cx, cy, side) 归一化 -> (T, 3, size, size) float [0, 1]。

    正方形像素框(边长 = side × H,横纵同一像素数 → 等比例;各帧边长相同),越界部分补 0,
    再用与整帧相同的 bilinear + antialias 缩放。
    """
    t, c, h, w = frames.shape
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0).expand(t, 3)
    side = max(int(round(float(boxes[0, 2]) * h)), 8)
    canvas = torch.zeros((t, c, side, side), dtype=frames.dtype)
    for f in range(t):
        x0 = int(round(float(boxes[f, 0]) * w - side / 2))
        y0 = int(round(float(boxes[f, 1]) * h - side / 2))
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0 + side, w), min(y0 + side, h)
        if sx1 > sx0 and sy1 > sy0:
            canvas[f, :, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frames[f, :, sy0:sy1, sx0:sx1]
    return F.interpolate(
        canvas.float().div_(255.0), size=(size, size), mode="bilinear", align_corners=False, antialias=True
    )


def build_region_clips(
    frames: torch.Tensor,
    pose: torch.Tensor,
    inverse: torch.Tensor,
    n_chunks: int,
    num_samples: int,
    names: Sequence[str],
    size: int,
    track: str = "follow",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把一条视频裁成各 gait 段的身体段小视频。

    Args:
        frames: (U, 3, H, W) uint8,采样计划里去重后的帧(原始分辨率,未缩放)。
        pose:   (U, 17, 3) 同一批帧的 2D 骨架,归一化 (x, y, score)。
        inverse: 采样计划 (n_chunks × num_samples) 到 frames 下标的映射(torch.unique 的 return_inverse)。
        track:  follow(逐帧跟随地标中心)| fixed(段内中位数,不动)。
    Returns:
        clips (n_chunks, R, 3, T, size, size) float [0, 1];
        boxes (n_chunks, R, T, 3) = 逐帧的 (cx, cy, side)。
    """
    if pose.shape[-1] != 3:
        raise ValueError(
            f"区域裁剪需要 2D 骨架 (x, y, score),收到最后一维 {pose.shape[-1]};"
            "paths.skeleton_path 请指向 2D 的 seg_skeleton_pkl(3D pkl 是相机坐标,不能用来裁图)"
        )
    if track not in TRACK_MODES:
        raise ValueError(f"data.region_track 只支持 {TRACK_MODES},收到 {track}")
    r = len(names)
    idx = inverse.view(n_chunks, num_samples)
    trunk = trunk_length(pose)
    video_center = {n: _nanmedian_rows(frame_centers(pose, SEGMENTS[n][0])) for n in names}
    clips = torch.zeros((n_chunks, r, frames.shape[1], num_samples, size, size))
    boxes = torch.zeros((n_chunks, r, num_samples, 3))
    for ci in range(n_chunks):
        sel = idx[ci]
        chunk_frames = frames.index_select(0, sel)
        b = segment_boxes(pose.index_select(0, sel), names, trunk, video_center, track)   # (R, T, 3)
        boxes[ci] = b
        for ri in range(r):
            clips[ci, ri] = crop_resize(chunk_frames, b[ri], size).permute(1, 0, 2, 3)   # (3, T, S, S)
    return clips, boxes

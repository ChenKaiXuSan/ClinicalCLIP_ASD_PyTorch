#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: whole_video_dataset.py
Project: dataloader
Created Date: 2026-02-03
Author: Kaixu Chen
-----
Comment:
Load full video clips with clinician attention maps.

采样计划先行:先根据视频总帧数算出最终会被保留的帧下标,再只解码、只生成
这些帧。旧实现解码全部帧、在原始分辨率上为每一帧生成注意力图,而后续按秒
切段 + 每段取 8 帧只保留约四分之一,且缩到 224,绝大部分计算被丢弃。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import av
import numpy as np
import torch
import torch.nn.functional as F

from .med_attn_map import MedAttnMap

logger = logging.getLogger(__name__)

# 旧实现对 video 和 attn_map 共用同一个 Compose,其中的 Div255 也作用在
# 注意力图上,使高斯图从 [0,1] 变成 [0,1/255]。模型里 downsample_attn_to_tokens
# 会做逐样本 min-max 归一化,该缩放基本被抵消;但 ChannelMapGuidedVideoEncoder
# 直接把原始均值送进 MLP,尺度是有影响的。此处保持与既有实验一致,如需修正
# 把此常量改为 False。
LEGACY_ATTN_DIV255 = True


def read_video(filename: str, output_format: str = "TCHW", pts_unit: str = "sec"):
    """torchvision>=0.26 移除了 read_video，用 PyAV 实现兼容替代。
    返回 (vframes, aframes, info)，与原接口一致。
    """
    frames = []
    fps = 0.0
    with av.open(filename) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate or 0)
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))  # H x W x C

    if frames:
        vframes = torch.from_numpy(np.stack(frames))  # T x H x W x C
    else:
        vframes = torch.zeros((0, 0, 0, 3), dtype=torch.uint8)

    if output_format == "TCHW":
        vframes = vframes.permute(0, 3, 1, 2)  # T x C x H x W

    info = {"video_fps": fps}
    return vframes, torch.tensor([]), info


@dataclass
class ClinicalAttnVideoData:
    video: torch.Tensor
    label: int
    attn_map: torch.Tensor
    disease: str
    video_name: str
    video_index: int


def _probe(video_path: str) -> Tuple[int, float]:
    """读容器元信息拿总帧数和 fps,不解码像素。"""
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        return int(stream.frames or 0), float(stream.average_rate or 0)


def _plan_frame_index(total: int, fps: int, num_samples: int) -> torch.Tensor:
    """复刻旧的“按秒切段 + 每段均匀取 num_samples 帧”得到的全局帧下标。

    Returns:
        (n_chunks, num_samples) 的下标,段内不足时重复最近帧(同 UniformTemporalSubsample)。
    """
    chunks = []
    for start in range(0, total, fps):
        length = min(start + fps, total) - start
        offset = torch.round(
            torch.linspace(0, max(length - 1, 0), num_samples)
        ).long()
        chunks.append(offset + start)
    return torch.stack(chunks, dim=0)


def _decode_selected(video_path: str, wanted: torch.Tensor) -> torch.Tensor:
    """顺序解码,只对需要的帧做 rgb24 转换。

    帧间预测决定了必须逐帧走解码器,但 to_ndarray 的色彩空间转换和拷贝是大头,
    跳过无用帧即可省下这部分。

    Returns:
        (len(wanted), 3, H, W) uint8,顺序与 wanted 一致。
    """
    wanted_set = {int(i) for i in wanted}
    last = int(wanted[-1])
    collected: dict[int, np.ndarray] = {}

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted_set:
                collected[i] = frame.to_ndarray(format="rgb24")
            if i >= last:
                break

    if not collected:
        raise RuntimeError(f"no frame decoded from {video_path}")

    # 个别容器声明的帧数多于实际可解码帧,用最后一帧补齐
    fallback = collected[max(collected)]
    stacked = np.stack([collected.get(int(i), fallback) for i in wanted])
    return torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()


class LabeledGaitVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        experiment: str,
        labeled_video_paths: list,
        img_size: int = 224,
        num_samples: int = 8,
        doctor_res_path: str = "",
        skeleton_path: str = "",
        attn_map: Optional[MedAttnMap] = None,
        region_supervision: bool = False,
        region_map_size: int = 28,
        return_pose: bool = False,
        return_video: bool = True,
    ) -> None:
        super().__init__()

        self._labeled_videos = labeled_video_paths
        self._experiment = experiment
        self._img_size = img_size
        self._num_samples = num_samples
        # 概念架构走 grounding 监督:按区域拆开的低分辨率图 + 区域软标签
        self._region_supervision = region_supervision
        self._region_map_size = region_map_size
        # 纯姿态基线用,与视频共用同一批帧下标
        self._return_pose = return_pose
        # 纯姿态基线不需要像素,跳过解码与缩放能省掉绝大部分数据加载开销
        self._return_video = return_video

        # 优先复用外部传入的实例:骨架 pkl 有 97MB,train/val/test 各建一份纯属浪费
        if attn_map is not None:
            self.attn_map = attn_map
        elif doctor_res_path and skeleton_path:
            self.attn_map = MedAttnMap(doctor_res_path, skeleton_path)
        else:
            self.attn_map = None

    def __len__(self) -> int:
        return len(self._labeled_videos)

    def _resolve_video_path(self, json_path, recorded_path: str) -> str:
        # json 里写死的是生成时环境的绝对路径,只取末尾三段接到当前数据根目录下
        prefix = str(json_path).split("json_mix/")[0]
        return prefix + "video/" + "/".join(recorded_path.split("/")[-3:])

    def __getitem__(self, index) -> dict[str, Any]:
        json_path = self._labeled_videos[index]
        with open(json_path) as f:
            file_info_dict = json.load(f)

        video_name = file_info_dict["video_name"]
        video_path = self._resolve_video_path(json_path, file_info_dict["video_path"])

        total, fps = _probe(video_path)
        frames = None
        if total <= 0 or fps <= 0:
            # 容器没写帧数时退回全解码
            vframes, _, info = read_video(video_path, output_format="TCHW")
            total, fps = vframes.shape[0], float(info["video_fps"] or 30.0)
            plan = _plan_frame_index(total, max(int(fps), 1), self._num_samples)
            wanted, inverse = torch.unique(plan.flatten(), return_inverse=True)
            frames = vframes.index_select(0, wanted)
        else:
            plan = _plan_frame_index(total, max(int(fps), 1), self._num_samples)
            wanted, inverse = torch.unique(plan.flatten(), return_inverse=True)
            if self._return_video:
                frames = _decode_selected(video_path, wanted)

        n_chunks = plan.shape[0]

        if frames is not None:
            # 先 /255 再 resize,与旧的 Div255 → Resize 顺序一致(uint8 上插值会有量化损失)
            video = F.interpolate(
                frames.float().div_(255.0),
                size=(self._img_size, self._img_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            video = video.index_select(0, inverse).view(
                n_chunks, self._num_samples, *video.shape[1:]
            )
            video = video.permute(0, 2, 1, 3, 4).contiguous()  # (n_chunks, C, T, H, W)
        else:
            video = None

        sample = {
            "label": file_info_dict["label"],
            "disease": file_info_dict["disease"],
            "video_name": video_name,
            "video_index": index,
            # 段数,供 collate 在没有 video 时也能展开视频级标签
            "num_chunks": n_chunks,
        }

        if video is not None:
            sample["video"] = video

            if self.attn_map is not None:
                attn = self.attn_map.build(
                    video_name=video_name,
                    frame_idx=wanted,
                    out_size=(self._img_size, self._img_size),
                )
            else:
                attn = torch.zeros((wanted.numel(), 1, self._img_size, self._img_size))

            if LEGACY_ATTN_DIV255:
                attn = attn / 255.0

            attn = attn.index_select(0, inverse).view(
                n_chunks, self._num_samples, *attn.shape[1:]
            )
            sample["attn_map"] = attn.permute(0, 2, 1, 3, 4).contiguous()

        if self._region_supervision and self.attn_map is not None:
            size = (self._region_map_size, self._region_map_size)
            region_map = self.attn_map.build_regions(
                video_name=video_name, frame_idx=wanted, out_size=size
            )
            region_map = region_map.index_select(0, inverse).view(
                n_chunks, self._num_samples, *region_map.shape[1:]
            )
            # (n_chunks, R, T, H, W),与 token 的 (B, d, T', H', W') 对齐
            sample["region_map"] = region_map.permute(0, 2, 1, 3, 4).contiguous()
            sample["region_target"] = self.attn_map.presence_for(video_name)

        if self._return_pose and self.attn_map is not None:
            pose = self.attn_map.pose_for(video_name, wanted)  # (U, 17, 3)
            # (n_chunks, T, 17, 3),时间采样与 video 完全一致
            sample["pose"] = (
                pose.index_select(0, inverse)
                .view(n_chunks, self._num_samples, *pose.shape[1:])
                .contiguous()
            )

        return sample


def whole_video_dataset(
    experiment: str,
    dataset_idx: list = [],
    img_size: int = 224,
    num_samples: int = 8,
    doctor_res_path: str = "",
    skeleton_path: str = "",
    attn_map: Optional[MedAttnMap] = None,
    region_supervision: bool = False,
    region_map_size: int = 28,
    return_pose: bool = False,
    return_video: bool = True,
    clip_duration: int = 1,
) -> LabeledGaitVideoDataset:
    return LabeledGaitVideoDataset(
        experiment=experiment,
        labeled_video_paths=dataset_idx,
        img_size=img_size,
        num_samples=num_samples,
        doctor_res_path=doctor_res_path,
        skeleton_path=skeleton_path,
        attn_map=attn_map,
        region_supervision=region_supervision,
        region_map_size=region_map_size,
        return_pose=return_pose,
        return_video=return_video,
    )

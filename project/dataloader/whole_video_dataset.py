#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: whole_video_dataset.py
Project: dataloader
Created Date: 2026-02-03
Author: OpenAI Assistant
-----
Comment:
Load full video clips with clinician attention maps.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import av
import numpy as np
import torch

from project.dataloader.med_attn_map import MedAttnMap


def read_video(filename: str, output_format: str = "TCHW", pts_unit: str = "sec"):
    """torchvision>=0.26 移除了 read_video，用 PyAV 实现兼容替代。
    返回 (vframes, aframes, info)，与原接口一致。
    """
    frames = []
    fps = 0.0
    with av.open(filename) as container:
        stream = container.streams.video[0]
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

logger = logging.getLogger(__name__)

@dataclass
class ClinicalAttnVideoData:
    video: torch.Tensor
    label: int
    attn_map: torch.Tensor
    disease: str
    video_name: str
    video_index: int

class LabeledGaitVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        experiment: str,
        labeled_video_paths: list[Tuple[str, Optional[dict]]],
        transform: Optional[Callable[[dict], Any]] = None,
        doctor_res_path: str = "",
        skeleton_path: str = "",
    ) -> None:
        super().__init__()

        self._transform = transform
        self._labeled_videos = labeled_video_paths
        self._experiment = experiment

        if doctor_res_path and skeleton_path:
            self.attn_map = MedAttnMap(doctor_res_path, skeleton_path)
        else:
            self.attn_map = None

    def __len__(self) -> int:
        return len(self._labeled_videos)

    def move_transform(self, vframes: torch.Tensor, fps: int) -> torch.Tensor:
        t, *_ = vframes.shape
        batch_res = []

        for f in range(0, t, fps):
            one_sec_vframes = vframes[f : f + fps, :, :, :]

            if self._transform is not None:
                transformed_img = self._transform(one_sec_vframes)
                batch_res.append(transformed_img.permute(1, 0, 2, 3))
            else:
                logger.warning("no transform")
                batch_res.append(one_sec_vframes.permute(1, 0, 2, 3))

        return torch.stack(batch_res, dim=0)

    def __getitem__(self, index) -> dict[str, Any]:
        with open(self._labeled_videos[index]) as f:
            file_info_dict = json.load(f)

        video_name = file_info_dict["video_name"]
        video_path = file_info_dict["video_path"]

        # TODO: 之后修改为从video_path里面读取video path进行load
        # * json mix里面是提前写好的，但是在超算的环境下，video path是需要改变的
        # video_path = "/" + "/".join(self._labeled_videos[index].parts[1:4]) + "/video/" + "/".join(video_path.split("/")[-3:])
        _pref = str(self._labeled_videos[index]).split("json_mix/")[0]
        video_path = _pref + "video/" + "/".join(video_path.split("/")[-3:])
        # logger.info(f"Loading video from path: {video_path}")
        vframes, _, info = read_video(video_path, output_format="TCHW", pts_unit="sec")

        label = file_info_dict["label"]
        disease = file_info_dict["disease"]

        if self.attn_map is not None:
            attn_map = self.attn_map(
                video_name=video_name,
                video_path=video_path,
                disease=disease,
                vframes=vframes,
            )
        else:
            attn_map = torch.zeros((vframes.shape[0], 1, vframes.shape[2], vframes.shape[3]))

        transformed_vframes = self.move_transform(vframes, int(info["video_fps"]))
        transformed_attn_map = self.move_transform(attn_map, int(info["video_fps"]))

        return {
            "video": transformed_vframes,
            "label": label,
            "attn_map": transformed_attn_map,
            "disease": disease,
            "video_name": video_name,
            "video_index": index,
        }


def whole_video_dataset(
    experiment: str,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    dataset_idx: list = [],
    doctor_res_path: str = "",
    skeleton_path: str = "",
    clip_duration: int = 1,
) -> LabeledGaitVideoDataset:
    return LabeledGaitVideoDataset(
        experiment=experiment,
        transform=transform,
        labeled_video_paths=dataset_idx,
        doctor_res_path=doctor_res_path,
        skeleton_path=skeleton_path,
    )

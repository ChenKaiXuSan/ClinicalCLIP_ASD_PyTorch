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
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torchvision.io import read_video

from project.dataloader.med_attn_map import MedAttnMap

logger = logging.getLogger(__name__)


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

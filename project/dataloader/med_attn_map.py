#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: med_attn_map.py
Project: dataloader
Created Date: 2026-02-03
Author: OpenAI Assistant
-----
Comment:
Generate clinician attention maps aligned to video frames.
"""

from typing import Any, Dict, List

import csv
import os
import pickle
import torch
from torchvision.utils import save_image


COCO_KEYPOINTS: Dict[int, str] = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

REGION_TO_KEYPOINTS: Dict[str, List[int]] = {
    "foot": [15, 16],
    "wrist": [9, 10],
    "shoulder": [5, 6],
    "lumbar_pelvis": [11, 12],
    "head": [0, 1, 2, 3, 4],
}


class MedAttnMap:

    def __init__(self, doctor_res_path: str, skeleton_path: str) -> None:
        self.doctor_res = self._load_doctor_res(doctor_res_path)
        self.skeleton = self._load_skeleton(skeleton_path)

    @staticmethod
    def _load_doctor_res(doctor_res_path: str) -> list[list[dict[str, str]]]:
        def load_csv(path: str) -> list[dict[str, str]]:
            with open(path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                return [row for row in reader]

        doctor_1 = load_csv(os.path.join(doctor_res_path, "doctor1.csv"))
        doctor_2 = load_csv(os.path.join(doctor_res_path, "doctor2.csv"))
        return [doctor_1, doctor_2]

    @staticmethod
    def _load_skeleton(skeleton_path: str) -> dict[str, Any]:
        with open(os.path.join(skeleton_path, "whole_annotations.pkl"), "rb") as f:
            return pickle.load(f)

    def _find_doctor_res(self, video_name: str) -> tuple[set[str], set[int]]:
        doctor_attn: List[str] = []
        keypoint_num: List[int] = []

        for one_doctor in self.doctor_res:
            for row in one_doctor:
                if row["video file name"] in video_name:
                    region = row["attention"][2:-6]
                    doctor_attn.append(region)
                    for i in REGION_TO_KEYPOINTS.get(region, []):
                        keypoint_num.append(i)

        return set(doctor_attn), set(keypoint_num)

    def _find_skeleton(self, video_name: str) -> list[dict[str, Any]]:
        res: List[dict[str, Any]] = []
        for one in self.skeleton["annotations"]:
            _video_name = one["frame_dir"].split("/")[-1]
            if video_name in _video_name:
                res.append(one)
        return res

    @staticmethod
    def _generate_attention_map(
        vframes: torch.Tensor,
        mapped_keypoint: set[int],
        keypoint: torch.Tensor,
        confidence_score: torch.Tensor,
    ) -> torch.Tensor:
        t, _, h, w = vframes.shape
        sigma = 0.1 * min(h, w)

        y_grid, x_grid = torch.meshgrid(
            torch.arange(h), torch.arange(w), indexing="ij"
        )

        res = []
        for frame in range(t):
            attn_maps = []
            for i in mapped_keypoint:
                x = keypoint[0, frame, i, 0] * w
                y = keypoint[0, frame, i, 1] * h

                if x < 0 or y < 0:
                    attn_maps.append(torch.zeros((h, w)))
                    continue

                dist_squared = (x_grid - x) ** 2 + (y_grid - y) ** 2
                heatmap = torch.exp(-dist_squared / (2 * sigma**2))
                curr_confidence = confidence_score[0, frame, i]
                if curr_confidence > 0.8:
                    heatmap *= curr_confidence
                attn_maps.append(heatmap)

            if not attn_maps:
                attn_maps.append(torch.zeros((h, w)))

            attn_stack = torch.stack(attn_maps, dim=0)
            attn_mean = torch.mean(attn_stack, dim=0).unsqueeze(0)
            res.append(attn_mean)

        return torch.stack(res, dim=0)

    @staticmethod
    def save_attention_map(attention_map: torch.Tensor, save_path: str, video_name: str) -> None:
        t, *_ = attention_map.shape
        save_pth = os.path.join(save_path, "attention_map", video_name)
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)

        for i in range(t):
            save_image(attention_map[i], os.path.join(save_pth, f"attn_{i}.png"), normalize=True)

    def __call__(self, video_path: str, disease: str, vframes: torch.Tensor, video_name: str) -> torch.Tensor:
        _, mapped_keypoint = self._find_doctor_res(video_name)
        skeleton = self._find_skeleton(video_name)

        if not skeleton:
            return torch.zeros((vframes.shape[0], 1, vframes.shape[2], vframes.shape[3]))

        attn_map = self._generate_attention_map(
            vframes,
            mapped_keypoint,
            skeleton[0]["keypoint"],
            confidence_score=skeleton[0]["keypoint_score"],
        )

        return attn_map

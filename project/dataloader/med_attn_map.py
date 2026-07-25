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

        # 两个查找都是对全表的子串匹配(骨架表 2277 条),按 video_name 记忆化
        self._keypoint_cache: Dict[str, set] = {}
        self._skeleton_cache: Dict[str, Any] = {}

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

    def keypoints_for(self, video_name: str) -> set:
        """医生关注区域对应的关键点下标(记忆化)。"""
        if video_name not in self._keypoint_cache:
            _, keypoints = self._find_doctor_res(video_name)
            self._keypoint_cache[video_name] = keypoints
        return self._keypoint_cache[video_name]

    def skeleton_for(self, video_name: str):
        """该视频的第一条骨架标注,没有则 None(记忆化)。"""
        if video_name not in self._skeleton_cache:
            found = self._find_skeleton(video_name)
            self._skeleton_cache[video_name] = found[0] if found else None
        return self._skeleton_cache[video_name]

    def build(
        self,
        video_name: str,
        frame_idx: torch.Tensor,
        out_size: tuple[int, int],
    ) -> torch.Tensor:
        """只为指定帧、直接在目标分辨率上生成注意力图。

        与逐帧逐关键点的 _generate_attention_map 等价:高斯核可分离为
        exp(-dx²/2σ²)·exp(-dy²/2σ²),因此对关键点求和退化成一次批量矩阵乘,
        无需为每个 (帧, 关键点) 物化 H×W 网格。σ 取 0.1*min(H,W),与原实现
        同为边长的十分之一,故直接在 224 上生成等价于在 512 上生成后缩放。

        Args:
            frame_idx: 需要的帧下标 (F,)
            out_size: (H, W) 目标分辨率

        Returns:
            (F, 1, H, W)
        """
        height, width = out_size
        num_frames = int(frame_idx.numel())

        annotation = self.skeleton_for(video_name)
        keypoints = sorted(self.keypoints_for(video_name))

        if annotation is None or not keypoints:
            return torch.zeros((num_frames, 1, height, width))

        keypoint = torch.as_tensor(annotation["keypoint"], dtype=torch.float32)
        score = torch.as_tensor(annotation["keypoint_score"], dtype=torch.float32)

        # 骨架帧数与视频帧数实测一致,越界时钳位以防个别标注偏短
        idx = frame_idx.clamp(max=keypoint.shape[1] - 1)
        coords = keypoint[0].index_select(0, idx)[:, keypoints, :]  # (F, K, 2)
        conf = score[0].index_select(0, idx)[:, keypoints]  # (F, K)

        x = coords[..., 0] * width
        y = coords[..., 1] * height

        sigma = 0.1 * min(height, width)
        denom = 2.0 * sigma**2

        # 负坐标表示该帧缺失该关键点,原实现记为全零图
        valid = (x >= 0) & (y >= 0)
        # 原实现只在置信度 > 0.8 时按置信度缩放,否则保持原样
        scale = torch.where(conf > 0.8, conf, torch.ones_like(conf)) * valid

        x_range = torch.arange(width, dtype=torch.float32)
        y_range = torch.arange(height, dtype=torch.float32)

        gx = torch.exp(-((x_range - x[..., None]) ** 2) / denom)  # (F, K, W)
        gy = torch.exp(-((y_range - y[..., None]) ** 2) / denom)  # (F, K, H)
        gy = gy * scale[..., None]

        attn = torch.einsum("fkh,fkw->fhw", gy, gx) / len(keypoints)
        return attn.unsqueeze(1)

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

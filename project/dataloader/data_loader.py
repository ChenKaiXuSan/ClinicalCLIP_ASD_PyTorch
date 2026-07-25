"""
File: data_loader.py
Project: dataloader
Created Date: 2026-02-03
Author: OpenAI Assistant
-----
Comment:
Lightning data module for gait videos with optional clinician attention maps.
"""

from typing import Any, Dict, Optional

from torchvision.transforms import Compose, Resize
from pytorch_lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader

from pytorchvideo.data import make_clip_sampler
from pytorchvideo.data.labeled_video_dataset import labeled_video_dataset

from .whole_video_dataset import whole_video_dataset
from .med_attn_map import MedAttnMap
from .utils import Div255, UniformTemporalSubsample, ApplyTransformToKey


DISEASE_TO_NUM_MAPPING: Dict = {
    2: {"ASD": 0, "non-ASD": 1},
    3: {"ASD": 0, "DHS": 1, "LCS_HipOA": 2},
    4: {"ASD": 0, "DHS": 1, "LCS_HipOA": 2, "normal": 3},
}


class WalkDataModule(LightningDataModule):
    def __init__(self, opt, dataset_idx: Dict = None):
        super().__init__()

        self._batch_size = opt.data.batch_size
        self._num_workers = opt.data.num_workers
        self._img_size = opt.data.img_size
        self._clip_duration = opt.train.clip_duration
        self.uniform_temporal_subsample_num = opt.train.uniform_temporal_subsample_num

        self._dataset_idx = dataset_idx
        self._doctor_res_path = opt.paths.doctor_results_path
        self._skeleton_path = opt.paths.skeleton_path
        self._class_num = opt.model.model_class_num
        self._experiment = opt.train.experiment
        self._attn_map = opt.train.attn_map
        self._med_attn_map: Optional[MedAttnMap] = None
        # concept 架构需要按区域拆开的 grounding 目标
        self._region_supervision = opt.model.backbone == "concept"
        self._region_map_size = getattr(opt.model, "region_map_size", 28)

        self.train_video_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            Div255(),
                            Resize(size=[self._img_size, self._img_size]),
                            UniformTemporalSubsample(
                                self.uniform_temporal_subsample_num
                            ),
                        ]
                    ),
                ),
            ]
        )

        self.val_video_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            Div255(),
                            Resize(size=[self._img_size, self._img_size]),
                            UniformTemporalSubsample(
                                self.uniform_temporal_subsample_num
                            ),
                        ]
                    ),
                ),
            ]
        )

    def prepare_data(self) -> None:
        ...

    def setup(self, stage: Optional[str] = None) -> None:
        if self._attn_map:
            # 骨架 pkl 97MB,只建一次给三个 dataset 共用(worker fork 时也只有一份)
            if self._med_attn_map is None:
                self._med_attn_map = MedAttnMap(
                    self._doctor_res_path, self._skeleton_path
                )

            common = dict(
                experiment=self._experiment,
                img_size=self._img_size,
                num_samples=self.uniform_temporal_subsample_num,
                attn_map=self._med_attn_map,
                region_supervision=self._region_supervision,
                region_map_size=self._region_map_size,
                clip_duration=self._clip_duration,
            )

            self.train_gait_dataset = whole_video_dataset(
                dataset_idx=self._dataset_idx['train'], **common
            )

            self.val_gait_dataset = whole_video_dataset(
                dataset_idx=self._dataset_idx['val'], **common
            )

            self.test_gait_dataset = whole_video_dataset(
                dataset_idx=self._dataset_idx['val'], **common
            )

        else:
            self.train_gait_dataset = labeled_video_dataset(
                data_path=self._dataset_idx['train'],
                clip_sampler=make_clip_sampler("uniform", self._clip_duration),
                transform=self.train_video_transform,
            )

            self.val_gait_dataset = labeled_video_dataset(
                data_path=self._dataset_idx['val'],
                clip_sampler=make_clip_sampler("uniform", self._clip_duration),
                transform=self.val_video_transform,
            )

            self.test_gait_dataset = labeled_video_dataset(
                data_path=self._dataset_idx['test'],
                clip_sampler=make_clip_sampler("uniform", self._clip_duration),
                transform=self.val_video_transform,
            )

    def collate_fn(self, batch: list[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_label = []
        batch_video = []
        batch_attn_map = []
        batch_region_map = []
        batch_region_target = []

        for i in batch:
            gait_num, *_ = i["video"].shape
            disease = i["disease"]

            batch_video.append(i["video"])
            batch_attn_map.append(i["attn_map"])

            if "region_map" in i:
                batch_region_map.append(i["region_map"])
                # 区域标签是场次级的,展开到该视频的每个 gait 段
                batch_region_target.append(
                    i["region_target"].unsqueeze(0).expand(gait_num, -1)
                )

            for _ in range(gait_num):
                if disease in DISEASE_TO_NUM_MAPPING[self._class_num].keys():
                    assert (
                        DISEASE_TO_NUM_MAPPING[self._class_num][disease]
                        == i["label"]
                    ), "The disease label mapping is not correct!"

                    batch_label.append(
                        DISEASE_TO_NUM_MAPPING[self._class_num][disease]
                    )
                else:
                    batch_label.append(
                        DISEASE_TO_NUM_MAPPING[self._class_num]["non-ASD"]
                    )

        out = {
            "video": torch.cat(batch_video, dim=0),
            "label": torch.tensor(batch_label),
            "attn_map": torch.cat(batch_attn_map, dim=0),
            # 只带元信息:原先整个 batch 连 video/attn 张量一起放进来,等于让同样
            # 的数据再穿一次 worker→主进程的 IPC
            "info": [
                {
                    k: v
                    for k, v in i.items()
                    if k not in ("video", "attn_map", "region_map", "region_target")
                }
                for i in batch
            ],
        }

        if batch_region_map:
            out["region_map"] = torch.cat(batch_region_map, dim=0)
            out["region_target"] = torch.cat(batch_region_target, dim=0)

        return out

    def _dataloader(self, dataset, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=self.collate_fn,
            # 每个 epoch 重建 worker 要重新 fork 一遍 97MB 的骨架数据
            persistent_workers=self._num_workers > 0,
            prefetch_factor=4 if self._num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_gait_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        # 验证/测试不能丢样本,否则指标算在残缺集合上
        return self._dataloader(self.val_gait_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_gait_dataset, shuffle=False, drop_last=False)

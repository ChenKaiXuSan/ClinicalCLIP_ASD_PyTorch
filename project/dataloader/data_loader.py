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

from project.dataloader.whole_video_dataset import whole_video_dataset
from project.dataloader.utils import Div255, UniformTemporalSubsample, ApplyTransformToKey


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
        self._doctor_res_path = opt.data.doctor_results_path
        self._skeleton_path = opt.data.skeleton_path
        self._class_num = opt.model.model_class_num
        self._experiment = opt.train.experiment
        self._attn_map = opt.train.attn_map

        self.mapping_transform = Compose(
            [
                UniformTemporalSubsample(self.uniform_temporal_subsample_num),
                Div255(),
                Resize(size=[self._img_size, self._img_size]),
            ]
        )

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
            self.train_gait_dataset = whole_video_dataset(
                experiment=self._experiment,
                dataset_idx=self._dataset_idx[0],
                transform=self.mapping_transform,
                skeleton_path=self._skeleton_path,
                doctor_res_path=self._doctor_res_path,
                clip_duration=self._clip_duration,
            )

            self.val_gait_dataset = whole_video_dataset(
                experiment=self._experiment,
                dataset_idx=self._dataset_idx[1],
                transform=self.mapping_transform,
                doctor_res_path=self._doctor_res_path,
                skeleton_path=self._skeleton_path,
                clip_duration=self._clip_duration,
            )

            self.test_gait_dataset = whole_video_dataset(
                experiment=self._experiment,
                dataset_idx=self._dataset_idx[1],
                transform=self.mapping_transform,
                doctor_res_path=self._doctor_res_path,
                skeleton_path=self._skeleton_path,
                clip_duration=self._clip_duration,
            )

        else:
            self.train_gait_dataset = labeled_video_dataset(
                data_path=self._dataset_idx[2],
                clip_sampler=make_clip_sampler("uniform", self._clip_duration),
                transform=self.train_video_transform,
            )

            self.val_gait_dataset = labeled_video_dataset(
                data_path=self._dataset_idx[3],
                clip_sampler=make_clip_sampler("uniform", self._clip_duration),
                transform=self.val_video_transform,
            )

            self.test_gait_dataset = labeled_video_dataset(
                data_path=self._dataset_idx[3],
                clip_sampler=make_clip_sampler("uniform", self._clip_duration),
                transform=self.val_video_transform,
            )

    def collate_fn(self, batch: list[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_label = []
        batch_video = []
        batch_attn_map = []

        for i in batch:
            gait_num, *_ = i["video"].shape
            disease = i["disease"]

            batch_video.append(i["video"])
            batch_attn_map.append(i["attn_map"])

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

        return {
            "video": torch.cat(batch_video, dim=0),
            "label": torch.tensor(batch_label),
            "attn_map": torch.cat(batch_attn_map, dim=0),
            "info": batch,
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_gait_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_gait_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=False,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_gait_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=False,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

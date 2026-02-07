#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/cross_validation.py
Project: /workspace/skeleton/project
Created Date: Friday March 22nd 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday May 1st 2025 8:34:05 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

22-03-2024	Kaixu Chen	add different class number mapping, and add the cross validation process.
"""


import os, json, shutil, copy, random
from typing import Any, Dict, List, Tuple

from sklearn.model_selection import StratifiedGroupKFold, train_test_split, GroupKFold
from pathlib import Path

class_num_mapping_Dict: Dict = {
    2: {0: "ASD", 1: "non-ASD"},
    3: {0: "ASD", 1: "DHS", 2: "LCS_HipOA"},
    4: {0: "ASD", 1: "DHS", 2: "LCS_HipOA", 3: "normal"},
}


class DefineCrossValidation(object):
    """Process cross validation for gait analysis dataset.
    
    Workflow:
        1. Cross validation split using StratifiedGroupKFold
        2. Train/Val split for each fold
        3. Save index mapping (no video file copying)
    
    Returns:
        fold: {'train': [path], 'val': [path]}
    """

    def __init__(self, config) -> None:

        self.video_path: Path = Path(config.paths.data_info_path)  # json file path
        self.gait_seg_idx_path: Path = Path(
            config.paths.index_mapping
        )  # used for training path mapping

        self.K: int = config.train.fold
        self.class_num: int = getattr(config.model, 'model_class_num', 2)
        self.clip_duration: int = config.train.clip_duration

    def process_cross_validation(self, video_dict: dict) -> Tuple[List, List, List]:

        _path = video_dict

        X = []  # patient index
        y = []  # patient class index
        groups = []  # different patient groups

        disease_to_num = {
            disease: idx
            for idx, disease in class_num_mapping_Dict[self.class_num].items()
        }
        element_to_num = {}

        name_map = set()

        # process one disease in one loop.
        for disease, path in _path.items():
            patient_list = sorted(list(path))

            for p in patient_list:
                name, _ = p.name.split("-")
                # FIXME: Filter out HipOA to address data imbalance
                if "HipOA" not in name:
                    name_map.add(name)

        for idx, element in enumerate(name_map):
            element_to_num[element] = idx

        for disease, path in _path.items():
            patient_list = sorted(list(path))
            for i in range(len(patient_list)):

                name, _ = patient_list[i].name.split("-")

                label = disease_to_num[disease]

                # FIXME: Filter out HipOA to address data imbalance
                if "HipOA" not in name:
                    X.append(patient_list[i])  # true path in Path
                    y.append(label)  # label, 0, 1, 2
                    groups.append(element_to_num[name])  # number of different patient

        return X, y, groups

    @staticmethod
    def magic_move(train_mapped_path, val_mapped_path):

        new_train_mapped_path = copy.deepcopy(train_mapped_path)
        new_val_mapped_path = copy.deepcopy(val_mapped_path)

        # train magic
        train_tmp_dict = {}
        for i in train_mapped_path:
            # not move ASD
            if "ASD" in i.name:
                continue

            train_tmp_dict[i.name.split("-")[0]] = i

        val_tmp_dict = {}
        for i in val_mapped_path:
            # not move ASD
            if "ASD" in i.name:
                continue
            val_tmp_dict[i.name.split("-")[0]] = i

        for k, v in train_tmp_dict.items():
            new_val_mapped_path.append(v)

            rm_idx = new_train_mapped_path.index(v)
            new_train_mapped_path.pop(rm_idx)

        for k, v in val_tmp_dict.items():
            new_train_mapped_path.append(v)

            rm_idx = new_val_mapped_path.index(v)
            new_val_mapped_path.pop(rm_idx)

        return new_train_mapped_path, new_val_mapped_path

    @staticmethod
    def map_class_num(class_num: int, raw_video_path: Path) -> Dict:

        _class_num = class_num_mapping_Dict[class_num]

        res_dict = {v: [] for k, v in _class_num.items()}

        for disease in raw_video_path.iterdir():

            for one_json_file in disease.iterdir():

                if disease.name in res_dict.keys():
                    res_dict[disease.name].append(one_json_file)
                elif disease.name == "log":
                    continue
                else:
                    res_dict["non-ASD"].append(one_json_file)

        return res_dict

    def prepare(self):
        """Define K-fold cross validation splits.
        
        Returns:
            tuple: (ans_fold, X, y, groups)
                - ans_fold: Dict with fold -> {'train': [paths], 'val': [paths]}
                - X: List of video paths
                - y: List of labels
                - groups: List of patient group indices
        """
        K = self.K

        ans_fold = {}

        mapped_class_Dict = self.map_class_num(self.class_num, self.video_path)

        # Process dataset: extract paths, labels, and patient groups
        # X: video path in Path format (e.g., len=1954)
        # y: label list (0, 1, 2, ...) defined by class_num_mapping_Dict
        # groups: unique patient indices (e.g., 54 patients)
        X, y, groups = self.process_cross_validation(mapped_class_Dict)

        sgkf = StratifiedGroupKFold(n_splits=K)

        for i, (train_index, test_index) in enumerate(
            sgkf.split(X=X, y=y, groups=groups)
        ):
            # Use original train/val split from StratifiedGroupKFold
            train_mapped_path = [X[i] for i in train_index]
            val_mapped_path = [X[i] for i in test_index]

            # FIXME: magic move
            train_mapped_path, val_mapped_path = self.magic_move(
                train_mapped_path, val_mapped_path
            )

            # * Save index mapping only, no video file copying
            ans_fold[i] = {
                'train': train_mapped_path,
                'val': val_mapped_path,
            }

        return ans_fold, X, y, groups

    def __call__(self, *args: Any, **kwds: Any) -> Any:

        target_path = self.gait_seg_idx_path / str(self.class_num)

        # * Create index mapping when it doesn't exist or JSON changed
        if not os.path.exists(target_path):

            fold_dataset_idx, *_ = self.prepare()

            json_fold_dataset_idx = copy.deepcopy(fold_dataset_idx)

            for k, v in fold_dataset_idx.items():
                # Convert Path objects to strings for JSON serialization
                json_fold_dataset_idx[k] = {
                    'train': [str(i) for i in v['train']],
                    'val': [str(i) for i in v['val']],
                }

            os.makedirs(target_path, exist_ok=True)
            
            with open(target_path / "index.json", "w") as f:
                json.dump(json_fold_dataset_idx, f, sort_keys=True, indent=4)

        elif os.path.exists(target_path):
            with open(target_path / "index.json", "r") as f:
                fold_dataset_idx = json.load(f)

            # Convert string paths back to Path objects
            for k, v in fold_dataset_idx.items():
                fold_dataset_idx[k] = {
                    'train': [Path(i) for i in v['train']],
                    'val': [Path(i) for i in v['val']],
                }

        else:
            raise ValueError(
                "The index mapping path does not exist, please check the path."
            )

        return fold_dataset_idx

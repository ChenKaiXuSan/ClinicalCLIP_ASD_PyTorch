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

    # NOTE: magic_move 已移除(2026-07)。它给每个非 ASD 患者在 train/val 之间对搬一个
    # 片段,直接制造患者级泄漏:5/5 折、46.8% 的验证样本来自训练见过的患者,而且只发生
    # 在 DHS 与 LCS_HipOA 两类(ASD 被显式跳过),macro 指标被不对称地抬高。
    # 它原本大概是为了让每折的 val 都凑齐三类;现在改用 train/val/test 三分,
    # 内外两层都按患者分组,不需要再搬样本。旧实现见 git 历史。
    @staticmethod
    def _unused_magic_move(train_mapped_path, val_mapped_path):

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

        每折产出 train / val / test 三份,三者按患者分组互不相交:

            外层 StratifiedGroupKFold(K)     -> 留出 test(1/K)
            内层 StratifiedGroupKFold(K-1)   -> 把剩下的开发集切成 train / val

        K=5 时大致是 60 / 20 / 20。**val 只用来选 checkpoint,test 只用来报指标**,
        测试集全程不参与任何决策。之前 val 与 test 是同一批数据,所有 test/* 都是
        "在测试集上挑最好的 epoch 再报测试集成绩",是模型选择后的有偏估计。

        两层都不 shuffle,所以划分是确定的,换机器重建结果一致。

        Returns:
            tuple: (ans_fold, X, y, groups)
                - ans_fold: Dict with fold -> {'train': [paths], 'val': [paths], 'test': [paths]}
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

        for fold, (dev_index, test_index) in enumerate(
            sgkf.split(X=X, y=y, groups=groups)
        ):
            dev_X = [X[j] for j in dev_index]
            dev_y = [y[j] for j in dev_index]
            dev_groups = [groups[j] for j in dev_index]

            inner = StratifiedGroupKFold(n_splits=K - 1)
            train_local, val_local = next(
                inner.split(X=dev_X, y=dev_y, groups=dev_groups)
            )

            ans_fold[fold] = {
                'train': [dev_X[j] for j in train_local],
                'val': [dev_X[j] for j in val_local],
                'test': [X[j] for j in test_index],
            }

        return ans_fold, X, y, groups

    def __call__(self, *args: Any, **kwds: Any) -> Any:

        target_path = self.gait_seg_idx_path / str(self.class_num)

        # * Create index mapping when it doesn't exist or JSON changed
        if not os.path.exists(target_path):

            fold_dataset_idx, *_ = self.prepare()

            json_fold_dataset_idx = {
                # split 名不再写死,加了 test 之后也不用再改这里
                k: {split: [str(p) for p in paths] for split, paths in v.items()}
                for k, v in fold_dataset_idx.items()
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
                    split: [Path(p) for p in paths] for split, paths in v.items()
                }

            # 缓存一旦存在就直接加载,不会重新划分 —— 旧缓存只有 train/val,
            # 拿它跑新代码会静默退回 "val 即 test" 的有偏评估。这里必须拦住。
            missing = [k for k, v in fold_dataset_idx.items() if "test" not in v]
            if missing:
                raise ValueError(
                    f"{target_path / 'index.json'} 是旧格式(缺 test 划分,折 {missing[:3]}...)。"
                    "现在每折需要 train/val/test 三份。请先跑 pegasus/prepare_index.sh 重建"
                    "(旧缓存会被备份,不会删)。"
                )

        else:
            raise ValueError(
                "The index mapping path does not exist, please check the path."
            )

        return fold_dataset_idx

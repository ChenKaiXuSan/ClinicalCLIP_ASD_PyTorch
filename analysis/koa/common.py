#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: common.py
Project: analysis/koa
Author: Kaixu Chen
-----
Comment:
KOA-PD-NM 各脚本共用:受试者表、交叉验证划分、先验测量集合。

划分与 ASD 的做法一致:外层 5 折留出 test,内层把开发集切成 train / val(4 折取第一份),全部按受试者分层
(按 NM / KOA / PD 三类),固定随机种子,所以测量分支、视频分支、骨架网络用的是同一批 train / val / test 受试者。
"""
from __future__ import annotations

import pickle

import numpy as np
from sklearn.model_selection import StratifiedKFold

CLASSES = ["NM", "KOA", "PD"]
# 先验测量集合,写在看任何结果之前:
#   帕金森 = MDS-UPDRS 3.10 步态条目点名的量(与 CARE-PD 用的 8 个量完全相同)
#   膝骨关节炎 = 教科书里的 KOA 步态特征:膝关节屈伸幅度减小、患侧代偿造成的左右不对称(膝幅度、步长)、双支撑期延长
PRIOR_PD = ["step_len", "speed", "cadence", "ank_clear", "toe_clear", "foot_pitch_hs", "arm_swing", "arm_swing_min"]
PRIOR_KOA = ["knee_rom", "knee_rom_asym", "step_len_asym", "double_support"]
PRIOR = PRIOR_PD + PRIOR_KOA


def load_gait(path: str = "logs/koa/gait.pkl") -> dict:
    return pickle.load(open(path, "rb"))


def subject_table(G: dict):
    """受试者级:名字、标签、特征(该受试者各段视频的均值)。"""
    subs = sorted(set(G["subject"]))
    y = np.array([int(G["label"][G["subject"] == s][0]) for s in subs])
    X = np.stack([np.nanmean(G["X"][G["subject"] == s], 0) for s in subs])
    return np.array(subs), y, X


def splits(subs: np.ndarray, y: np.ndarray, seed: int = 0, k: int = 5) -> list[dict]:
    """[{train, val, test}](受试者下标)。"""
    out = []
    for dev, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(subs, y):
        tr, va = next(iter(StratifiedKFold(4, shuffle=True, random_state=seed).split(dev, y[dev])))
        out.append({"train": dev[tr], "val": dev[va], "test": te})
    return out


def balanced_acc(y: np.ndarray, p: np.ndarray, n_class: int = 3) -> float:
    return float(np.mean([(p[y == c] == c).mean() for c in range(n_class) if (y == c).any()]))

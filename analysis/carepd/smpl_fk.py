#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: smpl_fk.py
Project: analysis/carepd
Author: Kaixu Chen
-----
Comment:
CARE-PD 的 SMPL 参数 -> 24 个关节点的 3D 位置(正向运动学)。

CARE-PD 为了匿名把体型参数 beta 全部置零,所有受试者共用同一副标准骨架,所以不需要(有许可限制的)SMPL 模型文件:
用 SMPL 中性模型在 beta = 0 时的静止关节位置 J_REST 与运动学树做正向运动学即可。角度类的量只取决于关节旋转;
长度类的量都按躯干长归一化,J_REST 的毫米级误差不影响结果。

坐标(Canonicalized_SMPL_pickles):x = 左右, y = 向上, z = 前进。
"""
from __future__ import annotations

import numpy as np

PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])
JOINT_NAMES = ["pelvis", "hip_L", "hip_R", "spine1", "kn_L", "kn_R", "spine2", "ank_L", "ank_R", "spine3", "foot_L", "foot_R",
               "neck", "collar_L", "collar_R", "head", "sh_L", "sh_R", "el_L", "el_R", "wr_L", "wr_R", "hand_L", "hand_R"]
# SMPL 中性模型 beta = 0 的静止关节位置(米)
J_REST = np.array([
    [-0.0018, -0.2233, 0.0282], [0.0677, -0.3147, 0.0214], [-0.0695, -0.3139, 0.0239], [-0.0043, -0.1144, 0.0015],
    [0.1020, -0.6899, 0.0169], [-0.1078, -0.6964, 0.0150], [0.0016, 0.0208, 0.0026], [0.0884, -1.0879, -0.0268],
    [-0.0920, -1.0932, -0.0273], [0.0026, 0.0737, 0.0280], [0.1146, -1.1437, 0.0925], [-0.1174, -1.1430, 0.0961],
    [-0.0002, 0.2877, -0.0148], [0.0814, 0.1955, -0.0060], [-0.0791, 0.1926, -0.0106], [0.0051, 0.3526, 0.0365],
    [0.1725, 0.2260, -0.0149], [-0.1758, 0.2251, -0.0197], [0.4323, 0.2132, -0.0424], [-0.4289, 0.2118, -0.0411],
    [0.6813, 0.2222, -0.0435], [-0.6847, 0.2195, -0.0467], [0.7653, 0.2140, -0.0585], [-0.7692, 0.2134, -0.0570],
])
OFFSETS = J_REST.copy()
OFFSETS[1:] = J_REST[1:] - J_REST[PARENTS[1:]]


def rodrigues(aa: np.ndarray) -> np.ndarray:
    """轴角 (..., 3) -> 旋转矩阵 (..., 3, 3)。"""
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.clip(theta, 1e-8, None)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    K = np.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], -1).reshape(*aa.shape[:-1], 3, 3)
    s, c = np.sin(theta)[..., None], np.cos(theta)[..., None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def forward_kinematics(pose: np.ndarray, trans: np.ndarray | None = None) -> np.ndarray:
    """pose (T, 72) 轴角, trans (T, 3) -> 关节点 (T, 24, 3),世界坐标。"""
    T = pose.shape[0]
    R = rodrigues(pose.reshape(T, 24, 3).astype(np.float64))
    G = np.zeros((T, 24, 3, 3)); P = np.zeros((T, 24, 3))
    G[:, 0] = R[:, 0]; P[:, 0] = OFFSETS[0]
    for j in range(1, 24):
        p = PARENTS[j]
        G[:, j] = G[:, p] @ R[:, j]
        P[:, j] = P[:, p] + G[:, p] @ OFFSETS[j]
    if trans is not None:
        P = P + np.asarray(trans, dtype=np.float64)[:, None, :]
    return P


def body_frame(P: np.ndarray) -> np.ndarray:
    """把关节点转到逐帧的身体坐标系:原点 = 骨盆,y 向上,x = 左右(右髋 -> 左髋的水平分量),z = 前进。
    对转身、绕行的行走也成立(不依赖全局前进方向)。返回 (T, 24, 3)。"""
    up = np.array([0.0, 1.0, 0.0])
    lat = P[:, 1] - P[:, 2]
    lat = lat - (lat @ up)[:, None] * up
    lat /= np.clip(np.linalg.norm(lat, axis=1, keepdims=True), 1e-8, None)
    fwd = np.cross(lat, up)             # x × y = z:左 × 上 = 前
    Q = P - P[:, :1]
    return np.stack([np.einsum("tjc,tc->tj", Q, lat), Q[..., 1], np.einsum("tjc,tc->tj", Q, fwd)], -1)

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: train_deep.py
Project: analysis/carepd
Author: Kaixu Chen
-----
Comment:
训练样本量曲线(端到端侧):ST-GCN(SMPL 24 关节图)直接吃关节点序列,与 size_curve.py 用同样的 6 折、
同样的 k、同样的受试者抽样(size_curve.subsample),所以两条曲线逐点可比。

输入:身体坐标系关节点 (x 左右, y 向上, z 前进) + 骨盆水平速度,共 4 通道;2 秒片段(60 帧,不足循环补齐)。
训练:每个 epoch 每次行走随机截一个片段,类别加权交叉熵,AdamW + 余弦;测试:滑窗(步长 30)概率取平均得到行走级预测。
指标:各折预测拼起来的宏平均 F1(0/1/2),与测量侧一致。

    python analysis/carepd/train_deep.py --cohort PD-GaM --reps 3 --out logs/carepd/size_curve_deep_PD-GaM.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_bank import FPS  # noqa: E402
from size_curve import KS, six_folds, subsample  # noqa: E402
from smpl_fk import PARENTS, body_frame, forward_kinematics  # noqa: E402

ROOT = Path("/work/1/SKIING/chenkaixu/data/care_pd")
CLIP = 60


def load_walks(cohort: str, cache: Path):
    f = cache / f"joints_{cohort}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    d = pickle.load(open(ROOT / "Canonicalized_SMPL_pickles" / f"{cohort}_canonical.pkl", "rb"))
    walks = []
    for s, wd in d.items():
        for w, r in wd.items():
            if r.get("UPDRS_GAIT") is None:
                continue
            step = max(1, int(round(int(r["fps"]) / FPS)))
            pose, trans = np.asarray(r["pose"])[::step], np.asarray(r["trans"])[::step]
            if len(pose) < FPS:
                continue
            P = forward_kinematics(pose, trans)
            B = body_frame(P).astype(np.float32)                                   # (T, 24, 3)
            v = np.linalg.norm(np.diff(P[:, 0, [0, 2]], axis=0, prepend=P[:1, 0, [0, 2]]), axis=1).astype(np.float32) * FPS
            x = np.concatenate([B, np.repeat(v[:, None, None], 24, 1)], -1)        # (T, 24, 4)
            walks.append({"x": x, "y": int(r["UPDRS_GAIT"]), "subject": str(s)})
    cache.mkdir(parents=True, exist_ok=True)
    pickle.dump(walks, open(f, "wb"))
    return walks


def adjacency() -> torch.Tensor:
    A = np.eye(24, dtype=np.float32)
    for j, p in enumerate(PARENTS):
        if p >= 0:
            A[j, p] = A[p, j] = 1
    D = A.sum(1)
    return torch.tensor(A / np.sqrt(D[:, None] * D[None, :]))


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1, drop=0.2):
        super().__init__()
        self.gcn = nn.Conv2d(cin, cout, 1)
        self.tcn = nn.Sequential(nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, (9, 1), (stride, 1), (4, 0)), nn.BatchNorm2d(cout), nn.Dropout(drop))
        self.res = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout))

    def forward(self, x, A):                      # x (B, C, T, V)
        y = torch.einsum("bctv,vw->bctw", self.gcn(x), A)
        return F.relu(self.tcn(y) + self.res(x))


class STGCN(nn.Module):
    def __init__(self, n_class: int, cin: int = 4):
        super().__init__()
        self.register_buffer("A", adjacency())
        self.bn = nn.BatchNorm1d(cin * 24)
        self.blocks = nn.ModuleList([Block(cin, 64), Block(64, 64), Block(64, 128, 2), Block(128, 128)])
        self.fc = nn.Linear(128, n_class)

    def forward(self, x):                         # (B, T, V, C)
        b, t, v, c = x.shape
        x = self.bn(x.permute(0, 2, 3, 1).reshape(b, v * c, t)).reshape(b, v, c, t).permute(0, 2, 3, 1)
        for blk in self.blocks:
            x = blk(x, self.A)
        return self.fc(x.mean(dim=(2, 3)))


def crop(x: np.ndarray, start: int) -> np.ndarray:
    idx = (start + np.arange(CLIP)) % len(x)
    return x[idx]


def run(walks, tr_idx, te_idx, n_class, seed, epochs, device):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    ytr = np.array([walks[i]["y"] for i in tr_idx])
    cls = np.bincount(ytr, minlength=n_class).astype(np.float32)
    w = torch.tensor(np.where(cls > 0, len(ytr) / (np.maximum(cls, 1) * (cls > 0).sum()), 0.0), device=device)
    model = STGCN(n_class).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    steps = epochs * max(1, int(np.ceil(len(tr_idx) / 64)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    model.train()
    for _ in range(epochs):
        perm = rng.permutation(tr_idx)
        for b0 in range(0, len(perm), 64):
            ids = perm[b0:b0 + 64]
            if len(ids) < 2:
                continue
            xb = torch.tensor(np.stack([crop(walks[i]["x"], int(rng.integers(len(walks[i]["x"])))) for i in ids]), device=device)
            yb = torch.tensor([walks[i]["y"] for i in ids], device=device)
            loss = F.cross_entropy(model(xb), yb, weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval(); pred = {}
    with torch.no_grad():
        for i in te_idx:
            x = walks[i]["x"]
            starts = list(range(0, max(1, len(x) - CLIP + 1), 30))
            p = torch.softmax(model(torch.tensor(np.stack([crop(x, s) for s in starts]), device=device)), -1).mean(0)
            pred[i] = int(p.argmax())
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--ks", default=None, help="逗号分隔,all 表示全部;默认同 size_curve.KS")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="logs/carepd/joints")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    walks = load_walks(args.cohort, Path(args.cache))
    sub = np.array([w["subject"] for w in walks]); y = np.array([w["y"] for w in walks])
    n_class = int(y.max()) + 1
    folds = six_folds(args.cohort)
    ks = KS if args.ks is None else [None if k == "all" else int(k) for k in args.ks.split(",")]
    print(f"{args.cohort}: 行走 {len(walks)}, 受试者 {len(set(sub))}, 类别 {np.bincount(y).tolist()}, 设备 {device}", flush=True)
    out = {}
    for k in ks:
        vals = []
        for rep in range(args.reps):
            pred = np.full(len(y), -1)
            for fi, (tr_s, te_s) in enumerate(folds):
                tr = np.where(np.isin(sub, subsample(tr_s, k, rep, fi)))[0]; te = np.where(np.isin(sub, te_s))[0]
                if len(set(y[tr])) < 2:
                    pred[te] = y[tr][0]
                    continue
                for i, p in run(walks, tr, te, n_class, 1000 * rep + fi, args.epochs, device).items():
                    pred[i] = p
            ok = pred >= 0
            vals.append(float(f1_score(y[ok], pred[ok], labels=[0, 1, 2], average="macro", zero_division=0)))
        out["all" if k is None else str(k)] = vals
        print(f"k={'全部' if k is None else k}: F1_0-2 {np.mean(vals):.3f} ± {np.std(vals):.3f}  {['%.3f' % v for v in vals]}", flush=True)
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

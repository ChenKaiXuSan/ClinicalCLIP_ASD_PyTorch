#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: train_skel.py
Project: analysis/koa
Author: Kaixu Chen
-----
Comment:
KOA-PD-NM 的骨架网络对照:ST-GCN 直接吃 SAM-3D-Body 的 3D 关节序列(与测量分支同一批关节点),
划分、checkpoint 选择方式与视频分支相同。用途是"把门控的第二个分支从像素换成关节点"这条消融。
模型沿用 analysis/carepd/train_deep.py 的 ST-GCN(SMPL 24 关节图,身体系坐标 + 行走速度 4 通道,2 秒片段)。

输出 logs/koa/video/skel_s<seed>.pkl,格式同 train_video.py。

    python analysis/koa/train_skel.py --seeds 42 1337 2024 7 11
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "carepd"))
from common import balanced_acc, load_gait, splits, subject_table  # noqa: E402
from train_deep import STGCN  # noqa: E402

CLIP, STRIDE = 50, 25          # 25 fps 下 2 秒片段,滑窗步长 1 秒


def crop(x: np.ndarray, start: int) -> np.ndarray:
    return x[(start + np.arange(CLIP)) % len(x)]


def windows(x: np.ndarray) -> np.ndarray:
    return np.stack([crop(x, s) for s in range(0, max(1, len(x) - CLIP + 1), STRIDE)])


@torch.no_grad()
def predict(model, xs: list[np.ndarray], device) -> list[np.ndarray]:
    model.eval()
    return [torch.softmax(model(torch.tensor(windows(x), device=device)), -1).cpu().numpy() for x in xs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gait", default="logs/koa/gait.pkl")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2024, 7, 11])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", default="logs/koa/video")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = load_gait(args.gait)
    subs, y, _ = subject_table(G)
    J = {v: np.nan_to_num(G["joints"][v]).astype(np.float32) for v in G["videos"]}
    vid_of = {s: [v for v, sv in zip(G["videos"], G["subject"]) if sv == s] for s in subs}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        prob = np.full((len(subs), 3), np.nan); fold_of = np.full(len(subs), -1); val_accs, eps = [], []
        for k, sp in enumerate(splits(subs, y)):
            torch.manual_seed(seed * 100 + k); rng = np.random.default_rng(seed * 100 + k)
            tr_v = [(v, y[i]) for i in sp["train"] for v in vid_of[subs[i]]]
            va_v = [(v, y[i]) for i in sp["val"] for v in vid_of[subs[i]]]
            cnt = np.bincount([l for _, l in tr_v], minlength=3).astype(np.float32)
            w = torch.tensor(len(tr_v) / (3 * np.maximum(cnt, 1)), device=device)
            model = STGCN(3).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            nb = int(np.ceil(len(tr_v) / 32))
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * nb)
            best = (-1.0, np.inf, None, -1)
            for ep in range(args.epochs):
                model.train()
                perm = rng.permutation(len(tr_v))
                for b0 in range(0, len(perm), 32):
                    ids = perm[b0: b0 + 32]
                    if len(ids) < 2:
                        continue
                    xb = torch.tensor(np.stack([crop(J[tr_v[j][0]], int(rng.integers(len(J[tr_v[j][0]])))) for j in ids]), device=device)
                    yb = torch.tensor([int(tr_v[j][1]) for j in ids], device=device)
                    loss = F.cross_entropy(model(xb), yb, weight=w)
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                ps = predict(model, [J[v] for v, _ in va_v], device)
                yy = np.concatenate([np.full(len(p), l) for p, (_, l) in zip(ps, va_v)]); pp = np.concatenate(ps)
                acc = balanced_acc(yy, pp.argmax(1)); vloss = float(-np.log(pp[np.arange(len(yy)), yy] + 1e-6).mean())
                if acc > best[0] + 1e-9 or (abs(acc - best[0]) < 1e-9 and vloss < best[1]):
                    best = (acc, vloss, {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}, ep)
            model.load_state_dict(best[2])
            for i in sp["test"]:
                prob[i] = np.concatenate(predict(model, [J[v] for v in vid_of[subs[i]]], device)).mean(0); fold_of[i] = k
            val_accs.append(best[0]); eps.append(best[3])
        print(f"ST-GCN s{seed}: 受试者级平衡准确率 {balanced_acc(y, prob.argmax(1)):.3f}, 逐折 val {np.round(val_accs, 3).tolist()}", flush=True)
        pickle.dump({"subjects": subs, "y": y, "prob": prob, "fold": fold_of, "val_acc": val_accs, "epoch": eps}, open(out / f"skel_s{seed}.pkl", "wb"))


if __name__ == "__main__":
    main()

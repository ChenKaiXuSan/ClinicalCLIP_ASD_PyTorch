#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: train_video.py
Project: analysis/koa
Author: Kaixu Chen
-----
Comment:
KOA-PD-NM 的视频分支:与 ASD 上的 B0 同一个模型和训练设置 —— Kinetics 预训练的 slow_r50,1 秒一段、每段均匀取 8 帧、
224×224、只除以 255,Adam(lr 1e-4, wd 1e-3)+ 余弦,一步 = 一段视频的全部片段,50 轮,按 val 的片段级平衡准确率选 checkpoint。
与 B0 唯一的不同:三类人数是 50 / 30 / 16,交叉熵按训练集类别频率加权(不加权时少数类基本学不到;给视频分支更强的设置,
对门控的检验更保守)。

输入三种(--mode):
  rgb     人物居中的原始裁剪(与 ASD 的形式相同,但这份数据三组的拍摄场地不同,背景是捷径)
  masked  同一裁剪,人物以外全部置黑
  sil     只有人物轮廓(二值),不带衣着、压缩、运动模糊的任何痕迹
输出 logs/koa/video/<mode>_s<seed>.pkl: {"subjects", "y", "prob" (n_sub, 3) 每个受试者在它作为 test 的那一折的平均概率,
"fold", "val_acc" 逐折, "epoch" 逐折选中的轮次}

    python analysis/koa/train_video.py --mode masked --seed 42
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import balanced_acc, load_gait, splits, subject_table  # noqa: E402

FPS, T = 25, 8


def segments(n: int) -> list[np.ndarray]:
    """n 帧 -> 每秒一段、每段 8 帧的帧下标;末尾不足半秒的丢掉,不足一秒的按 round(linspace) 重复补齐。"""
    out = []
    for s in range(0, n, FPS):
        L = min(FPS, n - s)
        if L < FPS // 2 and out:
            break
        out.append(s + np.round(np.linspace(0, L - 1, T)).astype(int))
    return out


def load_video(proc: Path, name: str, valid: tuple[int, int], mode: str) -> torch.Tensor:
    d = np.load(proc / "crops" / f"{name}.npz")
    c, m = d["crops"][valid[0]: valid[1]], d["mask"][valid[0]: valid[1]]
    if mode == "masked":
        c = c * m[..., None]
    elif mode == "sil":
        c = np.repeat((m * 255).astype(np.uint8)[..., None], 3, -1)
    idx = np.stack(segments(len(c)))                              # (S, 8)
    return torch.from_numpy(np.ascontiguousarray(c[idx])).permute(0, 4, 1, 2, 3)   # (S, 3, 8, 224, 224) uint8


def build_model(n_class: int) -> nn.Module:
    m = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
    m.blocks[-1].proj = nn.Linear(m.blocks[-1].proj.in_features, n_class)
    return m


@torch.no_grad()
def predict(model, vids: list[torch.Tensor], device) -> list[np.ndarray]:
    model.eval(); out = []
    for v in vids:
        ps = []
        for i in range(0, len(v), 16):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ps.append(torch.softmax(model(v[i: i + 16].to(device).float() / 255).float(), -1).cpu())
        out.append(torch.cat(ps).numpy())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="/work/1/SKIING/chenkaixu/data/koa_pd_nm/proc")
    ap.add_argument("--gait", default="logs/koa/gait.pkl")
    ap.add_argument("--mode", default="masked", choices=["rgb", "masked", "sil"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--max-seg", type=int, default=16, help="训练时每段视频最多取多少个片段(长录像随机抽)")
    ap.add_argument("--folds", default=None, help="只跑这些折(调试用),逗号分隔")
    ap.add_argument("--out", default="logs/koa/video")
    args = ap.parse_args()
    device = "cuda"
    G = load_gait(args.gait)
    subs, y, _ = subject_table(G)
    proc = Path(args.proc)
    t0 = time.time()
    data = {v: load_video(proc, v, G["valid"][v], args.mode) for v in G["videos"]}
    vid_of = {s: [v for v, sv in zip(G["videos"], G["subject"]) if sv == s] for s in subs}
    print(f"模式 {args.mode} 种子 {args.seed}: 视频 {len(data)}, 片段 {sum(len(v) for v in data.values())}, 载入 {time.time() - t0:.0f}s", flush=True)
    prob = np.full((len(subs), 3), np.nan); fold_of = np.full(len(subs), -1); val_accs, eps = [], []
    todo = range(5) if args.folds is None else [int(k) for k in args.folds.split(",")]
    for k, sp in enumerate(splits(subs, y)):
        if k not in todo:
            continue
        torch.manual_seed(args.seed * 100 + k); rng = np.random.default_rng(args.seed * 100 + k)
        tr_v = [(v, y[i]) for i in sp["train"] for v in vid_of[subs[i]]]
        va_v = [(v, y[i]) for i in sp["val"] for v in vid_of[subs[i]]]
        cnt = np.bincount([l for _, l in tr_v], minlength=3).astype(np.float32)
        w = torch.tensor(len(tr_v) / (3 * np.maximum(cnt, 1)), device=device)
        model = build_model(3).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(tr_v))
        best = (-1.0, np.inf, None, -1)
        for ep in range(args.epochs):
            model.train()
            for j in rng.permutation(len(tr_v)):
                v, l = tr_v[j]
                x = data[v]
                if len(x) > args.max_seg:
                    x = x[np.sort(rng.choice(len(x), args.max_seg, replace=False))]
                x = x.to(device).float() / 255
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit = model(x).float()
                loss = F.cross_entropy(logit, torch.full((len(x),), int(l), device=device), weight=w)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            ps = predict(model, [data[v] for v, _ in va_v], device)
            yy = np.concatenate([np.full(len(p), l) for p, (_, l) in zip(ps, va_v)]); pp = np.concatenate(ps)
            acc = balanced_acc(yy, pp.argmax(1)); vloss = float(-np.log(pp[np.arange(len(yy)), yy] + 1e-6).mean())
            if acc > best[0] + 1e-9 or (abs(acc - best[0]) < 1e-9 and vloss < best[1]):
                best = (acc, vloss, {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}, ep)
        model.load_state_dict(best[2])
        for i in sp["test"]:
            ps = predict(model, [data[v] for v in vid_of[subs[i]]], device)
            prob[i] = np.concatenate(ps).mean(0); fold_of[i] = k
        te = sp["test"]
        val_accs.append(best[0]); eps.append(best[3])
        print(f"  折 {k}: 训练视频 {len(tr_v)}, val 平衡准确率 {best[0]:.3f}(第 {best[3]} 轮), "
              f"test 受试者级 {balanced_acc(y[te], prob[te].argmax(1)):.3f}  用时 {time.time() - t0:.0f}s", flush=True)
        del model, opt
        torch.cuda.empty_cache()
    done = fold_of >= 0
    print(f"== {args.mode} s{args.seed}: 受试者级平衡准确率 {balanced_acc(y[done], prob[done].argmax(1)):.3f} (n={done.sum()})", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.folds is None:
        pickle.dump({"subjects": subs, "y": y, "prob": prob, "fold": fold_of, "val_acc": val_accs, "epoch": eps},
                    open(out / f"{args.mode}_s{args.seed}.pkl", "wb"))


if __name__ == "__main__":
    main()

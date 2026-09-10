#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: eval_qwen_attention.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
Q2:生成式 VLM 自己的注意力和医生一致吗?不训练。

对每个概念 r,用 vlm_prompts.py 的 region_<r> 指令("只看腰椎骨盆,描述它怎么动"),
取回答位置对视频 token 的注意力(最后 K 层、所有头平均)作为该概念的定位图,
与医生区域图算 attn_align —— 与 train_clinical_concept 的 test/attn_align 同口径,
可直接并排 M0 0.537 / Grad-CAM 0.135 / uniform 0.068。

这不需要任何 grounding 损失:如果 VLM 在被要求看某个部位时注意力真的落在那里,
可解释性就是免费的;如果落不到,说明通用 VLM 对步态视频的部位定位不可靠,
grounding 监督仍是必要的。

用法(计算节点,需 eager 注意力,8B 约 5 组 prompt x 2800 段,H100 上约 1-2 小时):
    python analysis/eval_qwen_attention.py --root-path /work/SKIING/chenkaixu/data/asd_dataset \
        --model Qwen/Qwen3-VL-8B-Instruct --img-size 448 --fold 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))
sys.path.insert(0, str(ROOT / "analysis"))

from dataloader.med_attn_map import MedAttnMap, REGIONS  # noqa: E402
from dataloader.whole_video_dataset import LabeledGaitVideoDataset  # noqa: E402
from models.vlm_encoder import VLMTokenEncoder  # noqa: E402
from eval_attention_alignment import alignment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--img-size", type=int, default=448, help="448 -> 14x14 token;224 -> 7x7")
    parser.add_argument("--fold", default="0")
    parser.add_argument("--class-num", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="0 = 整个 test 集")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--region-map-size", type=int, default=28)
    parser.add_argument("--last-layers", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=4, help="一次前向的段数")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"], help="bf16 隐状态误差可达 20%%,默认 fp32")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root_path)
    info = root / "clinical_CLIP_dataset"
    folds = json.load(open(info / "index_mapping" / str(args.class_num) / "index.json"))
    paths = [info / "json_mix" / p.split("json_mix/")[-1] for p in folds[args.fold]["test"]]
    if args.limit:
        paths = paths[: args.limit]

    med = MedAttnMap(str(info / "doctor_result"), str(info / "seg_skeleton_pkl"))
    dataset = LabeledGaitVideoDataset(
        "qwen_attn", paths, img_size=args.img_size, num_samples=args.num_samples,
        attn_map=med, region_supervision=True, region_map_size=args.region_map_size,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    enc = VLMTokenEncoder(
        "qwen3vl", args.model, hidden_dim=256, img_size=args.img_size,
        attn_implementation="eager", dtype=args.dtype,
    ).to(device).eval()

    total = {"qwen": 0.0, "uniform": 0.0}
    weight = 0.0
    per_region = {r: [0.0, 0.0] for r in REGIONS}
    for i in range(len(dataset)):
        sample = dataset[i]
        video = sample["video"].to(device)
        region_map = sample["region_map"].to(device)
        target = sample["region_target"].to(device).unsqueeze(0).expand(video.shape[0], -1)
        maps = []
        for r in REGIONS:
            chunks = [enc.tower.attention_maps(v, f"region_{r}", args.last_layers)
                      for v in video.split(args.chunk)]
            maps.append(torch.cat(chunks, 0))  # (B, t', h', w')
        attn = torch.stack(maps, dim=1)  # (B, R, t', h', w')
        s, w = alignment(attn, region_map, target)
        if s is None:
            continue
        uni = torch.full_like(attn, 1.0 / attn[0, 0].numel())
        u, _ = alignment(uni, region_map, target)
        total["qwen"] += s
        total["uniform"] += u
        weight += w
        # 逐区域:只对医生提到的区域记分
        for k, r in enumerate(REGIONS):
            tk = target[:, k]
            if tk.sum() > 0:
                sk, wk = alignment(attn[:, k:k + 1], region_map[:, k:k + 1], tk.unsqueeze(1))
                if sk is not None:
                    per_region[r][0] += sk
                    per_region[r][1] += wk
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(dataset)}] attn_align={total['qwen']/weight:.3f} uniform={total['uniform']/weight:.3f}", flush=True)

    result = {
        "model": args.model, "img_size": args.img_size, "fold": args.fold, "n_videos": len(dataset),
        "last_layers": args.last_layers,
        "attn_align": total["qwen"] / max(weight, 1e-6),
        "attn_align_uniform": total["uniform"] / max(weight, 1e-6),
        "per_region": {r: (v[0] / v[1] if v[1] > 0 else None) for r, v in per_region.items()},
        "note": "回答位置对视频 token 的注意力,最后 K 层所有头平均;每概念一条 region_<r> 指令",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: eval_qwen_attributes.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
方案 2:VLM 做**临床属性瓶颈**。不要它的隐状态,只要它对一组有临床定义的 yes/no 问题的回答
(models/vlm_prompts.py ATTRIBUTES:躯干前倾、骨盆后倾、膝屈曲……),每段视频得到一个
属性分数向量 logit(yes) - logit(no)。分类交给 analysis/attribute_classifier.py 的逻辑回归。

机制上这才贴合"医生注意力":医生看的就是这些属性。而且能回答审稿人的问题 —— VLM 到底看到了什么。

输出 <out_dir>/<attr>.json:{video_name: {"scores": [逐段], "label": 原始三分类标签, "json": 相对路径}}
每个属性一个作业并行(pegasus/qwen_attr_job.sh),8B bf16 全库约 50 分钟/属性。

    python analysis/eval_qwen_attributes.py --root-path $DATA --attr trunk_forward_lean --out-dir logs/qwen_attributes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))

from dataloader.whole_video_dataset import LabeledGaitVideoDataset  # noqa: E402
from models.vlm_encoder import VLMTokenEncoder  # noqa: E402
from models.vlm_prompts import ATTRIBUTES, attribute_prompt  # noqa: E402


def list_all_videos(root: Path, class_num: int) -> list[Path]:
    index = root / "clinical_CLIP_dataset" / "index_mapping" / str(class_num) / "index.json"
    folds = json.load(open(index))
    first = next(iter(folds.values()))
    paths = sorted({p for split in first.values() for p in split})
    return [root / "clinical_CLIP_dataset" / "json_mix" / p.split("json_mix/")[-1] for p in paths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--img-size", type=int, default=448)
    parser.add_argument("--class-num", type=int, default=2)
    parser.add_argument("--attr", required=True, choices=sorted(ATTRIBUTES), help="属性名")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    root = Path(args.root_path)
    paths = list_all_videos(root, args.class_num)
    if args.limit:
        paths = paths[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.attr}.json"
    done = json.load(open(out_path)) if out_path.is_file() else {}

    dataset = LabeledGaitVideoDataset("attr", paths, img_size=args.img_size, num_samples=args.num_samples)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    enc = VLMTokenEncoder("qwen3vl", args.model, hidden_dim=256, img_size=args.img_size, dtype=args.dtype).to(device).eval()
    prompt = attribute_prompt(args.attr)
    print(f"属性 {args.attr}: {prompt}\n全库 {len(paths)} 条,已完成 {len(done)} 条", flush=True)

    t0 = time.time()
    n_new = 0
    for i in range(len(dataset)):
        name = json.load(open(paths[i]))["video_name"]
        if name in done:
            continue
        sample = dataset[i]
        video = sample["video"].to(device)
        lg = torch.cat([enc.tower.answer_logits(v, prompt, ["yes", "no"]) for v in video.split(args.chunk)], 0)
        done[name] = {
            "scores": (lg[:, 0] - lg[:, 1]).cpu().tolist(),
            "label": int(sample["label"]),
            "json": str(paths[i]).split("json_mix/")[-1],
        }
        n_new += 1
        if n_new % 100 == 0:
            json.dump(done, open(out_path, "w"))
            el = time.time() - t0
            print(f"[{i+1}/{len(dataset)}] {el/60:.1f} min, 预计剩余 {(len(dataset)-i-1)*el/n_new/60:.1f} min", flush=True)
    json.dump(done, open(out_path, "w"))
    print(f"完成: {len(done)} 条 -> {out_path}, {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

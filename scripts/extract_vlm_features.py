#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: extract_vlm_features.py
Project: scripts
Author: Kaixu Chen
-----
Comment:
离线抽取冻结 VLM 的视觉 token,写成训练时直接读取的缓存。

为什么要离线:V0-V2 的视觉塔全部冻结,同一条视频在 100 个 epoch x 5 折 x 3 种子里
会被重复编码 1500 次;so400m 前向比 slow_r50 训练还贵。抽一次之后训练不再解码
像素也不再过视觉塔,单次实验从小时级降到分钟级,三种子方差实验才跑得起。

输出:
    <out_dir>/<video_name>.pt      {"tokens": fp16 (n_chunks, d, T, h, w), "video_name": ...}
    <out_dir>/manifest.json        backend / model / img_size / num_samples / grid / token_dim
训练时 data.feature_cache_dir=<out_dir>,dataset 会校验段数是否与采样计划一致。

帧采样计划与训练完全相同(dataloader.whole_video_dataset._plan_frame_index),
只依赖视频总帧数与 fps,所以缓存与在线编码逐帧对应。

用法(计算节点,HF 权重需已在 HF_HOME 里):
    python scripts/extract_vlm_features.py \
        --root-path /work/SKIING/chenkaixu/data/asd_dataset \
        --backend siglip2 --model google/siglip2-so400m-patch16-384 --img-size 224 \
        --out-dir /work/SKIING/chenkaixu/data/asd_dataset/vlm_cache/siglip2_so400m_224
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from dataloader.whole_video_dataset import LabeledGaitVideoDataset  # noqa: E402
from models.vlm_encoder import VLMTokenEncoder  # noqa: E402


def list_all_videos(root: Path, class_num: int) -> list[Path]:
    """index.json 里任一折的 train+val+test 就是全库(三份互不相交、并集为全体)。"""
    index = root / "clinical_CLIP_dataset" / "index_mapping" / str(class_num) / "index.json"
    if not index.is_file():
        raise FileNotFoundError(f"找不到划分缓存 {index},先跑 pegasus/prepare_index.sh")
    folds = json.load(open(index))
    first = next(iter(folds.values()))
    paths = sorted({p for split in first.values() for p in split})
    # index.json 存的是生成时的绝对路径,只取 json_mix/ 之后的部分接到当前根目录
    fixed = []
    for p in paths:
        tail = p.split("json_mix/")[-1]
        fixed.append(root / "clinical_CLIP_dataset" / "json_mix" / tail)
    return fixed


def collate_keep(batch):
    return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", required=True)
    parser.add_argument("--class-num", type=int, default=2)
    parser.add_argument("--backend", default="siglip2", choices=["siglip2", "internvideo2", "qwen3vl"])
    parser.add_argument("--model", default="google/siglip2-so400m-patch16-384")
    parser.add_argument("--img-size", type=int, default=224, help="送入视觉塔的边长")
    parser.add_argument("--prompt", default="generic",
                        help="qwen3vl 的指令:models/vlm_prompts.py 里的预设名或原文。缓存目录应含 prompt 名")
    parser.add_argument("--layer", type=int, default=-1, help="qwen3vl 取第几层隐状态,-1 为末层")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                        help="qwen3vl 权重精度。bf16 经 28 层累积后隐状态相对误差可达 20%%(tests/check_qwen_batch.py),"
                             "H100 80GB 放得下 8B 的 fp32(32GB),默认 fp32")
    parser.add_argument("--num-samples", type=int, default=8, help="= train.uniform_temporal_subsample_num")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk", type=int, default=64, help="视觉塔一次前向的帧数")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="只抽前 N 条,调试用")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = list_all_videos(root, args.class_num)
    if args.limit:
        videos = videos[: args.limit]
    todo = []
    for p in videos:
        name = json.load(open(p))["video_name"]
        if args.overwrite or not (out_dir / f"{name}.pt").is_file():
            todo.append(p)
    print(f"全库 {len(videos)} 条视频,待抽取 {len(todo)} 条 -> {out_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder = VLMTokenEncoder(
        backend=args.backend, model_name=args.model, hidden_dim=256,
        img_size=args.img_size, trainable_blocks=0, forward_chunk=args.chunk,
        prompt=args.prompt, layer=args.layer, dtype=args.dtype,
    ).to(device).eval()

    from models.vlm_prompts import get_prompt

    manifest = {
        "backend": args.backend, "model": args.model, "img_size": args.img_size,
        "num_samples": args.num_samples, "grid": encoder.grid, "token_dim": encoder.token_dim,
        "dtype": "float16",
        "prompt_name": args.prompt if args.backend == "qwen3vl" else None,
        "prompt": get_prompt(args.prompt) if args.backend == "qwen3vl" else None,
        "layer": args.layer,
        "weights_dtype": args.dtype if args.backend == "qwen3vl" else "float32+fp16 autocast",
    }
    json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=2, ensure_ascii=False)

    if not todo:
        return

    # dataset 输出 [0,1] 的 (n_chunks,3,T,S,S);S 取 img_size,编码器内部不再缩放。
    # 不带 attn_map:抽特征不需要医生标注
    dataset = LabeledGaitVideoDataset(
        "extract", todo, img_size=args.img_size, num_samples=args.num_samples,
    )
    loader = DataLoader(
        dataset, batch_size=1, num_workers=args.num_workers, collate_fn=collate_keep,
        pin_memory=device.type == "cuda",
    )

    t0 = time.time()
    n_done = 0
    n_chunks_total = 0
    # siglip2 用 fp16 autocast 加速;qwen3vl 精度由 --dtype 决定,不再 autocast
    use_autocast = device.type == "cuda" and args.backend == "siglip2"
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=use_autocast):
        for items in loader:
            sample = items[0]
            video = sample["video"].to(device, non_blocking=True)
            tokens, pooled = encoder.encode_with_pooled(video)
            tokens = tokens.to(torch.float16).cpu()
            torch.save(
                {
                    "tokens": tokens,
                    # qwen3vl:回答位置的隐状态(看得到视频与指令);其它后端为 token 均值
                    "pooled": pooled.to(torch.float16).cpu(),
                    "video_name": sample["video_name"], "label": sample["label"],
                },
                out_dir / f"{sample['video_name']}.pt",
            )
            n_done += 1
            n_chunks_total += tokens.shape[0]
            if n_done % 50 == 0 or n_done == len(todo):
                el = time.time() - t0
                print(
                    f"[{n_done}/{len(todo)}] {el/60:.1f} min, {n_chunks_total} 段, "
                    f"{tokens.shape[1:]} / 段, 预计剩余 {(len(todo)-n_done)*el/n_done/60:.1f} min",
                    flush=True,
                )
    print(f"完成:{n_done} 条视频,{n_chunks_total} 段,{(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

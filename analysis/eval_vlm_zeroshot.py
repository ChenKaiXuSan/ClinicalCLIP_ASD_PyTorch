#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: eval_vlm_zeroshot.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
V2_vlm_zeroshot:不训练,直接问 SigLIP 2 "医生关注腰椎骨盆时会看哪里",看它的
patch token 与概念 prompt 的相似度图能不能落到医生标注的区域上。

指标与 train_clinical_concept 的 test/attn_align 同口径(alignment() 来自
eval_attention_alignment.py),可直接与 M0 的 0.537、Grad-CAM 的 0.135、
uniform 的 0.068 并排。附带零样本区域存在性 AP(每个概念取相似度图的最大值当分数)。

patch token 与文本的对齐方式:SigLIP 的图像 embedding 是注意力池化 head 的输出,
patch token 本身不在文本空间。这里用 MaskCLIP 式近似 —— 把每个 patch token
当作 head 的池化结果走一遍 value 投影 + 输出投影 + 残差 MLP,得到逐 patch 的
"伪 pooled embedding",再与文本 embedding 算余弦。这是近似而非模型原生输出,
结论里要注明。

用法(计算节点):
    python analysis/eval_vlm_zeroshot.py --root-path /work/SKIING/chenkaixu/data/asd_dataset \
        --model google/siglip2-so400m-patch16-384 --img-size 224 --fold 0 --limit 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))
sys.path.insert(0, str(ROOT / "analysis"))

from dataloader.med_attn_map import MedAttnMap, REGIONS  # noqa: E402
from dataloader.whole_video_dataset import LabeledGaitVideoDataset  # noqa: E402
from models.clinical_concept import CONCEPT_PROMPTS  # noqa: E402
from models.vlm_encoder import _normalize  # noqa: E402
from eval_attention_alignment import alignment  # noqa: E402


def _average_precision(score: torch.Tensor, target: torch.Tensor) -> float:
    score = score.flatten()
    target = target.flatten().float()
    if target.sum() == 0:
        return float("nan")
    order = torch.argsort(score, descending=True)
    hit = target[order]
    tp = torch.cumsum(hit, dim=0)
    precision = tp / torch.arange(1, len(hit) + 1, dtype=tp.dtype)
    return float((precision * hit).sum() / target.sum())


class SigLIPZeroShot:
    def __init__(self, model_name: str, img_size: int, device: torch.device) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        vcfg = self.model.config.vision_config
        self.native = int(vcfg.image_size)
        self.img_size = img_size
        self.grid = img_size // int(vcfg.patch_size)

        batch = self.tok(
            list(CONCEPT_PROMPTS), padding="max_length", max_length=64,
            truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            self.text = F.normalize(self.model.text_model(**batch).pooler_output.float(), dim=-1)  # (R, d)

    @torch.no_grad()
    def patch_embed(self, frames: torch.Tensor) -> torch.Tensor:
        """frames (N,3,S,S) in [0,1] -> 文本空间的逐 patch embedding (N, L, d)。"""
        if frames.shape[-1] != self.img_size:
            frames = F.interpolate(frames, size=(self.img_size,) * 2, mode="bilinear",
                                   align_corners=False, antialias=True)
        frames = _normalize(frames, "siglip2")
        vm = self.model.vision_model
        tokens = vm(pixel_values=frames, interpolate_pos_encoding=self.img_size != self.native).last_hidden_state
        head = vm.head  # SiglipMultiheadAttentionPoolingHead
        attn = head.attention
        d = tokens.shape[-1]
        w_v = attn.in_proj_weight[2 * d:]
        b_v = attn.in_proj_bias[2 * d:] if attn.in_proj_bias is not None else None
        v = F.linear(tokens, w_v, b_v)
        o = attn.out_proj(v)
        o = o + head.mlp(head.layernorm(o))
        return F.normalize(o.float(), dim=-1)

    @torch.no_grad()
    def concept_maps(self, video: torch.Tensor, temperature: float) -> torch.Tensor:
        """video (B,3,T,S,S) -> (B, R, T, h, w) 每个概念一张、和为 1 的注意力分布。"""
        b, c, t = video.shape[:3]
        frames = video.transpose(1, 2).reshape(b * t, c, *video.shape[-2:])
        emb = torch.cat([self.patch_embed(ch) for ch in frames.split(64)], dim=0)  # (BT, L, d)
        sim = torch.einsum("nld,rd->nrl", emb, self.text)  # (BT, R, L)
        sim = sim.view(b, t, len(REGIONS), -1).permute(0, 2, 1, 3).reshape(b, len(REGIONS), -1)
        attn = (sim / temperature).softmax(dim=-1)
        return attn.view(b, len(REGIONS), t, self.grid, self.grid), sim.view(b, len(REGIONS), t, -1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", required=True)
    parser.add_argument("--model", default="google/siglip2-so400m-patch16-384")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--fold", default="0")
    parser.add_argument("--class-num", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="0 = 整个 test 集")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--region-map-size", type=int, default=28)
    parser.add_argument("--temperature", type=float, default=0.02,
                        help="相似度 softmax 温度;越小注意力越尖")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None, help="结果 json 路径")
    args = parser.parse_args()

    root = Path(args.root_path)
    info = root / "clinical_CLIP_dataset"
    folds = json.load(open(info / "index_mapping" / str(args.class_num) / "index.json"))
    paths = []
    for p in folds[args.fold]["test"]:
        paths.append(info / "json_mix" / p.split("json_mix/")[-1])
    if args.limit:
        paths = paths[: args.limit]

    med = MedAttnMap(str(info / "doctor_result"), str(info / "seg_skeleton_pkl"))
    dataset = LabeledGaitVideoDataset(
        "zeroshot", paths, img_size=args.img_size, num_samples=args.num_samples,
        attn_map=med, region_supervision=True, region_map_size=args.region_map_size,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    zs = SigLIPZeroShot(args.model, args.img_size, device)

    total = {"zeroshot": 0.0, "uniform": 0.0}
    weight = 0.0
    region_scores, region_targets = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        video = sample["video"].to(device)
        region_map = sample["region_map"].to(device)
        target = sample["region_target"].to(device).unsqueeze(0).expand(video.shape[0], -1)
        attn, sim = zs.concept_maps(video, args.temperature)

        s, w = alignment(attn, region_map, target)
        if s is None:
            continue
        uni = torch.full_like(attn, 1.0 / attn[0, 0].numel())
        u, _ = alignment(uni, region_map, target)
        total["zeroshot"] += s
        total["uniform"] += u
        weight += w
        # 零样本存在性:每个概念取相似度图的最大值(视频内所有段取均值)
        region_scores.append(sim.amax(dim=-1).amax(dim=-1).mean(dim=0).cpu())
        region_targets.append(sample["region_target"].float())
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(dataset)}] attn_align={total['zeroshot']/weight:.3f} "
                  f"uniform={total['uniform']/weight:.3f}", flush=True)

    result = {
        "model": args.model, "img_size": args.img_size, "fold": args.fold,
        "n_videos": len(dataset), "temperature": args.temperature,
        "attn_align": total["zeroshot"] / max(weight, 1e-6),
        "attn_align_uniform": total["uniform"] / max(weight, 1e-6),
        "region_ap": _average_precision(torch.stack(region_scores), torch.stack(region_targets) > 0)
        if region_scores else float("nan"),
        "note": "patch token 经 MaskCLIP 式近似投影到文本空间,非模型原生输出",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

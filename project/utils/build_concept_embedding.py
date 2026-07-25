#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: build_concept_embedding.py
Project: utils
Author: Kaixu Chen
-----
Comment:
把临床概念的文本描述编码成向量,供 models/clinical_concept.py 的 ConceptBank 加载。

产出 (R, dim) 的 .pt,行顺序与 dataloader.med_attn_map.REGIONS 严格一致。
训练时用 model.concept_text_embedding=<路径> 指定。

用法:
    python project/utils/build_concept_embedding.py \
        --model openai/clip-vit-base-patch32 \
        --output /mnt/data/xchen/asd_data/concepts/clip_vit_b32.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.clinical_concept import CONCEPT_PROMPTS, REGIONS  # noqa: E402


def encode_clip(model_name: str, prompts: list[str]) -> torch.Tensor:
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModel.from_pretrained(model_name, use_safetensors=True).eval()

    batch = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**batch)
    # CLIP 的句向量取 EOS 位置的 pooler_output
    return out.pooler_output.float()


def encode_bert(model_name: str, prompts: list[str]) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, use_safetensors=True).eval()

    batch = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**batch)
    # 掩码均值池化,比直接取 [CLS] 更稳
    mask = batch["attention_mask"].unsqueeze(-1).float()
    summed = (out.last_hidden_state * mask).sum(dim=1)
    return (summed / mask.sum(dim=1).clamp_min(1e-6)).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--encoder", choices=["clip", "bert"], default=None,
        help="留空则按模型名自动判断",
    )
    args = parser.parse_args()

    kind = args.encoder or ("clip" if "clip" in args.model.lower() else "bert")
    encode = encode_clip if kind == "clip" else encode_bert

    print(f"编码器 {kind} / 模型 {args.model}")
    for region, prompt in zip(REGIONS, CONCEPT_PROMPTS):
        print(f"  {region:16s} <- {prompt}")

    embedding = encode(args.model, CONCEPT_PROMPTS)
    assert embedding.shape[0] == len(REGIONS), "行数必须与 REGIONS 一致"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embedding, out_path)

    # 概念之间必须可区分,否则 grounding 学不出差异
    normed = torch.nn.functional.normalize(embedding, dim=-1)
    sim = normed @ normed.t()
    off_diag = sim[~torch.eye(len(REGIONS), dtype=torch.bool)]
    print(f"\n已保存 {tuple(embedding.shape)} -> {out_path}")
    print(f"概念间余弦相似度: 均值 {off_diag.mean():.3f}, 最大 {off_diag.max():.3f}")
    if off_diag.max() > 0.98:
        print("⚠ 概念向量过于接近,交叉注意力可能无法区分不同区域")


if __name__ == "__main__":
    main()

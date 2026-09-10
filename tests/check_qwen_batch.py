#!/usr/bin/env python3
"""Qwen3-VL 批量前向与单段前向是否一致 —— 精度选择的依据。

    HF_HOME=... python tests/check_qwen_batch.py sdpa float32
    HF_HOME=... python tests/check_qwen_batch.py eager bfloat16

2026-09-10 登录节点 CPU 实测(2B):
    sdpa  float32   batch2 vs single: 0.0000 / 0.0000   -> 批处理逻辑正确
    sdpa  bfloat16  batch2 vs single: 0.2255 / 0.1671
    eager bfloat16  batch2 vs single: 0.2054 / 0.2247   -> 与注意力实现无关,是 bf16 经 28 层累积的误差
所以离线抽特征默认 fp32(pegasus/extract_job.sh DTYPE)。GPU 上 bf16 用 fp32 累加会好得多,
但没有实测前不要默认它。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))
from models.vlm_encoder import VLMTokenEncoder  # noqa: E402

attn = sys.argv[1] if len(sys.argv) > 1 else "sdpa"
dtype = sys.argv[2] if len(sys.argv) > 2 else "float32"
model = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3-VL-2B-Instruct"
torch.manual_seed(0)
enc = VLMTokenEncoder("qwen3vl", model, hidden_dim=256, img_size=224,
                      prompt="clinical", attn_implementation=attn, dtype=dtype).eval()
rel = lambda x, y: float((x - y).norm() / x.norm())  # noqa: E731
video = torch.rand(2, 3, 8, 224, 224)
b1 = enc.tower.forward_video(video)
s0 = enc.tower.forward_video(video[:1])
s1 = enc.tower.forward_video(video[1:])
print(f"[{attn} {dtype}] batch2[0] vs single0: {rel(b1[0:1], s0):.4f}  batch2[1] vs single1: {rel(b1[1:2], s1):.4f}")

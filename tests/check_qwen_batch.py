#!/usr/bin/env python3
"""Qwen3-VL 批量前向与单段前向是否一致 —— 精度选择的依据。

    python tests/check_qwen_batch.py sdpa float32  [model]
    python tests/check_qwen_batch.py sdpa bfloat16 [model]

有 GPU 就在 GPU 上跑(pegasus/gpu_check_job.sh)。

2026-09-10 登录节点 CPU 实测(2B):
    sdpa  float32   batch2 vs single: 0.0000 / 0.0000   -> 批处理逻辑正确
    sdpa  bfloat16  batch2 vs single: 0.2255 / 0.1671
    eager bfloat16  batch2 vs single: 0.2054 / 0.2247   -> 与注意力实现无关,是 bf16 经 28 层累积的误差
所以离线抽特征默认 fp32(pegasus/extract_job.sh DTYPE)。GPU 上 bf16 用 fp32 累加会好得多,
本脚本在 GPU 上的输出决定要不要改回 bf16(阈值:相对差 < 0.01)。
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))
from models.vlm_encoder import VLMTokenEncoder  # noqa: E402

attn = sys.argv[1] if len(sys.argv) > 1 else "sdpa"
dtype = sys.argv[2] if len(sys.argv) > 2 else "float32"
model = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3-VL-2B-Instruct"
img = int(sys.argv[4]) if len(sys.argv) > 4 else 224
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
t0 = time.time()
enc = VLMTokenEncoder("qwen3vl", model, hidden_dim=256, img_size=img,
                      prompt="clinical", attn_implementation=attn, dtype=dtype).to(device).eval()
load_s = time.time() - t0
rel = lambda x, y: float((x - y).norm() / x.norm())  # noqa: E731
video = torch.rand(2, 3, 8, img, img, device=device)
if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.time()
b1 = enc.tower.forward_video(video)
if device.type == "cuda":
    torch.cuda.synchronize()
fwd_s = time.time() - t0
s0 = enc.tower.forward_video(video[:1])
s1 = enc.tower.forward_video(video[1:])
mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
print(
    f"[{model} {attn} {dtype} img={img} {device.type}] "
    f"batch2[0] vs single0: {rel(b1[0:1], s0):.4f}  batch2[1] vs single1: {rel(b1[1:2], s1):.4f}  "
    f"| load {load_s:.0f}s, 2 段前向 {fwd_s:.2f}s, 峰值显存 {mem:.1f} GB, token {tuple(b1.shape[1:])}",
    flush=True,
)

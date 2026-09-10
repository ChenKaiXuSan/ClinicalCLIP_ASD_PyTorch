#!/usr/bin/env python3
"""Qwen3-VL 后端的冒烟测试。登录节点 CPU 用 2B 模型约 5-10 分钟。

    HF_HOME=/work/SKIING/chenkaixu/hf_cache HF_HUB_OFFLINE=1 python tests/smoke_qwen.py [--model Qwen/Qwen3-VL-2B-Instruct]

覆盖:token 网格形状、pooled、指令是否真的改变视频 token、注意力图、零样本回答 logit、
以及 vlm_probe(probe_pool=pooled)一步前向。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))

from models.vlm_encoder import VLMTokenEncoder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()
    torch.manual_seed(0)

    t0 = time.time()
    # fp32:2B 约 9GB,登录节点 16GB 上限内;bf16 在 CPU 上误差太大(tests/check_qwen_batch.py)
    enc = VLMTokenEncoder(
        "qwen3vl", args.model, hidden_dim=256, img_size=args.img_size,
        prompt="clinical", attn_implementation="eager", dtype="float32",
    ).eval()
    print(f"[0] 加载 {round(time.time() - t0)} s; token_dim {enc.token_dim} grid {enc.grid}")

    video = torch.rand(2, 3, 8, args.img_size, args.img_size)
    t0 = time.time()
    tok, pooled = enc.encode_with_pooled(video)
    g = enc.grid
    assert tok.shape == (2, enc.token_dim, 4, g, g), tok.shape
    assert pooled.shape == (2, enc.token_dim), pooled.shape
    print(f"[1] tokens {tuple(tok.shape)} pooled {tuple(pooled.shape)} 耗时 {round(time.time() - t0)} s")

    # LLM 隐状态量级达数百(Qwen 有 massive activation),bf16 下绝对差可到几十,只看相对差
    tok2 = enc(video, return_raw=True)[1]
    noise = float((tok - tok2).norm() / tok.norm())
    assert noise < 0.05, f"同一输入两次前向相对差 {noise:.3f}"
    print(f"[2] encode_raw 与 encode_with_pooled 一致(bf16 噪声相对差 {noise:.4f})")

    tok_g = enc.tower.forward_video(video, prompt="generic")
    rel = float((tok - tok_g).norm() / tok.norm())
    assert rel > max(3 * noise, 1e-3), f"指令没有改变视频 token(差 {rel:.4f} vs 噪声 {noise:.4f})—— 指令是否放在了视频之后?"
    print(f"[3] 指令调制:clinical vs generic 的 token 相对差 {rel:.3f}(噪声 {noise:.4f})")

    m = enc.tower.attention_maps(video, "region_lumbar_pelvis", last_layers=4)
    assert m.shape == (2, 4, g, g), m.shape
    assert abs(float(m[0].sum()) - 1.0) < 1e-3
    print(f"[4] 注意力图 {tuple(m.shape)},和为 1,最大格 {float(m[0].max()):.3f}")

    lg = enc.tower.answer_logits(video, "diagnose", ["yes", "no"])
    assert lg.shape == (2, 2)
    print(f"[5] 零样本回答 logit yes/no: {[[round(x, 2) for x in r] for r in lg.tolist()]}")

    from omegaconf import OmegaConf
    from trainer.train_vlm_probe import VLMProbe

    cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
    cfg.model.backbone = "vlm_probe"
    cfg.model.token_backbone = "vlm"
    cfg.model.vlm_backend = "qwen3vl"
    cfg.model.vlm_name = args.model
    cfg.model.probe_pool = "pooled"
    cfg.model.model_class_num = 2
    probe = VLMProbe(cfg).eval()
    out = probe(raw_tokens=tok, pooled=pooled)
    assert out["logits"].shape == (2, 2)
    print(f"[6] vlm_probe(pooled) logits {tuple(out['logits'].shape)}")
    print("ALL OK")


if __name__ == "__main__":
    main()

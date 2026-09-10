#!/usr/bin/env python3
"""VLM backbone 的形状级冒烟测试,登录节点 CPU 即可跑(用 siglip2-base 约 2 分钟)。

    HF_HOME=/work/SKIING/chenkaixu/hf_cache python tests/smoke_vlm.py \
        --root-path /work/SKIING/chenkaixu/data/asd_dataset

覆盖:编码器输出形状 / concept 网络在线与缓存两条路一致 / 探针前向 /
真实视频抽特征 -> 缓存 -> dataset 读回 -> collate -> trainer 一步。
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))

from models.clinical_concept import ClinicalConceptNet  # noqa: E402
from models.vlm_encoder import VLMTokenEncoder  # noqa: E402
from trainer.train_vlm_probe import VLMProbe  # noqa: E402


def make_cfg(root: str, model: str, **model_overrides):
    cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
    cfg.paths.root_path = root
    cfg.model.backbone = "concept"
    cfg.model.token_backbone = "vlm"
    cfg.model.vlm_backend = "siglip2"
    cfg.model.vlm_name = model
    cfg.model.model_class_num = 2
    cfg.log_path = tempfile.mkdtemp()
    for k, v in model_overrides.items():
        cfg.model[k] = v
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", required=True)
    parser.add_argument("--model", default="google/siglip2-base-patch16-224")
    args = parser.parse_args()
    torch.manual_seed(0)

    # 1. 编码器形状
    enc = VLMTokenEncoder("siglip2", args.model, hidden_dim=256, img_size=224).eval()
    video = torch.rand(2, 3, 8, 224, 224)
    tokens, raw = enc(video, return_raw=True)
    grid = 224 // enc.tower.patch
    assert raw.shape == (2, enc.token_dim, 8, grid, grid), raw.shape
    assert tokens.shape == (2, 256, 8, grid, grid), tokens.shape
    assert not any(p.requires_grad for p in enc.tower.parameters())
    print(f"[1] 编码器 ok: raw {tuple(raw.shape)} tokens {tuple(tokens.shape)}")

    # 部分解冻
    enc_ft = VLMTokenEncoder("siglip2", args.model, hidden_dim=256, img_size=224, trainable_blocks=2)
    n_train = sum(p.numel() for p in enc_ft.tower.parameters() if p.requires_grad)
    assert n_train > 0
    print(f"[1b] 解冻最后 2 个 block: {n_train/1e6:.1f}M 可训练参数")
    del enc_ft

    # 2. concept 网络:在线 vs 缓存两条路输出一致
    cfg = make_cfg(args.root_path, args.model)
    net = ClinicalConceptNet(cfg).eval()
    with torch.no_grad():
        out_online = net(video)
        out_cached = net(raw_tokens=raw.half())
    assert out_online["attn"].shape == (2, 5, 8, grid, grid), out_online["attn"].shape
    diff = (out_online["logits"] - out_cached["logits"]).abs().max().item()
    assert diff < 5e-2, f"在线与缓存 logits 差 {diff}"
    print(f"[2] concept 网络 ok: attn {tuple(out_online['attn'].shape)}, 在线/缓存 logits 最大差 {diff:.4f}")

    # 3. 探针
    probe = VLMProbe(make_cfg(args.root_path, args.model, backbone="vlm_probe")).eval()
    with torch.no_grad():
        p_out = probe(raw_tokens=raw)
    assert p_out["logits"].shape == (2, 2)
    n_train = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    print(f"[3] 探针 ok: logits {tuple(p_out['logits'].shape)}, 可训练参数 {n_train}(含未用的 token_proj)")

    # 4. 真实视频:抽特征 -> 缓存 -> dataset -> collate -> trainer 一步
    from dataloader.data_loader import WalkDataModule
    from dataloader.med_attn_map import MedAttnMap
    from dataloader.whole_video_dataset import LabeledGaitVideoDataset
    from trainer.train_clinical_concept import ClinicalConceptModule

    info = Path(args.root_path) / "clinical_CLIP_dataset"
    index = json.load(open(info / "index_mapping" / "2" / "index.json"))
    paths = [str(info / "json_mix" / p.split("json_mix/")[-1]) for p in index["0"]["test"][:2]]
    cache_dir = Path(tempfile.mkdtemp())
    ds = LabeledGaitVideoDataset("smoke", paths, img_size=224, num_samples=8)
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            t = enc.encode_raw(s["video"]).half()
            torch.save({"tokens": t, "video_name": s["video_name"]}, cache_dir / f"{s['video_name']}.pt")
            print(f"    抽取 {s['video_name']}: video {tuple(s['video'].shape)} -> tokens {tuple(t.shape)}")

    cfg = make_cfg(args.root_path, args.model)
    cfg.data.feature_cache_dir = str(cache_dir)
    cfg.data.num_workers = 0
    cfg.train.attn_map = True
    cfg.train.experiment = "smoke"
    dm = WalkDataModule(cfg, {"train": paths, "val": paths, "test": paths})
    dm.setup()
    batch = next(iter(dm.test_dataloader()))
    assert "tokens" in batch and "video" not in batch, batch.keys()
    assert batch["region_map"].shape[0] == batch["tokens"].shape[0]
    print(f"[4] 缓存 dataset ok: tokens {tuple(batch['tokens'].shape)} region_map {tuple(batch['region_map'].shape)} label {batch['label'].tolist()}")

    module = ClinicalConceptModule(cfg)
    loss = module._shared_step(batch, "train")
    loss.backward()
    grads = [n for n, p in module.named_parameters() if p.grad is not None]
    assert grads and not any("tower" in n for n in grads), grads[:5]
    print(f"[5] trainer 一步 ok: loss {loss.item():.3f}, {len(grads)} 个参数有梯度, 视觉塔无梯度")
    print("ALL OK")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""离线缓存 -> DataModule -> collate -> 探针(pooled/mean)与 concept 各一步反传,且不加载视觉塔。

    python tests/smoke_cache_step.py <DATA_ROOT> <cache_dir>

cache_dir 用 scripts/extract_vlm_features.py --limit 2 生成即可(任意后端)。
"""
import json, sys, tempfile
from pathlib import Path
import torch
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))
from dataloader.data_loader import WalkDataModule
from trainer.train_vlm_probe import VLMProbeModule
from trainer.train_clinical_concept import ClinicalConceptModule

root, cache = sys.argv[1], sys.argv[2]
names = [p.stem for p in Path(cache).glob("*.pt")]
info = Path(root) / "clinical_CLIP_dataset"
paths = [str(p) for p in (info / "json_mix").rglob("*.json") if json.load(open(p))["video_name"] in names]
print("videos:", names, "json:", len(paths))

cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
cfg.paths.root_path = root
cfg.model.token_backbone = "vlm"; cfg.model.vlm_backend = "qwen3vl"; cfg.model.model_class_num = 2
cfg.data.feature_cache_dir = cache; cfg.data.num_workers = 0; cfg.train.experiment = "smoke"; cfg.log_path = tempfile.mkdtemp()
for backbone, pool in [("vlm_probe", "pooled"), ("vlm_probe", "mean"), ("concept", None)]:
    cfg.model.backbone = backbone
    if pool: cfg.model.probe_pool = pool
    dm = WalkDataModule(cfg, {"train": paths, "val": paths, "test": paths}); dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert "tokens" in batch and "pooled" in batch and "video" not in batch, batch.keys()
    module = VLMProbeModule(cfg) if backbone == "vlm_probe" else ClinicalConceptModule(cfg)
    enc = module.model.encoder if backbone == "vlm_probe" else module.model.backbone
    assert enc.tower is None, "缓存模式不应加载模型"
    loss = module._shared_step(batch, "train"); loss.backward()
    n = sum(p.numel() for p in module.parameters() if p.grad is not None)
    print(f"{backbone}/{pool}: tokens {tuple(batch['tokens'].shape)} pooled {tuple(batch['pooled'].shape)} loss {loss.item():.3f} grad params {n}")
print("ALL OK")

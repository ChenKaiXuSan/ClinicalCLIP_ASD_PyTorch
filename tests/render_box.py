#!/usr/bin/env python3
"""把 visual_prompt=box_lumbar 的帧渲染成 PNG,肉眼检查红框是否落在腰椎骨盆上。

    python tests/render_box.py --root-path $DATA --out /tmp/box.png
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))
from dataloader.med_attn_map import MedAttnMap  # noqa: E402
from dataloader.whole_video_dataset import LabeledGaitVideoDataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root-path", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=4, help="取几条视频(每条第一段第一帧)")
args = ap.parse_args()
info = Path(args.root_path) / "clinical_CLIP_dataset"
index = json.load(open(info / "index_mapping/2/index.json"))
paths = [str(info / "json_mix" / p.split("json_mix/")[-1]) for p in index["0"]["test"]]
paths = paths[:: max(1, len(paths) // args.n)][: args.n]
med = MedAttnMap(str(info / "doctor_result"), str(info / "seg_skeleton_pkl"))
ds = LabeledGaitVideoDataset("render", paths, img_size=448, num_samples=8, attn_map=med, visual_prompt="box_lumbar")
tiles = []
for i in range(len(ds)):
    v = ds[i]["video"]  # (S,3,T,448,448)
    frames = [v[0, :, t] for t in (0, 4)]  # 第一段的第 0、4 帧
    tiles.append(torch.cat(frames, dim=2))
grid = torch.cat(tiles, dim=1)  # 纵向拼
img = (grid.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
Image.fromarray(img).save(args.out)
print("saved", args.out, img.shape, [Path(p).stem for p in paths])

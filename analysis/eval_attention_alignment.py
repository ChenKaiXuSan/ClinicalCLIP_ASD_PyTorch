#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: eval_attention_alignment.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
可解释性对照:模型的注意力与医生标注的对齐程度,和几个参照基线放在一起比。

为什么必须做:如果只报"我们的注意力与医生对齐度 = 0.4",审稿人一定会问
"随便一个模型做 Grad-CAM 是不是也有 0.35?"。所以这里同时给出:

  uniform   均匀注意力(零信息下界)
  random    随机注意力(随机性下界)
  center    中心先验(人在画面中央的平凡解)
  gradcam   纯 3D CNN 基线的 Grad-CAM(事后解释方法的对照)
  model     concept 架构自己预测的注意力

对齐度定义与 train_clinical_concept._attn_alignment 一致:
    score = Σ_n A(n) · M̂(n),  M̂ = M / max(M),A 为归一化到和为 1 的分布
该式对 A 是线性的,不会因为注意力更集中而被惩罚(软 IoU 会,已弃用)。

用法:
    python analysis/eval_attention_alignment.py \
        --ckpt logs/train/B0_3dcnn/.../checkpoint/0/xxx.ckpt \
        --fold 0 --root-path /mnt/data/xchen/asd_data --limit 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from dataloader.med_attn_map import MedAttnMap, REGIONS  # noqa: E402
from dataloader.whole_video_dataset import LabeledGaitVideoDataset  # noqa: E402


def alignment(attn: torch.Tensor, region_map: torch.Tensor, target: torch.Tensor):
    """attn: (B,R,T,H,W) 每个概念一张、和为 1;region_map: (B,R,t,h,w);target: (B,R)"""
    b, r = attn.shape[:2]
    if region_map.shape[2:] != attn.shape[2:]:
        region_map = F.interpolate(
            region_map.reshape(b * r, 1, *region_map.shape[2:]),
            size=attn.shape[2:], mode="trilinear", align_corners=False,
        ).reshape(b, r, *attn.shape[2:])

    a = attn.reshape(b, r, -1)
    a = a / a.sum(-1, keepdim=True).clamp_min(1e-8)
    m = region_map.reshape(b, r, -1)
    m = m / m.amax(-1, keepdim=True).clamp_min(1e-6)

    score = (a * m).sum(-1)
    weight = target * (m.sum(-1) > 1e-6)
    if weight.sum() <= 0:
        return None, None
    return float((score * weight).sum()), float(weight.sum())


class GradCAM3D:
    """3D CNN 的 Grad-CAM。取最后一个卷积 stage 的激活与梯度加权求和。

    产出的是"与类别相关的重要性图",与医生区域图对分,用来回答:内建的概念
    注意力是否比事后解释更贴近临床关注点。
    """

    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module) -> None:
        self.model = model
        self.activations = None
        self.gradients = None
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, video: torch.Tensor, target_class: torch.Tensor) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(video)
        if isinstance(logits, dict):
            logits = logits["logits"]
        selected = logits.gather(1, target_class[:, None]).sum()
        selected.backward()

        # 通道权重 = 梯度的时空平均
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        return cam.detach()  # (B, 1, T', H', W')


def make_reference_maps(shape, device, kind: str) -> torch.Tensor:
    """构造参照注意力,形状 (B, 1, T, H, W),和为 1。"""
    b, _, t, h, w = shape
    if kind == "uniform":
        cam = torch.ones(b, 1, t, h, w, device=device)
    elif kind == "random":
        cam = torch.rand(b, 1, t, h, w, device=device)
    elif kind == "center":
        yy = torch.linspace(-1, 1, h, device=device)[:, None]
        xx = torch.linspace(-1, 1, w, device=device)[None, :]
        gauss = torch.exp(-(yy**2 + xx**2) / 0.5)
        cam = gauss[None, None, None].expand(b, 1, t, h, w).contiguous()
    else:
        raise ValueError(kind)
    return cam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", default="/mnt/data/xchen/asd_data")
    parser.add_argument("--fold", default="0")
    parser.add_argument("--ckpt", default=None, help="3dcnn 基线 checkpoint,用于 Grad-CAM")
    parser.add_argument("--limit", type=int, default=60, help="评估多少条视频")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--region-map-size", type=int, default=28)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    info = Path(args.root_path) / "clinical_CLIP_dataset"
    folds = json.load(open(info / "index_mapping/3/index.json"))
    paths = [Path(p) for p in folds[args.fold]["val"]][: args.limit]

    med = MedAttnMap(str(info / "doctor_result"), str(info / "seg_skeleton_pkl"))
    dataset = LabeledGaitVideoDataset(
        "eval", paths, img_size=args.img_size, num_samples=args.num_samples,
        attn_map=med, region_supervision=True, region_map_size=args.region_map_size,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cam_engine = None
    if args.ckpt:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))
        from omegaconf import OmegaConf
        from trainer.train_res_3dcnn import SingleModule

        # checkpoint 里存的超参含 Hydra 的 ${now:...} 插值,脱离 Hydra 无法解析
        if not OmegaConf.has_resolver("now"):
            OmegaConf.register_new_resolver("now", lambda pattern="": "eval")

        module = SingleModule.load_from_checkpoint(args.ckpt, map_location=device)
        module.eval().to(device)
        # 取最后一个非 Identity 的卷积 stage
        convs = [m for m in module.modules() if isinstance(m, torch.nn.Conv3d)]
        cam_engine = GradCAM3D(module, convs[-1])
        print(f"Grad-CAM 挂载在 {convs[-1]}")

    kinds = ["uniform", "random", "center"] + (["gradcam"] if cam_engine else [])
    totals = {k: 0.0 for k in kinds}
    weights = {k: 0.0 for k in kinds}

    for i in range(len(dataset)):
        sample = dataset[i]
        video = sample["video"].to(device)
        region_map = sample["region_map"].to(device)
        target = sample["region_target"].to(device).unsqueeze(0).expand(video.shape[0], -1)
        label = torch.full((video.shape[0],), sample["label"], device=device, dtype=torch.long)

        # 参照图与 concept 架构的 token 分辨率对齐
        t_tok = args.num_samples
        h_tok = w_tok = args.img_size // 32
        shape = (video.shape[0], 1, t_tok, h_tok, w_tok)

        for kind in kinds:
            if kind == "gradcam":
                cam = cam_engine(video, label)
            else:
                cam = make_reference_maps(shape, device, kind)
            if cam.shape[2:] != (t_tok, h_tok, w_tok):
                cam = F.interpolate(cam, size=(t_tok, h_tok, w_tok),
                                    mode="trilinear", align_corners=False)
            # 单张重要性图对所有区域共用:事后解释方法不区分临床概念
            attn = cam.expand(-1, len(REGIONS), -1, -1, -1)
            s, w = alignment(attn, region_map, target)
            if s is not None:
                totals[kind] += s
                weights[kind] += w

        if (i + 1) % 20 == 0:
            print(f"  已评估 {i+1}/{len(dataset)} 条")

    print(f"\n=== 注意力与医生标注的对齐度 (fold {args.fold}, {len(dataset)} 条视频) ===")
    print("越高越贴近医生关注区域;uniform 是零信息下界")
    for kind in kinds:
        if weights[kind] > 0:
            print(f"  {kind:10s} {totals[kind]/weights[kind]:.4f}")
    print("\n把 concept 架构训练日志里的 test/attn_align 与上表对比。"
          "若不明显高于 gradcam,可解释性主张需要重新论证。")


if __name__ == "__main__":
    main()

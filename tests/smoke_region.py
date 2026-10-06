#!/usr/bin/env python3
"""多分支区域模型(model.backbone=region)的冒烟测试,CPU 可跑。

    python tests/smoke_region.py                                   # 只跑合成数据部分(不读数据集)
    python tests/smoke_region.py --root-path $DATA --sheet /tmp/region_crops.png   # 加真实数据 + 裁剪对照图

合成部分:裁剪几何(等比例、越界补零、段内固定框、地标缺失回退)与各配置的前向 / 反向。
真实数据部分:DataModule 出的 region_clips 形状、把整帧上的框和裁出的小视频画成一张图供肉眼核对、
Lightning 模块跑一步。
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

from dataloader.region_crops import (  # noqa: E402
    DEFAULT_TRUNK, REGION_SETS, SEGMENTS, build_region_clips, crop_resize, resolve_region_set, trunk_length,
)
from models.region_branch import FEATURE_DIM, RegionBranchNet  # noqa: E402


def make_cfg(**model_over):
    cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
    cfg.model.backbone = "region"
    cfg.model.model_class_num = 2
    cfg.data.num_workers = 0
    cfg.train.experiment = "smoke_region"
    cfg.log_path = tempfile.mkdtemp()
    for k, v in model_over.items():
        cfg.model[k] = v
    return cfg


def synthetic_pose(u: int, lean: float = 0.0) -> torch.Tensor:
    """直立(或前倾 lean)的侧视骨架,归一化坐标,置信度全 1。"""
    p = torch.zeros((u, 17, 3))
    y = {0: 0.108, 1: 0.09, 2: 0.09, 3: 0.096, 4: 0.094, 5: 0.20, 6: 0.20, 7: 0.356, 8: 0.359, 9: 0.477, 10: 0.484,
         11: 0.485, 12: 0.486, 13: 0.695, 14: 0.70, 15: 0.905, 16: 0.91}
    for k, yy in y.items():
        p[:, k, 0] = 0.5 + lean * (0.485 - yy)   # 髋以上向前偏
        p[:, k, 1] = yy
        p[:, k, 2] = 1.0
    return p


def test_crops() -> None:
    # 1) 等比例 + 越界补零:横向渐变图,框一半在画面外
    h = w = 64
    frame = torch.arange(w, dtype=torch.float32).view(1, 1, 1, w).expand(2, 3, h, w)
    frames = (frame / (w - 1) * 255).to(torch.uint8)
    out = crop_resize(frames, torch.tensor([0.0, 0.5, 0.5]), 16)   # 中心在左边缘,左半越界
    assert out.shape == (2, 3, 16, 16)
    assert float(out[:, :, :, :7].abs().max()) == 0.0, "越界部分应补 0"
    assert float(out[:, :, :, 9:].min()) > 0.0 and float(out.max()) <= 1.0
    # 纵向不变(渐变只沿 x):等比例缩放不引入纵向结构
    assert float((out[:, :, 0] - out[:, :, -1]).abs().max()) < 1e-5

    # 2) 形状 / 边长 / 两种跟随方式
    n_chunks, t, u = 3, 8, 24
    pose = synthetic_pose(u, lean=0.3)
    sway = 0.06 * torch.sin(torch.arange(u) * 1.7)          # 模拟源视频裁剪窗随步态的晃动(整个人一起平移)
    pose[:, :, 0] += sway.unsqueeze(1)
    frames = torch.randint(0, 255, (u, 3, 96, 96), dtype=torch.uint8)
    inverse = torch.arange(u)
    names = REGION_SETS["measure"]
    clips, boxes = build_region_clips(frames, pose, inverse, n_chunks, t, names, 32, track="follow")
    assert clips.shape == (n_chunks, 3, 3, t, 32, 32) and boxes.shape == (n_chunks, 3, t, 3)
    assert 0.0 <= float(clips.min()) and float(clips.max()) <= 1.0
    trunk = trunk_length(pose)
    assert abs(trunk - 0.287) < 0.03, trunk
    for ri, n in enumerate(names):
        assert torch.allclose(boxes[:, ri, :, 2], torch.full((n_chunks, t), SEGMENTS[n][1] * trunk)), "边长应全程不变"
    # follow:框中心跟着晃动走,躯干段的 x 与(肩中点+髋中点)/2 一致
    want = (pose[:, [5, 6], 0].mean(1) + pose[:, [11, 12], 0].mean(1)) / 2
    assert torch.allclose(boxes[:, 1, :, 0].reshape(-1), want, atol=1e-5)
    # fixed:段内不动
    _, bf = build_region_clips(frames, pose, inverse, n_chunks, t, names, 32, track="fixed")
    assert float((bf[:, :, :, :2] - bf[:, :, :1, :2]).abs().max()) == 0.0
    assert float((boxes[:, :, :, 0] - bf[:, :, :, 0]).abs().max()) > 0.03, "follow 应明显不同于 fixed"
    # 前倾:头肩段的中心应在躯干段中心的前方(x 更大)、上方(y 更小)
    assert float(bf[0, 0, 0, 0]) > float(bf[0, 1, 0, 0]) and float(bf[0, 0, 0, 1]) < float(bf[0, 1, 0, 1])

    # 3) 地标组:远侧耳朵时有时无不应把头肩段中心拉偏(每组各自取有效点均值,再对组取均值)
    pose_ear = pose.clone(); pose_ear[::2, 4, 2] = 0.0; pose_ear[::2, 4, 0] += 0.3   # 无效点即使坐标离谱也不参与
    _, be = build_region_clips(frames, pose_ear, inverse, n_chunks, t, names, 32)
    assert float((be[:, 0, :, 0] - boxes[:, 0, :, 0]).abs().max()) < 0.02

    # 4) 回退:某帧缺 -> 段中位数;整段缺 -> 整条视频的中位数;整条都缺 -> 全库典型位置
    pose1 = pose.clone(); pose1[2, :, 2] = 0.0
    _, b1 = build_region_clips(frames, pose1, inverse, n_chunks, t, names, 32)
    assert torch.isfinite(b1).all() and torch.allclose(b1[0, 1, 2, :2], bf[0, 1, 0, :2], atol=0.03)
    pose2 = pose.clone(); pose2[:t, :, 2] = 0.0
    _, b2 = build_region_clips(frames, pose2, inverse, n_chunks, t, names, 32)
    assert torch.isfinite(b2).all() and float((b2[0, :, :, :2] - b2[0, :, :1, :2]).abs().max()) == 0.0
    pose3 = pose.clone(); pose3[:, :, 2] = 0.0
    _, b3 = build_region_clips(frames, pose3, inverse, n_chunks, t, names, 32)
    assert abs(float(b3[0, 1, 0, 0]) - SEGMENTS["trunk"][2][0]) < 1e-6 and abs(float(b3[0, 1, 0, 2]) - 1.5 * DEFAULT_TRUNK) < 1e-6
    # 5) 3D 骨架(4 通道)与未知跟随方式应明确报错
    for bad_kwargs in (dict(pose=torch.zeros((u, 17, 4))), dict(pose=pose, track="smooth")):
        try:
            build_region_clips(frames, bad_kwargs["pose"], inverse, n_chunks, t, names, 32, track=bad_kwargs.get("track", "follow"))
            raise AssertionError("应报错")
        except ValueError:
            pass
    assert resolve_region_set("trunk, knee") == ["trunk", "knee"] and resolve_region_set("none") == []
    print("[crops] 等比例 / 越界补零 / follow 与 fixed / 地标组 / 回退 / 报错: ok")


def test_model() -> None:
    torch.manual_seed(0)
    b, t, s_full, s_crop = 2, 8, 64, 32
    video = torch.rand(b, 3, t, s_full, s_full)
    label = torch.tensor([0, 1])
    combos = [
        dict(region_set="measure", region_fusion="score"),
        dict(region_set="measure", region_fusion="feature"),
        dict(region_set="parts", region_fusion="score", region_branch_weights=[2.0, 1.0, 1.0, 1.0]),
        dict(region_set="unmarked", region_fusion="score", region_use_global=False),
        dict(region_set="measure", region_fusion="score", region_share_backbone=False),
        dict(region_set="none", region_fusion="score"),
    ]
    for over in combos:
        cfg = make_cfg(**over)
        cfg.train.transfer_learning = False   # 合成测试不依赖 torch hub 缓存
        net = RegionBranchNet(cfg)
        r = len(net.region_names)
        clips = torch.rand(b, r, 3, t, s_crop, s_crop) if r else None
        out = net(video if net.use_global else None, clips)
        n = len(net.branch_names)
        assert out["branch_logits"].shape == (b, n, 2), out["branch_logits"].shape
        assert out["fused_prob"].shape == (b, 2)
        assert torch.allclose(out["fused_prob"].sum(-1), torch.ones(b), atol=1e-5)
        assert len(net.backbones) == (1 if net.share else n)
        assert abs(float(net.branch_weights.sum()) - 1.0) < 1e-6
        ce = torch.stack([torch.nn.functional.cross_entropy(out["branch_logits"][:, i], label) for i in range(n)]).mean()
        if net.fusion == "feature":
            # 探针损失不应回传到骨干:只对它求导,骨干梯度必须全为 None / 0
            ce.backward(retain_graph=True)
            g_backbone = sum(float(p.grad.abs().sum()) for p in net.backbones.parameters() if p.grad is not None)
            assert g_backbone == 0.0, f"探针损失漏进了骨干: {g_backbone}"
            torch.nn.functional.cross_entropy(out["fused_logits"], label).backward()
            assert out["fused_logits"].shape == (b, 2) and net.fuse_head.in_features == FEATURE_DIM * n
        else:
            ce.backward()
        g = sum(float(p.grad.abs().sum()) for p in net.backbones.parameters() if p.grad is not None)
        assert g > 0, "骨干没有梯度"
        n_param = sum(p.numel() for p in net.parameters())
        print(f"[model] {over}: 分支 {net.branch_names}, 参数 {n_param / 1e6:.1f}M, fused {tuple(out['fused_prob'].shape)} ok")


def test_real(root_path: str, sheet: str | None) -> None:
    from dataloader.data_loader import WalkDataModule
    from trainer.train_region_branch import RegionBranchModule

    info = Path(root_path) / "clinical_CLIP_dataset"
    index = json.load(open(info / "index_mapping/2/index.json"))
    paths = [str(info / "json_mix" / p.split("json_mix/")[-1]) for p in index["0"]["test"][:2]]
    cfg = make_cfg(region_set="measure")
    cfg.paths.root_path = root_path
    dm = WalkDataModule(cfg, {"train": paths, "val": paths, "test": paths}); dm.setup()
    batch = next(iter(dm.val_dataloader()))
    v, c, bx = batch["video"], batch["region_clips"], batch["region_boxes"]
    assert c.shape[0] == v.shape[0] == batch["label"].shape[0] and c.shape[1:] == (3, 3, 8, 112, 112), c.shape
    assert bx.shape == (v.shape[0], 3, 8, 3), bx.shape
    assert 0.0 <= float(c.min()) and float(c.max()) <= 1.0 and float(c.mean()) > 0.02
    print(f"[real] video {tuple(v.shape)}, region_clips {tuple(c.shape)}, boxes {tuple(bx.shape)}; 第 0 段第 0 帧的框 {bx[0, :, 0].tolist()}")

    if sheet:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        names = REGION_SETS["measure"]
        # 一个段的 8 帧:第一行整帧 + 逐帧的框,下面每行一个身体段裁出的小视频
        seg = v.shape[0] // 2
        fig, axes = plt.subplots(1 + len(names), 8, figsize=(16, 2.1 * (1 + len(names))))
        for fr in range(8):
            ax = axes[0, fr]
            ax.imshow(v[seg, :, fr].permute(1, 2, 0).numpy()); ax.set_title(f"segment {seg}, frame {fr}", fontsize=7); ax.axis("off")
            for k, col in enumerate(("red", "lime", "cyan")):
                cx, cy, side = [float(x) for x in bx[seg, k, fr]]
                ax.add_patch(Rectangle(((cx - side / 2) * 224, (cy - side / 2) * 224), side * 224, side * 224, fill=False, ec=col, lw=1.0))
            for k, n in enumerate(names):
                a = axes[1 + k, fr]
                a.imshow(c[seg, k, :, fr].permute(1, 2, 0).numpy()); a.axis("off")
                if fr == 0:
                    a.set_title(n, fontsize=8, loc="left")
        fig.tight_layout(); fig.savefig(sheet, dpi=80); plt.close(fig)
        print(f"[real] 裁剪对照图 -> {sheet}")

    m = RegionBranchModule(cfg)
    sub = {k: (x[:2] if torch.is_tensor(x) and x.shape[0] == v.shape[0] else x) for k, x in batch.items()}
    loss, out, label = m._shared_step(sub, "train")
    loss.backward()
    print(f"[real] RegionBranchModule 一步: loss {float(loss):.3f}, fused_prob {out['fused_prob'].tolist()}, 分支 {m.branch_names}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-path", default=None)
    ap.add_argument("--sheet", default=None, help="真实数据裁剪对照图的输出路径(png)")
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()
    test_crops()
    if not args.skip_model:
        test_model()
    if args.root_path:
        test_real(args.root_path, args.sheet)
    print("smoke_region: 通过")


if __name__ == "__main__":
    main()

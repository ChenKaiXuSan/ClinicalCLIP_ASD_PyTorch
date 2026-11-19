#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: run_sam3d.py
Project: analysis/koa
Author: Kaixu Chen
-----
Comment:
KOA-PD-NM 的第二步:SAM-3D-Body 逐帧估计 3D 关键点,与 ASD 数据同一个模型、同一套输出字段
(pred_keypoints_3d 70 点、pred_joint_coords 127 点、pred_cam_t)。人框用 detect.py 挑出的行走者;
相机内参全库统一(--focal,见参数说明);--focal 0 时退回每个视频用 MoGe-2 估一次。

在 sam_3d_body 环境里跑,代码取自 Drive_Face_Mesh_PyTorch/SAM3Dbody(--code-root)。
输出 sam3d/<video>.npz: frame_idx, kp3d (N,70,3), joints (N,127,3), cam_t (N,3), kp2d (N,70,2), focal, ok (N,)

    python analysis/koa/run_sam3d.py --video-root $KOA/video --proc $KOA/proc [--chunk 0/4]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

EXT = {".mov", ".mp4", ".avi", ".m4v"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--proc", required=True)
    ap.add_argument("--code-root", default="/work/1/SKIING/chenkaixu/code/Drive_Face_Mesh_PyTorch")
    ap.add_argument("--ckpt-root", default="/work/1/SKIING/chenkaixu/ckpt/sam-3d-body-dinov3")
    ap.add_argument("--chunk", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-fov", action="store_true", help="不用 MoGe-2,交给模型的默认内参")
    ap.add_argument("--focal", type=float, default=4000.0,
                    help="固定焦距(像素),主点取画面中心;> 0 时不跑 MoGe-2。KOA-PD-NM 的相机固定在 8 米处,三组的人框高度 / 真实身高"
                         "都是约 500 像素每米,换算焦距约 4000;逐视频估计的焦距有约 8% 的波动,会直接进到步速里")
    args = ap.parse_args()
    sys.path.insert(0, args.code_root)
    import cv2
    import torch
    from SAM3Dbody.sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    proc = Path(args.proc); (proc / "sam3d").mkdir(parents=True, exist_ok=True)
    vids = sorted(p for p in Path(args.video_root).rglob("*") if p.suffix.lower() in EXT and p.stat().st_size > 0)
    k, n = (int(x) for x in args.chunk.split("/"))
    vids = [v for i, v in enumerate(vids) if i % n == k]
    if args.limit:
        vids = vids[: args.limit]
    model, model_cfg = load_sam_3d_body(f"{args.ckpt_root}/model.ckpt", device="cuda", mhr_path=f"{args.ckpt_root}/assets/mhr_model.pt")
    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg, human_detector=None, human_segmentor=None, fov_estimator=None)
    fov = None
    if not args.no_fov and args.focal <= 0:
        from SAM3Dbody.tools.build_fov_estimator import FOVEstimator
        fov = FOVEstimator(name="moge2", device="cuda")
    print(f"视频 {len(vids)} 个(分片 {args.chunk})", flush=True)
    for v in vids:
        name = v.stem
        dst = proc / "sam3d" / f"{name}.npz"
        det = proc / "det" / f"{name}.npz"
        if dst.exists() or not det.exists():
            continue
        D = np.load(det)
        want = {int(f): j for j, f in enumerate(D["frame_idx"])}
        N = len(want)
        kp3d = np.full((N, 70, 3), np.nan, np.float32); joints = np.full((N, 127, 3), np.nan, np.float32)
        cam_t = np.full((N, 3), np.nan, np.float32); kp2d = np.full((N, 70, 2), np.nan, np.float32); ok = np.zeros(N, bool)
        cam_int, focal = None, np.nan
        cap = cv2.VideoCapture(str(v))
        i = -1
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            i += 1
            if i > int(D["frame_idx"][-1]):
                break
            j = want.get(i)
            if j is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if cam_int is None and args.focal > 0:
                h, w = rgb.shape[:2]
                cam_int = torch.tensor([[[args.focal, 0.0, w / 2], [0.0, args.focal, h / 2], [0.0, 0.0, 1.0]]], dtype=torch.float32)
                focal = args.focal
            if cam_int is None and fov is not None:
                cam_int = torch.as_tensor(np.asarray(fov.get_cam_intrinsics(rgb)), dtype=torch.float32)
                focal = float(cam_int[0, 1, 1])
            with contextlib.redirect_stdout(io.StringIO()):
                outs = est.process_one_image(img=rgb, bboxes=D["boxes"][j][None].astype(np.float32), cam_int=cam_int)
            if not outs:
                continue
            o = outs[0]
            kp3d[j], joints[j], cam_t[j], kp2d[j], ok[j] = o["pred_keypoints_3d"], o["pred_joint_coords"], o["pred_cam_t"], o["pred_keypoints_2d"], True
            if np.isnan(focal):
                focal = float(np.asarray(o["focal_length"]).reshape(-1)[0])
        cap.release()
        np.savez_compressed(dst, frame_idx=D["frame_idx"], kp3d=kp3d, joints=joints, cam_t=cam_t, kp2d=kp2d, focal=focal, ok=ok,
                            eff_fps=D["eff_fps"], size=D["size"])
        print(f"{name}: 帧 {N}, 成功 {ok.sum()}, 焦距 {focal:.0f}px, 相机系位移 x {np.nanmax(cam_t[:, 0]) - np.nanmin(cam_t[:, 0]):.2f} m, "
              f"深度中位 {np.nanmedian(cam_t[:, 2]):.1f} m", flush=True)
    print("[done]")


if __name__ == "__main__":
    main()

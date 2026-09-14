#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: gate_strata_diag.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
方法 2 的分层诊断:按几何分支的可靠性特征(离群度 outl_g / 置信度 conf_g)把患者分成三档,
每档里报 视频 / 几何 / 等权融合 的患者级准确率(micro,按患者数;三档各约 26 人,不做 macro)。
回答:几何分支拿不准的那一档里,视频分支是不是更准?若不是,逐患者门控没有可利用的空间。

    python analysis/gate_strata_diag.py --data-root $DATA --video ... --geom ...
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patient_level_stats import build_patient_map  # noqa: E402
from uncertainty_fusion import fuse, geom_reliability, load_geom, load_video  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--video", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2024])
    ap.add_argument("--geom", nargs="+", required=True)
    ap.add_argument("--attr-dir", default="logs/geom_attributes_3d_doctor")
    ap.add_argument("--by", default="outl_g", choices=["outl_g", "conf_g", "miss_g"])
    args = ap.parse_args()
    pmap = build_patient_map(args.data_root)
    R = geom_reliability(Path(args.attr_dir), pmap)
    for gpath in args.geom:
        G = load_geom(gpath)
        print(f"\n===== 几何分支 {Path(gpath).stem}, 按 {args.by} 分三档(低 / 中 / 高), 每档: 视频 / 几何 / 融合0.5 的准确率, n")
        tot = defaultdict(lambda: np.zeros(8))
        for vtag in args.video:
            for seed in args.seeds:
                V = load_video(Path(args.root), vtag, seed, pmap)
                pids = sorted(set(V) & set(G) & set(R))
                y = np.array([V[p]["label"] for p in pids])
                Pv = np.stack([V[p]["p"] for p in pids]); Pg = np.stack([G[p]["p"] for p in pids])
                key = {"outl_g": np.array([R[p][1] for p in pids]), "miss_g": np.array([R[p][0] for p in pids]),
                       "conf_g": np.abs(Pg[:, 1] - 0.5) * 2}[args.by]
                q = np.quantile(key, [1 / 3, 2 / 3])
                band = np.digitize(key, q)
                ok = {"v": Pv.argmax(1) == y, "g": Pg.argmax(1) == y, "f": fuse(Pv, Pg, np.full(len(y), 0.5)).argmax(1) == y}
                asd = (y == 0)  # 标签 0 = ASD(多数类)
                row = f"{vtag:24s} {seed:5d} "
                for b in range(3):
                    m = band == b
                    row += (f"| {ok['v'][m].mean():.2f} {ok['g'][m].mean():.2f} {ok['f'][m].mean():.2f} n={m.sum():2d} "
                            f"ASD占比 {asd[m].mean():.2f} 几何在非ASD上对 {ok['g'][m & ~asd].mean():.2f} ")
                    tot[b] += [ok['v'][m].sum(), ok['g'][m].sum(), ok['f'][m].sum(), m.sum(), asd[m].sum(),
                               ok['g'][m & ~asd].sum(), (m & ~asd).sum(), ok['v'][m & ~asd].sum()]
                print(row)
        print(f"{'合计(9 组)':24s} {'':5s} " + "".join(
            f"| 视频 {tot[b][0] / tot[b][3]:.2f} 几何 {tot[b][1] / tot[b][3]:.2f} 融合 {tot[b][2] / tot[b][3]:.2f} n={int(tot[b][3]):3d} "
            f"ASD占比 {tot[b][4] / tot[b][3]:.2f} 非ASD上 几何 {tot[b][5] / max(tot[b][6], 1):.2f} 视频 {tot[b][7] / max(tot[b][6], 1):.2f} "
            for b in range(3)))


if __name__ == "__main__":
    main()

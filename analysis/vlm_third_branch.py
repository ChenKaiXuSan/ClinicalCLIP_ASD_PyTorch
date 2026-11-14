#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: vlm_third_branch.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
VLM 属性分支作为证据门控里的"备选意见":测量偏离常态时信测量,否则让给备选分支。
备选分支三种:视频(现状)、VLM 属性(患者级 LR 概率)、视频与 VLM 的平均。全部走 uncertainty_fusion.run_one
的同一套嵌套流程(阈值与两档权重在训练折上选),67 人口径。

    python analysis/vlm_third_branch.py --data-root $DATA --vlm logs/geom_probs/vlm_attr_lr_probs.json \
        --exclude-missing logs/measure_bank_3d/bank.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from late_fusion import macro  # noqa: E402
from patient_level_stats import build_patient_map  # noqa: E402
from uncertainty_fusion import geom_reliability, load_geom, load_video, run_one  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--root", default="logs/train")
    ap.add_argument("--video", nargs="+", default=["B0_3dcnn_e50", "M0_concept_learned_e50", "A1_no_grounding_e50"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2024])
    ap.add_argument("--geom", default="logs/geom_probs/geom3d_doctor_lr_probs.json")
    ap.add_argument("--vlm", nargs="+", required=True, help="VLM 分支的患者级概率 json(可多个,逐个评)")
    ap.add_argument("--attr-dir", default="logs/geom_attributes_3d_doctor")
    ap.add_argument("--exclude-missing", default=None)
    ap.add_argument("--perm", type=int, default=0, help="把 VLM 概率在患者间置换 N 次作对照")
    args = ap.parse_args()

    drop = set()
    if args.exclude_missing:
        drop = {Path(v["json"]).stem.split("-")[0] for v in pickle.load(open(args.exclude_missing, "rb"))["videos"].values() if v["frac_ok"] == 0}
    pmap = build_patient_map(args.data_root)
    R = {k: v for k, v in geom_reliability(Path(args.attr_dir), pmap).items() if k not in drop}
    G = load_geom(args.geom)
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    lams = [1.0]
    Vs = {(t, s): load_video(Path(args.root), t, s, pmap) for t in args.video for s in args.seeds}
    fold_of = {p: v["fold"] for p, v in next(iter(Vs.values())).items()}

    def as_branch(P: dict) -> dict:
        return {p: {"p": v["p"], "std": 0.0, "n": 1, "fold": fold_of[p], "label": v["label"]} for p, v in P.items() if p in fold_of}

    def gate_score(V):
        r = run_one(V, G, R, lams, grid, light=True)
        y = r["y"]
        return {k: macro(r["pred"][k], y) for k in ("video", "geom", "fixed0.5", "nested_w", "step_outl")}, r["ceiling"]

    base = [gate_score(V) for V in Vs.values()]
    print(f"{'备选分支':34s} {'备选单独':>8s} {'测量':>6s} {'固定0.5':>7s} {'嵌套w':>6s} {'阶梯门控':>8s} {'上限':>6s}")
    print(f"{'视频(9 组均值)':34s} " + " ".join(f"{np.mean([b[0][k] for b in base]):8.3f}" for k in ("video", "geom", "fixed0.5", "nested_w", "step_outl"))
          + f" {np.mean([b[1] for b in base]):6.3f}")
    for path in args.vlm:
        A = as_branch(load_geom(path))
        s, c = gate_score(A)
        name = Path(path).stem.replace("_lr_probs", "")
        print(f"{'VLM: ' + name:34s} " + " ".join(f"{s[k]:8.3f}" for k in ("video", "geom", "fixed0.5", "nested_w", "step_outl")) + f" {c:6.3f}")
        # 视频 + VLM 平均作为备选
        mix = []
        for V in Vs.values():
            M = {p: {**V[p], "p": 0.5 * V[p]["p"] + 0.5 * A[p]["p"]} for p in V if p in A}
            mix.append(gate_score(M))
        print(f"{'  视频 + VLM 平均(9 组均值)':34s} " + " ".join(f"{np.mean([b[0][k] for b in mix]):8.3f}" for k in ("video", "geom", "fixed0.5", "nested_w", "step_outl"))
              + f" {np.mean([b[1] for b in mix]):6.3f}")
        if args.perm:
            rng = np.random.default_rng(0)
            ps = sorted(A)
            vals, vals_mix = [], []
            for _ in range(args.perm):
                perm = rng.permutation(len(ps))
                Ap = {p: {**A[p], "p": A[ps[j]]["p"]} for p, j in zip(ps, perm)}
                vals.append(gate_score(Ap)[0]["step_outl"])
                V = Vs[(args.video[0], args.seeds[0])]
                vals_mix.append(gate_score({p: {**V[p], "p": 0.5 * V[p]["p"] + 0.5 * Ap[p]["p"]} for p in V if p in Ap})[0]["step_outl"])
            print(f"{'  置换 VLM 概率(' + str(args.perm) + ' 次)':34s} 门控 {np.mean(vals):.3f} ± {np.std(vals):.3f};与 {args.video[0]} s{args.seeds[0]} 平均后 {np.mean(vals_mix):.3f} ± {np.std(vals_mix):.3f}")


if __name__ == "__main__":
    main()

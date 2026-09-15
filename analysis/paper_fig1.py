#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Fig 1 流程示意(matplotlib 画框图,免去 drawio 导出):
医生区域标注 → 两种用法:(上)当监督 → grounded 视频分支;(下)当测量先验 → 5 个量 → LR;→ 证据强度门控。

    python analysis/paper_fig1.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "paper/figs"
C_VIDEO, C_GEOM, C_GATE, C_ANN = "#4C72B0", "#DD8452", "#55A868", "#8172B2"


def box(ax, x, y, w, h, text, color, fs=6.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=color, ec="none", alpha=0.16))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc="none", ec=color, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, color="#444444", text=None, fs=5.2, ls="-", dy=0.06):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=color, linestyle=ls))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha="center", va="bottom", fontsize=fs, color=color)


def main() -> None:
    fig = plt.figure(figsize=(4.8, 1.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.0); ax.axis("off")
    # 输入
    box(ax, 0.15, 2.55, 1.75, 1.25, "clinician region\nannotations\n(lumbar–pelvis,\nhead, shoulder, …)", C_ANN)
    box(ax, 0.15, 0.35, 1.75, 1.25, "lateral-view\ngait video\n1-s segments,\n8 frames", "#333333")
    # 上路:监督
    box(ax, 2.45, 2.55, 2.75, 1.25, "(A) attention as supervision\n3D CNN + region grounding\n→ verifiable attention maps", C_VIDEO)
    arrow(ax, (1.9, 3.35), (2.45, 3.35), C_ANN, "training-only loss", dy=0.05)
    arrow(ax, (1.9, 1.3), (2.45, 2.75), C_VIDEO)
    box(ax, 5.55, 2.75, 1.6, 0.85, "video branch\n$P_v(\\mathrm{ASD})$", C_VIDEO)
    arrow(ax, (5.2, 3.17), (5.55, 3.17), C_VIDEO)
    # 下路:测量先验
    box(ax, 2.45, 0.35, 2.75, 1.25, "(B) attention as measurement prior\nmarked regions → 5 sagittal\nquantities from 3D keypoints", C_GEOM)
    arrow(ax, (1.9, 2.7), (2.45, 1.45), C_ANN, ls="--")
    arrow(ax, (1.9, 0.95), (2.45, 0.95), C_GEOM)
    box(ax, 5.55, 0.55, 1.6, 0.85, "logistic regression\n$P_m(\\mathrm{ASD})$", C_GEOM)
    arrow(ax, (5.2, 0.97), (5.55, 0.97), C_GEOM)
    # 门控
    box(ax, 7.55, 1.2, 2.3, 1.75, "(C) evidence-strength gate\n$d$ = mean $|z|$ of the 5 quantities\n$d<\\tau$: defer to video\n$d\\geq\\tau$: trust measurements", C_GATE)
    arrow(ax, (7.15, 3.17), (7.55, 2.5), C_VIDEO)
    arrow(ax, (7.15, 0.97), (7.55, 1.6), C_GEOM)
    ax.text(8.7, 0.7, "annotations are never\nused at inference", ha="center", va="center", fontsize=5.2, style="italic", color="#444444")
    fig.savefig(OUT / "fig1_pipeline.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "fig1_pipeline.png", dpi=250, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print("fig1 ok")


if __name__ == "__main__":
    main()

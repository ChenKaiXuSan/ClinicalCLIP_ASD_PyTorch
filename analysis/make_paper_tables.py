#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: make_paper_tables.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
把矩阵结果整理成论文可用的表格(Markdown + LaTeX)。

两条硬规则:
1. **分类指标一律从 best_preds 的原始预测重算**,不读训练日志。日志里的
   video_acc 是逐 batch 的 macro 值再平均,而 batch_size=1 时每个 batch 只含
   一条视频、只有一个类别,torchmetrics 会忽略缺席的类,"全预测多数类"也能得
   高分。实测 clip_old 日志 0.698 而真实 macro 0.720,pose 日志 0.543 而真实
   macro 0.333(已完全退化成单类)。
2. **同时给出被选中的值和验证曲线后段均值**。val 与 test 是同一批数据,
   报告的 test 就是 100 个 epoch 里 val 曲线的最大值,实测选择偏差
   +0.13~+0.31,且幅度因配置而异 —— 只看被选中的值会得出错误的排名。

用法:
    python analysis/make_paper_tables.py --root logs/train --fold 0
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import statistics
from pathlib import Path

import torch

DISPLAY = {
    "B0_3dcnn": ("基线", "3D CNN (slow\\_r50)"),
    "B1_2dcnn": ("基线", "2D CNN"),
    "B2_cnn_lstm": ("基线", "CNN + LSTM"),
    "B3_pose": ("基线", "ST-GCN (纯姿态)"),
    "B4_clip_old": ("基线", "CLIP 对齐 (推理需标注)"),
    "M0_concept_learned": ("本方法", "概念引导 (可学习概念)"),
    "M1_concept_cliptext": ("本方法", "概念引导 (CLIP 文本概念)"),
    "A0_shuffle_region": ("消融", "打乱区域标注"),
    "A1_no_grounding": ("消融", "去空间 grounding"),
    "A2_no_presence": ("消融", "去区域存在性"),
    "A3_no_concept_loss": ("消融", "去概念对比"),
    "A4_no_prior": ("消融", "去全部临床先验"),
    "D1_doctor1_only": ("标注者", "仅医生 1"),
    "D2_doctor2_only": ("标注者", "仅医生 2"),
}


def collect(root: Path, exp: str, fold: int, seed: int) -> dict | None:
    run = f"{exp}__f{fold}_s{seed}"
    out: dict = {"exp": exp}

    pred_dirs = glob.glob(str(root / run / "*/*/best_preds"))
    if pred_dirs:
        pf = sorted(glob.glob(pred_dirs[0] + "/*_pred.pt"))
        if pf:
            prob = torch.load(pf[0], map_location="cpu", weights_only=False).float()
            lab = torch.load(pf[0].replace("_pred.pt", "_label.pt"),
                             map_location="cpu", weights_only=False).long()
            pred = prob.argmax(1)
            rec = [float((pred[lab == c] == c).float().mean()) if (lab == c).any() else float("nan")
                   for c in range(prob.shape[1])]
            valid = [r for r in rec if r == r]
            out["macro"] = sum(valid) / len(valid)
            out["micro"] = float((pred == lab).float().mean())
            out["recall"] = rec
            out["collapsed"] = sum(1 for r in valid if r > 0.01) <= 1

    csvs = glob.glob(str(root / run / "*/*/csv/*/version_*/metrics.csv"))
    if csvs:
        rows = list(csv.DictReader(open(csvs[0])))
        if rows:
            key = next((k for k in ("val/video_acc", "val/video_acc_epoch") if k in rows[0]), None)
            vals = [float(r[key]) for r in rows if key and r.get(key) not in (None, "")]
            late = vals[20:] or vals
            if late:
                out["val_mean"] = statistics.mean(late)
                out["val_std"] = statistics.stdev(late) if len(late) > 1 else 0.0
                out["val_best"] = max(vals)

    for tf in glob.glob(str(root / run / "*/*/tensorboard/*/*/test_metrics.txt")):
        try:
            out["extra"] = ast.literal_eval(Path(tf).read_text().strip())[0]
        except (ValueError, SyntaxError, IndexError):
            pass

    return out if len(out) > 1 else None


def fmt(v, digits=3, dash="—"):
    return f"{v:.{digits}f}" if isinstance(v, float) and v == v else dash


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="写入 Markdown 文件")
    args = ap.parse_args()

    root = Path(args.root)
    results = {}
    for exp in DISPLAY:
        r = collect(root, exp, args.fold, args.seed)
        if r:
            results[exp] = r

    lines = []
    add = lines.append
    add(f"# fold {args.fold} 结果表(seed {args.seed})\n")
    add("分类指标由 `best_preds` 原始预测重算,非训练日志。\n")

    add("\n## 表 1:分类性能\n")
    add("| 组别 | 方法 | macro | micro | ASD | DHS | LCS_HipOA | val 后段均值 |")
    add("|---|---|---|---|---|---|---|---|")
    for exp, r in results.items():
        grp, name = DISPLAY[exp]
        rec = r.get("recall", [float("nan")] * 3)
        flag = " ⚠退化" if r.get("collapsed") else ""
        vm = (f"{r['val_mean']:.3f} ± {r['val_std']:.3f}" if "val_mean" in r else "—")
        add(f"| {grp} | {name} | {fmt(r.get('macro'))}{flag} | {fmt(r.get('micro'))} | "
            f"{fmt(rec[0])} | {fmt(rec[1])} | {fmt(rec[2])} | {vm} |")
    add("\nmacro 随机基线 0.333;micro 多数类基线 0.543。")
    add("val 后段均值取 epoch ≥ 20,是比「被选中的最佳值」更诚实的估计"
        "(val 与 test 同源,实测选择偏差 +0.13~+0.31)。")

    add("\n## 表 2:可解释性——注意力与医生标注的对齐\n")
    add("| 方法 | 注意力对齐 | 区域 AP | 区域 F1(any) |")
    add("|---|---|---|---|")
    for exp, r in results.items():
        ex = r.get("extra", {})
        if "test/attn_align" not in ex:
            continue
        add(f"| {DISPLAY[exp][1]} | {fmt(ex.get('test/attn_align'))} | "
            f"{fmt(ex.get('test/region_ap'))} | {fmt(ex.get('test/region_f1_any'))} |")
    add("\n**参照基线**(`analysis/eval_attention_alignment.py`,fold 0,60 条视频):")
    add("\n| 参照 | 对齐度 |")
    add("|---|---|")
    add("| center(人在画面中央的平凡先验) | 0.160 |")
    add("| uniform(零信息下界) | 0.068 |")
    add("| random | 0.067 |")
    add("| **Grad-CAM(纯 3D CNN 的事后解释)** | **0.035** |")

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()

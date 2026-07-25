#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: compare_concept_runs.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
汇总 concept 架构的对照实验,重点回答两个问题:

1. 打乱区域标注后指标是否下降 —— 若不降,说明增益来自姿态渲染而非临床知识,
   论文的核心主张不成立;
2. CLIP 文本概念相比可学习概念是否有优势。

用法:
    python analysis/compare_concept_runs.py --root logs/train
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

METRIC_KEYS = [
    "test/video_acc",
    "test/video_f1_score",
    "test/region_f1",
    "test/attn_iou",
    "test/loss",
]


def load_run(exp_dir: Path) -> list[dict]:
    """一个实验目录下可能有多次运行(按日期/时刻),每折一个 test_metrics.txt。"""
    runs = []
    for metrics_file in sorted(exp_dir.rglob("test_metrics.txt")):
        try:
            payload = ast.literal_eval(metrics_file.read_text().strip())
        except (ValueError, SyntaxError):
            continue
        if isinstance(payload, list) and payload:
            record = dict(payload[0])
            # tensorboard/<fold>/version_x/test_metrics.txt
            record["_fold"] = metrics_file.parent.parent.name
            record["_path"] = str(metrics_file)
            runs.append(record)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="logs/train")
    parser.add_argument("--json", default=None, help="额外导出为 json")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"找不到 {root}")

    results = {}
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        runs = load_run(exp_dir)
        if runs:
            results[exp_dir.name] = runs

    if not results:
        raise SystemExit(f"{root} 下没有找到 test_metrics.txt,实验可能还没跑完")

    header = f"{'实验':32s}" + "".join(f"{k.replace('test/',''):>16s}" for k in METRIC_KEYS)
    print(header)
    print("-" * len(header))
    summary = {}
    for name, runs in results.items():
        # 多折时取均值
        row = {}
        for key in METRIC_KEYS:
            vals = [r[key] for r in runs if key in r]
            row[key] = sum(vals) / len(vals) if vals else float("nan")
        summary[name] = {"n_fold": len(runs), **row}
        print(f"{name:32s}" + "".join(f"{row[k]:16.4f}" for k in METRIC_KEYS))

    # 消融对照:同名去掉 _shuffled 的两个实验配对
    print("\n=== 区域消融对照 (正常 vs 打乱区域) ===")
    found = False
    for name in summary:
        if name.endswith("_shuffled"):
            base = name[: -len("_shuffled")]
            if base in summary:
                found = True
                print(f"\n{base}:")
                for key in ["test/video_acc", "test/video_f1_score", "test/region_f1", "test/attn_iou"]:
                    a, b = summary[base][key], summary[name][key]
                    delta = a - b
                    print(f"  {key.replace('test/',''):16s} 正常 {a:.4f}  打乱 {b:.4f}  差值 {delta:+.4f}")
                acc_gap = summary[base]["test/video_acc"] - summary[name]["test/video_acc"]
                if acc_gap < 0.01:
                    print("  ⚠ 打乱区域后精度几乎不变 —— 增益可能来自姿态而非临床区域选择")
                else:
                    print(f"  ✅ 打乱区域使精度下降 {acc_gap:.4f},支持临床先验确实起作用")
    if not found:
        print("(没有成对的 *_shuffled 实验)")

    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\n已导出 {args.json}")


if __name__ == "__main__":
    main()

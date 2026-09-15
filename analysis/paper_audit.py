#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: paper_audit.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
稿件数字审计:
  1. 已撤回的表述 / 数字不得出现在 main.tex(AUC 0.85、宽松硬过滤 0.803、随机森林 0.812、"head is the key region"、
     "independent of the skeleton source" 等);
  2. 正文里出现的每个三位小数,要么能在 paper/tables/numbers.json 或 paper/tables/*.tex 里找到(允许 ±0.001 的四舍五入),
     要么在白名单里(来自 docs/vlm_backbone.md 的、不由 paper_figs.py 生成的数),否则列为"未溯源";
  3. 输出一份来源清单,写稿改数时对照。

    python analysis/paper_audit.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper/main.tex"
TABLES = ROOT / "paper/tables"

# 已撤回:出现即报错
FORBIDDEN = [
    (r"0\.85\b", "AUC 0.85(79 人固定划分)已撤回, 67 人 10 份划分是 0.79"),
    (r"0\.803", "宽松硬过滤 0.803 已撤回(划分运气)"),
    (r"0\.812", "随机森林固定划分 0.812 已撤回, 10 份划分 0.732"),
    (r"0\.845", "AUC 0.845(79 人)已撤回"),
    (r"head (region )?is (the )?key", "'头部是关键区域'的测量库排序已撤回(只在门控信号里关键)"),
    (r"independent of (the )?skeleton", "'门控与骨架来源无关'已撤回(2D 离群度 67 人上无收益)"),
    (r"\+0\.085", "严格边界对补集 +0.085 是 79 人含填充的数, 67 人上 +0.03"),
]

# 白名单:正文里来自 docs/vlm_backbone.md、不在 numbers.json 里的数(每项写来源)
WHITELIST = {
    "0.533": "attn_align M0 50ep×3seeds (docs 三个补丁)", "0.001": "attn_align std", "0.135": "Grad-CAM (docs/findings.md)",
    "0.034": "Grad-CAM std", "0.144": "centre prior (findings)", "0.068": "uniform map (findings)", "0.104": "A1 attn_align 50ep",
    "0.012": "A1 attn_align std", "0.065": "random-location control (findings, 100ep s42)", "0.571": "TimeSformer lr1e-4 三种子",
    "0.715": "G5 3D concat (docs 3D 进编码器)", "0.008": "G5 std", "0.795": "阈值敏感性 q=0.1 (docs 67 人补算)", "0.809": "q=0.4",
    "0.729": "门控去头前伸 (docs 67 人复核)", "0.740": "几何单独去头前伸", "0.743": "置换 30 次 (docs; 表里是 10 次 0.743)",
    "0.733": "离群度改用未标部位 4 量 (docs 67 人)", "0.785": "门控换 TimeSformer (docs 三个补丁)",
    "0.44": "VLM zero-shot AUC (docs VLM 线)", "45.7": "两医生一致率", "1,889": "视频数", "14,048": "段数", "1,272": "测量库大小",
    "0.90": "上限(任一分支判对), numbers.json 无", "0.60": "随机 5 量未标池均值 0.595", "0.63": "随机 5 量全库池 0.627", "0.64": "随机 5 量医生池 0.637",
    "0.53": "attn 0.533 / 端到端下限", "0.14": "Grad-CAM 0.135", "0.66": "M0 0.663 / 未标 4 量 0.660", "0.72": "5 量 0.717 / TimeSformer AUC 0.716",
    "0.79": "AUC 0.787 / 门控 0.794", "0.75": "几何 0.751 / 嵌套 w 0.745", "0.82": "M0 门控 0.819", "0.49": "CTR-GCN 2D 0.491", "0.54": "ST-GCN 3D 0.538",
    "0.57": "2D CNN 0.572", "0.74": "AUC 去头 0.739 / RF 0.732", "0.73": "均匀 L1 0.730", "0.41": "head forward alone 0.409",
    "0.77": "中档几何 0.77 (fig3)", "0.96": "高档几何 0.96 (fig3)", "0.69": "低档视频 0.69 (fig3)", "0.50": "低档几何 0.50 (fig3)",
    "0.03": "严格−补集 +0.031", "0.01": "严格−全库 +0.012", "0.05": "std 0.052", "0.04": "门控 +0.043", "0.08": "单种子噪声阈值",
    "0.10": "CI 半宽", "0.81": "region_ap M0 50ep×3seeds 0.810 (docs 三个补丁)",
}


def main() -> int:
    tex = TEX.read_text(encoding="utf-8")
    body = re.sub(r"(?m)^%.*$", "", tex)
    body = body.split(r"\begin{document}", 1)[-1]
    body = re.sub(r"width=\d\.\d+\\linewidth", "width=W", body)  # 图宽不是数据
    bad = 0
    print("== 已撤回表述检查")
    for pat, why in FORBIDDEN:
        for m in re.finditer(pat, body):
            line = body[:m.start()].count("\n") + 1
            print(f"  FORBIDDEN 第 {line} 行: '{m.group(0)}' — {why}"); bad += 1
    if not bad:
        print("  通过")

    # 可溯源数字池:numbers.json 的全部数值 + tables/*.tex 的全部三位小数
    pool = set()
    nums = json.load(open(TABLES / "numbers.json"))

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, (int, float)):
            pool.add(round(float(o), 3)); pool.add(round(float(o), 2))
        elif isinstance(o, str):
            for x in re.findall(r"[-+]?\d\.\d{2,3}", o):
                pool.add(round(float(x), 3)); pool.add(round(float(x), 2))
    walk(nums)
    for t in TABLES.glob("*.tex"):
        for x in re.findall(r"[-+]?\d\.\d{2,3}", t.read_text()):
            pool.add(round(float(x), 3)); pool.add(round(float(x), 2))

    print("\n== 正文数字溯源(三位与两位小数)")
    seen = {}
    for m in re.finditer(r"(?<![\d.])(\d\.\d{2,3})(?![\d])", body):
        s = m.group(1)
        seen.setdefault(s, body[:m.start()].count("\n") + 1)
    untraced = []
    for s, line in sorted(seen.items(), key=lambda kv: kv[1]):
        v = float(s)
        ok = round(v, 3) in pool or (len(s) == 4 and any(abs(round(p, 2) - v) < 1e-9 for p in pool))
        src = "numbers.json/tables" if ok else WHITELIST.get(s)
        if src is None:
            untraced.append((line, s))
        else:
            print(f"  第 {line:3d} 行 {s:6s} ← {src}")
    if untraced:
        print("\n  未溯源(补白名单或改数):")
        for line, s in untraced:
            print(f"    第 {line} 行: {s}")
    print(f"\n数字 {len(seen)} 个, 未溯源 {len(untraced)} 个, 已撤回表述 {bad} 处")
    return 1 if (bad or untraced) else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: patient_level_stats.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
论文主表:患者级指标 + 置信区间 + 配置间的配对显著性检验。

为什么不是"每折算一个 macro 再求均值":
    那样有效样本量是 **5**(折数),标准差 0.07-0.14 而效应量只有 0.06-0.12,
    做不出任何显著性。

为什么可以直接汇总:
    5 折的 test 集**恰好划分**全部 79 个患者(已校验:无重复、无遗漏),
    每个患者被且只被一个"没见过他"的模型预测过一次。把 5 折的患者级预测拼起来,
    就得到 79 个独立预测 —— 这正是留一患者交叉验证想要的覆盖,而不必跑 79 折。
    区别仅在训练集大小(78 vs 约 63 个患者)。

段级预测按患者取概率均值再 argmax(软投票)聚合到患者级。

指标口径:
    macro = 各类召回的均值(平衡准确率),二分类的多数类基线是 0.5
    micro = 患者级总体正确率

区间与检验:
    micro / 逐类召回 -> Wilson 区间
    macro           -> 对患者做 bootstrap(与 macro 不是简单二项比例)
    配置两两比较    -> McNemar 精确检验(79 个配对的对错向量)

用法:
    python analysis/patient_level_stats.py --root logs/train
    python analysis/patient_level_stats.py --compare B0_3dcnn M0_concept_learned
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

DEFAULT_DATA_ROOT = "/work/SKIING/chenkaixu/data/asd_dataset"


def build_patient_map(root_path: str) -> dict[str, str]:
    """video_name -> 患者键。

    数据里有两套命名:JSON 文件名是 `<患者>-0001.json`,而内容里的 video_name
    有时是 `<患者>-0001`、有时是 `<患者>__ (1)`,下划线数量还不固定。交叉验证按
    文件名分组,预测里存的却是 video_name,靠正则对不齐(实测 fold4 会数出 40 个
    "患者",实际 17 个)。所以读一遍 json_mix 建精确映射。
    """
    mapping: dict[str, str] = {}
    for jf in Path(root_path, "clinical_CLIP_dataset", "json_mix").rglob("*.json"):
        try:
            mapping[json.loads(jf.read_text())["video_name"]] = jf.stem.split("-")[0]
        except (ValueError, KeyError, OSError):
            continue
    return mapping


def collect_patients(exp_root: Path, tag: str, pmap: dict[str, str]) -> dict[str, tuple[int, int]]:
    """把一个配置的 5 折结果汇总成 {患者: (预测, 真值)}。"""
    out: dict[str, tuple[int, int]] = {}
    for fold in range(5):
        runs = sorted(
            {p.parent.parent for p in (exp_root / f"{tag}__f{fold}_s42").rglob("best_preds/*_pred.pt")},
            key=lambda p: p.stat().st_mtime,
        )
        if not runs:
            continue
        best = runs[-1] / "best_preds"
        for pf in sorted(best.glob("*_pred.pt")):
            lf = pf.with_name(pf.name.replace("_pred.pt", "_label.pt"))
            nf = pf.with_name(pf.name.replace("_pred.pt", "_video_name.json"))
            if not (lf.exists() and nf.exists()):
                continue
            prob = torch.load(pf, map_location="cpu", weights_only=False).float()
            label = torch.load(lf, map_location="cpu", weights_only=False).long()
            names = json.loads(nf.read_text())
            if len(names) != label.numel():
                continue
            by: dict[str, list[int]] = defaultdict(list)
            for i, v in enumerate(names):
                by[pmap.get(v, v)].append(i)
            for pid, idx in by.items():
                sel = torch.tensor(idx)
                out[pid] = (int(prob[sel].mean(0).argmax()), int(label[sel].mode().values))
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - half), min(1.0, c + half)


def macro_ci(pred: np.ndarray, label: np.ndarray, n_boot: int = 10000, seed: int = 0):
    """对患者做 bootstrap 的平衡准确率区间。"""
    rng = np.random.default_rng(seed)
    classes = np.unique(label)
    n = len(label)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        rec = [((pred[i] == c) & (label[i] == c)).sum() / max((label[i] == c).sum(), 1)
               for c in classes]
        vals.append(np.mean(rec))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarize(res: dict[str, tuple[int, int]]):
    pids = sorted(res)
    pred = np.array([res[p][0] for p in pids])
    label = np.array([res[p][1] for p in pids])
    classes = np.unique(label)
    recalls = [((pred == c) & (label == c)).sum() / max((label == c).sum(), 1) for c in classes]
    correct = int((pred == label).sum())
    return {
        "n": len(pids), "pred": pred, "label": label,
        "macro": float(np.mean(recalls)),
        "macro_ci": macro_ci(pred, label),
        "micro": correct / len(pids),
        "micro_ci": wilson(correct, len(pids)),
        "recalls": recalls,
        "support": {int(c): int((label == c).sum()) for c in classes},
    }


def mcnemar(a: dict, b: dict) -> tuple[int, int, float]:
    """两个配置在同一批患者上的配对检验。返回 (a 独对, b 独对, 双侧 p)。"""
    ca, cb = a["pred"] == a["label"], b["pred"] == b["label"]
    n01 = int((ca & ~cb).sum())
    n10 = int((~ca & cb).sum())
    if n01 + n10 == 0:
        return n01, n10, 1.0
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue
    return n01, n10, float(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--tags", nargs="*", default=None, help="留空则自动发现所有配置")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("A", "B"))
    args = ap.parse_args()

    root = Path(args.root)
    pmap = build_patient_map(args.data_root)

    tags = args.tags or sorted({d.name.split("__f")[0] for d in root.glob("*__f*_s42")})
    results = {}
    for tag in tags:
        r = collect_patients(root, tag, pmap)
        if len(r) >= 70:          # 少于 70 说明有折没跑完,汇总会失真
            results[tag] = summarize(r)
        elif r:
            print(f"[跳过] {tag}: 只覆盖 {len(r)} 个患者(应为 79),折未齐")

    if not results:
        raise SystemExit("没有可汇总的结果")

    print(f"\n患者级汇总 (5 折 test 恰好划分全部患者, 每人被测一次)")
    print(f"{'实验':<24}{'macro [95% CI]':>26}{'micro [95% CI]':>26}{'n':>5}")
    print("-" * 82)
    for tag, s in sorted(results.items(), key=lambda kv: -kv[1]["macro"]):
        m, mc = s["macro"], s["macro_ci"]
        u, uc = s["micro"], s["micro_ci"]
        print(f"{tag:<24}{f'{m:.3f} [{mc[0]:.3f}, {mc[1]:.3f}]':>26}"
              f"{f'{u:.3f} [{uc[0]:.3f}, {uc[1]:.3f}]':>26}{s['n']:>5}")
    any_s = next(iter(results.values()))
    print(f"\n类别支持数(患者): {any_s['support']}   macro 基线 0.500")

    pairs = [tuple(args.compare)] if args.compare else [
        ("B0_3dcnn", "M0_concept_learned"),
        ("A1_no_grounding", "M0_concept_learned"),
        ("M0_concept_learned", "A0_shuffle_region"),
        ("M0_concept_learned", "G3_random_map"),
        ("M0_concept_learned", "A4_no_prior"),
        ("M0_concept_learned", "A2_no_presence"),
    ]
    print("\nMcNemar 配对检验 (同一批患者, 双侧精确检验)")
    print(f"{'A vs B':<44}{'A独对':>7}{'B独对':>7}{'p':>10}")
    print("-" * 68)
    for a, b in pairs:
        if a not in results or b not in results:
            continue
        n01, n10, p = mcnemar(results[a], results[b])
        star = " *" if p < 0.05 else ""
        print(f"{a + ' vs ' + b:<44}{n01:>7}{n10:>7}{p:>10.3f}{star}")


if __name__ == "__main__":
    main()

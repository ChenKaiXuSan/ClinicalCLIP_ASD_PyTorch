#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: pool_qwen_diag.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
把 eval_qwen_zeroshot_diag.py 各折的逐患者分数拼成全库 79 个患者,报患者级 AUC(bootstrap 区间)、
阈值 0 与最佳阈值下的平衡准确率。口径与 patient_level_stats.py 一致(5 折 test 恰好划分全部患者)。

    python analysis/pool_qwen_diag.py logs/qwen_analysis/diag_qwen3-vl-8b-instruct_448_fold*.json
"""
from __future__ import annotations

import json
import sys

import numpy as np


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def bal_acc(s, y, thr):
    p = (np.asarray(s) > thr).astype(int)
    y = np.asarray(y)
    r1 = (p[y == 1] == 1).mean() if (y == 1).any() else 0.0
    r0 = (p[y == 0] == 0).mean() if (y == 0).any() else 0.0
    return 0.5 * (r1 + r0)


def main() -> None:
    files = sys.argv[1:]
    patients = {}
    for f in files:
        d = json.load(open(f))
        if "patients" not in d:
            print(f"{f}: 没有逐患者分数(旧版脚本输出),跳过")
            continue
        for p, v in d["patients"].items():
            assert p not in patients, f"患者 {p} 出现在多个折"
            patients[p] = v
    if not patients:
        sys.exit("没有可用的逐患者结果")
    s = [v["score"] for v in patients.values()]
    y = [v["label"] for v in patients.values()]
    rng = np.random.default_rng(0)
    idx = np.arange(len(s))
    boots = [auc(np.asarray(s)[b], np.asarray(y)[b]) for b in (rng.choice(idx, len(idx)) for _ in range(2000))]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    best = max(bal_acc(s, y, t) for t in sorted(set(s)))
    print(f"患者 n={len(s)} (ASD {sum(y)} / non-ASD {len(y) - sum(y)})")
    print(f"患者级 AUC {auc(s, y):.3f} [95% bootstrap {lo:.3f}, {hi:.3f}]   随机 = 0.5")
    print(f"平衡准确率 @阈值0 {bal_acc(s, y, 0.0):.3f}   @最佳阈值(乐观上界) {best:.3f}")
    print(f"阈值 0 下判为 ASD 的患者比例 {np.mean(np.asarray(s) > 0):.3f}")


if __name__ == "__main__":
    main()

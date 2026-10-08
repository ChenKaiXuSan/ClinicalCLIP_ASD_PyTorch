#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: bout_from_clips.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
设计 B("行走过程中的动态代偿")的零成本版本:短片段的编号就是同一次录像里经过的时间顺序
(35 个 session 里相邻编号的行进方向交替率中位 0.97,见 docs/bout.md),所以把每个片段已经算好的
医生区域量按编号排列,就得到患者在一次录像里的轨迹,不需要任何新处理。

每名患者(= 一次录像):片段级值 = 该片段各段的均值;按编号排序,时间轴用归一化秩 r ∈ [0, 1]。
  static_<q>   整次录像均值(即 MICCAI 用的静态量)
  slope_<q>    值对 r 的最小二乘斜率 = 从开始到结束的总变化量(单位同该量)
  lastfirst_<q> 末四分之一片段均值 - 首四分之一片段均值
  rho_<q>      编号与值的 Spearman 相关
  z_<q>        斜率相对"片段顺序随机打乱"零分布的 z 分数(每人 N_PERM 次)
组间比较 ASD vs non-ASD(Mann-Whitney,单量 AUC),再做 10 份划分逻辑回归:静态 5 量 / 静态 + 动态 / 只动态。

    python analysis/bout_from_clips.py --attr-root logs/geom_attributes_3d_doctor --exclude-missing logs/measure_bank_3d/bank.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gait_phase import repeated_lr  # noqa: E402

DOCTOR5 = ["trunk_lean", "shoulder_offset", "head_forward", "hip_flexion_max", "hip_range"]
N_PERM = 300
MIN_CLIPS = 6


def load_clips(attr_root: Path, attrs: list[str]) -> dict[str, dict]:
    """患者 -> {y, clips: {idx: {attr: 片段均值}}}"""
    pat: dict[str, dict] = {}
    for a in attrs:
        for v, rec in json.load(open(attr_root / f"{a}.json")).items():
            stem = Path(rec["json"]).stem
            pid, idx = stem.split("-")[0], int(stem.split("-")[1])
            d = pat.setdefault(pid, {"y": int(rec["label"] == 0), "clips": {}})
            d["clips"].setdefault(idx, {})[a] = float(np.mean(rec["scores"]))
    return pat


def dynamics(pat: dict[str, dict], attrs: list[str], rng: np.random.Generator) -> dict[str, dict]:
    out = {}
    for pid, d in pat.items():
        idxs = sorted(d["clips"])
        n = len(idxs)
        row = {"y": d["y"], "n_clips": n}
        r = (np.arange(n) / max(n - 1, 1)) if n > 1 else np.zeros(1)
        k = max(1, n // 4)
        for a in attrs:
            v = np.array([d["clips"][i].get(a, np.nan) for i in idxs])
            row[f"static_{a}"] = float(np.nanmean(v))
            if n < MIN_CLIPS or np.isnan(v).any():
                for key in ("slope", "lastfirst", "rho", "z", "sd_resid"):
                    row[f"{key}_{a}"] = np.nan
                continue
            A = np.c_[r, np.ones(n)]
            slope = float(np.linalg.lstsq(A, v, rcond=None)[0][0])
            row[f"slope_{a}"] = slope
            row[f"lastfirst_{a}"] = float(v[-k:].mean() - v[:k].mean())
            row[f"rho_{a}"] = float(spearmanr(r, v)[0])
            resid = v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
            row[f"sd_resid_{a}"] = float(resid.std())
            null = np.array([np.linalg.lstsq(A, rng.permutation(v), rcond=None)[0][0] for _ in range(N_PERM)])
            row[f"z_{a}"] = float((slope - null.mean()) / (null.std() + 1e-9))
        out[pid] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr-root", default="logs/geom_attributes_3d_doctor")
    ap.add_argument("--exclude-missing", default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default=None, help="患者级动态量 json")
    args = ap.parse_args()

    pat = load_clips(Path(args.attr_root), DOCTOR5)
    miss = set()
    if args.exclude_missing:
        bank = pickle.load(open(args.exclude_missing, "rb"))
        miss = {Path(v["json"]).stem.split("-")[0] for v in bank["videos"].values() if v["frac_ok"] == 0}
    pat = {p: d for p, d in pat.items() if p not in miss}
    rows = dynamics(pat, DOCTOR5, np.random.default_rng(0))
    pids = sorted(rows)
    y = np.array([rows[p]["y"] for p in pids])
    nclips = np.array([rows[p]["n_clips"] for p in pids])
    ok = nclips >= MIN_CLIPS
    print(f"患者 {len(pids)} (ASD {y.sum()});片段数 中位 {np.median(nclips):.0f} 范围 [{nclips.min()}, {nclips.max()}];"
          f"片段 ≥ {MIN_CLIPS} 的 {ok.sum()} 人 (ASD {y[ok].sum()} / non-ASD {(1 - y[ok]).sum()})")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)

    def col(key):
        return np.array([rows[p].get(key, np.nan) for p in pids])

    print(f"\n{'量':18s} {'ASD 中位':>9s} {'non 中位':>9s} {'MWU p':>7s} {'AUC':>6s}   (片段 ≥ {MIN_CLIPS} 的患者;AUC 取 max(a, 1-a) 并标方向)")
    for a in DOCTOR5:
        for key, lab in (("static", "静态均值"), ("slope", "斜率(总变化)"), ("lastfirst", "末-首 1/4"), ("rho", "Spearman"), ("z", "斜率 z"), ("sd_resid", "去趋势波动")):
            v = col(f"{key}_{a}")
            m = ok & np.isfinite(v)
            if m.sum() < 10:
                continue
            p = mannwhitneyu(v[m & (y == 1)], v[m & (y == 0)]).pvalue
            auc = roc_auc_score(y[m], v[m])
            print(f"{a + ' ' + lab:24s} {np.median(v[m & (y == 1)]):+9.2f} {np.median(v[m & (y == 0)]):+9.2f} {p:7.3f} {max(auc, 1 - auc):6.3f} {'ASD高' if auc > 0.5 else 'ASD低'}")
        print()

    # 有多少患者的趋势超出随机打乱的零分布(|z| > 1.96)
    for a in ("trunk_lean", "head_forward", "shoulder_offset"):
        z = col(f"z_{a}"); m = ok & np.isfinite(z)
        for c, nm in ((1, "ASD"), (0, "non-ASD")):
            mm = m & (y == c)
            print(f"{a:16s} {nm:8s}: |z|>1.96 的比例 {np.mean(np.abs(z[mm]) > 1.96):.2f} (n={mm.sum()}), 斜率为正(前倾增加)的比例 {np.mean(col(f'slope_{a}')[mm] > 0):.2f}")

    # 分类:静态 5 / 静态 + 动态 / 只动态;动态量缺失(片段不足)用中位数填
    def matrix(keys):
        M = np.array([[rows[p].get(k, np.nan) for k in keys] for p in pids], dtype=float)
        med = np.nanmedian(M, 0)
        return np.where(np.isnan(M), med, M)
    static = [f"static_{a}" for a in DOCTOR5]
    dyn_slope = [f"slope_{a}" for a in DOCTOR5]
    dyn_lf = [f"lastfirst_{a}" for a in DOCTOR5]
    dyn_sd = [f"sd_resid_{a}" for a in DOCTOR5]
    sets = [("静态 5 量(均值)", static), ("静态 5 + 斜率 5", static + dyn_slope), ("静态 5 + 末-首 5", static + dyn_lf),
            ("静态 5 + 去趋势波动 5", static + dyn_sd), ("只斜率 5", dyn_slope), ("只末-首 5", dyn_lf), ("只去趋势波动 5", dyn_sd),
            ("静态 + 斜率 + 波动 (15)", static + dyn_slope + dyn_sd)]
    print(f"\n{'集合':28s} {'macro 均值±std':>16s} {'AUC':>7s}   L1(C=0.5)")
    for name, keys in sets:
        X = matrix(keys)
        m, a = repeated_lr(X, y, args.n, 1.0); m1, a1 = repeated_lr(X, y, args.n, 0.5, "l1")
        print(f"{name:28s} {m.mean():.3f} ± {m.std():.3f}   {a.mean():.3f}   {m1.mean():.3f} / {a1.mean():.3f}")

    # 顺序打乱对照:动态量在片段顺序随机化后重算(所有患者同一次打乱),看"静态 + 动态"是否掉回静态
    print("\n顺序打乱对照(片段顺序随机化后重算动态量, 5 次):")
    vals = []
    for s in range(5):
        rng = np.random.default_rng(100 + s)
        shuffled = {p: {"y": d["y"], "clips": {i: d["clips"][j] for i, j in zip(sorted(d["clips"]), rng.permutation(sorted(d["clips"])))}} for p, d in pat.items()}
        rows_s = dynamics(shuffled, DOCTOR5, np.random.default_rng(s))
        M = np.array([[rows_s[p].get(k, np.nan) for k in static + dyn_slope] for p in pids], dtype=float)
        M = np.where(np.isnan(M), np.nanmedian(M, 0), M)
        m, a = repeated_lr(M, y, args.n, 1.0)
        vals.append((m.mean(), a.mean()))
    print(f"  静态 5 + 打乱后的斜率 5: macro {np.mean([v[0] for v in vals]):.3f} ± {np.std([v[0] for v in vals]):.3f}  AUC {np.mean([v[1] for v in vals]):.3f}")


if __name__ == "__main__":
    main()

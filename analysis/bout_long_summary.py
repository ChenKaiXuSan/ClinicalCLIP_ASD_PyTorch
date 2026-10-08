#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: bout_long_summary.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
汇总 bout_trunk.py 在侧视长录像上的输出(logs/bout_trunk/<类>__<日期>__<文件>.json):
每次录像 -> 只用单人同框的经过 -> 三个量对"自第一次经过起的分钟数"的斜率 / 末-首 1/4 / Spearman / 去趋势波动,
按患者(录像日期)汇总后比较 ASD vs non-ASD;年轻正常人给出斜率的噪声地板;与短片段顺序版(bout_from_clips)的
同一患者动态量对比,看两个独立来源是否一致。

    python analysis/bout_long_summary.py --exclude-missing logs/measure_bank_3d/bank.pkl --clips-json logs/bout_trunk/bout_from_clips_3d.json
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

QS = ["trunk_lean", "head_forward", "shoulder_offset", "hip_flexion_max", "hip_range", "knee_flexion_max"]
MIN_PASS = 6


def rec_dynamics(passes: list[dict], single_only: bool) -> dict | None:
    ps = [p for p in passes if (p["n_person_max"] == 1 or not single_only) and p.get("trunk_lean") is not None]
    if len(ps) < MIN_PASS:
        return None
    t0 = ps[0]["t_start"]
    t = np.array([(p["t_start"] - t0) / 60.0 for p in ps])
    out = {"n_pass": len(ps), "span_min": float(t[-1]), "frac_multi": float(np.mean([p["n_person_max"] > 1 for p in passes]))}
    k = max(1, len(ps) // 4)
    for q in QS:
        v = np.array([p.get(q) if p.get(q) is not None else np.nan for p in ps], dtype=float)
        m = np.isfinite(v)
        if m.sum() < MIN_PASS:
            continue
        A = np.c_[t[m], np.ones(m.sum())]
        coef = np.linalg.lstsq(A, v[m], rcond=None)[0]
        out[f"static_{q}"] = float(np.nanmean(v))
        out[f"slope_{q}"] = float(coef[0])                       # 每分钟
        out[f"total_{q}"] = float(coef[0] * t[m][-1])            # 整次录像的总变化
        out[f"lastfirst_{q}"] = float(np.nanmean(v[-k:]) - np.nanmean(v[:k]))
        out[f"rho_{q}"] = float(spearmanr(t[m], v[m])[0])
        out[f"sd_resid_{q}"] = float((v[m] - A @ coef).std())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/bout_trunk")
    ap.add_argument("--exclude-missing", default=None)
    ap.add_argument("--clips-json", default=None, help="bout_from_clips 的患者级输出,用于两个来源的对比")
    ap.add_argument("--all-passes", action="store_true", help="不限制单人同框")
    args = ap.parse_args()

    recs = {}
    for f in sorted(glob.glob(f"{args.root}/*__*.json")):
        d = json.load(open(f))
        tag = Path(f).stem
        cls, sess, fname = tag.split("__", 2)
        dyn = rec_dynamics(d["passes"], single_only=not args.all_passes)
        recs[tag] = {"cls": cls, "sess": sess, "file": fname, "dyn": dyn, "n_pass_all": len(d["passes"]),
                     "frac_multi": float(np.mean([p["n_person_max"] > 1 for p in d["passes"]])) if d["passes"] else 1.0}
    n_ok = sum(1 for r in recs.values() if r["dyn"])
    print(f"录像 {len(recs)} 条,单人经过 ≥ {MIN_PASS} 的 {n_ok} 条")
    for cls in ("ASD", "DHS", "LCS", "HipOA", "normal"):
        rr = [r for r in recs.values() if r["cls"] == cls]
        ok = [r for r in rr if r["dyn"]]
        print(f"  {cls:7s} 录像 {len(rr):3d}, 可用 {len(ok):3d}, 多人同框经过占比中位 {np.median([r['frac_multi'] for r in rr]):.2f}, "
              f"可用录像的经过数中位 {np.median([r['dyn']['n_pass'] for r in ok]) if ok else 0:.0f}, 时长中位 {np.median([r['dyn']['span_min'] for r in ok]) if ok else 0:.1f} min")

    # 患者 = 类 + 日期(一个日期一次录像;同日两人的目录名带 _1/_2,与片段命名一致)
    pat = defaultdict(list)
    for r in recs.values():
        if r["dyn"]:
            pat[(r["cls"], r["sess"])].append(r["dyn"])
    rows = {}
    for (cls, sess), ds in pat.items():
        row = {"cls": cls, "y": int(cls == "ASD")}
        for key in ds[0]:
            vals = [d[key] for d in ds if key in d]
            row[key] = float(np.mean(vals)) if vals else np.nan
        rows[(cls, sess)] = row
    keys = sorted(rows)
    y = np.array([rows[k]["y"] for k in keys])
    is_pat = np.array([rows[k]["cls"] != "normal" for k in keys])
    is_norm = ~is_pat
    print(f"\n患者 {is_pat.sum()} (ASD {y[is_pat].sum()} / non-ASD {(is_pat & (y == 0)).sum()}),正常人 {is_norm.sum()}")

    def col(key):
        return np.array([rows[k].get(key, np.nan) for k in keys])

    print(f"\n{'量':30s} {'ASD 中位':>9s} {'non 中位':>9s} {'正常中位':>9s} {'MWU p':>7s} {'AUC':>6s}")
    for q in QS:
        for key, lab in (("static", "静态均值"), ("slope", "斜率 /min"), ("total", "总变化"), ("lastfirst", "末-首 1/4"), ("rho", "Spearman"), ("sd_resid", "去趋势波动")):
            v = col(f"{key}_{q}")
            m = is_pat & np.isfinite(v)
            if m.sum() < 10 or (m & (y == 0)).sum() < 3:
                continue
            p = mannwhitneyu(v[m & (y == 1)], v[m & (y == 0)]).pvalue
            auc = roc_auc_score(y[m], v[m])
            nv = v[is_norm & np.isfinite(v)]
            print(f"{q + ' ' + lab:30s} {np.median(v[m & (y == 1)]):+9.2f} {np.median(v[m & (y == 0)]):+9.2f} {np.median(nv) if len(nv) else float('nan'):+9.2f} {p:7.3f} {max(auc, 1 - auc):6.3f} {'ASD高' if auc > 0.5 else 'ASD低'}")
        print()
    # 噪声地板:正常人斜率的分布 vs 患者
    for q in QS:
        s = col(f"slope_{q}")
        nv, pv = s[is_norm & np.isfinite(s)], s[is_pat & np.isfinite(s)]
        if len(nv):
            thr = np.percentile(np.abs(nv), 95)
            print(f"{q:16s} 正常人斜率 中位 {np.median(nv):+.2f} °/min, |斜率| 95 分位 {thr:.2f};患者 |斜率| 超过它的比例 ASD {np.mean(np.abs(pv[y[is_pat & np.isfinite(s)] == 1]) > thr):.2f} / non-ASD {np.mean(np.abs(pv[y[is_pat & np.isfinite(s)] == 0]) > thr):.2f}")

    # 与短片段顺序版的对比(同一患者)
    if args.clips_json:
        clips = json.load(open(args.clips_json))
        # 片段患者键形如 20170130_ASD_lat_ / 20160523_1_ASD_lat__V1 -> 日期部分
        def date_of(pid):
            for c in ("_ASD", "_DHS", "_LCS", "_HipOA"):
                if c in pid:
                    return pid.split(c)[0]
            return pid
        cl = {date_of(p): d for p, d in clips.items()}
        for q, cq in (("trunk_lean", "trunk_lean"), ("shoulder_offset", "shoulder_offset"), ("head_forward", "head_forward"),
                      ("hip_flexion_max", "hip_flexion_max"), ("hip_range", "hip_range")):
            a, b, lab = [], [], []
            for k in keys:
                if rows[k]["cls"] == "normal" or k[1] not in cl:
                    continue
                va, vb = rows[k].get(f"lastfirst_{q}", np.nan), cl[k[1]].get(f"lastfirst_{cq}", np.nan)
                if np.isfinite(va) and np.isfinite(vb):
                    a.append(va); b.append(vb); lab.append(rows[k]["y"])
            if len(a) >= 8:
                print(f"{q:16s} 长录像 vs 短片段顺序 的 末-首 变化:同一患者 n={len(a)}, Spearman {spearmanr(a, b)[0]:+.2f};静态均值 Spearman "
                      f"{spearmanr([rows[k].get(f'static_{q}', np.nan) for k in keys if rows[k]['cls'] != 'normal' and k[1] in cl], [cl[k[1]].get(f'static_{cq}', np.nan) for k in keys if rows[k]['cls'] != 'normal' and k[1] in cl], nan_policy='omit')[0]:+.2f}")


if __name__ == "__main__":
    main()

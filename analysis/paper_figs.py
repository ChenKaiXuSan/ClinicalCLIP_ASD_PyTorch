#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: paper_figs.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
论文图表一键生成(不重跑训练,只读 logs/ 里已有的预测、概率、属性与测量库结果):
  fig2    随机 5 量分布(三种池)+ 手挑 5 量位置;测量驱动变体 10 份划分条形图
  fig3    证据门控分层:离群度三档 × {视频, 几何, 门控}
  fig4    2–3 名患者的可读解释卡片(矢状面骨架、5 个数与常模、离群度、门控判断)
  tables  Table 1–3 的 LaTeX 片段 + 全部数字的 numbers.json(供正文审计)
  all     以上全部
口径:67 名 3D 完整患者;视频 / 骨架 50 轮 × 3 种子;测量 10 份随机划分;融合 固定划分 × 视频种子。

    python analysis/paper_figs.py all --data-root $DATA
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from attribute_repeated_splits import DOCTOR5, OTHER4, SETS, load as load_attr, macro as macro_np  # noqa: E402
from patient_level_stats import build_patient_map  # noqa: E402
from patient_metrics import load_tag, metrics  # noqa: E402
from uncertainty_fusion import fuse, geom_reliability, load_geom, load_video, outl_by_fold, run_one  # noqa: E402

BANK = ROOT / "logs/measure_bank_3d/bank.pkl"
GEOM_JSON = ROOT / "logs/geom_probs/geom3d_doctor_lr_probs.json"
ATTR_DIR = ROOT / "logs/geom_attributes_3d_doctor"
SAM3D = Path("/work/1/SKIING/chenkaixu/luoxi_data/luoxi_data/data/asd_dataset/skeleton_rgb_dataset/sam3d_body_results")
VIDEO_BRANCHES = ["B0_3dcnn_e50", "M0_concept_learned_e50", "A1_no_grounding_e50"]
SEEDS = [42, 1337, 2024]
LAMS, GRID = [0.3, 1, 3, 10], np.round(np.arange(0, 1.0001, 0.05), 2)
ATTR_LABEL = {"trunk_lean": "trunk lean (°)", "head_forward": "head forward (°)", "shoulder_offset": "shoulder offset (/torso)",
              "hip_flexion_max": "max hip flexion (°)", "hip_range": "hip range (°)"}

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C_VIDEO, C_GEOM, C_GATE, C_GREY = "#4C72B0", "#DD8452", "#55A868", "#8C8C8C"


def missing_patients() -> set:
    bank = pickle.load(open(BANK, "rb"))
    return {Path(v["json"]).stem.split("-")[0] for v in bank["videos"].values() if v["frac_ok"] == 0}


def rep_attr(attr_root: Path, attrs: list[str], drop: set, n: int = 10, C: float = 1.0):
    """属性级患者 LR,10 份划分:返回 (macro 数组, auc 数组)。与 attribute_repeated_splits.py 完全一致。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    pids, X, y = load_attr(attr_root, attrs)
    keep = np.array([p not in drop for p in pids])
    Xk, yk = X[keep], y[keep]
    ms, aucs = [], []
    for s in range(n):
        pred, prob = np.zeros(len(yk), int), np.zeros(len(yk))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=1000 + s).split(Xk, yk):
            sc = StandardScaler().fit(Xk[tr])
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000).fit(sc.transform(Xk[tr]), yk[tr])
            pred[te] = clf.predict(sc.transform(Xk[te])); prob[te] = clf.predict_proba(sc.transform(Xk[te]))[:, 1]
        ms.append(macro_np(pred, yk)); aucs.append(roc_auc_score(yk, prob))
    return np.array(ms), np.array(aucs)


def pm(v):
    return f"{np.mean(v):.3f}$\\pm${np.std(v):.3f}"


# ----------------------------------------------------------------------------- 融合(所有图表共用)
def fusion_runs(pmap, drop, branches=VIDEO_BRANCHES, seeds=SEEDS, light=False):
    G = load_geom(str(GEOM_JSON))
    R = {k: v for k, v in geom_reliability(ATTR_DIR, pmap).items() if k not in drop}
    out = {}
    for b in branches:
        for s in seeds:
            V = load_video(ROOT / "logs/train", b, s, pmap)
            V = {k: v for k, v in V.items() if k not in drop}
            out[(b, s)] = run_one(V, G, R, LAMS, GRID, light=light)
    return out, G, R


# ----------------------------------------------------------------------------- Fig 2
def fig2(out_dir: Path, drop: set, numbers: dict):
    pools = {"all 1,272": "全库", "clinician regions": "医生区域内", "unmarked regions": "非医生区域内"}
    rnd = {k: np.load(ROOT / f"logs/geom_probs/p0/random5_{v}_nomiss_segment.npy") for k, v in pools.items()}
    hand_fixed = 0.768  # 手挑 5 量, 同一固定划分, 段级, 67 人(docs "完整指标(67 人,固定划分)")
    rep = json.load(open(ROOT / "logs/geom_probs/p0/repeated_splits_nomiss_auc.json"))
    nl = json.load(open(ROOT / "logs/geom_probs/p1_nomiss/nonlinear_repeated_nomiss.json"))
    m5, a5 = rep_attr(ROOT / "logs/geom_attributes_3d", DOCTOR5, drop)
    m4, a4 = rep_attr(ROOT / "logs/geom_attributes_3d", OTHER4, drop)
    bars = [("clinician 5 + LR", m5, C_GATE), ("unmarked 4 + LR", m4, C_GREY),
            ("L1, clinician regions", rep["macro"]["strict_doctor"], C_GEOM), ("L1, all 1,272", rep["macro"]["uniform"], C_GEOM),
            ("L1, unmarked regions", rep["macro"]["hard_other"], C_GREY), ("RF, all 1,272", nl["rf/全库"]["macro"], C_GEOM),
            ("GB, all 1,272", nl["hgb/全库"]["macro"], C_GEOM)]
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.8), gridspec_kw={"width_ratios": [1.0, 1.35]})
    ax = axes[0]
    bins = np.linspace(0.45, 0.85, 25)
    for (k, v), c in zip(rnd.items(), ("#7F7F7F", "#BCBD22", "#C7C7C7")):
        ax.hist(v, bins=bins, alpha=0.5, color=c, label=f"random 5, {k}")
    ax.axvline(hand_fixed, color=C_GATE, lw=1.8, ls="--")
    ax.set_xlabel("balanced accuracy (fixed split)", fontsize=7); ax.set_ylabel("draws (100 per pool)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=5.5, loc="upper left", frameon=False)
    ax.set_title("(a) random five vs. clinician five (dashed)", fontsize=7, loc="left")
    ax = axes[1]
    x = np.arange(len(bars))
    ax.bar(x, [np.mean(v) for _, v, _ in bars], yerr=[np.std(v) for _, v, _ in bars], color=[c for _, _, c in bars], capsize=2, width=0.7)
    ax.set_xticks(x); ax.set_xticklabels([n for n, _, _ in bars], fontsize=5.5, rotation=32, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylim(0.55, 0.82); ax.set_ylabel("balanced accuracy\n(10 random splits)", fontsize=7)
    ax.axhline(np.mean(m5), color=C_GATE, lw=0.8, ls=":")
    ax.set_title("(b) selection over 1,272 candidates", fontsize=7, loc="left")
    fig.tight_layout(); fig.savefig(out_dir / "fig2_selection.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(out_dir / "fig2_selection.png", bbox_inches="tight"); plt.close(fig)
    numbers["fig2"] = {"random5_mean": {k: float(v.mean()) for k, v in rnd.items()},
                       "random5_pct_of_hand": {k: float((v < hand_fixed).mean()) for k, v in rnd.items()},
                       "hand5_fixed_segment": hand_fixed, "bars": {n.replace("\n", " "): pm(v) for n, v, _ in bars},
                       "hand5_auc_10split": pm(a5), "other4_auc_10split": pm(a4)}
    print("fig2 ok")


# ----------------------------------------------------------------------------- Fig 3
def fig3(out_dir: Path, runs: dict, numbers: dict):
    tot = defaultdict(lambda: np.zeros(5))  # v, g, gate, n, asd
    for (b, s), r in runs.items():
        y, pr, U = r["y"], r["pred"], r["U"]
        outl = U[:, 5]
        q = np.quantile(outl, [1 / 3, 2 / 3]); band = np.digitize(outl, q)
        for k in range(3):
            m = band == k
            tot[k] += [(pr["video"][m] == y[m]).sum(), (pr["geom"][m] == y[m]).sum(), (pr["step_outl"][m] == y[m]).sum(), m.sum(), (y[m] == 0).sum()]
    fig, ax = plt.subplots(figsize=(3.2, 2.1))
    x = np.arange(3); w = 0.26
    names = ["video branch", "5 measurements", "evidence gate"]
    for j, (c, n) in enumerate(zip((C_VIDEO, C_GEOM, C_GATE), names)):
        vals = [tot[k][j] / tot[k][3] for k in range(3)]
        ax.bar(x + (j - 1) * w, vals, w, color=c, label=n)
        for xi, v in zip(x + (j - 1) * w, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([f"{t}\n(ASD {tot[k][4] / tot[k][3]:.0%})" for k, t in enumerate(("low", "mid", "high"))], fontsize=7)
    ax.set_xlabel("deviation from the norm, $d$ (terciles)", fontsize=7)
    ax.set_ylabel("patient-level accuracy", fontsize=7); ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylim(0.3, 1.02); ax.legend(fontsize=6, loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(out_dir / "fig3_gate_strata.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(out_dir / "fig3_gate_strata.png", bbox_inches="tight"); plt.close(fig)
    numbers["fig3"] = {k: {"video": tot[k][0] / tot[k][3], "geom": tot[k][1] / tot[k][3], "gate": tot[k][2] / tot[k][3],
                           "n": int(tot[k][3]), "asd_frac": tot[k][4] / tot[k][3]} for k in range(3)}
    print("fig3 ok")


# ----------------------------------------------------------------------------- Fig 4
_EDGES = [(3, 5), (4, 6), (5, 6), (5, 9), (6, 10), (9, 10), (9, 11), (11, 13), (10, 12), (12, 14), (5, 7), (6, 8)]
_JOINTS = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


def _stick(ax, pid: str, attr_json: dict):
    """该患者第一条视频中间帧的矢状面骨架(x 行进方向, y 向上)。只画躯干 / 头 / 腿 / 上臂,手腕估计不稳不画。"""
    vids = [v for v, rec in attr_json.items() if Path(rec["json"]).stem.split("-")[0] == pid]
    ax.axis("off")
    for v in vids:
        fs = sorted(glob.glob(f"{SAM3D}/*/{v}/video/*_sam3d_body.npz"))
        if not fs:
            continue
        k = np.load(fs[len(fs) // 2], allow_pickle=True)["output"].item()["pred_keypoints_3d"]
        pts = np.c_[k[_JOINTS, 0], -k[_JOINTS, 1]]
        for a, b in _EDGES:
            ax.plot([k[a, 0], k[b, 0]], [-k[a, 1], -k[b, 1]], color="k", lw=1.3, solid_capstyle="round")
        ax.scatter(pts[:, 0], pts[:, 1], s=7, color="k", zorder=3)
        # 铅垂参考线(髋中点)
        hip = (k[9] + k[10]) / 2
        ax.plot([hip[0], hip[0]], [-hip[1] - 0.05, -hip[1] + 0.9], color=C_GREY, lw=0.8, ls=":")
        lo, hi = pts.min(0) - 0.1, pts.max(0) + 0.1
        span = max(hi - lo)
        c = (lo + hi) / 2
        ax.set_xlim(c[0] - span / 2, c[0] + span / 2); ax.set_ylim(c[1] - span / 2, c[1] + span / 2)
        ax.set_aspect("equal")
        return


def fig4(out_dir: Path, runs: dict, G: dict, R: dict, pmap: dict, numbers: dict, branch=("M0_concept_learned_e50", 2024)):
    r = runs[branch]
    pids, y, pr, U = r["pids"], r["y"], r["pred"], r["U"]
    V = load_video(ROOT / "logs/train", branch[0], branch[1], pmap)
    fold = np.array([V[p]["fold"] for p in pids])
    step = r["chosen_step"]  # 逐折 (阈值, w_lo, w_hi)
    attr_json = json.load(open(ATTR_DIR / "trunk_lean.json"))
    per = {a: json.load(open(ATTR_DIR / f"{a}.json")) for a in DOCTOR5}
    miss = missing_patients()
    allv = {a: np.concatenate([np.asarray(rec["scores"]) for v, rec in per[a].items() if Path(rec["json"]).stem.split("-")[0] not in miss]) for a in DOCTOR5}
    mu, sd = {a: allv[a].mean() for a in DOCTOR5}, {a: allv[a].std() for a in DOCTOR5}
    outl = U[:, 5]
    thr_i = np.array([step[f][0] for f in fold])
    w_i = np.array([step[f][1] if outl[i] < step[f][0] else step[f][2] for i, f in enumerate(fold)])
    ok_v, ok_g, ok_f = pr["video"] == y, pr["geom"] == y, pr["step_outl"] == y
    # 三个例子:几何强证据判对且视频错;几何不离群、视频救回;门控也错的失败例
    cands = [np.where(ok_g & ~ok_v & (outl >= thr_i))[0], np.where(ok_v & ~ok_g & (outl < thr_i) & ok_f)[0], np.where(~ok_f)[0]]
    titles = ["(a) deviating measurements decide\n     (video was wrong)",
              "(b) unremarkable: video decides\n     (measurements were wrong)",
              "(c) failure case"]
    picks = [int(c[np.argsort(-np.abs(outl[c] - thr_i[c]))[0]]) if len(c) else None for c in cands]
    fig = plt.figure(figsize=(4.8, 2.0))
    rows = []
    for j, (i, t) in enumerate(zip(picks, titles)):
        if i is None:
            continue
        p = pids[i]
        x0 = 0.01 + j * 0.335
        fig.text(x0, 0.99, t, fontsize=5.6, va="top", weight="bold")
        ax = fig.add_axes([x0 + 0.10, 0.50, 0.12, 0.36]); _stick(ax, p, attr_json)
        vals = {a: np.mean(np.concatenate([np.asarray(rec["scores"]) for v, rec in per[a].items() if Path(rec["json"]).stem.split("-")[0] == p])) for a in DOCTOR5}
        short = {"trunk_lean": "trunk lean (°)", "head_forward": "head forward (°)", "shoulder_offset": "shoulder offset", "hip_flexion_max": "hip flexion max", "hip_range": "hip range (°)"}
        lines = [f"{short[a]:16s}{vals[a]:6.2f}  z{(vals[a] - mu[a]) / sd[a]:+.1f}" for a in DOCTOR5]
        trust = "measurements" if outl[i] >= thr_i[i] else "video"
        pf = fuse(V[p]["p"][None], G[p]["p"][None], np.array([w_i[i]]))[0, 0]
        lines += [f"mean|z| {outl[i]:.2f} {'≥' if outl[i] >= thr_i[i] else '<'} thr {thr_i[i]:.2f}",
                  f"→ trust {trust} (w={w_i[i]:.2f})",
                  f"P(ASD) video {V[p]['p'][0]:.2f} meas {G[p]['p'][0]:.2f}",
                  f"       fused {pf:.2f}  truth {'ASD' if y[i] == 0 else 'non-ASD'}",
                  f"gate {'✓' if ok_f[i] else '✗'}  video {'✓' if ok_v[i] else '✗'}  meas {'✓' if ok_g[i] else '✗'}"]
        fig.text(x0, 0.47, "\n".join(lines), fontsize=5.0, va="top", family="monospace", linespacing=1.28)
        rows.append({"patient": p, "vals": vals, "outl": float(outl[i]), "thr": float(thr_i[i]), "w": float(w_i[i]),
                     "p_video": float(V[p]["p"][0]), "p_geom": float(G[p]["p"][0]), "p_fused": float(pf), "label_asd": bool(y[i] == 0)})
    fig.savefig(out_dir / "fig4_explanations.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(out_dir / "fig4_explanations.png", bbox_inches="tight"); plt.close(fig)
    numbers["fig4"] = {"branch": f"{branch[0]} s{branch[1]}", "norm_mean": mu, "norm_std": sd, "examples": rows}
    print("fig4 ok", [r["patient"] for r in rows])


# ----------------------------------------------------------------------------- Tables
def tables(out_dir: Path, pmap: dict, drop: set, runs: dict, numbers: dict, runs_b5b: dict | None):
    # Table 1 视频 / 骨架行
    tags = [("3D CNN (slow\\_r50, K400)", "B0_3dcnn_e50", "video"), ("TimeSformer-B (K400, 8 frames)", "B5b_timesformer_lr2e5_e50", "video"),
            ("CLIP-style alignment (annotation at test)", "B4_clip_old_e50", "video+annot."),
            ("3D CNN + region grounding (Sec.~3.1)", "M0_concept_learned_e50", "video"),
            ("ST-GCN, 2D skeleton", "B3_pose_e50", "skeleton"), ("ST-GCN, 3D skeleton", "B3_pose_3d_e50", "skeleton"),
            ("CTR-GCN, 2D skeleton", "B6_ctrgcn_e50", "skeleton"), ("CTR-GCN, 3D skeleton", "B6_ctrgcn_3d_e50", "skeleton")]
    t1 = {}
    for name, tag, inp in tags:
        ms, aucs = [], []
        for s in SEEDS:
            P, yy = load_tag(str(ROOT / "logs/train"), tag, pmap, s, drop)
            m = metrics(P, yy); ms.append(m["macro"]); aucs.append(m["auc"])
        t1[name] = (inp, pm(ms), pm(aucs))
    rep = json.load(open(ROOT / "logs/geom_probs/p0/repeated_splits_nomiss_auc.json"))
    nl = json.load(open(ROOT / "logs/geom_probs/p1_nomiss/nonlinear_repeated_nomiss.json"))
    m5, a5 = rep_attr(ROOT / "logs/geom_attributes_3d", DOCTOR5, drop)
    m4, a4 = rep_attr(ROOT / "logs/geom_attributes_3d", OTHER4, drop)
    m5_2d, a5_2d = rep_attr(ROOT / "logs/geom_attributes", DOCTOR5, drop)
    t1.update({
        "Random forest, 1,272 measurements": ("3D meas.", pm(nl["rf/全库"]["macro"]), pm(nl["rf/全库"]["auc"])),
        "Gradient boosting, 1,272 measurements": ("3D meas.", pm(nl["hgb/全库"]["macro"]), pm(nl["hgb/全库"]["auc"])),
        "L1 logistic regression, 1,272 measurements": ("3D meas.", pm(rep["macro"]["uniform"]), pm(rep["auc"]["uniform"])),
        "L1 logistic regression, 300 within clinician regions": ("3D meas.", pm(rep["macro"]["strict_doctor"]), pm(rep["auc"]["strict_doctor"])),
        "4 quantities at unmarked regions + LR": ("3D meas.", pm(m4), pm(a4)),
        "\\textbf{5 clinician-defined quantities + LR}": ("3D meas.", "\\textbf{" + pm(m5) + "}", "\\textbf{" + pm(a5) + "}"),
        "same 5 quantities from the 2D skeleton": ("2D meas.", pm(m5_2d), pm(a5_2d)),
    })
    # 融合行(9 组)
    def agg(key):
        return np.mean([macro_np(r["pred"][key], r["y"]) for r in runs.values()])
    fus = {k: agg(k) for k in ("video", "geom", "fixed0.5", "nested_w", "stack", "gate", "step_outl")}
    m0 = np.mean([macro_np(r["pred"]["step_outl"], r["y"]) for (b, s), r in runs.items() if b.startswith("M0")])
    lines = ["\\begin{tabular}{llcc}", "\\toprule", "Method & Input & Balanced acc. & AUC \\\\", "\\midrule"]
    for name, (inp, m, a) in t1.items():
        lines.append(f"{name} & {inp} & {m} & {a} \\\\")
    lines += ["\\midrule",
              f"5 quantities + video, fixed $w{{=}}0.5$ / nested global $w$ / sigmoid gate & fusion & {fus['fixed0.5']:.3f} / {fus['nested_w']:.3f} / {fus['gate']:.3f} & \\\\",
              f"\\textbf{{5 quantities + video, evidence gate}} & fusion & \\textbf{{{fus['step_outl']:.3f}}} (grounded video: {m0:.3f}) & \\\\",
              "\\bottomrule", "\\end{tabular}"]
    (out_dir / "table1_main.tex").write_text("\n".join(lines) + "\n")
    numbers["table1"] = {k: v for k, v in t1.items()} | {"fusion_9groups": fus, "gate_M0": m0}

    # Table 2 对照
    rnd = {k: np.load(ROOT / f"logs/geom_probs/p0/random5_{v}_nomiss_segment.npy") for k, v in
           {"all": "全库", "clinician": "医生区域内", "unmarked": "非医生区域内"}.items()}
    hand_fixed = 0.768
    m9, a9 = rep_attr(ROOT / "logs/geom_attributes_3d", DOCTOR5 + OTHER4, drop)
    d_sd = np.array(rep["macro"]["strict_doctor"]) - np.array(rep["macro"]["hard_other"])
    d_su = np.array(rep["macro"]["strict_doctor"]) - np.array(rep["macro"]["uniform"])
    lines = ["\\begin{tabular}{lcc}", "\\toprule", "Control & Balanced acc. & Note \\\\", "\\midrule"]
    for k, v in rnd.items():
        lines.append(f"5 random measurements from {k} regions ($n{{=}}100$) & {v.mean():.3f}$\\pm${v.std():.3f} & clinician-defined five at the {100 * (v < hand_fixed).mean():.0f}th pct. \\\\")
    lines += [f"5 clinician-defined + 4 unmarked (9) & {pm(m9)} & adding uninformative quantities hurts \\\\",
              f"L1 within clinician regions $-$ L1 within unmarked & {d_sd.mean():+.3f}$\\pm${d_sd.std():.3f} & positive in {int((d_sd > 0).sum())}/10 splits \\\\",
              f"L1 within clinician regions $-$ L1 over all & {d_su.mean():+.3f}$\\pm${d_su.std():.3f} & positive in {int((d_su > 0).sum())}/10 splits \\\\",
              "\\bottomrule", "\\end{tabular}"]
    (out_dir / "table2_controls.tex").write_text("\n".join(lines) + "\n")
    numbers["table2"] = {"random5": {k: pm(v) for k, v in rnd.items()}, "nine": pm(m9), "strict_minus_other": pm(d_sd), "strict_minus_uniform": pm(d_su)}

    # Table 3 消融:5 量留一 / 区域整组 / 单量;门控对照
    abl = {}
    for name, attrs in [("all five", DOCTOR5)] + [(f"$-$ {ATTR_LABEL[a].split(' (')[0]}", [b for b in DOCTOR5 if b != a]) for a in DOCTOR5] + \
            [("$-$ lumbar--pelvis group (3)", SETS["去腰椎骨盆组(剩 2)"]), ("trunk lean alone", ["trunk_lean"]), ("head forward alone", ["head_forward"])]:
        m, a = rep_attr(ROOT / "logs/geom_attributes_3d", attrs, drop)
        abl[name] = (pm(m), pm(a))
    # 门控对照(9 组均值)
    perm = []
    G = load_geom(str(GEOM_JSON)); Rr = {k: v for k, v in geom_reliability(ATTR_DIR, pmap).items() if k not in drop}
    for (b, s) in runs:
        V = {k: v for k, v in load_video(ROOT / "logs/train", b, s, pmap).items() if k not in drop}
        perm.append(np.mean([macro_np(run_one(V, G, Rr, LAMS, GRID, light=True, perm_seed=ps)["pred"]["step_outl"], runs[(b, s)]["y"]) for ps in range(10)]))
    gate_rows = [("video branch alone", fus["video"]), ("5 quantities alone", fus["geom"]), ("fixed $w{=}0.5$ / nested global $w$ / stacking", (fus["fixed0.5"], fus["nested_w"], fus["stack"])),
                 ("sigmoid gate (6 features)", fus["gate"]),
                 ("\\textbf{evidence gate}", fus["step_outl"]), ("evidence gate, deviation permuted (10$\\times$)", float(np.mean(perm)))]
    if runs_b5b:
        gate_rows.append(("evidence gate with TimeSformer video branch", np.mean([macro_np(r["pred"]["step_outl"], r["y"]) for r in runs_b5b.values()])))
    lines = ["\\begin{tabular}{lcc}", "\\toprule", "Measurement ablation (10 splits) & Balanced acc. & AUC \\\\", "\\midrule"]
    lines += [f"{n} & {m} & {a} \\\\" for n, (m, a) in abl.items()]
    lines += ["\\midrule", "Fusion (3 video branches $\\times$ 3 seeds) & Balanced acc. & \\\\", "\\midrule"]
    lines += [f"{n} & {' / '.join(f'{x:.3f}' for x in v) if isinstance(v, tuple) else f'{v:.3f}'} & \\\\" for n, v in gate_rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / "table3_ablation.tex").write_text("\n".join(lines) + "\n")
    numbers["table3"] = {"measurement_ablation": abl, "gate": {n: (list(v) if isinstance(v, tuple) else v) for n, v in gate_rows}}
    print("tables ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["fig2", "fig3", "fig4", "tables", "all"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-figs", default=str(ROOT / "paper/figs"))
    ap.add_argument("--out-tables", default=str(ROOT / "paper/tables"))
    args = ap.parse_args()
    figs, tabs = Path(args.out_figs), Path(args.out_tables)
    figs.mkdir(parents=True, exist_ok=True); tabs.mkdir(parents=True, exist_ok=True)
    pmap = build_patient_map(args.data_root)
    drop = missing_patients()
    numbers = {}
    nfile = tabs / "numbers.json"
    if nfile.exists():
        numbers = json.load(open(nfile))
    need_runs = args.what in ("fig3", "fig4", "tables", "all")
    runs = G = R = None
    if need_runs:
        # 只出图时跳过 sigmoid 门控与堆叠(慢,且图里不用);表格需要它们
        runs, G, R = fusion_runs(pmap, drop, light=args.what in ("fig3", "fig4"))
    if args.what in ("fig2", "all"):
        fig2(figs, drop, numbers)
    if args.what in ("fig3", "all"):
        fig3(figs, runs, numbers)
    if args.what in ("fig4", "all"):
        fig4(figs, runs, G, R, pmap, numbers)
    if args.what in ("tables", "all"):
        runs_b5b, _, _ = fusion_runs(pmap, drop, branches=["B5b_timesformer_lr2e5_e50"], light=True)
        tables(tabs, pmap, drop, runs, numbers, runs_b5b)
    json.dump(numbers, open(nfile, "w"), indent=1, ensure_ascii=False, default=float)
    print(f"-> {figs}, {tabs}")


if __name__ == "__main__":
    main()

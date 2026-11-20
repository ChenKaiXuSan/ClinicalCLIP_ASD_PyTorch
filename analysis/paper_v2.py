#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: paper_v2.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
三个数据集版本(ASD + CARE-PD + KOA-PD-NM)的论文数字、表格与图。取代 paper_figs.py 的 tables / fig2 / fig3。
不重跑训练,只读 logs/ 里已有的预测与测量;所有数字先进 paper/tables/numbers_v2.json,表格与图再从它生成。

  asd      ASD 的全部数字:端到端模型(5 个种子 42/1337/2024/7/11)、测量分支(10 份随机划分)、随机 5 量对照
           (与测量分支同一口径:患者级 LR,10 份划分)、测量库上的数据驱动选择、证据门控(第二个分支 = 视频 / 定位监督 /
           TimeSformer / 骨架网络)及置换对照、按偏离度分层
  public   CARE-PD 与 KOA-PD-NM:汇总 logs/carepd、logs/koa 下各评测脚本落盘的 json
  tables   Table 1–3 的 LaTeX
  figs     Fig. 2(样本量曲线)、Fig. 3(偏离度分层)
  all      以上全部

口径:ASD 67 名有 3D 的患者,患者级平衡准确率;端到端模型 50 轮 × 5 折 × 5 种子;测量 10 份随机 5 折划分;
融合在训练用的固定划分上逐种子做,报种子均值。

    python analysis/paper_v2.py all --data-root $DATA
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from attribute_repeated_splits import DOCTOR5, OTHER4, SETS, macro as macro_np  # noqa: E402
from paper_figs import ATTR_DIR, BANK, GEOM_JSON, GRID, LAMS, missing_patients, rep_attr  # noqa: E402
from patient_level_stats import build_patient_map  # noqa: E402
from patient_metrics import load_tag, metrics  # noqa: E402
from uncertainty_fusion import geom_reliability, load_geom, load_video, run_one  # noqa: E402

SEEDS = [42, 1337, 2024, 7, 11]
NUM = ROOT / "paper/tables/numbers_v2.json"
E2E = [  # (表里的名字, 实验名, 输入)
    ("3D CNN (slow\\_r50, K400)", "B0_3dcnn_e50", "video"),
    ("TimeSformer-B (K400)", "B5b_timesformer_lr2e5_e50", "video"),
    ("2D CNN, frame-wise", "B1_2dcnn_e50", "video"),
    ("CNN--LSTM", "B2_cnn_lstm_e50", "video"),
    ("CLIP-style alignment (annot.\\ at test)", "B4_clip_old_e50", "video+annot."),
    ("3D CNN + region grounding", "M0_concept_learned_e50", "video"),
    ("ST-GCN, 2D skeleton", "B3_pose_e50", "skeleton"),
    ("ST-GCN, 3D skeleton", "B3_pose_3d_wfix_e50", "skeleton"),
    ("CTR-GCN, 2D skeleton", "B6_ctrgcn_e50", "skeleton"),
    ("CTR-GCN, 3D skeleton", "B6_ctrgcn_3d_wfix_e50", "skeleton"),
]
FUSE = [("video", "B0_3dcnn_e50"), ("grounded", "M0_concept_learned_e50"), ("timesformer", "B5b_timesformer_lr2e5_e50"),
        ("skel3d", "B3_pose_3d_wfix_e50"), ("skel2d", "B3_pose_e50")]
C_VIDEO, C_GEOM, C_GATE, C_GREY = "#4C72B0", "#DD8452", "#55A868", "#8C8C8C"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def ms(v) -> dict:
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std()), "n": int(len(v)), "values": v.tolist()}


def have(tag: str, seed: int) -> bool:
    return all(any((ROOT / "logs/train" / f"{tag}__f{k}_s{seed}").rglob("best_preds/*_pred.pt")) for k in range(5))


# ----------------------------------------------------------------------------- ASD
def asd(data_root: str, numbers: dict, n_random: int = 200, n_perm: int = 20) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    from prior_sparse_lr import DOCTOR_REGIONS, load_bank
    pmap = build_patient_map(data_root)
    drop = missing_patients()
    A: dict = {"seeds": SEEDS}

    # ---- 端到端模型
    A["e2e"] = {}
    for name, tag, inp in E2E:
        seeds = [s for s in SEEDS if have(tag, s)]
        if not seeds:
            print(f"[缺] {tag}: 没有可用的种子"); continue
        mm, aa = [], []
        for s in seeds:
            P, yy = load_tag(str(ROOT / "logs/train"), tag, pmap, s, drop)
            m = metrics(P, yy); mm.append(m["macro"]); aa.append(m["auc"])
        A["e2e"][tag] = {"name": name, "input": inp, "seeds": seeds, "bacc": ms(mm), "auc": ms(aa), "n_patients": int(len(yy))}
        print(f"{tag:32s} 种子 {len(seeds)}  {np.mean(mm):.3f} ± {np.std(mm):.3f}  AUC {np.mean(aa):.3f}" + ("" if len(seeds) == len(SEEDS) else "   <-- 种子不全"), flush=True)

    # ---- 测量分支(10 份划分)
    A["meas"] = {}
    sets = {"prior5": (ROOT / "logs/geom_attributes_3d", DOCTOR5), "unmarked4": (ROOT / "logs/geom_attributes_3d", OTHER4),
            "prior5+unmarked4": (ROOT / "logs/geom_attributes_3d", DOCTOR5 + OTHER4), "prior5_2d": (ROOT / "logs/geom_attributes", DOCTOR5),
            "trunk_lean": (ROOT / "logs/geom_attributes_3d", ["trunk_lean"]), "head_forward": (ROOT / "logs/geom_attributes_3d", ["head_forward"]),
            "no_lumbar_pelvis": (ROOT / "logs/geom_attributes_3d", SETS["去腰椎骨盆组(剩 2)"])}
    for a in DOCTOR5:
        sets[f"minus_{a}"] = (ROOT / "logs/geom_attributes_3d", [b for b in DOCTOR5 if b != a])
    for k, (root, attrs) in sets.items():
        m, a = rep_attr(root, attrs, drop)
        A["meas"][k] = {"bacc": ms(m), "auc": ms(a)}
    print("测量分支:", {k: round(v["bacc"]["mean"], 3) for k, v in A["meas"].items()}, flush=True)

    # ---- 随机集合对照。必须同口径:参照是先验 5 量在测量库语法里的那几列(prior_sparse_lr.HAND5,髋角左右分开共 7 列),
    #      与随机列走同一条流水线(患者级均值 -> 标准化 -> LR C=1 balanced,10 份划分取均值)。
    #      A["meas"]["prior5"](0.717)是 geom_attributes_3d.py 的专用几何代码算的,口径不同,不能拿来和随机列比
    #      (2026-11-20 之前的 paper_v2 这样比过,得到的"第 91 百分位"偏高,已改)。
    Xp, _, _, y, pids, feats, regions, _ = load_bank(str(BANK), data_root, 2)
    keep = np.array([p not in drop for p in pids])
    Xk, yk = Xp[keep], y[keep]
    Xk = np.where(np.isnan(Xk), np.nanmedian(Xk, 0), Xk)
    splits10 = [list(StratifiedKFold(5, shuffle=True, random_state=1000 + s).split(Xk, yk)) for s in range(10)]

    def lr10(cols) -> float:
        out = []
        for sp in splits10:
            pred = np.zeros(len(yk), int)
            for tr, te in sp:
                sc = StandardScaler().fit(Xk[tr][:, cols])
                pred[te] = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(sc.transform(Xk[tr][:, cols]), yk[tr]).predict(sc.transform(Xk[te][:, cols]))
            out.append(macro_np(pred, yk))
        return float(np.mean(out))
    within = np.array([all(r in DOCTOR_REGIONS for r in rs) for rs in regions])
    touching = np.array([any(r in DOCTOR_REGIONS for r in rs) for rs in regions])
    pools = {"all": np.arange(len(feats)), "clinician": np.where(within)[0], "unmarked": np.where(~touching)[0]}
    from prior_sparse_lr import HAND5
    hand_cols = [feats.index(f) for f in HAND5]
    ref = lr10(hand_cols)
    rng = np.random.default_rng(0)
    A["random5"] = {"ref_bank": ref, "ref_bank_ncols": len(hand_cols), "ref_geom": A["meas"]["prior5"]["bacc"]["mean"], "n_bank": int(len(feats))}
    for k in (len(hand_cols), 5):                     # 主口径:与参照同样多的列;另报 5 列
        for pn, pool in pools.items():
            v = np.array([lr10(rng.choice(pool, k, replace=False)) for _ in range(n_random)])
            key = pn if k == len(hand_cols) else f"{pn}_k5"
            A["random5"][key] = {"mean": float(v.mean()), "std": float(v.std()), "pct": float((v < ref).mean()), "n": n_random, "k": k, "pool": int(len(pool)), "values": v.tolist()}
            print(f"随机 {k} 列 [{pn}, 池 {len(pool)}]: {v.mean():.3f} ± {v.std():.3f}; 先验 5 量(库内 {len(hand_cols)} 列){ref:.3f} 处于第 {(v < ref).mean() * 100:.0f} 百分位", flush=True)

    # ---- 测量库上的数据驱动选择(p0_controls repeated-splits / p1_robustness nonlinear 落盘的结果)
    A["bank"] = {}
    f = ROOT / "logs/geom_probs/p0/repeated_splits_nomiss_auc.json"
    if f.exists():
        rep = json.load(open(f))
        for k in ("uniform", "strict_doctor", "hard_other"):
            A["bank"][f"l1_{k}"] = {"bacc": ms(rep["macro"][k]), "auc": ms(rep["auc"][k])}
        d = np.array(rep["macro"]["strict_doctor"]) - np.array(rep["macro"]["hard_other"])
        A["bank"]["l1_clinician_minus_unmarked"] = {"mean": float(d.mean()), "std": float(d.std()), "pos": int((d > 0).sum())}
    f = ROOT / "logs/geom_probs/p1_nomiss/nonlinear_repeated_nomiss.json"
    if f.exists():
        nl = json.load(open(f))
        for k, kk in (("rf", "rf/全库"), ("gb", "hgb/全库")):
            A["bank"][k] = {"bacc": ms(nl[kk]["macro"]), "auc": ms(nl[kk]["auc"])}
    print("测量库:", {k: round(v["bacc"]["mean"], 3) for k, v in A["bank"].items() if "bacc" in v}, flush=True)

    # ---- 证据门控
    G = load_geom(str(GEOM_JSON))
    R = {k: v for k, v in geom_reliability(ATTR_DIR, pmap).items() if k not in drop}
    R_other = {k: v for k, v in geom_reliability(ROOT / "logs/geom_attributes_3d_other", pmap).items() if k not in drop}
    keys = ["video", "geom", "fixed0.5", "nested_w", "stack", "gate", "step_outl"]
    A["fusion"] = {}; strata = np.zeros((3, 5))
    for short, tag in FUSE:
        seeds = [s for s in SEEDS if have(tag, s)]
        if not seeds:
            print(f"[缺] 融合分支 {tag}"); continue
        rows, perm, perm_frac, ceil, other = [], [], [], [], []
        for s in seeds:
            V = {k: v for k, v in load_video(ROOT / "logs/train", tag, s, pmap).items() if k not in drop}
            r = run_one(V, G, R, LAMS, GRID, light=short != "video")
            yy = r["y"]
            row = {k: macro_np(r["pred"][k], yy) for k in keys if k in r["pred"] and (short == "video" or k not in ("stack", "gate"))}
            rows.append(row); ceil.append(r["ceiling"])
            ps = [macro_np(run_one(V, G, R, LAMS, GRID, light=True, perm_seed=p)["pred"]["step_outl"], yy) for p in range(n_perm)]
            perm.append(float(np.mean(ps))); perm_frac.append(float(np.mean(np.array(ps) < row["step_outl"])))
            if short == "video":
                other.append(macro_np(run_one(V, G, R_other, LAMS, GRID, light=True)["pred"]["step_outl"], yy))
                outl = r["U"][:, 5]; band = np.digitize(outl, np.quantile(outl, [1 / 3, 2 / 3]))
                for b in range(3):
                    m = band == b
                    strata[b] += [(r["pred"]["video"][m] == yy[m]).sum(), (r["pred"]["geom"][m] == yy[m]).sum(), (r["pred"]["step_outl"][m] == yy[m]).sum(), m.sum(), (yy[m] == 0).sum()]
        F = {k: ms([r[k] for r in rows]) for k in rows[0]}
        gain = np.array([r["step_outl"] - max(r["video"], r["geom"]) for r in rows])
        F.update({"seeds": seeds, "perm": ms(perm), "gate_gt_perm": float(np.mean(perm_frac)), "ceiling": ms(ceil),
                  "gain": float(gain.mean()), "gain_pos": int((gain > 0).sum()),
                  "gate_minus_measure": float(np.mean([r["step_outl"] - r["geom"] for r in rows]))})
        if other:
            F["deviation_from_unmarked"] = ms(other)
        A["fusion"][short] = F
        print(f"融合 [{short}] 种子 {len(seeds)}: 第二分支 {F['video']['mean']:.3f}  测量 {F['geom']['mean']:.3f}  嵌套 w {F['nested_w']['mean']:.3f}  "
              f"门控 {F['step_outl']['mean']:.3f} ± {F['step_outl']['std']:.3f}  置换 {F['perm']['mean']:.3f}  门控 > 置换 {F['gate_gt_perm']:.0%}  上限 {F['ceiling']['mean']:.3f}", flush=True)
    A["strata"] = [{"video": strata[b][0] / strata[b][3], "measure": strata[b][1] / strata[b][3], "gate": strata[b][2] / strata[b][3],
                    "n": int(strata[b][3]), "asd_frac": strata[b][4] / strata[b][3]} for b in range(3)] if strata.sum() else []
    # ---- 注意力对齐:定位监督模型训练日志里的 test/attn_align(全部 test 视频);Grad-CAM 等参照来自
    #      pegasus/attn_gradcam_job.sh(同一视频基线的验证集最佳 checkpoint,每折 test 的前 60 条视频)
    import csv
    import re
    al = []
    for s in SEEDS:
        v = []
        for k in range(5):
            runs = sorted({q.parent.parent for q in (ROOT / "logs/train" / f"M0_concept_learned_e50__f{k}_s{s}").rglob("best_preds/*_pred.pt")}, key=lambda q: q.stat().st_mtime)
            for f in (runs[-1].rglob("metrics.csv") if runs else []):
                for row in csv.DictReader(open(f)):
                    if row.get("test/attn_align") not in (None, ""):
                        v.append(float(row["test/attn_align"]))
        if v:
            al.append(float(np.mean(v)))
    A["align"] = {"grounded": ms(al)}
    ref: dict = {}
    for s in SEEDS:
        per = {}
        for k in range(5):
            f = ROOT / "logs/attn_align" / f"B0_3dcnn_e50_s{s}_f{k}.log"
            if f.exists():
                for line in open(f):
                    m = re.match(r"\s+(uniform|random|center|gradcam)\s+([0-9.]+)", line)
                    if m:
                        per.setdefault(m.group(1), []).append(float(m.group(2)))
        for kind, v in per.items():
            if len(v) == 5:
                ref.setdefault(kind, []).append(float(np.mean(v)))
    for kind, v in ref.items():
        A["align"][kind] = ms(v)
    print("注意力对齐:", {k: f"{v['mean']:.3f}±{v['std']:.3f} (n={v['n']})" for k, v in A["align"].items()}, flush=True)

    # ---- LLM 由医生区域提出的测量(analysis/llm_measure_eval.py,修正后的测量库)
    f = ROOT / "logs/llm_measure/eval.json"
    if f.exists():
        ev = json.load(open(f))
        A["llm"] = {k: {"bacc": ms(v["macro"]), "auc": ms(v["auc"]), "n_cols": float(np.mean(v["n_cols"])) if isinstance(v["n_cols"], list) else float(v["n_cols"])}
                    for k, v in ev.items() if k.split("|")[0] in ("Qwen3-32B", "gpt-oss-20b")}
    numbers["asd"] = A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["asd", "public", "tables", "figs", "all"])
    ap.add_argument("--data-root", default="/work/1/SKIING/chenkaixu/data/asd_dataset")
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=20)
    args = ap.parse_args()
    numbers = json.load(open(NUM)) if NUM.exists() else {}
    if args.what in ("asd", "all"):
        asd(args.data_root, numbers, args.n_random, args.n_perm)
        json.dump(numbers, open(NUM, "w"), indent=1, ensure_ascii=False, default=float)
    if args.what in ("public", "all"):
        from paper_v2_public import public
        public(numbers)
        json.dump(numbers, open(NUM, "w"), indent=1, ensure_ascii=False, default=float)
    if args.what in ("tables", "all"):
        from paper_v2_public import tables
        tables(numbers)
    if args.what in ("figs", "all"):
        from paper_v2_public import figs
        figs(numbers)
    print("->", NUM)


if __name__ == "__main__":
    main()

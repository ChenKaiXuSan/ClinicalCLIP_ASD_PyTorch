#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: paper_v2_public.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
paper_v2.py 的后半:公开数据集(CARE-PD、KOA-PD-NM)的数字汇总,以及全文的表格与图。入口在 paper_v2.py。

官方 CARE-PD 基线(arXiv 2510.04312 的 Fig. 3,宏平均 F1 0–2 类,队列内留一受试者,冻结编码器 + 线性头;
论文只给热力图,两位小数,单次运行)抄在 OFFICIAL 里,顺序 PD-GaM / BMCLab / T-SDU-PD / 3DGait。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "paper/tables", ROOT / "paper/figs"
COHORTS = ["PD-GaM", "BMCLab", "T-SDU-PD", "3DGait"]
OFFICIAL = {  # PD-GaM, BMCLab, T-SDU-PD, 3DGait
    "MixSTE": [0.67, 0.67, 0.41, 0.43], "MotionAGFormer": [0.65, 0.67, 0.39, 0.32], "MotionBERT": [0.66, 0.68, 0.36, 0.38],
    "PoseFormerV2": [0.66, 0.60, 0.38, 0.26], "POTR": [0.49, 0.45, 0.38, 0.34], "MoMask": [0.64, 0.57, 0.54, 0.35],
    "MotionCLIP": [0.65, 0.54, 0.37, 0.24],
}
OFFICIAL_RF = [0.38, 0.66, 0.30, 0.28]
C_VIDEO, C_GEOM, C_GATE, C_GREY, C_ANN = "#4C72B0", "#DD8452", "#55A868", "#8C8C8C", "#8172B2"


def _j(p):
    return json.load(open(ROOT / p))


# ----------------------------------------------------------------------------- 汇总
def public(numbers: dict) -> None:
    C: dict = {"cohorts": COHORTS, "official": OFFICIAL, "official_rf": OFFICIAL_RF}
    lo = _j("logs/carepd/eval_gait_loso.json")
    C["loso"] = {c: lo[f"loso|lr|{c}"] for c in COHORTS}
    C["random8"] = {}
    for c in COHORTS:
        r = lo[f"random|{c}"]; g, a = np.array(r["gait30_pool"]), np.array(r["all_pool"])
        C["random8"][c] = {"ref": r["rubric8"], "gait_mean": float(g.mean()), "gait_std": float(g.std()), "gait_pct": float((g < r["rubric8"]).mean()),
                           "all_mean": float(a.mean()), "all_std": float(a.std()), "all_pct": float((a < r["rubric8"]).mean())}
    C["llm"] = {}
    for f, model in (("logs/carepd/eval_gait_llm_qwen.json", "Qwen3-32B"), ("logs/carepd/eval_gait_llm_gpt.json", "gpt-oss-20b")):
        d = _j(f)
        for cond in ("prior_task", "prior_blind", "noprior_task", "noprior_blind"):
            C["llm"][f"{model}|{cond}"] = {c: {"mean": float(np.mean(d[f"llm|{model[:12]} {cond}|{c}"])), "std": float(np.std(d[f"llm|{model[:12]} {cond}|{c}"])),
                                               "n": len(d[f"llm|{model[:12]} {cond}|{c}"])} for c in COHORTS}
    sc = _j("logs/carepd/size_curve_measure_v2.json")
    C["size_curve"] = {}
    for c in COHORTS:
        deep = _j(f"logs/carepd/size_curve_deep_{c}.json")
        C["size_curve"][c] = {k: {"measure": sc[c][k]["rubric8_lr"], "stgcn": [float(np.mean(deep[k])), float(np.std(deep[k]))]} for k in sc[c] if k in deep}
    C["fusion"] = _j("logs/carepd/fusion_gate.json")
    C["fusion_binary"] = _j("logs/carepd/fusion_gate_bin1.json")
    cx = _j("logs/carepd/eval_gait_cross.json")
    C["lodo"] = {p: {c: cx[f"{p}|lr|{c}"]["rubric8"] for c in COHORTS} for p in ("lodo", "lodo_cn")}
    numbers["carepd"] = C
    numbers["koa"] = {"three": _j("logs/koa/eval_gate.json"), "binary": _j("logs/koa/eval_gate_binary.json")}
    r8 = [C["loso"][c]["rubric8"] for c in COHORTS]
    best = max(OFFICIAL, key=lambda k: np.mean(OFFICIAL[k]))
    print(f"CARE-PD: 先验 8 量 {np.round(r8, 3).tolist()} 平均 {np.mean(r8):.3f}; 官方单个编码器最好 {best} {np.mean(OFFICIAL[best]):.3f}; "
          f"逐队列最好的平均 {np.mean(np.max(list(OFFICIAL.values()), 0)):.3f}")
    k3 = numbers["koa"]["three"]
    print(f"KOA: 先验 12 量 {k3['measure']['先验 12 量']['mean']:.3f}; 第二分支 " + ", ".join(f"{t} {v['second']['mean']:.3f}" for t, v in k3["gate"].items()))


# ----------------------------------------------------------------------------- 表格
def pm(d, nd=3) -> str:
    return f"{d['mean']:.{nd}f}$\\pm${d['std']:.{nd}f}"


def tables(numbers: dict) -> None:
    A, C, K = numbers["asd"], numbers["carepd"], numbers["koa"]
    # ---- Table 1: ASD
    L = ["\\begin{tabular}{@{}llcc@{}}", "\\toprule", "Method & Input & Bal.\\ acc. & AUC \\\\", "\\midrule",
         "\\multicolumn{4}{@{}l}{\\emph{End-to-end (5 folds $\\times$ 5 seeds)}} \\\\"]
    order = ["B0_3dcnn_e50", "B5b_timesformer_lr2e5_e50", "B4_clip_old_e50", "M0_concept_learned_e50", "B3_pose_e50", "B3_pose_3d_wfix_e50", "B6_ctrgcn_e50", "B6_ctrgcn_3d_wfix_e50"]
    for t in order:
        if t in A["e2e"]:
            e = A["e2e"][t]
            star = "" if len(e["seeds"]) == len(A["seeds"]) else f"$^{{({len(e['seeds'])})}}$"
            L.append(f"{e['name']}{star} & {e['input']} & {pm(e['bacc'])} & {pm(e['auc'])} \\\\")
    L += ["\\midrule", "\\multicolumn{4}{@{}l}{\\emph{Measurements from 3D keypoints (10 random partitions)}} \\\\"]
    b, m, r = A["bank"], A["meas"], A["random5"]
    if "gb" in b:
        L.append(f"Gradient boosting, 1,272 candidates & meas. & {pm(b['gb']['bacc'])} & {pm(b['gb']['auc'])} \\\\")
    L += [f"5 random candidates + LR ({r['all']['n']} draws) & meas. & {r['all']['mean']:.3f}$\\pm${r['all']['std']:.3f} & -- \\\\",
          f"4 quantities at unmarked regions + LR & meas. & {pm(m['unmarked4']['bacc'])} & {pm(m['unmarked4']['auc'])} \\\\",
          f"\\textbf{{5 prior-defined quantities + LR}} & meas. & \\textbf{{{pm(m['prior5']['bacc'])}}} & \\textbf{{{pm(m['prior5']['auc'])}}} \\\\",
          f"same 5 quantities from 2D keypoints & meas. & {pm(m['prior5_2d']['bacc'])} & {pm(m['prior5_2d']['auc'])} \\\\",
          "\\bottomrule", "\\end{tabular}"]
    (TAB / "v2_table1_asd.tex").write_text("\n".join(L) + "\n")

    # ---- Table 2: 公开数据集
    def row(name, v, bold=False, nd=2):
        cells = [f"{x:.{nd}f}" for x in v] + [f"{np.mean(v):.{nd}f}"]
        if bold:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        return f"{name} & " + " & ".join(cells) + " \\\\"
    L = ["\\begin{tabular}{@{}lccccc@{}}", "\\toprule", "Method & PD-GaM & BMCLab & T-SDU-PD & 3DGait & Mean \\\\", "\\midrule"]
    for k in ("MixSTE", "MotionBERT", "MotionAGFormer", "MoMask", "PoseFormerV2"):      # 另两个更弱的(POTR, MotionCLIP)在表注里给均值
        L.append(row(f"{k}$^\\dagger$", C["official"][k]))
    L.append(row("Gait features + RF$^\\dagger$", C["official_rf"]))
    L.append("\\midrule")
    L.append(row("random 8 + LR", [C["random8"][c]["gait_mean"] for c in COHORTS]))
    L.append(row("LLM 8, no rubric + LR", [np.mean([C["llm"][f"{mdl}|noprior_task"][c]["mean"] for mdl in ("Qwen3-32B", "gpt-oss-20b")]) for c in COHORTS]))
    L.append(row("LLM 8, from rubric + LR", [np.mean([C["llm"][f"{mdl}|prior_task"][c]["mean"] for mdl in ("Qwen3-32B", "gpt-oss-20b")]) for c in COHORTS]))
    L.append(row("rubric-defined 8 + LR (ours)", [C["loso"][c]["rubric8"] for c in COHORTS], bold=True))
    L += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "v2_table2a_carepd.tex").write_text("\n".join(L) + "\n")
    k3 = K["three"]
    g = k3["gate"]
    L = ["\\begin{tabular}{@{}lc@{}}", "\\toprule", "(b) KOA-PD-NM, 3-class & Bal.\\ acc. \\\\", "\\midrule",
         f"Video (raw frames)$^\\ddagger$ & {pm(g['rgb']['second'])} \\\\",
         f"Video (background removed) & {pm(g['masked']['second'])} \\\\",
         f"Video (silhouette) & {pm(g['sil']['second'])} \\\\",
         f"ST-GCN (3D keypoints) & {pm(g['skel']['second'])} \\\\",
         "\\midrule",
         f"Age, sex, height + LR & {k3['demo']['age_sex_height']:.3f} \\\\",
         f"random 12 + LR & {k3['random']['mean']:.3f}$\\pm${k3['random']['std']:.3f} \\\\",
         f"\\textbf{{prior-defined 12 + LR (ours)}} & \\textbf{{{pm(k3['measure']['先验 12 量'])}}} \\\\",
         "\\bottomrule", "\\end{tabular}"]
    (TAB / "v2_table2b_koa.tex").write_text("\n".join(L) + "\n")

    # ---- Table 3: 融合
    L = ["\\begin{tabular}{@{}llcccccc@{}}", "\\toprule",
         "Dataset & Second branch & Second & Meas. & Global $w$ & Gate & Gate, $d$ perm. & Either \\\\", "\\midrule"]
    names = {"video": "video (3D CNN)", "timesformer": "video (TimeSformer)", "grounded": "video (grounded)", "skel3d": "ST-GCN, 3D joints", "skel2d": "ST-GCN, 2D joints"}
    first = True
    for s in ("video", "timesformer", "grounded", "skel3d", "skel2d"):
        if s in A["fusion"]:
            f = A["fusion"][s]
            gate = f"\\textbf{{{f['step_outl']['mean']:.3f}}}" if s == "video" else f"{f['step_outl']['mean']:.3f}"
            L.append(f"{'ASD' if first else ''} & {names[s]} & {f['video']['mean']:.3f} & {f['geom']['mean']:.3f} & {f['nested_w']['mean']:.3f} & {gate} & {f['perm']['mean']:.3f} & {f['ceiling']['mean']:.3f} \\\\")
            first = False
    kn = {"masked": "video, bg.\\ removed", "sil": "video, silhouette", "skel": "ST-GCN, 3D joints"}
    for label, key in (("KOA, binary", "binary"), ("KOA, 3-class", "three")):
        L.append("\\midrule")
        for i, t in enumerate(("masked", "sil", "skel")):
            f = K[key]["gate"][t]
            L.append(f"{label if i == 0 else ''} & {kn[t]} & {f['second']['mean']:.3f} & {f['measure']['mean']:.3f} & {f['nested_w']['mean']:.3f} & {f['gate']['mean']:.3f} & {f['perm']['mean']:.3f} & {f['ceiling']['mean']:.3f} \\\\")
    L.append("\\midrule")
    fz = C["fusion"]
    avg = lambda k: np.mean([fz[c][k]["mean"] for c in COHORTS])     # noqa: E731
    L.append(f"CARE-PD (4 cohorts) & ST-GCN, 3D joints & {avg('deep'):.3f} & {avg('measure'):.3f} & {avg('nested_w'):.3f} & {avg('gate'):.3f} & {avg('perm'):.3f} & -- \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "v2_table3_fusion.tex").write_text("\n".join(L) + "\n")
    macros(numbers)
    print("tables ok ->", TAB)


REQUIRED = ["asdGradcam", "asdTsf", "asdTsfAuc", "asdLone", "asdGb", "asdGateUnmDev", "asdGateGrounded", "asdGateTsf", "asdGateSkelThree", "asdGateSkelThreeDelta",
            "asdGateSkelTwo", "asdStrLowVideo", "asdStrLowMeas", "asdStrHighMeas", "asdStrLowAsd", "asdSkelLo", "asdSkelHi"]


def macros(numbers: dict) -> None:
    """正文里用到的每个数字一个 \\newcommand,写到 paper/tables/v2_macros.tex。宏名只能用字母。"""
    A, C, K = numbers["asd"], numbers["carepd"], numbers["koa"]
    M: dict[str, str] = {}
    f3 = lambda x: f"{x:.3f}"            # noqa: E731
    f2 = lambda x: f"{x:.2f}"            # noqa: E731
    pct = lambda x: f"{round(100 * x):d}"   # noqa: E731
    e, m, r, b, fu = A["e2e"], A["meas"], A["random5"], A["bank"], A["fusion"]
    vid = [e[t]["bacc"]["mean"] for t in ("B0_3dcnn_e50", "B5b_timesformer_lr2e5_e50", "M0_concept_learned_e50") if t in e]
    sk = [e[t]["bacc"]["mean"] for t in ("B3_pose_e50", "B3_pose_3d_wfix_e50", "B6_ctrgcn_e50", "B6_ctrgcn_3d_wfix_e50") if t in e]
    M.update({"asdVideo": f3(e["B0_3dcnn_e50"]["bacc"]["mean"]), "asdVideoSd": f3(e["B0_3dcnn_e50"]["bacc"]["std"]),
              "asdGrounded": f3(e["M0_concept_learned_e50"]["bacc"]["mean"]),
              "asdVidLo": f2(min(vid)), "asdVidHi": f2(max(vid)), "asdSkelLo": f2(min(sk)) if sk else "??", "asdSkelHi": f2(max(sk)) if sk else "??"})
    if "B5b_timesformer_lr2e5_e50" in e:
        M["asdTsf"] = f3(e["B5b_timesformer_lr2e5_e50"]["bacc"]["mean"]); M["asdTsfAuc"] = f2(e["B5b_timesformer_lr2e5_e50"]["auc"]["mean"])
    for k, name in (("B1_2dcnn_e50", "asdTwoDcnn"), ("B2_cnn_lstm_e50", "asdCnnLstm"), ("B4_clip_old_e50", "asdClipAlign")):
        if k in e:
            M[name] = f2(e[k]["bacc"]["mean"])
    M.update({"asdPrior": f3(m["prior5"]["bacc"]["mean"]), "asdPriorSd": f3(m["prior5"]["bacc"]["std"]), "asdPriorTwo": f2(m["prior5"]["bacc"]["mean"]),
              "asdPriorAuc": f2(m["prior5"]["auc"]["mean"]), "asdPriorTwoD": f3(m["prior5_2d"]["bacc"]["mean"]),
              "asdUnmarked": f3(m["unmarked4"]["bacc"]["mean"]), "asdNine": f3(m["prior5+unmarked4"]["bacc"]["mean"]),
              "asdTrunk": f3(m["trunk_lean"]["bacc"]["mean"]), "asdHeadAlone": f2(m["head_forward"]["bacc"]["mean"]),
              "asdMinusHead": f3(m["minus_head_forward"]["bacc"]["mean"]), "asdMinusHeadAuc": f2(m["minus_head_forward"]["auc"]["mean"]),
              "asdRandAll": f2(r["all"]["mean"]), "asdRandClin": f2(r["clinician"]["mean"]), "asdRandUnm": f2(r["unmarked"]["mean"]),
              "asdRandAllPct": pct(r["all"]["pct"]), "asdRandClinPct": pct(r["clinician"]["pct"]), "asdRandUnmPct": pct(r["unmarked"]["pct"]),
              "asdRandAllPctFive": pct(r["all_k5"]["pct"]), "asdRandN": str(r["all"]["n"]), "asdRandK": str(r["all"]["k"]),
              "asdPriorBank": f2(r["ref_bank"])})
    if "align" in A:
        M["asdAlign"] = f2(A["align"]["grounded"]["mean"])
        for kind in ("gradcam", "center", "uniform"):
            if kind in A["align"] and A["align"][kind]["n"] == len(A["seeds"]):
                M[{"gradcam": "asdGradcam", "center": "asdCenter", "uniform": "asdUniform"}[kind]] = f2(A["align"][kind]["mean"])
    if "annotation" in A:
        an = A["annotation"]
        M.update({"annClips": str(an["clips"]), "annPatients": str(an["recordings"]), "annAgree": pct(an["agree"]),
                  "annLumbar": pct(an["any"]["lumbar_pelvis"]), "annHead": pct(an["any"]["head"]), "annShoulder": pct(an["any"]["shoulder"]),
                  "annWrist": pct(an["any"]["wrist"]), "annFoot": pct(an["any"]["foot"])})
    if "llm" in A:
        L = A["llm"]
        mk = lambda cond: float(np.mean([L[f"{mdl}|{cond}"]["bacc"]["mean"] for mdl in ("Qwen3-32B", "gpt-oss-20b")]))      # noqa: E731
        M.update({"asdLlmMarked": f2(mk("marked_task")), "asdLlmUnmarked": f2(mk("unmarked_task")), "asdLlmNoRegion": f2(mk("noregion_task"))})
    for k, name in (("l1_uniform", "asdLone"), ("gb", "asdGb"), ("rf", "asdRf"), ("l1_strict_doctor", "asdLoneClin")):
        if k in b:
            M[name] = f3(b[k]["bacc"]["mean"]); M[name + "Sd"] = f3(b[k]["bacc"]["std"]); M[name + "Auc"] = f2(b[k]["auc"]["mean"])
    v = fu["video"]
    M.update({"asdFusVideo": f3(v["video"]["mean"]), "asdFusMeas": f3(v["geom"]["mean"]), "asdFixed": f3(v["fixed0.5"]["mean"]), "asdFusMeasTwo": f2(v["geom"]["mean"]), "asdNested": f3(v["nested_w"]["mean"]),
              "asdStack": f3(v["stack"]["mean"]), "asdSigmoid": f3(v["gate"]["mean"]), "asdGate": f3(v["step_outl"]["mean"]), "asdGateSd": f3(v["step_outl"]["std"]),
              "asdGateTwo": f2(v["step_outl"]["mean"]), "asdPerm": f3(v["perm"]["mean"]), "asdGateGtPerm": pct(v["gate_gt_perm"]), "asdEither": f2(v["ceiling"]["mean"]),
              "asdGateGainPos": str(v["gain_pos"]), "asdNseed": str(len(v["seeds"])), "asdGateMinusMeas": f"{v['gate_minus_measure']:+.3f}"})
    if "deviation_from_unmarked" in v:
        M["asdGateUnmDev"] = f3(v["deviation_from_unmarked"]["mean"])
    for k, name in (("grounded", "asdGateGrounded"), ("timesformer", "asdGateTsf"), ("skel3d", "asdGateSkelThree"), ("skel2d", "asdGateSkelTwo")):
        if k in fu:
            M[name] = f3(fu[k]["step_outl"]["mean"]); M[name + "Alone"] = f3(fu[k]["video"]["mean"]); M[name + "Delta"] = f"{fu[k]['gate_minus_measure']:+.3f}"
    if A.get("strata"):
        st = A["strata"]
        M.update({"asdStrLowVideo": f2(st[0]["video"]), "asdStrLowMeas": f2(st[0]["measure"]), "asdStrMidMeas": f2(st[1]["measure"]),
                  "asdStrHighMeas": f2(st[2]["measure"]), "asdStrLowAsd": pct(st[0]["asd_frac"])})
    r8 = [C["loso"][c]["rubric8"] for c in COHORTS]
    best = max(C["official"], key=lambda k: np.mean(C["official"][k]))
    llm = lambda cond: float(np.mean([C["llm"][f"{mdl}|{cond}"][c]["mean"] for mdl in ("Qwen3-32B", "gpt-oss-20b") for c in COHORTS]))   # noqa: E731
    M.update({"pdMean": f3(float(np.mean(r8))), "pdMeanTwo": f2(float(np.mean(r8))), "pdBestEnc": f3(float(np.mean(C["official"][best]))), "pdBestEncTwo": f2(float(np.mean(C["official"][best]))),
              "pdBestEncName": best, "pdOracle": f2(float(np.mean(np.max(list(C["official"].values()), 0)))), "pdRf": f2(float(np.mean(C["official_rf"]))),
              "pdVone": f3(float(np.mean([C["loso"][c]["v1_prior12"] for c in COHORTS]))),
              "pdLlmRubric": f3(llm("prior_task")), "pdLlmNoRubric": f3(llm("noprior_task")), "pdRand": f3(float(np.mean([C["random8"][c]["gait_mean"] for c in COHORTS])))})
    for c, nm in zip(COHORTS, ("Gam", "Bmc", "Tsdu", "Tdg")):
        M[f"pd{nm}"] = f3(C["loso"][c]["rubric8"]); M[f"pd{nm}Pct"] = pct(C["random8"][c]["gait_pct"])
        ci = C["loso"][c]["rubric8_ci"]; M[f"pd{nm}Ci"] = f"{ci[0]:.2f}--{ci[1]:.2f}"
        M[f"pd{nm}Lodo"] = f2(C["lodo"]["lodo_cn"][c])
    k3, kb = K["three"], K["binary"]
    M.update({"koaPrior": f3(k3["measure"]["先验 12 量"]["mean"]), "koaPriorTwo": f2(k3["measure"]["先验 12 量"]["mean"]), "koaRand": f3(k3["random"]["mean"]), "koaRandPct": pct(k3["random"]["pct"]),
              "koaAge": f2(k3["demo"]["age"]), "koaDemo": f2(k3["demo"]["age_sex_height"]),
              "koaVideo": f3(k3["gate"]["masked"]["second"]["mean"]), "koaVideoTwo": f2(k3["gate"]["masked"]["second"]["mean"]), "koaSil": f3(k3["gate"]["sil"]["second"]["mean"]),
              "koaRaw": f3(k3["gate"]["rgb"]["second"]["mean"]), "koaSkel": f3(k3["gate"]["skel"]["second"]["mean"]),
              "koaGateThree": f3(k3["gate"]["masked"]["gate"]["mean"]), "koaFixedThree": f3(k3["gate"]["masked"]["fixed0.5"]["mean"]), "koaPermThree": f3(k3["gate"]["masked"]["perm"]["mean"]),
              "koaMeasThree": f3(k3["gate"]["masked"]["measure"]["mean"]), "koaGateSkelThree": f3(k3["gate"]["skel"]["gate"]["mean"]),
              "koaMeasBin": f3(kb["gate"]["masked"]["measure"]["mean"]), "koaVideoBin": f3(kb["gate"]["masked"]["second"]["mean"]), "koaGateBin": f3(kb["gate"]["masked"]["gate"]["mean"]),
              "koaGateBinTwo": f2(kb["gate"]["masked"]["gate"]["mean"]), "koaMeasBinTwo": f2(kb["gate"]["masked"]["measure"]["mean"]),
              "koaPermBin": f3(kb["gate"]["masked"]["perm"]["mean"]), "koaGateGtPermBin": pct(kb["gate"]["masked"]["gate_gt_perm"]),
              "koaSkelBin": f3(kb["gate"]["skel"]["second"]["mean"]), "koaGateSkelBin": f3(kb["gate"]["skel"]["gate"]["mean"]),
              "koaEitherVideoBin": f2(kb["gate"]["masked"]["ceiling"]["mean"]), "koaEitherSkelBin": f2(kb["gate"]["skel"]["ceiling"]["mean"]),
              "koaStrLowVideo": f2(k3["strata"]["masked"]["acc"][0]), "koaStrLowMeas": f2(k3["strata"]["measure"][0]), "koaStrHighMeas": f2(k3["strata"]["measure"][2]),
              "koaStrHighVideo": f2(k3["strata"]["masked"]["acc"][2])})
    sg = C["size_curve"]["PD-GaM"]
    M["pdSizeGapTwo"] = f2(sg["2"]["measure"][0] - sg["2"]["stgcn"][0]); M["pdSizeGapFour"] = f2(sg["4"]["measure"][0] - sg["4"]["stgcn"][0])
    for k in REQUIRED:                      # 依赖的实验还没跑完时先占位,正文能编译,PDF 里一眼能看到缺什么
        M.setdefault(k, "??")
    lines = ["% 由 analysis/paper_v2.py tables 生成,不要手改。数字来源:paper/tables/numbers_v2.json"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
    (TAB / "v2_macros.tex").write_text("\n".join(lines) + "\n")
    numbers["macros"] = M


# ----------------------------------------------------------------------------- 图
def _box(ax, x, y, w, h, text, color, fs=6.0, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=color, ec="none", alpha=0.14))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc="none", ec=color, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", linespacing=1.3)


def _arrow(ax, p, q, color="#444444", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.9, color=color, linestyle=ls, shrinkA=1, shrinkB=1))


def fig1() -> None:
    fig = plt.figure(figsize=(4.8, 1.75))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12); ax.set_ylim(0, 4.3); ax.axis("off")
    # 上路:先验 -> 测量 -> LR
    _box(ax, 0.1, 2.65, 2.5, 1.5, "clinical prior\n(offline, once)\nmarked regions /\nrating rubric / textbook", C_ANN)
    _box(ax, 3.2, 2.65, 2.3, 1.5, "measurement set\n5–12 named\nquantities\n(expert or LLM)", C_GEOM)
    _box(ax, 6.1, 2.65, 2.2, 1.5, "geometry on\n3D keypoints\n$\\rightarrow$ $\\mathbf{m}_i$,\ndeviation $d_i$", C_GEOM)
    _box(ax, 8.9, 2.65, 1.5, 1.5, "logistic\nregression\n$P^m_i$", C_GEOM)
    # 下路:视频
    _box(ax, 0.1, 0.2, 2.5, 1.5, "walking video\n(test-time input)", "#333333")
    _box(ax, 3.2, 0.2, 2.3, 1.5, "3D pose\nestimation", "#333333")
    _box(ax, 6.1, 0.2, 2.2, 1.5, "video network\n(pixels)", C_VIDEO)
    _box(ax, 8.9, 0.2, 1.5, 1.5, "$P^v_i$", C_VIDEO)
    _box(ax, 10.85, 1.3, 1.05, 1.7, "gate\n$d_i\\!<\\!\\tau$:\nvideo\nelse:\nmeas.", C_GATE, fs=5.6)
    for p, q in (((2.6, 3.4), (3.2, 3.4)), ((5.5, 3.4), (6.1, 3.4)), ((8.3, 3.4), (8.9, 3.4)), ((2.6, 0.95), (3.2, 0.95)), ((8.3, 0.95), (8.9, 0.95))):
        _arrow(ax, p, q)
    _arrow(ax, (4.35, 1.7), (6.6, 2.65))                     # 3D 关键点 -> 几何
    _arrow(ax, (1.35, 0.2), (1.35, 0.02)); ax.plot([1.35, 7.2], [0.02, 0.02], color="#444444", lw=0.9); _arrow(ax, (7.2, 0.02), (7.2, 0.2))
    _arrow(ax, (10.4, 3.2), (10.85, 2.6), C_GEOM); _arrow(ax, (10.4, 1.1), (10.85, 1.7), C_VIDEO)
    fig.savefig(FIG / "v2_fig1_pipeline.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(FIG / "v2_fig1_pipeline.png", bbox_inches="tight", dpi=220); plt.close(fig)


def figs(numbers: dict) -> None:
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    fig1()
    C, K, A = numbers["carepd"], numbers["koa"], numbers.get("asd", {})
    # ---- Fig 2: 样本量曲线
    fig, axes = plt.subplots(1, 4, figsize=(4.8, 1.2), sharey=True)
    for ax, c in zip(axes, COHORTS):
        d = C["size_curve"][c]
        ks = [k for k in d if k != "all"]; x = list(range(len(ks) + 1))
        for key, col, lab in (("measure", C_GEOM, "8 quantities + LR"), ("stgcn", C_VIDEO, "ST-GCN")):
            m = np.array([d[k][key][0] for k in ks + ["all"]]); s = np.array([d[k][key][1] for k in ks + ["all"]])
            ax.plot(x, m, "-o", color=col, ms=2.2, lw=1.0, label=lab); ax.fill_between(x, m - s, m + s, color=col, alpha=0.15, lw=0)
        ax.set_xticks(x); ax.set_xticklabels(ks + ["all"], fontsize=5.5); ax.set_title(c, fontsize=6.5, pad=2); ax.tick_params(axis="y", labelsize=5.5)
    axes[0].set_ylabel("macro-F1", fontsize=6.5); axes[0].legend(fontsize=5, frameon=False, loc="lower right")
    fig.text(0.5, -0.09, "training subjects per fold", ha="center", fontsize=6.5)
    fig.tight_layout(pad=0.3, w_pad=0.4)
    fig.savefig(FIG / "v2_fig2_size.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(FIG / "v2_fig2_size.png", bbox_inches="tight", dpi=220); plt.close(fig)
    # ---- Fig 3: 偏离度分层(ASD 与 KOA)
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.5), sharey=True)
    panels = []
    if A.get("strata"):
        panels.append(("ASD (video = 3D CNN)", [s["video"] for s in A["strata"]], [s["measure"] for s in A["strata"]]))
    st = K["three"]["strata"]
    panels.append(("KOA-PD-NM (video, background removed)", st["masked"]["acc"], st["measure"]))
    for ax, (title, v, m) in zip(axes, panels):
        x = np.arange(3); w = 0.36
        for j, (vals, col, lab) in enumerate(((v, C_VIDEO, "video branch"), (m, C_GEOM, "measurements"))):
            ax.bar(x + (j - 0.5) * w, vals, w, color=col, label=lab)
            for xi, val in zip(x + (j - 0.5) * w, vals):
                ax.text(xi, val + 0.012, f"{val:.2f}", ha="center", fontsize=5.5)
        ax.set_xticks(x); ax.set_xticklabels(["low", "mid", "high"], fontsize=6); ax.set_title(title, fontsize=6.5, pad=2)
        ax.set_ylim(0.3, 1.2); ax.set_yticks([0.4, 0.6, 0.8, 1.0]); ax.tick_params(axis="y", labelsize=5.5); ax.set_xlabel("deviation $d$ (terciles)", fontsize=6.5)
    axes[0].set_ylabel("accuracy", fontsize=6.5); axes[0].legend(fontsize=5.5, frameon=False, loc="upper left", ncol=2)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(FIG / "v2_fig3_strata.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(FIG / "v2_fig3_strata.png", bbox_inches="tight", dpi=220); plt.close(fig)
    print("figs ok ->", FIG)

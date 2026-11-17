"""8 个种子的最终对比:单独成绩、接门控、配对差(同种子);以及原 3 种子 vs 新 5 种子分开看(防挑选偏差)。"""
import sys, pickle
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "analysis")
from late_fusion import macro
from patient_level_stats import build_patient_map
from uncertainty_fusion import geom_reliability, load_geom, load_video, run_one
DATA = "/work/1/SKIING/chenkaixu/data/asd_dataset"
drop = {Path(v["json"]).stem.split("-")[0] for v in pickle.load(open("logs/measure_bank_3d/bank.pkl", "rb"))["videos"].values() if v["frac_ok"] == 0}
pmap = build_patient_map(DATA)
R = {k: v for k, v in geom_reliability(Path("logs/geom_attributes_3d_doctor"), pmap).items() if k not in drop}
G = load_geom("logs/geom_probs/geom3d_doctor_lr_probs.json")
grid = np.round(np.arange(0, 1.0001, 0.05), 2)
OLD, NEW = [42, 1337, 2024], [7, 11, 23, 101, 2025]
SEEDS = OLD + NEW
TAGS = {"B0 3D CNN": "B0_3dcnn_e50", "M0": "M0_concept_learned_e50", "F1 双分支 带知识": "F1_vvf_32b_clinical_feat_e50",
        "F2 双分支 不带知识": "F2_vvf_32b_generic_feat_e50", "VLM32 概念头 不带知识": "Q1g_qwen32b_concept_generic_e50",
        "VLM32 概念头 带知识": "Q1_qwen32b_concept_clinical_e50"}
cache = {}
def L(tag, s):
    if (tag, s) not in cache:
        cache[(tag, s)] = load_video(Path("logs/train"), tag, s, pmap)
    return cache[(tag, s)]
def score(V):
    r = run_one(V, G, R, [1.0], grid, light=True); y = r["y"]; pids = r["pids"]
    p_asd = np.array([V[p]["p"][0] for p in pids])
    return macro(r["pred"]["video"], y), roc_auc_score((y == 0).astype(int), p_asd), macro(r["pred"]["step_outl"], y), macro(r["pred"]["geom"], y)
def mix(a, b, s):
    A, B = L(a, s), L(b, s)
    return {p: {**A[p], "p": 0.5 * A[p]["p"] + 0.5 * B[p]["p"], "std": 0.5 * (A[p]["std"] + B[p]["std"])} for p in A if p in B}
res = {}
for name, tag in TAGS.items():
    res[name] = np.array([score(L(tag, s)) for s in SEEDS])
res["B0 + VLM32 概念头(概率平均)"] = np.array([score(mix(TAGS["B0 3D CNN"], TAGS["VLM32 概念头 不带知识"], s)) for s in SEEDS])
res["F2 + VLM32 概念头(概率平均)"] = np.array([score(mix(TAGS["F2 双分支 不带知识"], TAGS["VLM32 概念头 不带知识"], s)) for s in SEEDS])
print(f"测量单独 {res['B0 3D CNN'][0, 3]:.3f};种子顺序 {SEEDS}")
print(f"\n{'配置':30s} {'单独 8 种子':>15s} {'AUC':>6s} | {'门控 8 种子':>15s} {'原 3':>6s} {'新 5':>6s} | 逐种子门控")
for name, a in res.items():
    print(f"{name:30s} {a[:, 0].mean():.3f} ± {a[:, 0].std():.3f} {a[:, 1].mean():6.3f} | {a[:, 2].mean():.3f} ± {a[:, 2].std():.3f} {a[:3, 2].mean():6.3f} {a[3:, 2].mean():6.3f} | " + " ".join(f"{v:.3f}" for v in a[:, 2]))
print("\n配对差(同种子,8 对;Wilcoxon 双侧):")
def cmp(a, b, col, lab):
    d = res[a][:, col] - res[b][:, col]
    p = wilcoxon(res[a][:, col], res[b][:, col]).pvalue if np.any(d != 0) else 1.0
    print(f"  {lab:8s} {a} - {b}: {d.mean():+.3f} ± {d.std():.3f}, 为正 {(d > 0).sum()}/8, p {p:.3f};只看新 5 种子 {d[3:].mean():+.3f}")
for a, b in (("F2 双分支 不带知识", "B0 3D CNN"), ("F1 双分支 带知识", "B0 3D CNN"), ("F2 双分支 不带知识", "F1 双分支 带知识"), ("VLM32 概念头 不带知识", "B0 3D CNN"), ("M0", "B0 3D CNN")):
    cmp(a, b, 0, "单独")
for a, b in (("VLM32 概念头 不带知识", "B0 3D CNN"), ("VLM32 概念头 不带知识", "M0"), ("VLM32 概念头 带知识", "B0 3D CNN"), ("F2 双分支 不带知识", "B0 3D CNN"),
             ("F1 双分支 带知识", "B0 3D CNN"), ("M0", "B0 3D CNN"), ("B0 + VLM32 概念头(概率平均)", "B0 3D CNN"), ("F2 + VLM32 概念头(概率平均)", "B0 3D CNN")):
    cmp(a, b, 2, "门控")

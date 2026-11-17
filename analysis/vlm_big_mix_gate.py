"""决策级的"视频分支 + VLM 分支":两个已训练模型的患者概率取平均作为门控的备选意见(同种子配对)。"""
import sys, pickle
import numpy as np
from pathlib import Path
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
seeds = [42, 1337, 2024]
cache = {}
def L(tag, s):
    if (tag, s) not in cache:
        cache[(tag, s)] = load_video(Path("logs/train"), tag, s, pmap)
    return cache[(tag, s)]
def score(V):
    r = run_one(V, G, R, [1.0], grid, light=True); y = r["y"]
    return macro(r["pred"]["video"], y), macro(r["pred"]["fixed0.5"], y), macro(r["pred"]["step_outl"], y), r["ceiling"]
def mix(a, b, s):
    A, B = L(a, s), L(b, s)
    return {p: {**A[p], "p": 0.5 * A[p]["p"] + 0.5 * B[p]["p"], "std": 0.5 * (A[p]["std"] + B[p]["std"])} for p in A if p in B}
Q = "Q1g_qwen32b_concept_generic_e50"
print(f"{'备选意见':58s} {'单独':>6s} {'固定0.5':>7s} {'门控':>6s} {'上限':>6s}   逐种子门控")
rows = [("B0", lambda s: L("B0_3dcnn_e50", s)), ("M0", lambda s: L("M0_concept_learned_e50", s)), ("VLM32 概念头", lambda s: L(Q, s)),
        ("F2 双分支(特征融合)", lambda s: L("F2_vvf_32b_generic_feat_e50", s)),
        ("B0 + VLM32 概念头(概率平均)", lambda s: mix("B0_3dcnn_e50", Q, s)), ("M0 + VLM32 概念头(概率平均)", lambda s: mix("M0_concept_learned_e50", Q, s)),
        ("F2 + VLM32 概念头(概率平均)", lambda s: mix("F2_vvf_32b_generic_feat_e50", Q, s)), ("B0 + M0(概率平均, 参照)", lambda s: mix("B0_3dcnn_e50", "M0_concept_learned_e50", s))]
for name, fn in rows:
    sc = np.array([score(fn(s)) for s in seeds])
    print(f"{name:58s} {sc[:, 0].mean():6.3f} {sc[:, 1].mean():7.3f} {sc[:, 2].mean():6.3f} {sc[:, 3].mean():6.3f}   " + " / ".join(f"{v:.3f}" for v in sc[:, 2]))

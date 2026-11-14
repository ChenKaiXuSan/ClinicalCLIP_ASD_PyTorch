"""按测量偏离度三分层:测量 / 视频 / VLM 属性各自的准确率;低偏离层里各个 VLM 属性、各个非医生区域几何量的单变量 AUC。"""
import sys, json, pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "analysis")
from patient_level_stats import build_patient_map
from uncertainty_fusion import geom_reliability, load_geom, load_video
DATA = "/work/1/SKIING/chenkaixu/data/asd_dataset"
drop = {Path(v["json"]).stem.split("-")[0] for v in pickle.load(open("logs/measure_bank_3d/bank.pkl", "rb"))["videos"].values() if v["frac_ok"] == 0}
pmap = build_patient_map(DATA)
R = {k: v for k, v in geom_reliability(Path("logs/geom_attributes_3d_doctor"), pmap).items() if k not in drop}
G = load_geom("logs/geom_probs/geom3d_doctor_lr_probs.json"); A = load_geom("logs/geom_probs/vlm_attr_lr_probs.json")
Vs = [load_video(Path("logs/train"), t, s, pmap) for t in ("B0_3dcnn_e50", "M0_concept_learned_e50", "A1_no_grounding_e50") for s in (42, 1337, 2024)]
pids = sorted(set(R) & set(G) & set(A) & set(Vs[0]))
y = np.array([G[p]["label"] for p in pids])            # 0 = ASD
outl = np.array([R[p][1] for p in pids])
pg = np.array([G[p]["p"].argmax() for p in pids]); pa = np.array([A[p]["p"].argmax() for p in pids])
pv = np.array([[V[p]["p"].argmax() for p in pids] for V in Vs])
def bal(pred, m):
    return float(np.mean([(pred[m & (y == c)] == c).mean() for c in (0, 1) if (m & (y == c)).any()]))
q = np.quantile(outl, [1 / 3, 2 / 3])
strata = [("低偏离", outl <= q[0]), ("中", (outl > q[0]) & (outl <= q[1])), ("高偏离", outl > q[1])]
print(f"{'层':8s} {'n':>3s} {'ASD':>4s} {'测量':>6s} {'视频(9组)':>9s} {'VLM属性':>8s}")
for name, m in strata:
    print(f"{name:8s} {m.sum():3d} {(y[m] == 0).sum():4d} {bal(pg, m):6.3f} {np.mean([bal(v, m) for v in pv]):9.3f} {bal(pa, m):8.3f}")
low = strata[0][1] | strata[1][1]
print(f"\n低 + 中偏离层(门控会让出去的患者, n={low.sum()}, ASD {(y[low] == 0).sum()})内的单变量 AUC(max(a,1-a)):")
def patient_mean(d):
    by = {}
    for v, rec in d.items():
        by.setdefault(Path(rec["json"]).stem.split("-")[0], []).extend(rec["scores"])
    return np.array([np.mean(by[p]) if p in by else np.nan for p in pids])
yy = (y == 0).astype(int)
rows = []
for root, lab in (("logs/qwen_attributes/qwen3-vl-8b-instruct_448", "VLM"), ("logs/geom_attributes_3d", "几何")):
    for f in sorted(Path(root).glob("*.json")):
        d = json.load(open(f))
        if not all(isinstance(v, dict) and "scores" in v for v in d.values()):
            continue
        x = patient_mean(d); ok = np.isfinite(x)
        a_all = roc_auc_score(yy[ok], x[ok]); a_low = roc_auc_score(yy[ok & low], x[ok & low])
        rows.append((lab, f.stem, max(a_all, 1 - a_all), max(a_low, 1 - a_low), "ASD高" if a_low > 0.5 else "ASD低"))
for r in sorted(rows, key=lambda r: -r[3]):
    print(f"  {r[0]:4s} {r[1]:22s} 全体 {r[2]:.3f}   低中偏离层 {r[3]:.3f} {r[4]}")

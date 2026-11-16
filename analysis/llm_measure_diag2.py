"""大模型提案的诊断:区域遵从率、躯干倾角类测量的出现率、医生区域 vs 未标区域的差距检验。"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu
sys.path.insert(0, "analysis")
from llm_measure_eval import load_patients, evaluate, CONDS
from llm_measure_propose import ANGLES
feats, X, y = load_patients("logs/measure_bank_3d/bank.pkl")
col = {f: i for i, f in enumerate(feats)}
REG = {"nose": "head", "ear_mid": "head", "sh_L": "shoulder", "sh_R": "shoulder", "sh_mid": "shoulder", "spine37": "tspine",
       "el_L": "elbow", "el_R": "elbow", "spine35": "lumbar_pelvis", "wr_L": "wrist", "wr_R": "wrist", "hip_L": "lumbar_pelvis",
       "hip_R": "lumbar_pelvis", "hip_mid": "lumbar_pelvis", "kn_L": "knee", "kn_R": "knee", "ank_L": "foot", "ank_R": "foot"}
MARK, UNMARK = {"head", "shoulder", "lumbar_pelvis"}, {"knee", "foot", "wrist", "elbow"}
UP, LOW = {"sh_mid", "sh_L", "sh_R", "spine37"}, {"hip_mid", "hip_L", "hip_R", "spine35"}
def lms(it):
    return [it["upper"], it["lower"]] if it["type"] == "pair" else sorted({l for tri in ANGLES[it["angle"]][1] for l in tri.split("-")})
def trunk_like(it):   # 躯干整体的矢状倾角 / 前后偏移(上端在肩或胸椎,下端在髋或腰椎)
    return it["type"] == "pair" and it["quantity"] in ("incl", "dx") and it["stat"] == "mean" and \
        ({it["upper"], it["lower"]} & UP) and ({it["upper"], it["lower"]} & LOW)
for name in sys.argv[1:]:
    mdir = Path("logs/llm_measure") / name
    res = {}
    print(f"===== {name}")
    for cond in CONDS:
        d = json.load(open(mdir / f"{cond}.json"))
        target = MARK if cond.startswith("marked") else (UNMARK if cond.startswith("unmarked") else None)
        S = [s for s in d["samples"] if not s["greedy"] and s["columns"]]
        m = np.array([evaluate(X[:, [col[c] for c in s["columns"] if c in col]], y)[0] for s in S])
        res[cond] = m
        comp = [({REG[l] for l in lms(it)} <= target, bool({REG[l] for l in lms(it)} & target)) for s in S for it in s["items"]] if target else []
        tl = np.mean([any(trunk_like(it) for it in s["items"]) for s in S])
        c = np.array(comp) if comp else None
        print(f"  {cond:16s} macro {m.mean():.3f} ± {m.std():.3f}  含躯干整体倾角/偏移 {tl:.2f}" +
              (f"  条目全在给定区域 {c[:, 0].mean():.2f} / 触及 {c[:, 1].mean():.2f}" if c is not None else ""))
    for a, b in (("marked_task", "unmarked_task"), ("marked_blind", "unmarked_blind"), ("marked_task", "noregion_task"), ("marked_task", "marked_blind")):
        print(f"  {a} - {b}: {res[a].mean() - res[b].mean():+.3f}  (MWU p {mannwhitneyu(res[a], res[b]).pvalue:.4f})")
    allm = np.concatenate([res["marked_task"], res["marked_blind"]])
    print(f"  医生区域两条件合计 40 个集合: {allm.mean():.3f} ± {allm.std():.3f}, ≥ 手工 0.687 的比例 {np.mean(allm >= 0.687):.2f}, 最低 {allm.min():.3f} 最高 {allm.max():.3f}")

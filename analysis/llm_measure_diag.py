"""LLM 提案的诊断:区域遵从率;含 / 不含躯干倾角(sh_mid-hip_mid)的集合成绩;手工 5 量在同口径随机对照中的百分位。"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, "analysis")
from llm_measure_eval import load_patients, evaluate, HAND5, CONDS
from llm_measure_propose import ANGLES
feats, X, y = load_patients("logs/measure_bank_3d/bank.pkl")
col = {f: i for i, f in enumerate(feats)}
REG = {"nose": "head", "ear_mid": "head", "sh_L": "shoulder", "sh_R": "shoulder", "sh_mid": "shoulder", "spine37": "tspine",
       "el_L": "elbow", "el_R": "elbow", "spine35": "lumbar_pelvis", "wr_L": "wrist", "wr_R": "wrist", "hip_L": "lumbar_pelvis",
       "hip_R": "lumbar_pelvis", "hip_mid": "lumbar_pelvis", "kn_L": "knee", "kn_R": "knee", "ank_L": "foot", "ank_R": "foot"}
MARK, UNMARK = {"head", "shoulder", "lumbar_pelvis"}, {"knee", "foot", "wrist", "elbow"}
def lms(it):
    if it["type"] == "pair":
        return [it["upper"], it["lower"]]
    return sorted({l for tri in ANGLES[it["angle"]][1] for l in tri.split("-")})
LEAN = {"incl|sh_mid-hip_mid|mean", "dx|sh_mid-hip_mid|mean"}
rows = []
for mdir in sorted(p for p in Path("logs/llm_measure").iterdir() if p.is_dir()):
    for cond in CONDS:
        d = json.load(open(mdir / f"{cond}.json"))
        target = MARK if cond.startswith("marked") else (UNMARK if cond.startswith("unmarked") else None)
        comp = []
        for s in d["samples"]:
            if s["greedy"] or not s["columns"]:
                continue
            m = evaluate(X[:, [col[c] for c in s["columns"] if c in col]], y)[0]
            rows.append((mdir.name, cond, m, bool(LEAN & set(s["columns"]))))
            if target:
                # 条目遵从 = 涉及的地标全部落在给定区域内 / 至少一个落在区域内
                for it in s["items"]:
                    rs = {REG[l] for l in lms(it)}
                    comp.append((rs <= target, bool(rs & target)))
        if comp:
            c = np.array(comp)
            print(f"{mdir.name:22s} {cond:16s} 条目全部在给定区域内 {c[:, 0].mean():.2f}, 至少触及给定区域 {c[:, 1].mean():.2f}")
M = np.array([r[2] for r in rows]); L = np.array([r[3] for r in rows])
print(f"\n全部 {len(rows)} 个采样集合: 含躯干倾角/偏移(sh_mid-hip_mid) 的 {L.sum()} 个, macro {M[L].mean():.3f} ± {M[L].std():.3f};不含的 {(~L).sum()} 个, {M[~L].mean():.3f} ± {M[~L].std():.3f}")
print(f"最好的 5 个集合: {sorted(M)[-5:]}")
# 手工 5 量在同口径(患者级, 10 份划分)随机对照中的百分位
rng = np.random.default_rng(1)
ref7 = evaluate(X[:, [col[c] for c in HAND5]], y)[0]
r5 = np.array([evaluate(X[:, rng.choice(X.shape[1], 5, replace=False)], y)[0] for _ in range(300)])
r7 = np.array([evaluate(X[:, rng.choice(X.shape[1], 7, replace=False)], y)[0] for _ in range(300)])
print(f"\n同口径随机对照(各 300 次, 每次 10 份划分): 随机 5 列 {r5.mean():.3f} ± {r5.std():.3f}, 随机 7 列 {r7.mean():.3f} ± {r7.std():.3f}")
print(f"手工 5 量(库内 7 列) {ref7:.3f}: 在随机 5 列的第 {(r5 < ref7).mean() * 100:.0f} 百分位, 随机 7 列的第 {(r7 < ref7).mean() * 100:.0f} 百分位")
print(f"属性版 5 量的 0.717: 在随机 5 列的第 {(r5 < 0.717).mean() * 100:.0f} 百分位, 随机 7 列的第 {(r7 < 0.717).mean() * 100:.0f} 百分位")
one = evaluate(X[:, [col['incl|sh_mid-hip_mid|mean']]], y)
print(f"躯干倾角单列: macro {one[0]:.3f} AUC {one[1]:.3f}")

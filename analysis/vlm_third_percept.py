"""感知检验:VLM 对某个临床问题的回答分数,与同一段视频上几何测得的对应量,相关多少。
如果 VLM 连最显眼的躯干前倾都"看不见",那么换问题也救不了第三分支。"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
V = Path("logs/qwen_attributes/qwen3-vl-8b-instruct_448"); G = Path("logs/geom_attributes_3d")
PAIRS = [("trunk_forward_lean", "trunk_lean", +1), ("head_forward", "head_forward", +1), ("knee_flexion_stance", "knee_flexion_max", +1),
         ("short_stride", "step_length", -1), ("slow_walking", "gait_speed", -1), ("reduced_arm_swing", "arm_swing", -1),
         ("hip_extension_loss", "hip_range", -1), ("pelvic_retroversion", "lower_trunk_lean", +1)]
print(f"{'VLM 问题':22s} {'对应几何量':18s} {'预期方向':>6s} {'段级 Spearman':>14s} {'视频级':>8s} {'患者级':>8s} {'yes 率':>7s}")
for vq, gq, sign in PAIRS:
    dv, dg = json.load(open(V / f"{vq}.json")), json.load(open(G / f"{gq}.json"))
    seg_v, seg_g, vid_v, vid_g, pat = [], [], [], [], {}
    for name in dv:
        if name not in dg:
            continue
        a, b = np.asarray(dv[name]["scores"], float), np.asarray(dg[name]["scores"], float)
        n = min(len(a), len(b))
        if n == 0:
            continue
        seg_v.extend(a[:n]); seg_g.extend(b[:n]); vid_v.append(a.mean()); vid_g.append(b.mean())
        p = Path(dv[name]["json"]).stem.split("-")[0]
        pat.setdefault(p, [[], []]); pat[p][0].extend(a[:n]); pat[p][1].extend(b[:n])
    pv = [np.mean(x[0]) for x in pat.values()]; pg = [np.mean(x[1]) for x in pat.values()]
    print(f"{vq:22s} {gq:18s} {'+' if sign > 0 else '-':>6s} {spearmanr(seg_v, seg_g)[0]:+14.2f} {spearmanr(vid_v, vid_g)[0]:+8.2f} {spearmanr(pv, pg)[0]:+8.2f} {np.mean(np.array(seg_v) > 0):7.2f}")
print(f"段数 {len(seg_v)}, 视频 {len(vid_v)}, 患者 {len(pat)}")

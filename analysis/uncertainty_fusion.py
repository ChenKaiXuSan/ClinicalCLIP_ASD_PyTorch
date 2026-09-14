#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: uncertainty_fusion.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
方法 2:不确定性感知的患者级融合。现在的决策级融合用一个全局权重 w,而且是在全部 79 名患者上事后挑的。
这里改成**每个患者一个权重**,由两条分支各自的可靠性决定,门控在训练折上学,测试折只做前向:

    w_i = sigmoid(theta0 + theta . u_i),   p_fused_i = (1 - w_i) p_video_i + w_i p_geom_i

u_i(患者级可靠性特征):
  conf_v   视频分支置信度 |P - 0.5| * 2              std_v   视频分支段级概率的标准差(段间不一致)
  logn_v   log 段数                                   conf_g  几何分支置信度
  miss_g   该患者 3D 结果缺失的段占比(中位数填充的行)  outl_g  几何量的平均 |z|(离群程度)
门控目标:类别平衡的负对数似然 + L2(lambda 在训练患者上内层 4 折按 NLL 选)。

嵌套流程:外层就是训练用的 5 个患者折,患者 i 的分支概率都是它作为 test 时产生的(out-of-fold),
门控只用其它 4 折患者的 (概率, 特征, 标签) 训练,对第 k 折患者前向。所有对照走完全相同的嵌套:
  video / geom 单独;固定 w=0.5(无需选);嵌套全局 w(训练折上网格选);堆叠逻辑回归(两分支 logit);
  事后 w=0.7(泄漏,只作参考);上限 = 任一分支判对。

    python analysis/uncertainty_fusion.py --data-root $DATA --video B0_3dcnn_e50 M0_concept_learned_e50 A1_no_grounding_e50 \
        --seeds 42 1337 2024 --geom logs/geom_probs/geom3d_doctor_lr_probs.json --attr-dir logs/geom_attributes_3d_doctor
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from late_fusion import macro, mcnemar  # noqa: E402
from patient_level_stats import build_patient_map, macro_ci  # noqa: E402

FEATS = ["conf_v", "std_v", "logn_v", "conf_g", "miss_g", "outl_g"]


# ----------------------------------------------------------------------------- 数据
def load_video(root: Path, tag: str, seed: int, pmap: dict) -> dict:
    """{患者: {p:(2,), std, n, fold, label}},每个患者来自它作为 test 的那一折。"""
    out = {}
    for k in range(5):
        runs = sorted({p.parent.parent for p in (root / f"{tag}__f{k}_s{seed}").rglob("best_preds/*_pred.pt")},
                      key=lambda p: p.stat().st_mtime)
        if not runs:
            raise SystemExit(f"{tag} seed {seed} 缺第 {k} 折")
        best = runs[-1] / "best_preds"
        for pf in sorted(best.glob("*_pred.pt")):
            prob = torch.load(pf, map_location="cpu", weights_only=False).float().numpy()
            if not np.allclose(prob.sum(1), 1, atol=1e-3):
                prob = np.exp(prob - prob.max(1, keepdims=True)); prob /= prob.sum(1, keepdims=True)
            label = torch.load(pf.with_name(pf.name.replace("_pred.pt", "_label.pt")), map_location="cpu",
                               weights_only=False).long().numpy()
            names = json.loads(pf.with_name(pf.name.replace("_pred.pt", "_video_name.json")).read_text())
            by = defaultdict(list)
            for i, v in enumerate(names):
                by[pmap.get(v, v)].append(i)
            for pid, idx in by.items():
                out[pid] = {"p": prob[idx].mean(0), "std": float(prob[idx, 1].std()), "n": len(idx), "fold": k,
                            "label": int(np.bincount(label[idx]).argmax())}
    return out


def load_geom(path: str) -> dict:
    d = json.load(open(path))
    return {p: {"p": np.asarray(v["prob"], dtype=float), "label": int(v["label"])} for p, v in d.items()}


def geom_reliability(attr_dir: Path, pmap: dict) -> dict:
    """{患者: (缺失段占比, 全库 z 的平均 |z|, 该患者的原始段矩阵 (n_seg, A))}。
    缺失 = 该段全部属性都等于全库中位数(中位数填充的痕迹)。
    第三项供 outl_by_fold 用训练折统计量重算离群度(避免用到测试患者的均值/方差)。"""
    attrs = sorted(attr_dir.glob("*.json"))
    per_attr = [json.load(open(a)) for a in attrs]
    names = list(per_attr[0].keys())
    mat = {v: np.stack([np.asarray(d[v]["scores"], dtype=float) for d in per_attr], 1) for v in names}  # (n_seg, A)
    allv = np.concatenate(list(mat.values()))
    med, mu, sd = np.median(allv, 0), allv.mean(0), allv.std(0) + 1e-8
    by = defaultdict(list)
    for v, m in mat.items():
        by[pmap.get(v, v)].append(m)
    out = {}
    for pid, ms in by.items():
        m = np.concatenate(ms)
        missing = np.all(np.isclose(m, med), 1)
        z = np.abs((m - mu) / sd).mean(1)
        out[pid] = (float(missing.mean()), float(z.mean()), m)
    return out


def outl_by_fold(R: dict, pids: list, fold: np.ndarray) -> np.ndarray:
    """每个外层折:用训练折患者的段均值/标准差算所有患者的平均 |z|(测试患者的值只用训练统计量)。"""
    out = np.zeros(len(pids))
    for k in np.unique(fold):
        tr = [p for p, f in zip(pids, fold) if f != k]
        allv = np.concatenate([R[p][2] for p in tr])
        mu, sd = allv.mean(0), allv.std(0) + 1e-8
        for i, (p, f) in enumerate(zip(pids, fold)):
            if f == k:
                out[i] = float(np.abs((R[p][2] - mu) / sd).mean())
    return out


# ----------------------------------------------------------------------------- 门控
def _sig(x):
    return 1 / (1 + np.exp(-x))


def fit_gate(U, Pv, Pg, y, lam: float):
    """最小化类别平衡 NLL + lam*|theta|^2(不罚偏置)。U 已标准化。"""
    cw = np.array([len(y) / (2 * max((y == c).sum(), 1)) for c in (0, 1)])[y]
    X = np.c_[np.ones(len(y)), U]

    def loss(th):
        w = _sig(X @ th)[:, None]
        pf = (1 - w) * Pv + w * Pg
        return float((cw * -np.log(pf[np.arange(len(y)), y] + 1e-9)).sum() / cw.sum() + lam * (th[1:] ** 2).sum())

    res = minimize(loss, np.zeros(X.shape[1]), method="L-BFGS-B")
    return res.x


def gate_w(th, U):
    return _sig(np.c_[np.ones(len(U)), U] @ th)


def fuse(Pv, Pg, w):
    w = np.asarray(w, dtype=float).reshape(-1, 1)
    return (1 - w) * Pv + w * Pg


def nll(P, y):
    return float(-np.log(P[np.arange(len(y)), y] + 1e-9).mean())


def inner_folds(y, k=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = [rng.permutation(np.where(y == c)[0]) for c in (0, 1)]
    folds = [[] for _ in range(k)]
    for arr in idx:
        for j, i in enumerate(arr):
            folds[j % k].append(i)
    return [np.array(f) for f in folds]


def nested_global_w(Pv, Pg, y, grid):
    """训练折上按平衡准确率选全局 w,平手取 NLL 小的。"""
    best = None
    for w in grid:
        pf = fuse(Pv, Pg, np.full(len(y), w))
        key = (macro(pf.argmax(1), y), -nll(pf, y))
        if best is None or key > best[0]:
            best = (key, w)
    return best[1]


def step_gate_fit(key_tr, Pv, Pg, y, grid_q=(0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5), grid_lo=(0.0, 0.25, 0.5), grid_hi=(0.75, 1.0)):
    """阶梯门控:可靠性特征低于训练患者的 q 分位 -> w_lo, 否则 -> w_hi。训练折上按 (平衡准确率, -NLL) 选 (q, w_lo, w_hi)。"""
    best = None
    for q in grid_q:
        thr = float(np.quantile(key_tr, q))
        low = key_tr < thr
        for wl in grid_lo:
            for wh in grid_hi:
                w = np.where(low, wl, wh)
                pf = fuse(Pv, Pg, w)
                k = (macro(pf.argmax(1), y), -nll(pf, y))
                if best is None or k > best[0]:
                    best = (k, thr, wl, wh)
    return best[1:]


def run_one(V: dict, G: dict, R: dict, lams, grid, seed_inner=0, light=False, perm_seed=None):
    """light=True 只算阶梯门控与固定/嵌套对照(跳过 sigmoid 门控和堆叠);perm_seed 置换离群度特征(对照)。"""
    pids = sorted(set(V) & set(G) & set(R))
    y = np.array([V[p]["label"] for p in pids])
    fold = np.array([V[p]["fold"] for p in pids])
    Pv = np.stack([V[p]["p"] for p in pids])
    Pg = np.stack([G[p]["p"] for p in pids])
    U = np.array([[abs(V[p]["p"][1] - 0.5) * 2, V[p]["std"], np.log(V[p]["n"]),
                   abs(G[p]["p"][1] - 0.5) * 2, R[p][0], R[p][1]] for p in pids])
    U[:, FEATS.index("outl_g")] = outl_by_fold(R, pids, fold)  # 测试折的离群度只用训练折统计量
    if perm_seed is not None:
        j = FEATS.index("outl_g")
        U[:, j] = np.random.default_rng(perm_seed).permutation(U[:, j])

    pred = {k: np.zeros(len(pids), dtype=int) for k in
            ["video", "geom", "fixed0.5", "posthoc0.7", "nested_w", "stack", "gate", "step_outl", "step_conf"]}
    wg = np.zeros(len(pids))
    chosen_w, chosen_lam, thetas, chosen_step = [], [], [], []
    outl, confg = U[:, FEATS.index("outl_g")], U[:, FEATS.index("conf_g")]
    for k in range(5):
        te, tr = fold == k, fold != k
        pred["video"][te] = Pv[te].argmax(1)
        pred["geom"][te] = Pg[te].argmax(1)
        pred["fixed0.5"][te] = fuse(Pv[te], Pg[te], np.full(te.sum(), 0.5)).argmax(1)
        pred["posthoc0.7"][te] = fuse(Pv[te], Pg[te], np.full(te.sum(), 0.7)).argmax(1)
        w_star = nested_global_w(Pv[tr], Pg[tr], y[tr], grid)
        chosen_w.append(w_star)
        pred["nested_w"][te] = fuse(Pv[te], Pg[te], np.full(te.sum(), w_star)).argmax(1)
        if not light:
            # 堆叠:两分支 logit -> 逻辑回归
            L = np.c_[np.log(Pv[:, 1] + 1e-6) - np.log(Pv[:, 0] + 1e-6), np.log(Pg[:, 1] + 1e-6) - np.log(Pg[:, 0] + 1e-6)]
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(L[tr], y[tr])
            pred["stack"][te] = clf.predict(L[te])
            # 门控:标准化 + 内层选 lambda
            mu, sd = U[tr].mean(0), U[tr].std(0) + 1e-8
            Utr, Ute = (U[tr] - mu) / sd, (U[te] - mu) / sd
            ytr, Pvtr, Pgtr = y[tr], Pv[tr], Pg[tr]
            best = None
            for lam in lams:
                s = 0.0
                for f in inner_folds(ytr, 4, seed_inner):
                    m = np.ones(len(ytr), bool); m[f] = False
                    th = fit_gate(Utr[m], Pvtr[m], Pgtr[m], ytr[m], lam)
                    s += nll(fuse(Pvtr[f], Pgtr[f], gate_w(th, Utr[f])), ytr[f])
                if best is None or s < best[0]:
                    best = (s, lam)
            chosen_lam.append(best[1])
            th = fit_gate(Utr, Pvtr, Pgtr, ytr, best[1])
            thetas.append(th)
            wg[te] = gate_w(th, Ute)
            pred["gate"][te] = fuse(Pv[te], Pg[te], wg[te]).argmax(1)
        # 阶梯门控:单个可靠性特征 + 阈值 + 两档权重,全部在训练折上选
        for name, key in (("step_outl", outl), ("step_conf", confg)):
            thr, wl, wh = step_gate_fit(key[tr], Pv[tr], Pg[tr], y[tr])
            if name == "step_outl":
                chosen_step.append((round(thr, 2), wl, wh))
            pred[name][te] = fuse(Pv[te], Pg[te], np.where(key[te] < thr, wl, wh)).argmax(1)
    ceiling = float(np.mean([((pred["video"] == y) | (pred["geom"] == y))[y == c].mean() for c in (0, 1)]))
    return {"pids": pids, "y": y, "pred": pred, "w": wg, "U": U, "chosen_w": chosen_w, "chosen_lam": chosen_lam,
            "chosen_step": chosen_step, "theta": np.mean(thetas, 0) if thetas else np.zeros(len(FEATS) + 1),
            "ceiling": ceiling}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--video", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2024])
    ap.add_argument("--geom", nargs="+", required=True, help="患者级概率 json(几何分支)")
    ap.add_argument("--attr-dir", default="logs/geom_attributes_3d_doctor", help="算 3D 可靠性特征用的属性目录")
    ap.add_argument("--lams", default="0.3,1,3,10")
    ap.add_argument("--out", default=None, help="把每组的患者级门控权重与预测存成 json")
    ap.add_argument("--perm", type=int, default=0, help="置换对照:把离群度特征在患者间随机置换 N 次, 只报阶梯门控")
    args = ap.parse_args()

    pmap = build_patient_map(args.data_root)
    root = Path(args.root)
    R = geom_reliability(Path(args.attr_dir), pmap)
    lams = [float(x) for x in args.lams.split(",")]
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    keys = ["video", "geom", "fixed0.5", "nested_w", "stack", "gate", "step_outl", "step_conf", "posthoc0.7"]
    dump = {}
    agg = defaultdict(list)
    if args.perm > 0:
        print(f"置换对照: 离群度特征随机置换 {args.perm} 次, 阶梯门控(outl)的 9 组均值; 真实特征的值见正常运行")
        for gpath in args.geom:
            G = load_geom(gpath)
            Vs = {(v, s): load_video(root, v, s, pmap) for v in args.video for s in args.seeds}
            real = np.mean([macro(run_one(V, G, R, lams, grid, light=True)["pred"]["step_outl"], run_one(V, G, R, lams, grid, light=True)["y"])
                            for V in Vs.values()])
            perms = []
            for ps in range(args.perm):
                perms.append(np.mean([macro((r := run_one(V, G, R, lams, grid, light=True, perm_seed=ps))["pred"]["step_outl"], r["y"])
                                      for V in Vs.values()]))
            perms = np.array(perms)
            print(f"{Path(gpath).stem}: 真实 {real:.3f}; 置换 {perms.mean():.3f}±{perms.std():.3f} "
                  f"[{perms.min():.3f}, {perms.max():.3f}], 真实值的置换分位 {(perms < real).mean():.2f}")
        return
    for gpath in args.geom:
        G = load_geom(gpath)
        gname = Path(gpath).stem.replace("_probs", "").replace("_lr", "")
        print(f"\n===== 几何分支: {gname}   (对照 McNemar 写成 参照独对:门控独对 p)")
        print(f"{'视频分支':24s} {'seed':>5s} " + " ".join(f"{k:>10s}" for k in keys) +
              f" {'上限':>6s} | {'阶梯(outl) vs 视频':>17s} {'vs 几何':>10s} {'vs 嵌套w':>11s} {'vs sigmoid门控':>14s} | 阶梯 (阈值,w低,w高) 逐折 | 嵌套w 选值")
        for vtag in args.video:
            for seed in args.seeds:
                V = load_video(root, vtag, seed, pmap)
                r = run_one(V, G, R, lams, grid)
                y, pr = r["y"], r["pred"]
                m = {k: macro(pr[k], y) for k in keys}
                for k in keys:
                    agg[(gname, vtag, k)].append(m[k])
                agg[(gname, vtag, "ceiling")].append(r["ceiling"])
                def mc(a, b="step_outl"):
                    a_only, b_only, p = mcnemar(pr[a] == y, pr[b] == y)
                    return f"{a_only:2d}:{b_only:<2d} p{p:.2f}"
                print(f"{vtag:24s} {seed:5d} " + " ".join(f"{m[k]:10.3f}" for k in keys) + f" {r['ceiling']:6.3f} | "
                      f"{mc('video'):>17s} {mc('geom'):>10s} {mc('nested_w'):>11s} {mc('gate'):>14s} | "
                      f"{' '.join(f'({t},{wl},{wh})' for t, wl, wh in r['chosen_step'])} | "
                      f"{','.join(f'{w:.2f}' for w in r['chosen_w'])}")
                dump[f"{gname}|{vtag}|s{seed}"] = {
                    "theta": r["theta"].tolist(), "feats": FEATS, "chosen_lam": r["chosen_lam"],
                    "w": dict(zip(r["pids"], r["w"].tolist())),
                    "spearman_w_vs_feat": {f: float(stats.spearmanr(r["w"], r["U"][:, j])[0]) for j, f in enumerate(FEATS)},
                }
                lo, hi = macro_ci(pr["gate"], y, n_boot=2000)
                agg[(gname, vtag, "gate_ci")].append((lo, hi))
        print(f"\n--- {gname}: 三种子均值")
        print(f"{'视频分支':24s} " + " ".join(f"{k:>10s}" for k in keys) + f" {'上限':>6s}")
        for vtag in args.video:
            print(f"{vtag:24s} " + " ".join(f"{np.mean(agg[(gname, vtag, k)]):10.3f}" for k in keys) +
                  f" {np.mean(agg[(gname, vtag, 'ceiling')]):6.3f}")
        print(f"{'9 组均值':24s} " + " ".join(f"{np.mean([x for v in args.video for x in agg[(gname, v, k)]]):10.3f}" for k in keys) +
              f" {np.mean([x for v in args.video for x in agg[(gname, v, 'ceiling')]]):6.3f}")
        # 门控可解释性:系数均值(标准化特征)与 w 对特征的秩相关
        ths = np.array([v["theta"] for kk, v in dump.items() if kk.startswith(gname + "|")])
        sp = {f: np.mean([v["spearman_w_vs_feat"][f] for kk, v in dump.items() if kk.startswith(gname + "|")]) for f in FEATS}
        print("门控系数均值(偏置, " + ", ".join(FEATS) + "): " + ", ".join(f"{x:+.2f}" for x in ths.mean(0)))
        print("w 与特征的 Spearman 均值: " + ", ".join(f"{f} {v:+.2f}" for f, v in sp.items()))
    if args.out:
        json.dump(dump, open(args.out, "w"), indent=1)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()

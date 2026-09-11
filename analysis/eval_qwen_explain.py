#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: eval_qwen_explain.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
角色三:VLM 作为推理时的**解释生成器**。分类由主线 M0(concept 架构)负责,它给出预测、
关注区域(存在性头)和该区域的注意力图;把注意力峰值处画成红框交给 Qwen3-VL,
让它以骨科医生口吻描述框内支持该分类的步态特征。

没有医生评审时用三个自动指标:
  1. 反事实一致性:同一视频分别以"真实预测"和"翻转预测"为条件各生成一段解释,
     再让 VLM(不给标签)判断哪段更符合视频。若只有 ~50%,解释是从标签编出来的,不是看出来的。
  2. 视频错配对照:真实解释 vs "同标签、同区域但来自另一条视频"的解释。若判官也只有 ~50%,
     说明判官本身看不出差别(指标 1 的 50% 就不能归因于解释)。
  3. 词汇多样性 distinct-2 与平均两两 Jaccard:解释是不是同一套模板。
另报模型关注区域与医生标注区域的一致率(这是 M0 的性质,作为参照)。

输出 <out_dir>/explanations.jsonl(逐视频)、summary.json、samples.md(给医生看的样例)。

    python analysis/eval_qwen_explain.py --root-path $DATA --ckpt <M0 fold0 ckpt> --fold 0 --limit 120
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))

from dataloader.med_attn_map import MedAttnMap, REGIONS  # noqa: E402
from dataloader.whole_video_dataset import LabeledGaitVideoDataset  # noqa: E402
from models.vlm_encoder import VLMTokenEncoder  # noqa: E402
from trainer.train_clinical_concept import ClinicalConceptModule  # noqa: E402

REGION_PHRASE = {
    "foot": "feet and ankles", "wrist": "wrists and arm swing", "shoulder": "shoulders",
    "lumbar_pelvis": "lumbar spine and pelvis", "head": "head and neck",
}
SYSTEM = ("You are an orthopedic surgeon reviewing a lateral-view walking video of a patient "
          "in a gait laboratory.")
LABEL_TEXT = {0: "consistent with adult spinal deformity (ASD)", 1: "not consistent with adult spinal deformity"}


def explain_prompt(label: int, region: str) -> str:
    return (f"{SYSTEM} A diagnostic model classified this patient's gait as {LABEL_TEXT[label]}, and its attention "
            f"focused on the {REGION_PHRASE[region]}, marked by the red box on every frame. In two or three sentences, "
            f"describe the visible gait features inside the red box that support this classification. "
            f"Refer only to what can be seen in the video.")


def judge_prompt(a: str, b: str) -> str:
    return (f"{SYSTEM} Two surgeons each wrote a short description of the gait in this video.\n"
            f"Description A: {a}\nDescription B: {b}\n"
            f"Which description matches what is actually visible in the video better? Answer with exactly one letter: A or B.")


def draw_box(video: torch.Tensor, cx: float, cy: float, half: int, t: int = 4) -> torch.Tensor:
    out = video.clone()
    _, _, _, h, w = out.shape
    x0, x1 = int(max(0, cx - half)), int(min(w - 1, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(h - 1, cy + half))
    col = torch.tensor([1.0, 0.0, 0.0], dtype=out.dtype, device=out.device).view(1, 3, 1, 1, 1)
    out[:, :, :, y0:y0 + t, x0:x1 + 1] = col
    out[:, :, :, max(0, y1 - t + 1):y1 + 1, x0:x1 + 1] = col
    out[:, :, :, y0:y1 + 1, x0:x0 + t] = col
    out[:, :, :, y0:y1 + 1, max(0, x1 - t + 1):x1 + 1] = col
    return out


def distinct_n(texts: list[str], n: int = 2) -> float:
    grams, total = set(), 0
    for t in texts:
        w = t.lower().split()
        g = list(zip(*[w[i:] for i in range(n)]))
        grams.update(g)
        total += len(g)
    return len(grams) / max(total, 1)


def mean_jaccard(texts: list[str], k: int = 200, seed: int = 0) -> float:
    rng = random.Random(seed)
    sets = [set(t.lower().split()) for t in texts]
    if len(sets) < 2:
        return float("nan")
    vals = []
    for _ in range(k):
        i, j = rng.sample(range(len(sets)), 2)
        vals.append(len(sets[i] & sets[j]) / max(len(sets[i] | sets[j]), 1))
    return float(np.mean(vals))


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - half), min(1.0, c + half)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-path", required=True)
    ap.add_argument("--ckpt", required=True, help="主线 M0 的 checkpoint(与 --fold 同折)")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--class-num", type=int, default=2)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--max-new-tokens", type=int, default=110)
    ap.add_argument("--box-half", type=int, default=80, help="红框半边长(像素,448 坐标)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="logs/qwen_explain")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    root = Path(args.root_path)
    info = root / "clinical_CLIP_dataset"
    folds = json.load(open(info / "index_mapping" / str(args.class_num) / "index.json"))
    paths = [info / "json_mix" / p.split("json_mix/")[-1] for p in folds[args.fold]["test"]]
    # 均匀抽样而不是取前 N 条:test 列表按疾病排序,前 N 条会全是 ASD
    if args.limit and len(paths) > args.limit:
        step = len(paths) / args.limit
        paths = [paths[int(i * step)] for i in range(args.limit)]
    med = MedAttnMap(str(info / "doctor_result"), str(info / "seg_skeleton_pkl"))
    dataset = LabeledGaitVideoDataset("explain", paths, img_size=args.img_size, num_samples=8, attn_map=med)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver("now", lambda pattern="%Y-%m-%d": datetime.now().strftime(pattern))
    m0 = ClinicalConceptModule.load_from_checkpoint(args.ckpt, map_location=device).eval().to(device)
    vlm = VLMTokenEncoder("qwen3vl", args.model, hidden_dim=256, img_size=args.img_size, dtype=args.dtype).to(device).eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    pool_by_key: dict[tuple, list[str]] = {}
    scale = args.img_size / 224

    # ---- 第一遍:M0 预测 + 生成真实/反事实解释 ----
    for i in range(len(dataset)):
        s = dataset[i]
        v448 = s["video"].to(device)  # (S,3,8,448,448)
        S = v448.shape[0]
        v224 = F.interpolate(v448.transpose(1, 2).reshape(S * 8, 3, args.img_size, args.img_size),
                             size=(224, 224), mode="bilinear", align_corners=False, antialias=True
                             ).reshape(S, 8, 3, 224, 224).transpose(1, 2)
        with torch.no_grad():
            o = m0.model(v224)
        prob = torch.softmax(o["logits"], 1).mean(0)  # (2,)
        pred = int(prob.argmax())
        presence = torch.sigmoid(o["region_logits"]).mean(0)  # (R,)
        r = int(presence.argmax())
        region = REGIONS[r]
        # 该区域的注意力在段与时间上平均 -> (7,7) 峰值 -> 448 坐标
        heat = o["attn"][:, r].mean(dim=(0, 1))
        hy, hx = divmod(int(heat.argmax()), heat.shape[-1])
        cell = 224 / heat.shape[-1]
        cx, cy = (hx + 0.5) * cell * scale, (hy + 0.5) * cell * scale
        # 取对预测类最自信的一段给 VLM
        seg = int(torch.softmax(o["logits"], 1)[:, pred].argmax())
        clip = draw_box(v448[seg:seg + 1], cx, cy, args.box_half)

        exp_true = vlm.tower.generate(clip, explain_prompt(pred, region), args.max_new_tokens)[0]
        exp_cf = vlm.tower.generate(clip, explain_prompt(1 - pred, region), args.max_new_tokens)[0]
        label = int(int(s["label"]) != 0)  # 二分类:ASD=0
        doc = med.presence_for(s["video_name"])
        rec = {
            "video_name": s["video_name"], "json": str(paths[i]).split("json_mix/")[-1],
            "label": label, "pred": pred, "prob_asd": float(prob[0]), "region": region,
            "presence": [round(float(x), 3) for x in presence], "box": [round(cx), round(cy)], "segment": seg,
            "doctor_regions": [REGIONS[k] for k in range(len(REGIONS)) if doc[k] > 0],
            "explanation": exp_true, "explanation_counterfactual": exp_cf,
        }
        records.append(rec)
        pool_by_key.setdefault((pred, region), []).append(exp_true)
        print(f"[{i+1}/{len(dataset)}] {s['video_name']} label={label} pred={pred} region={region}", flush=True)

    # ---- 第二遍:判官(不给标签) ----
    win_cf, win_mismatch, n_mismatch = 0, 0, 0
    for i, rec in enumerate(records):
        s = dataset[i]
        v448 = s["video"].to(device)
        clip = draw_box(v448[rec["segment"]:rec["segment"] + 1], rec["box"][0], rec["box"][1], args.box_half)
        # 1. 真实 vs 反事实
        flip = random.random() < 0.5
        a, b = (rec["explanation_counterfactual"], rec["explanation"]) if flip else (rec["explanation"], rec["explanation_counterfactual"])
        lg = vlm.tower.answer_logits(clip, judge_prompt(a, b), ["A", "B"])[0]
        pick_true = (lg[1] > lg[0]) if flip else (lg[0] > lg[1])
        rec["judge_cf_true_wins"] = bool(pick_true)
        win_cf += int(pick_true)
        # 2. 真实 vs 视频错配(同预测、同区域的另一条视频的真实解释)
        others = [e for e in pool_by_key[(rec["pred"], rec["region"])] if e != rec["explanation"]]
        if others:
            other = random.choice(others)
            flip = random.random() < 0.5
            a, b = (other, rec["explanation"]) if flip else (rec["explanation"], other)
            lg = vlm.tower.answer_logits(clip, judge_prompt(a, b), ["A", "B"])[0]
            pick_true = (lg[1] > lg[0]) if flip else (lg[0] > lg[1])
            rec["judge_mismatch_true_wins"] = bool(pick_true)
            win_mismatch += int(pick_true)
            n_mismatch += 1

    with open(out_dir / "explanations.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(records)
    texts = [r["explanation"] for r in records]
    region_ok = [r["region"] in r["doctor_regions"] for r in records if r["doctor_regions"]]
    mention = [REGION_PHRASE[r["region"]].split()[0] in r["explanation"].lower() for r in records]
    summary = {
        "model": args.model, "ckpt": args.ckpt, "fold": args.fold, "n_videos": n,
        "n_patients": len({Path(r["json"]).stem.split("-")[0] for r in records}),
        "pred_dist": dict(Counter(r["pred"] for r in records)),
        "m0_video_level_balanced_acc": float(np.mean([
            np.mean([r["pred"] == c for r in records if r["label"] == c]) for c in (0, 1)
            if any(r["label"] == c for r in records)])),
        "counterfactual_true_win_rate": win_cf / n, "counterfactual_ci": wilson(win_cf, n),
        "video_mismatch_true_win_rate": win_mismatch / max(n_mismatch, 1), "video_mismatch_ci": wilson(win_mismatch, n_mismatch),
        "n_mismatch": n_mismatch,
        "model_region_in_doctor_regions": float(np.mean(region_ok)) if region_ok else None,
        "n_with_doctor_labels": len(region_ok),
        "explanation_mentions_region": float(np.mean(mention)),
        "distinct_2": distinct_n(texts, 2), "mean_pairwise_jaccard": mean_jaccard(texts),
        "mean_words": float(np.mean([len(t.split()) for t in texts])),
        "note": "判官为同一 VLM、不给标签;反事实 50% = 解释从标签编出;错配 50% = 判官看不出视频差别",
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    rng = random.Random(1)
    sample = rng.sample(records, min(8, n))
    with open(out_dir / "samples.md", "w") as f:
        f.write("# VLM 解释样例(角色三)\n\n每条:M0 的预测与关注区域 → Qwen3-VL 的解释;附反事实解释以对照。\n\n")
        for r in sample:
            f.write(f"## {r['video_name']}\n- 真实标签: {'ASD' if r['label']==0 else 'non-ASD'} | M0 预测: "
                    f"{'ASD' if r['pred']==0 else 'non-ASD'} (P(ASD)={r['prob_asd']:.2f}) | 关注区域: {r['region']} | "
                    f"医生标注: {', '.join(r['doctor_regions']) or '无'}\n"
                    f"- 判官(反事实): {'真实胜' if r.get('judge_cf_true_wins') else '反事实胜'}\n\n"
                    f"**解释**: {r['explanation']}\n\n**反事实解释**({'non-ASD' if r['pred']==0 else 'ASD'} 条件下): "
                    f"{r['explanation_counterfactual']}\n\n")
    print(f"样例 -> {out_dir/'samples.md'}")


if __name__ == "__main__":
    main()

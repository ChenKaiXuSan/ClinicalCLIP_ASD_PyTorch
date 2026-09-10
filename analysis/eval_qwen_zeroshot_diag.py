#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: eval_qwen_zeroshot_diag.py
Project: analysis
Author: Kaixu Chen
-----
Comment:
Q3:零样本诊断基线。直接问 Qwen3-VL "这个病人有没有成人脊柱畸形的步态特征",
用回答首 token 的 yes/no logit 差作为 ASD 分数,段级 -> 患者级(均值)汇总,
报患者级平衡准确率与 AUC。

预期接近随机;这是一个必须报告的负结果 —— 它说明通用 VLM 的语言知识本身不足以
从侧视步态识别 ASD,后续任何收益都来自监督训练而非 VLM 的"医学常识"。

用法(计算节点):
    python analysis/eval_qwen_zeroshot_diag.py --root-path /work/SKIING/chenkaixu/data/asd_dataset \
        --model Qwen/Qwen3-VL-8B-Instruct --img-size 448 --fold 0
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))

from dataloader.whole_video_dataset import LabeledGaitVideoDataset  # noqa: E402
from models.vlm_encoder import VLMTokenEncoder  # noqa: E402


def patient_of(json_path: Path) -> str:
    """与 analysis/patient_level_stats.py 的 build_patient_map 同规则:
    json 文件名 `<患者>-0001.json` 里 `-` 之前的部分(video_name 的命名不统一,不能用它)。"""
    return json_path.stem.split("-")[0]


def auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--img-size", type=int, default=448)
    parser.add_argument("--fold", default="0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--class-num", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--prompt", default="diagnose")
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"], help="bf16 隐状态误差可达 20%%,默认 fp32")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root_path)
    info = root / "clinical_CLIP_dataset"
    folds = json.load(open(info / "index_mapping" / str(args.class_num) / "index.json"))
    paths = [info / "json_mix" / p.split("json_mix/")[-1] for p in folds[args.fold][args.split]]
    if args.limit:
        paths = paths[: args.limit]
    dataset = LabeledGaitVideoDataset("qwen_diag", paths, img_size=args.img_size, num_samples=args.num_samples)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    enc = VLMTokenEncoder("qwen3vl", args.model, hidden_dim=256, img_size=args.img_size, dtype=args.dtype).to(device).eval()

    seg_scores, seg_labels, seg_patients = [], [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        video = sample["video"].to(device)
        # json 里 label 是三分类原始标签(ASD=0),二分类:ASD 为正类 1
        is_asd = int(int(sample["label"]) == 0)
        lg = torch.cat([enc.tower.answer_logits(v, args.prompt, ["yes", "no"]) for v in video.split(args.chunk)], 0)
        score = (lg[:, 0] - lg[:, 1]).cpu()  # yes - no
        seg_scores.extend(score.tolist())
        seg_labels.extend([is_asd] * len(score))
        seg_patients.extend([patient_of(Path(paths[i]))] * len(score))
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(dataset)}] seg AUC={auc(seg_scores, seg_labels):.3f}", flush=True)

    # 患者级:分数均值
    by_p = defaultdict(list)
    lab_p = {}
    for s, l, p in zip(seg_scores, seg_labels, seg_patients):
        by_p[p].append(s)
        lab_p[p] = l
    p_scores = [sum(v) / len(v) for v in by_p.values()]
    p_labels = [lab_p[p] for p in by_p]
    # 阈值取 0(yes 与 no 等可能),另报最佳阈值下的平衡准确率作为上界
    def bal_acc(thr: float) -> float:
        pred = [int(s > thr) for s in p_scores]
        tp = sum(p and l for p, l in zip(pred, p_labels)); fn = sum((not p) and l for p, l in zip(pred, p_labels))
        tn = sum((not p) and (not l) for p, l in zip(pred, p_labels)); fp = sum(p and (not l) for p, l in zip(pred, p_labels))
        rec1 = tp / max(tp + fn, 1); rec0 = tn / max(tn + fp, 1)
        return 0.5 * (rec1 + rec0)

    result = {
        "model": args.model, "fold": args.fold, "split": args.split, "prompt": args.prompt,
        "n_segments": len(seg_scores), "n_patients": len(p_scores),
        "n_asd_patients": sum(p_labels), "n_non_asd_patients": len(p_labels) - sum(p_labels),
        "segment_auc": auc(seg_scores, seg_labels),
        "patient_auc": auc(p_scores, p_labels),
        "patient_balanced_acc_thr0": bal_acc(0.0),
        "patient_balanced_acc_best_thr": max(bal_acc(t) for t in sorted(set(p_scores))) if p_scores else float("nan"),
        "yes_rate_segments": sum(s > 0 for s in seg_scores) / max(len(seg_scores), 1),
        "note": "分数 = logit(yes) - logit(no);best_thr 是在同一批数据上挑的阈值,只作上界参考",
        # 逐患者分数:5 折 test 恰好划分全部 79 个患者,拼起来就是患者级的 AUC / 平衡准确率
        "patients": {p: {"score": sum(v) / len(v), "label": lab_p[p], "n_segments": len(v)}
                     for p, v in by_p.items()},
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: llm_propose.py
Project: analysis/carepd
Author: Kaixu Chen
-----
Comment:
"医生看哪里 -> 量什么"这一步交给 LLM:输入医生关注的身体区域(群体级,不含任何患者数据或标签)和
可用的 3D 地标,LLM 在测量库的语法内提出 K 个测量;几何代码照着算(llm_measure_eval.py)。
目的是把论文里唯一手工的一步(5 个量是作者定的)变成可复现的步骤,并检验它是否仍然有效。

条件(2 × 2 + 1,区分"医生区域的作用"与"LLM 自带疾病知识的作用"):
  marked_task     医生标注的区域(腰椎骨盆 / 头 / 肩) + 告知鉴别任务
  marked_blind    医生标注的区域,不告知疾病
  unmarked_task   未标区域(膝 / 足踝 / 腕肘) + 告知鉴别任务
  unmarked_blind  未标区域,不告知疾病
  noregion_task   不给区域,只告知鉴别任务(纯 LLM 知识)

每个条件 1 次贪心 + N 次采样(temperature 0.7);输出 logs/llm_measure/<模型>/<条件>.json:
  [{"sample": i, "raw": 文本, "items": [解析后的条目], "columns": [测量库列名], "n_invalid": k}]
提示词是固定模板,写的人(也看过结果)没有在里面放任何具体测量的提示;地标、量、统计方式的列表就是测量库的全集。

    python analysis/llm_measure_propose.py --model Qwen/Qwen3-VL-8B-Instruct --n 10 --out logs/llm_measure
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LANDMARKS = {
    "head": "head centre", "neck": "base of the neck",
    "sh_L": "left shoulder", "sh_R": "right shoulder", "sh_mid": "midpoint of the two shoulders",
    "spine3": "upper (thoracic) spine joint",
    "el_L": "left elbow", "el_R": "right elbow",
    "spine1": "lower (lumbar) spine joint",
    "wr_L": "left wrist", "wr_R": "right wrist",
    "hip_L": "left hip", "hip_R": "right hip", "hip_mid": "midpoint of the two hips (pelvis centre)",
    "kn_L": "left knee", "kn_R": "right knee",
    "ank_L": "left ankle", "ank_R": "right ankle",
    "foot_L": "left forefoot", "foot_R": "right forefoot",
}
ORDER = list(LANDMARKS)
# 角名 -> 测量库里的列名片段(双侧的展开成左右两列)
ANGLES = {
    "head_trunk": ("angle at the neck between head and pelvis", ["head_trunk"]),
    "trunk_bend": ("angle at the lumbar spine between shoulders and pelvis", ["trunk_bend"]),
    "spine_curve": ("angle at the thoracic spine between neck and lumbar spine", ["spine_curve"]),
    "hip": ("hip joint angle between trunk and thigh, both sides", ["hip_L", "hip_R"]),
    "knee": ("knee joint angle, both sides", ["knee_L", "knee_R"]),
    "ankle": ("ankle joint angle between shank and foot, both sides", ["ankle_L", "ankle_R"]),
    "elbow": ("elbow joint angle, both sides", ["elbow_L", "elbow_R"]),
    "shoulder_arm": ("angle at the shoulder between trunk and upper arm, both sides", ["shoulder_arm_L", "shoulder_arm_R"]),
}
PAIR_Q = {"incl": "sagittal inclination of the line from the lower to the upper landmark relative to vertical, in degrees",
          "dz": "forward (walking-direction) offset of the upper landmark relative to the lower one, divided by trunk length",
          "dy": "height difference between the two landmarks, divided by trunk length",
          "dx": "left-right distance between the two landmarks, divided by trunk length"}
PAIR_STATS, TRI_STATS = ["mean", "range"], ["mean", "min", "max", "range"]

# 临床先验 = MDS-UPDRS 步态条目的评分要点(公开文本),相当于 ASD 里医生的关注区域
RUBRIC = "stride amplitude, stride speed, height of foot lift, heel strike during walking, turning, and arm swing"
TASK = ("The goal is to rate the severity of gait impairment in people with Parkinson's disease "
        "(the clinician-rated gait item of the MDS-UPDRS, 0 = normal to 3 = severe).")
BLIND = "The goal is to characterise each person's gait for a clinical assessment."
CONDITIONS = {
    "prior_task": (TASK, RUBRIC), "prior_blind": (BLIND, RUBRIC), "noprior_task": (TASK, None), "noprior_blind": (BLIND, None),
}
K = 8


def build_prompt(task: str, regions: list[str] | None) -> str:
    lm = "\n".join(f"  {k}: {v}" for k, v in LANDMARKS.items())
    ang = "\n".join(f"  {k}: {v[0]}" for k, v in ANGLES.items())
    pq = "\n".join(f"  {k}: {v}" for k, v in PAIR_Q.items())
    reg = ("" if regions is None else
           "Clinicians who rate these walks are instructed to observe the following aspects: "
           f"{regions}. The measurements you propose should quantify these aspects.\n\n")
    return (
        "We analyse recordings of people walking. For every frame we have the 3D positions of the landmarks listed below "
        "in a body-centred coordinate frame (x: left-right, y: vertical, z: walking direction). A further scalar, the "
        "walking speed of the pelvis in leg lengths per second, is also available.\n\n"
        f"{task}\n\n{reg}"
        f"Propose exactly {K} different measurements to compute from the landmarks. Each measurement is summarised over "
        "1-second windows of walking and must have ONE of these forms:\n\n"
        '(1) a two-landmark quantity: {"type": "pair", "upper": <landmark>, "lower": <landmark>, "quantity": <q>, "stat": <s>}\n'
        f"quantities q:\n{pq}\nstat s: mean (average over the window) or range (max minus min over the window)\n\n"
        '(2) a joint angle: {"type": "angle", "angle": <name>, "stat": <s>}\n'
        f"angle names:\n{ang}\nstat s: mean, min, max or range over the window\n\n"
        '(3) the walking speed: {"type": "speed"}\n\n'
        f"Landmarks:\n{lm}\n\n"
        f'Answer with a JSON array of exactly {K} objects, each with the fields above plus "rationale" (one short sentence). '
        "Output the JSON array only."
    )


def parse(raw: str) -> tuple[list[dict], list[str], int]:
    """抽出 JSON 数组并映射到测量库列名。非法条目计数后丢弃。"""
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return [], [], K
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        # 逐个对象抢救
        items = []
        for obj in re.findall(r"\{[^{}]*\}", m.group(0), re.S):
            try:
                items.append(json.loads(obj))
            except json.JSONDecodeError:
                pass
    cols, good, bad = [], [], 0
    for it in items:
        if not isinstance(it, dict):
            bad += 1
            continue
        t = it.get("type")
        if t == "pair":
            a, b, q, s = it.get("upper"), it.get("lower"), it.get("quantity"), it.get("stat")
            if a in LANDMARKS and b in LANDMARKS and a != b and q in PAIR_Q and s in PAIR_STATS:
                i, j = sorted((ORDER.index(a), ORDER.index(b)))      # 测量库按自上而下的地标顺序命名
                cols.append(f"{q}|{ORDER[i]}-{ORDER[j]}|{s}"); good.append(it)
            else:
                bad += 1
        elif t == "angle":
            n, s = it.get("angle"), it.get("stat")
            if n in ANGLES and s in TRI_STATS:
                cols.extend(f"ang|{tri}|{s}" for tri in ANGLES[n][1]); good.append(it)
            else:
                bad += 1
        elif t == "speed":
            cols.append("speed|pelvis|mean"); good.append(it)
        else:
            bad += 1
    return good, list(dict.fromkeys(cols)), bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--n", type=int, default=10, help="每个条件的采样次数(另加 1 次贪心)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--out", default="logs/carepd/llm")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--batch", type=int, default=5, help="每次并行采样的条数")
    ap.add_argument("--think", action=argparse.BooleanOptionalAction, default=True, help="文本模型是否开启思考模式(模板支持时)")
    ap.add_argument("--print-prompt", action="store_true")
    args = ap.parse_args()

    if args.print_prompt:
        print(build_prompt(*CONDITIONS["prior_task"]))
        return

    import torch

    system = "You are an expert in clinical gait analysis and musculoskeletal biomechanics."
    is_vl = "-VL-" in args.model
    if is_vl:      # Qwen3-VL 系列:纯文本也要走它自己的 processor
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        tok = AutoProcessor.from_pretrained(args.model)
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device).eval()
    else:          # 通用开源文本 LLM
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device).eval()
    name = args.model.split("/")[-1] + ("" if is_vl or args.think else "-nothink")
    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    def encode(prompt: str):
        if is_vl:
            msgs = [{"role": "system", "content": [{"type": "text", "text": system}]}, {"role": "user", "content": [{"type": "text", "text": prompt}]}]
            return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(args.device)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
                                       enable_thinking=args.think).to(args.device)

    def decode(ids, n_in: int) -> list[str]:
        if is_vl:
            return tok.batch_decode(ids[:, n_in:], skip_special_tokens=True)
        return [strip_reasoning(t) for t in tok.batch_decode(ids[:, n_in:], skip_special_tokens=False)]

    for cond in args.conditions:
        prompt = build_prompt(*CONDITIONS[cond])
        inputs = encode(prompt)
        n_in = inputs["input_ids"].shape[1]
        raws: list[str] = []
        with torch.no_grad():
            raws += decode(model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False), n_in)
            done = 0
            while done < args.n:          # 采样分批,省 KV 缓存
                b = min(args.batch, args.n - done)
                torch.manual_seed(1000 + done)
                raws += decode(model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature,
                                              top_p=0.9, num_return_sequences=b), n_in)
                done += b
        recs = []
        for i, raw in enumerate(raws):
            items, cols, bad = parse(raw)
            recs.append({"sample": i, "greedy": i == 0, "raw": raw, "items": items, "columns": cols, "n_invalid": bad})
            print(f"[{cond} #{i}] 合法 {len(items)} / 非法 {bad} -> {cols}", flush=True)
        json.dump({"model": args.model, "condition": cond, "prompt": prompt, "samples": recs}, open(out_dir / f"{cond}.json", "w"), indent=1)
    print("[done]", out_dir)


def strip_reasoning(raw: str) -> str:
    """去掉思考过程,只留最终回答:Qwen3 的 </think>,gpt-oss 的 final 通道;再去掉特殊符号。"""
    for mark in ("</think>", "<|channel|>final<|message|>"):
        if mark in raw:
            raw = raw.rsplit(mark, 1)[1]
    return re.sub(r"<\|[^|>]*\|>", "", raw).strip()


if __name__ == "__main__":
    main()

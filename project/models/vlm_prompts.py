#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: vlm_prompts.py
Project: models
Author: Kaixu Chen
-----
Comment:
生成式 VLM(Qwen3-VL)用的指令预设。医生的关注区域在这里**以文本指令**进入模型,
而不是热图 —— 这是与 ClinicalCLIP 主线本质不同的先验注入方式。

指令必须放在视频 token **之前**:因果注意力下视频 token 只能看到前文,放在后面
就调制不到视频特征(只有回答位置能看到)。models/vlm_encoder.py 按此拼 chat 模板。

所有指令都是**患者无关**的:测试患者的医生标注绝不能进推理输入,否则是泄漏。
"""
from __future__ import annotations

# 与 dataloader.med_attn_map.REGIONS 顺序一致
REGION_PHRASES = {
    "foot": "the feet and ankles",
    "wrist": "the wrists and arm swing",
    "shoulder": "the shoulders",
    "lumbar_pelvis": "the lumbar spine and pelvis",
    "head": "the head and neck",
}

_SYSTEM = (
    "You are an orthopedic surgeon assessing a lateral-view walking video of a patient "
    "in a gait laboratory."
)

PROMPTS: dict[str, str] = {
    # 无临床引导:只说这是步态视频
    "generic": f"{_SYSTEM} Watch the gait carefully.",
    # 临床引导:按两位医生标注的频率列出关注部位(lumbar_pelvis 60%, head, shoulder, ...)
    "clinical": (
        f"{_SYSTEM} Adult spinal deformity changes the sagittal alignment of the trunk. "
        "Focus on the lumbar spine and pelvis (pelvic tilt, forward trunk lean), "
        "then the head and neck position and the shoulders, and note any compensatory "
        "arm swing or foot clearance."
    ),
    # 零样本诊断:要求单词回答,logit 打分用
    "diagnose": (
        f"{_SYSTEM} Adult spinal deformity (ASD) typically shows forward trunk lean, "
        "posterior pelvic tilt and knee flexion during walking. Does this patient show "
        "the gait pattern of adult spinal deformity? Answer with exactly one word: yes or no."
    ),
}

# 每个概念一条:回答位置对视频 token 的注意力 = 该概念的定位图
for _region, _phrase in REGION_PHRASES.items():
    PROMPTS[f"region_{_region}"] = (
        f"{_SYSTEM} Focus only on {_phrase} of the patient and describe how they move "
        "during the gait cycle."
    )


def get_prompt(name_or_text: str) -> str:
    """预设名或原文;原文直接返回。"""
    return PROMPTS.get(name_or_text, name_or_text)

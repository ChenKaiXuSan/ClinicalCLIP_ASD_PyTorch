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
    # 视觉提示版:帧上已按骨架画出腰椎骨盆的红框(dataloader visual_prompt=box_lumbar),
    # 指令改为指向画面标记而不是文字描述部位。generic_box 是同一帧、不提框的对照
    "generic_box": f"{_SYSTEM} Watch the gait carefully.",
    "clinical_box": (
        f"{_SYSTEM} A red box is drawn on every frame marking the patient's lumbar spine "
        "and pelvis. Focus on the motion inside the red box: pelvic tilt and forward trunk "
        "lean during walking, then the head, neck and shoulders above it."
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


# 临床属性(概念瓶颈):每个属性一个 yes/no 问题,logit(yes)-logit(no) 作为属性分数。
# 属性来自 ASD 步态的临床描述(矢状面失衡的代偿),患者无关。
_ATTR_SYSTEM = (
    f"{_SYSTEM} Look at the whole gait cycle and answer the question about this patient. "
    "Answer with exactly one word: yes or no."
)
ATTRIBUTES: dict[str, str] = {
    "trunk_forward_lean": "Is the patient's trunk leaning forward (a stooped posture) while walking?",
    "pelvic_retroversion": "Is the patient's pelvis tilted backward (posterior pelvic tilt) while walking?",
    "knee_flexion_stance": "Are the patient's knees bent (flexed) during the stance phase instead of straight?",
    "hip_extension_loss": "Does the patient's hip fail to extend behind the body in late stance (short push-off)?",
    "short_stride": "Is the patient's stride length noticeably short?",
    "slow_walking": "Is the patient walking slowly or cautiously?",
    "head_forward": "Is the patient's head positioned forward of the shoulders (forward head posture)?",
    "reduced_arm_swing": "Is the patient's arm swing reduced or held stiffly?",
}


def attribute_prompt(name: str) -> str:
    return f"{_ATTR_SYSTEM} Question: {ATTRIBUTES[name]}"


def get_prompt(name_or_text: str) -> str:
    """预设名或原文;原文直接返回。"""
    return PROMPTS.get(name_or_text, name_or_text)

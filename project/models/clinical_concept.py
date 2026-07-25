#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: clinical_concept.py
Project: models
Author: Kaixu Chen
-----
Comment:
临床概念引导的步态分类模型 (Clinical Concept Grounding)。

与 clip_align.py 的关键区别在于医生标注的用法:那里把渲染出的注意力图当作
第二个模态,在推理时也要喂进模型,并用批内 InfoNCE 做对齐;这里把医生标注
降格为训练期监督,推理时只输入视频,模型自己产出注意力。

这么改是因为数据本身的三个性质:
  1. 关注区域单独预测疾病的准确率恰好等于多数类基线(66.7%),说明它不是
     一个能独立分类的"模态",而是一种先验;
  2. 全库只有 6 种不同的区域组合,批内 InfoNCE 会把临床标注完全相同的样本
     当作负例互相推开,假负例极多;
  3. 两位医生只有 45.7% 的一致率,标注应当按软目标处理而不是取并集。

前向输出:
  logits          (B, num_classes)   疾病分类
  region_logits   (B, R)             模型预测"医生会关注哪些区域"
  attn            (B, R, T', H', W') 每个概念的时空注意力(已归一化为分布)
  region_feat     (B, R, d)          概念条件下池化出的区域特征
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_align import ResNet3DTokenEncoder

logger = logging.getLogger(__name__)

# 与 dataloader.med_attn_map.REGIONS 保持同序
REGIONS = ["foot", "wrist", "shoulder", "lumbar_pelvis", "head"]

# 供外部预计算文本向量时使用,顺序同上
CONCEPT_PROMPTS = [
    "the clinician focuses on the patient's feet while assessing gait",
    "the clinician focuses on the patient's wrists while assessing gait",
    "the clinician focuses on the patient's shoulders while assessing gait",
    "the clinician focuses on the patient's lumbar spine and pelvis while assessing gait",
    "the clinician focuses on the patient's head while assessing gait",
]


class ConceptBank(nn.Module):
    """临床概念嵌入 P ∈ (R, d)。

    默认可学习。若给出 text_embedding_path,则加载外部预计算的文本向量
    (顺序须与 REGIONS 一致)并冻结,只训练一层投影——这样才是真正的
    视觉-语言对齐,且概念可扩展;环境里没有文本编码器时退回可学习模式。
    """

    def __init__(
        self,
        num_regions: int,
        embed_dim: int,
        text_embedding_path: Optional[str] = None,
        freeze_text: bool = True,
    ) -> None:
        super().__init__()
        self.num_regions = num_regions

        self.text_embedding = None
        if text_embedding_path:
            try:
                weight = torch.load(text_embedding_path, map_location="cpu")
                weight = torch.as_tensor(weight, dtype=torch.float32)
                if weight.shape[0] != num_regions:
                    raise ValueError(
                        f"concept embedding 行数 {weight.shape[0]} 与区域数 {num_regions} 不符"
                    )
                self.text_embedding = nn.Parameter(weight, requires_grad=not freeze_text)
                self.proj = nn.Linear(weight.shape[1], embed_dim)
                logger.info("concept bank 使用文本向量 %s", text_embedding_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("加载 %s 失败(%s),退回可学习概念", text_embedding_path, exc)
                self.text_embedding = None

        if self.text_embedding is None:
            self.embedding = nn.Parameter(torch.randn(num_regions, embed_dim) * 0.02)

    def forward(self) -> torch.Tensor:
        if self.text_embedding is not None:
            return self.proj(self.text_embedding)
        return self.embedding


class ConceptCrossAttention(nn.Module):
    """以概念为 query 对视频 token 做交叉注意力。

    A_r = softmax_{t,h,w}(<P_r, Z> / τ) 是概念 r 的时空注意力分布,
    F_r = Σ A_r · Z 是该概念条件下的区域特征。A_r 同时是 grounding 损失的
    预测端,也是推理时可直接可视化的解释。
    """

    def __init__(self, embed_dim: int, temperature: float = 0.07) -> None:
        super().__init__()
        self.key = nn.Conv3d(embed_dim, embed_dim, kernel_size=1)
        self.value = nn.Conv3d(embed_dim, embed_dim, kernel_size=1)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.log_temperature = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(1.0 / temperature)))))

    def forward(
        self, tokens: torch.Tensor, concepts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, d, t, h, w = tokens.shape

        k = self.key(tokens).flatten(2)  # (B, d, N)
        v = self.value(tokens).flatten(2)  # (B, d, N)
        q = F.normalize(self.query(concepts), dim=-1)  # (R, d)
        k = F.normalize(k, dim=1)

        scale = self.log_temperature.exp().clamp(max=100.0)
        logits = torch.einsum("rd,bdn->brn", q, k) * scale  # (B, R, N)
        attn = logits.softmax(dim=-1)

        feat = torch.einsum("brn,bdn->brd", attn, v)  # (B, R, d)
        return attn.view(b, -1, t, h, w), feat


class ClinicalConceptNet(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()

        cfg = hparams.model
        self.num_classes = int(getattr(cfg, "model_class_num", 3))
        self.embed_dim = int(getattr(cfg, "concept_embed_dim", 256))
        self.num_regions = len(REGIONS)

        self.backbone = ResNet3DTokenEncoder(
            in_channels=3,
            hidden_dim=self.embed_dim,
            backbone_depth=int(getattr(cfg, "clip_backbone_depth", 50)),
            pretrained=bool(getattr(cfg, "clip_backbone_pretrained", True)),
        )

        self.concepts = ConceptBank(
            num_regions=self.num_regions,
            embed_dim=self.embed_dim,
            text_embedding_path=getattr(cfg, "concept_text_embedding", None),
            freeze_text=bool(getattr(cfg, "concept_freeze_text", True)),
        )

        self.cross_attention = ConceptCrossAttention(
            embed_dim=self.embed_dim,
            temperature=float(getattr(cfg, "concept_temperature", 0.07)),
        )

        # 预测"医生会关注哪些区域",既是监督信号也是推理期的可解释输出
        self.presence_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim // 2, 1),
        )

        # 分类头同时看概念加权特征和全局特征:先验只是引导,不该成为唯一通路
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim * 2),
            nn.Linear(self.embed_dim * 2, self.num_classes),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.backbone(video)  # (B, d, T', H', W')
        concepts = self.concepts()  # (R, d)

        attn, region_feat = self.cross_attention(tokens, concepts)

        region_logits = self.presence_head(region_feat).squeeze(-1)  # (B, R)

        # 按"该区域被关注的概率"加权聚合,未被关注的概念不该主导表征
        weight = region_logits.sigmoid().unsqueeze(-1)  # (B, R, 1)
        concept_feat = (region_feat * weight).sum(dim=1) / (
            weight.sum(dim=1) + 1e-6
        )  # (B, d)
        global_feat = tokens.mean(dim=(2, 3, 4))  # (B, d)

        logits = self.classifier(torch.cat([concept_feat, global_feat], dim=-1))

        return {
            "logits": logits,
            "region_logits": region_logits,
            "attn": attn,
            "region_feat": region_feat,
            "concepts": concepts,
            "concept_feat": concept_feat,
            "global_feat": global_feat,
        }


def presence_loss(region_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """区域存在性的软目标 BCE。target 取值 {0, 0.5, 1},0.5 表示两位医生有分歧。"""
    return F.binary_cross_entropy_with_logits(region_logits, target)


def grounding_loss(
    attn: torch.Tensor, region_map: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """把模型的概念注意力对齐到医生渲染的区域图。

    两侧都归一化成时空分布后取 KL,并按区域软标签加权——医生没提到的区域
    不施加空间约束,分歧区域(0.5)的权重也相应减半。

    Args:
        attn:       (B, R, T', H', W')  已 softmax 的模型注意力
        region_map: (B, R, T,  H,  W)   医生区域图,分辨率可与 attn 不同
        target:     (B, R)              区域软标签
    """
    b, r = attn.shape[:2]

    if region_map.shape[2:] != attn.shape[2:]:
        region_map = F.interpolate(
            region_map.reshape(b * r, 1, *region_map.shape[2:]),
            size=attn.shape[2:],
            mode="trilinear",
            align_corners=False,
        ).reshape(b, r, *attn.shape[2:])

    flat_map = region_map.reshape(b, r, -1)
    flat_attn = attn.reshape(b, r, -1)

    mass = flat_map.sum(dim=-1, keepdim=True)
    valid = (mass.squeeze(-1) > 1e-6) & (target > 0)
    if not valid.any():
        return attn.new_zeros(())

    doctor = flat_map / mass.clamp_min(1e-6)
    kl = (doctor * ((doctor + 1e-8).log() - (flat_attn + 1e-8).log())).sum(dim=-1)

    weight = target * valid
    return (kl * weight).sum() / weight.sum().clamp_min(1e-6)


def concept_contrastive_loss(
    region_feat: torch.Tensor, concepts: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """区域特征与概念库的对比损失(对比对象是 R 个概念,不是批内样本)。

    这是本架构里"CLIP 的部分"。换成概念库之后不再有假负例——原先批内
    InfoNCE 会把临床标注相同的样本推开,而全库只有 6 种区域组合。
    该项的作用是防止概念塌缩、保持各概念可区分。
    """
    feat = F.normalize(region_feat, dim=-1)  # (B, R, d)
    proto = F.normalize(concepts, dim=-1)  # (R, d)

    logits = torch.einsum("brd,kd->brk", feat, proto) / temperature
    b, r, _ = logits.shape
    target = torch.arange(r, device=logits.device).expand(b, r).reshape(-1)
    return F.cross_entropy(logits.reshape(b * r, r), target)

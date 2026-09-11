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

from .clip_align import ResNet3DTokenEncoder  # noqa: F401  (旧 checkpoint 反序列化仍会引用)
from .vlm_encoder import VLMTokenEncoder, build_token_encoder

logger = logging.getLogger(__name__)

# 直接复用数据侧的定义:两处各写一份时,顺序一旦错配就会静默地把 grounding
# 目标接到错误的概念上,而且不会报任何错
from dataloader.med_attn_map import REGIONS  # noqa: E402

# 供外部预计算文本向量时使用,顺序同上
CONCEPT_PROMPTS = [
    "the clinician focuses on the patient's feet while assessing gait",
    "the clinician focuses on the patient's wrists while assessing gait",
    "the clinician focuses on the patient's shoulders while assessing gait",
    "the clinician focuses on the patient's lumbar spine and pelvis while assessing gait",
    "the clinician focuses on the patient's head while assessing gait",
]


def _count_aux(data_cfg) -> int:
    """data.aux_targets_dir 下 json 文件数 = 辅助目标维度;未设置返回 0。"""
    import glob
    import os

    d = str(getattr(data_cfg, "aux_targets_dir", "") or "") if data_cfg is not None else ""
    return len(glob.glob(os.path.join(d, "*.json"))) if d else 0


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
            # 不做 try/except 静默降级:文件路径写错时会悄悄变回可学习概念,
            # 于是"文本概念 vs 可学习概念"的对照实验实际跑成了两组同样的配置
            weight = torch.load(text_embedding_path, map_location="cpu", weights_only=False)
            weight = torch.as_tensor(weight, dtype=torch.float32)
            if weight.shape[0] != num_regions:
                raise ValueError(
                    f"concept embedding 行数 {weight.shape[0]} 与区域数 {num_regions} 不符"
                )
            self.text_embedding = nn.Parameter(weight, requires_grad=not freeze_text)
            self.proj = nn.Linear(weight.shape[1], embed_dim)
            # 投影层也必须冻结。5 个线性无关的 512 维向量经一个可训练的
            # Linear(512, 256) 能映射到任意 5 个 256 维目标(每个输出维只是
            # 5 个方程、512 个未知数),不冻结的话文本概念与可学习概念是
            # 同一个假设空间,对照实验失去意义。
            if freeze_text:
                for p in self.proj.parameters():
                    p.requires_grad_(False)
            logger.info(
                "concept bank 使用文本向量 %s (冻结=%s)", text_embedding_path, freeze_text
            )
        else:
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

    def __init__(
        self, embed_dim: int, temperature: float = 0.07, spatial_softmax: bool = False
    ) -> None:
        super().__init__()
        # 联合 softmax 恰好可分解为 逐帧空间 softmax × 时间维 softmax(logsumexp)。
        # spatial_softmax=True 时**池化仍用联合分布(与默认完全一致)**,只把交给
        # grounding 与可视化的那一份的时间边缘换成均匀 —— 因为医生区域图逐帧质量
        # 的变异系数只有 0.043,时间上几乎均匀,那个约束等于强制"每帧同等重要",
        # 没有临床依据,却剥夺了模型聚焦判别性时刻(如触地)的能力。
        # 这样这条消融是外科式的:唯一变量就是 grounding 是否约束时间轴。
        self.spatial_softmax = bool(spatial_softmax)
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

        r = logits.shape[1]
        lg = logits.view(b, r, t, h * w)
        spatial = lg.softmax(dim=-1)                       # (B,R,T,HW) 逐帧空间分布
        temporal = lg.logsumexp(dim=-1).softmax(dim=-1)    # (B,R,T)    时间边缘
        joint = spatial * temporal.unsqueeze(-1)           # 与 logits.softmax(-1) 逐元素相等

        # 池化恒用联合分布,两条臂完全一致
        feat = torch.einsum("brn,bdn->brd", joint.reshape(b, r, -1), v)  # (B, R, d)

        # 交给 grounding / 可视化的那一份:可选把时间边缘换成均匀
        out = spatial / t if self.spatial_softmax else joint
        return out.view(b, r, t, h, w), feat


class ClinicalConceptNet(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()

        cfg = hparams.model
        self.num_classes = int(getattr(cfg, "model_class_num", 3))
        self.embed_dim = int(getattr(cfg, "concept_embed_dim", 256))
        self.num_regions = len(REGIONS)

        # token 编码器:resnet3d(slow_r50,既有行为)或 vlm(冻结的 VLM 视觉塔)。
        # 后面的概念交叉注意力只依赖 (B, d, T', h, w) 这个形状,与 backbone 无关。
        self.backbone = build_token_encoder(
            cfg, hidden_dim=self.embed_dim, data_cfg=getattr(hparams, "data", None)
        )
        self.accepts_cached_tokens = isinstance(self.backbone, VLMTokenEncoder)

        self.concepts = ConceptBank(
            num_regions=self.num_regions,
            embed_dim=self.embed_dim,
            text_embedding_path=getattr(cfg, "concept_text_embedding", None),
            freeze_text=bool(getattr(cfg, "concept_freeze_text", True)),
        )

        self.cross_attention = ConceptCrossAttention(
            embed_dim=self.embed_dim,
            temperature=float(getattr(cfg, "concept_temperature", 0.07)),
            spatial_softmax=bool(getattr(cfg, "concept_spatial_softmax", False)),
        )

        # 背景概念:第 R+1 个槽位,**不参与 grounding**,给区域外的判别证据一个正当去处。
        # grounding 把概念注意力拉向 49 格中的约 2 格(医生区域图 50% 的质量只占
        # ~4% 的格子),区域外的信息进不了概念通路 —— 这是 grounding 分类代价的
        # 一个候选机制。
        # 与 G0_global_raw 的区别很关键:G0 给的是 2048 维旁路,占分类器输入 89%,
        # 模型直接绕开概念通路、退化到 A4 水平(0/5 全负)。背景槽位仍走同一套交叉
        # 注意力、同样的 256 维聚合,宽度不变,只是多一个位置。
        self.use_background = bool(getattr(cfg, "concept_background", False))
        if self.use_background:
            self.background = nn.Parameter(torch.randn(1, self.embed_dim) * 0.02)

        # 预测"医生会关注哪些区域",既是监督信号也是推理期的可解释输出
        self.presence_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim // 2, 1),
        )

        # 分类头同时看概念加权特征和全局特征:先验只是引导,不该成为唯一通路。
        #
        # 但"不该成为唯一通路"此前只写在注释里没做到:global_feat 也取自
        # token_proj 之后的 embed_dim(256)维 token,和概念通路共用同一个
        # 2048 -> 256 的瓶颈,而这个投影正被 grounding 损失塑形 —— 全局通路
        # 无处可逃。证据:A4_no_prior(先验全关)患者级 macro 只有 0.566,比
        # 朴素 slow_r50 基线低 14.5 点,而先验又找回 7.9 点。也就是说亏的是
        # 架构不是先验。
        #
        # concept_global_from_raw=true 时全局通路改取**投影前**的 token_dim
        # (2048)维,与 B0_3dcnn 的分类输入同宽。默认 false 保持既有行为,
        # 两条臂只差这一个变量。
        self.global_from_raw = bool(getattr(cfg, "concept_global_from_raw", False))
        global_dim = self.backbone.token_dim if self.global_from_raw else self.embed_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim + global_dim),
            nn.Linear(self.embed_dim + global_dim, self.num_classes),
        )

        # 角色二:辅助回归头,预测 VLM 给出的临床属性分数;与分类头共享同一份特征。
        # 属性数由 data.aux_targets_dir 里的 json 数决定
        self.aux_dim = _count_aux(getattr(hparams, "data", None))
        self.aux_head = (
            nn.Sequential(nn.LayerNorm(self.embed_dim + global_dim),
                          nn.Linear(self.embed_dim + global_dim, self.aux_dim))
            if self.aux_dim > 0 else None
        )

    def forward(
        self,
        video: Optional[torch.Tensor] = None,
        raw_tokens: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """video (B,3,T,H,W);或 raw_tokens (B,token_dim,T',h,w) —— 离线缓存的 VLM
        特征,跳过视觉塔。缓存只对 vlm backbone 有意义。"""
        if raw_tokens is not None:
            if not self.accepts_cached_tokens:
                raise ValueError("缓存 token 只能配 model.token_backbone=vlm 使用")
            tokens, raw_tokens = self.backbone(
                None, return_raw=True, raw_tokens=raw_tokens
            )
            if not self.global_from_raw:
                raw_tokens = None
        elif self.global_from_raw:
            tokens, raw_tokens = self.backbone(video, return_raw=True)
        else:
            tokens, raw_tokens = self.backbone(video), None
        concepts = self.concepts()  # (R, d)
        # 背景槽位只参与交叉注意力与特征聚合,不进 concept_contrastive_loss ——
        # 那个损失的语义是"region_feat[r] 对上 concept[r],负例是另外 4 个临床概念",
        # 多一个背景概念会把负例数改掉,输出里也就只暴露前 R 个。
        attn_concepts = (
            torch.cat([concepts, self.background], dim=0)
            if self.use_background
            else concepts
        )

        attn, feat_all = self.cross_attention(tokens, attn_concepts)

        # 背景槽位没有医生标注,不进存在性头、也不进 grounding:
        # attn 与 region_logits 都只保留前 R 个临床概念,损失与指标口径完全不变
        region_feat = feat_all[:, : self.num_regions]
        attn = attn[:, : self.num_regions]

        region_logits = self.presence_head(region_feat).squeeze(-1)  # (B, R)

        # 按"该区域被关注的概率"加权聚合,未被关注的概念不该主导表征
        weight = region_logits.sigmoid().unsqueeze(-1)  # (B, R, 1)
        # clamp 而非 +1e-6:后者在所有 logit 都被推到很负时(某视频没有任何医生
        # 标注)会破坏加权平均的尺度不变性,把特征模长静默压到接近 0,使这类
        # 样本的分类器输入分布与正常样本完全不同
        if self.use_background:
            # 背景槽位恒参与(权重 1),其余按"该区域被关注的概率"加权
            bg_feat = feat_all[:, self.num_regions :]                      # (B,1,d)
            feats = torch.cat([region_feat, bg_feat], dim=1)               # (B,R+1,d)
            weight = torch.cat([weight, weight.new_ones(weight.shape[0], 1, 1)], dim=1)
        else:
            feats = region_feat
        concept_feat = (feats * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(
            1e-2
        )  # (B, d)
        # 全局通路:概念注意力管不到的那条路
        global_feat = (raw_tokens if raw_tokens is not None else tokens).mean(dim=(2, 3, 4))

        fused = torch.cat([concept_feat, global_feat], dim=-1)
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "aux": self.aux_head(fused) if self.aux_head is not None else None,
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

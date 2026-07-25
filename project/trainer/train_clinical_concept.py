#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: train_clinical_concept.py
Project: trainer
Author: Kaixu Chen
-----
Comment:
Clinical Concept Grounding 的 Lightning 模块。

推理只吃视频,医生标注仅参与训练期的两项监督(区域存在性 + 空间 grounding)。
因此"可解释性"是模型的预测量而非输入,可以直接与留出的医生标注对分:
  test/region_f1      预测关注区域与医生标注的一致度
  test/attn_iou       模型注意力与医生区域图的软 IoU
  test/attn_acc_gap   注意力对齐得好的样本是否分类也更准
"""

import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from models.clinical_concept import (
    REGIONS,
    ClinicalConceptNet,
    concept_contrastive_loss,
    grounding_loss,
    presence_loss,
)
from utils.helper import save_helper

logger = logging.getLogger(__name__)


def _average_precision(score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """免阈值的平均精度,避免 region_f1 的结论被阈值选择左右。"""
    score = score.flatten()
    target = target.flatten().float()
    if target.sum() == 0:
        return score.new_zeros(())
    order = torch.argsort(score, descending=True)
    hit = target[order]
    tp = torch.cumsum(hit, dim=0)
    precision = tp / torch.arange(1, len(hit) + 1, device=hit.device, dtype=tp.dtype)
    return (precision * hit).sum() / target.sum()


class ClinicalConceptModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()

        loss_cfg = getattr(hparams, "loss", {})
        self.lr = float(getattr(loss_cfg, "lr", 0.01))
        self.weight_decay = float(getattr(loss_cfg, "weight_decay", 0.001))
        self.w_presence = float(getattr(loss_cfg, "presence_weight", 1.0))
        self.w_ground = float(getattr(loss_cfg, "grounding_weight", 1.0))
        self.w_concept = float(getattr(loss_cfg, "concept_weight", 0.1))

        self.num_classes = int(getattr(hparams.model, "model_class_num", 3))
        # 消融用:把医生区域换成随机区域,检验增益是否真来自临床知识
        self.shuffle_region = bool(getattr(hparams.model, "shuffle_region", False))

        self.model = ClinicalConceptNet(hparams)

        # torchmetrics 默认 average="macro":这里的 video_acc 是**平衡准确率**
        # (各类召回的均值),多数类预测器只得 1/num_classes 而非类别占比。
        # 论文里不要写成 accuracy。逐类与 micro 口径由
        # analysis/compare_concept_runs.py 从保存的预测中补算。
        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)

        self.save_root = hparams.log_path

    def forward(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(video)

    def _maybe_shuffle(self, region_map, region_target):
        """消融:沿区域维度逐样本置换,等价于"同一副骨架、换一个区域"。

        不能在 batch 维度打乱——batch_size=1 时一个 batch 全是同一条视频的
        gait 段,区域标签本来就相同,打乱不起作用。

        若置换后分类精度不掉,说明增益来自姿态渲染而非医生选的区域,论文的
        核心主张就不成立。
        """
        if not self.shuffle_region or region_target is None:
            return region_map, region_target

        b, r = region_target.shape
        perm = torch.argsort(
            torch.rand(b, r, device=region_target.device), dim=1
        )  # (B, R) 每个样本一个独立置换
        target = region_target.gather(1, perm)
        index = perm.view(b, r, *([1] * (region_map.dim() - 2))).expand_as(region_map)
        return region_map.gather(1, index), target

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        video = batch["video"].detach()
        label = batch["label"].detach().long()
        region_map = batch.get("region_map")
        region_target = batch.get("region_target")
        if region_map is not None:
            region_map = region_map.detach()
            region_target = region_target.detach().float()
        region_map, region_target = self._maybe_shuffle(region_map, region_target)

        outputs = self.model(video)
        logits = outputs["logits"]

        cls_loss = F.cross_entropy(logits, label)
        zero = logits.new_zeros(())

        if region_target is not None:
            pres_loss = presence_loss(outputs["region_logits"], region_target)
            ground = grounding_loss(outputs["attn"], region_map, region_target)
        else:
            pres_loss = ground = zero

        concept = concept_contrastive_loss(outputs["region_feat"], outputs["concepts"])

        loss = (
            cls_loss
            + self.w_presence * pres_loss
            + self.w_ground * ground
            + self.w_concept * concept
        )

        probs = torch.softmax(logits, dim=1)
        on_step = stage == "train"
        bs = label.size(0)
        for name, value in [
            ("loss", loss),
            ("loss_cls", cls_loss),
            ("loss_presence", pres_loss),
            ("loss_grounding", ground),
            ("loss_concept", concept),
        ]:
            self.log(
                f"{stage}/{name}", value, on_epoch=True, on_step=on_step,
                batch_size=bs, prog_bar=name in ("loss", "loss_cls"),
            )
        self.log_dict(
            {
                f"{stage}/video_acc": self._accuracy(probs, label),
                f"{stage}/video_f1_score": self._f1_score(probs, label),
            },
            on_epoch=True, on_step=on_step, batch_size=bs,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    ##############
    # test step
    ##############

    def on_test_start(self) -> None:
        self.test_pred_list: list[torch.Tensor] = []
        self.test_label_list: list[torch.Tensor] = []
        self.test_region_pred: list[torch.Tensor] = []
        self.test_region_true: list[torch.Tensor] = []
        # (对齐得分, 均匀基线, 权重) —— 按权重加权平均,避免长短视频等权
        self.test_attn_align: list[tuple] = []

    @staticmethod
    def _attn_alignment(attn: torch.Tensor, region_map: torch.Tensor, target: torch.Tensor):
        """注意力对齐度:模型注意力分布下,医生图归一化强度的期望值。

            score_r = Σ_n A_r(n) · M̂_r(n),  M̂ = M / max(M)

        为什么不用软 IoU:A 是 softmax 分布,峰值随集中程度上升,峰值归一化后
        "越集中→有效支撑越小",导致指标非单调——实测均匀注意力得 0.071,而
        定位正确但更锐的注意力只得 0.036,消融组反而会赢。本式对 A 是线性的:
        把质量放在医生图强的位置就得高分,与集中程度无关,定位正确的锐注意力
        趋近 1.0,定位错误趋近 0。

        同时返回均匀注意力的得分作为参照下界(等于 M̂ 的均值),否则单看一个
        绝对数无法判断好坏。
        """
        b, r = attn.shape[:2]
        if region_map.shape[2:] != attn.shape[2:]:
            region_map = F.interpolate(
                region_map.reshape(b * r, 1, *region_map.shape[2:]),
                size=attn.shape[2:], mode="trilinear", align_corners=False,
            ).reshape(b, r, *attn.shape[2:])

        a = attn.reshape(b, r, -1)
        m = region_map.reshape(b, r, -1)
        m = m / m.amax(dim=-1, keepdim=True).clamp_min(1e-6)

        score = (a * m).sum(-1)
        uniform = m.mean(-1)  # 均匀注意力下的期望,即随机基线

        weight = target * (m.sum(-1) > 1e-6)
        if weight.sum() <= 0:
            return None
        denom = weight.sum()
        return (
            (score * weight).sum() / denom,
            (uniform * weight).sum() / denom,
            denom,
        )

    def test_step(self, batch, batch_idx):
        video = batch["video"].detach()
        label = batch["label"].detach().long()
        region_map = batch.get("region_map")
        region_target = batch.get("region_target")

        outputs = self.model(video)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1)

        self.log("test/loss", F.cross_entropy(logits, label),
                 on_epoch=True, on_step=False, batch_size=label.size(0))
        self.log_dict(
            {
                "test/video_acc": self._accuracy(probs, label),
                "test/video_f1_score": self._f1_score(probs, label),
            },
            on_epoch=True, on_step=False, batch_size=label.size(0),
        )

        self.test_pred_list.append(probs.detach().cpu())
        self.test_label_list.append(label.detach().cpu())

        if region_target is not None:
            region_target = region_target.detach().float()
            self.test_region_pred.append(
                outputs["region_logits"].sigmoid().detach().cpu()
            )
            self.test_region_true.append(region_target.cpu())
            align = self._attn_alignment(
                outputs["attn"].detach(), region_map.detach(), region_target
            )
            if align is not None:
                self.test_attn_align.append(tuple(x.detach().cpu() for x in align))

    def _fold_name(self) -> str:
        """从 logger 的 root_dir 取折号;fast_dev_run 下 logger 被禁用,root_dir 为 None。"""
        root_dir = getattr(self.logger, "root_dir", None) if self.logger else None
        return root_dir.split("/")[-1] if root_dir else "fold"

    def on_test_epoch_end(self) -> None:
        if self.test_region_pred:
            score = torch.cat(self.test_region_pred)
            target = torch.cat(self.test_region_true)

            # 软目标下 BCE 的最优解就是 sigmoid==target,"仅一位医生提到"的区域
            # 最优输出恰好落在 0.5。若用 pred>0.5 判正、true>=0.5 判真,完美模型
            # 会把这些区域全判成假阴性 —— 阈值必须与目标口径对齐。
            # "至少一位医生提到" 对应 target>0,判决阈值取 0 与 0.5 的中点。
            for tag, thr_true, thr_pred in [("any", 0.0, 0.25), ("both", 0.75, 0.75)]:
                true = target > thr_true
                pred = score > thr_pred
                tp = (pred & true).sum().float()
                precision = tp / pred.sum().clamp_min(1)
                recall = tp / true.sum().clamp_min(1)
                f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
                self.log(f"test/region_f1_{tag}", f1, on_epoch=True)
                self.log(f"test/region_precision_{tag}", precision, on_epoch=True)
                self.log(f"test/region_recall_{tag}", recall, on_epoch=True)

            # 免阈值口径:按分数排序的平均精度,不受上面阈值选择影响
            self.log("test/region_ap", _average_precision(score, target > 0), on_epoch=True)

        if self.test_attn_align:
            w = torch.stack([x[2] for x in self.test_attn_align])
            s = torch.stack([x[0] for x in self.test_attn_align])
            u = torch.stack([x[1] for x in self.test_attn_align])
            total = w.sum().clamp_min(1e-6)
            score = (s * w).sum() / total
            uniform = (u * w).sum() / total
            self.log("test/attn_align", score, on_epoch=True)
            # 均匀注意力的得分,作为"零信息"参照;两者之差才是真正学到的对齐
            self.log("test/attn_align_uniform", uniform, on_epoch=True)
            self.log("test/attn_align_gain", score - uniform, on_epoch=True)

        fold_name = self._fold_name()
        save_helper(
            all_pred=self.test_pred_list,
            all_label=self.test_label_list,
            fold=fold_name,
            save_path=self.save_root,
            num_class=self.num_classes,
        )

        if self.test_region_pred and self.save_root:
            out_path = Path(self.save_root) / "region"
            out_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "region_pred": torch.cat(self.test_region_pred),
                    "region_true": torch.cat(self.test_region_true),
                    "regions": REGIONS,
                },
                out_path / f"{fold_name}_region.pt",
            )
            logger.info("saved region predictions to %s", out_path)

    def configure_optimizers(self):
        # log_temperature 和概念库必须免除 weight decay:
        # - log_temperature 初值 log(1/0.07)=2.66,L2 会以每步约 lr 的确定性拉力
        #   把它推向 0,几百步后 scale→1,注意力几乎均匀,grounding KL 卡在下界
        # - 概念库在使用前都会被 F.normalize,模长方向不受任何损失约束,只有
        #   weight decay 在推它,收缩后五个概念的 query 会互相靠拢
        no_decay, decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "log_temperature" in name or "concepts." in name:
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = torch.optim.Adam(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train/loss"},
        }

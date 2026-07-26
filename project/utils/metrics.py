#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: metrics.py
Project: utils
Author: Kaixu Chen
-----
Comment:
所有 trainer 共用的分类指标,统一口径。

原先每个 trainer 各自写 `acc = self._accuracy(probs, label)` 再
`self.log("val/video_acc", acc, on_epoch=True)`,lightning 拿到的是一个普通标量,
epoch 末做的是**逐 batch 求平均**。而本项目 `batch_size=1` 时一个 batch 就是一条
视频的全部 gait 段、标签完全相同 —— 三类里只有一类有样本,这种 batch 上的 macro
没有意义,把它们平均起来更没有意义。fold0 实测:A0_shuffle_region 这样算出来是
0.685,而把全测试集预测汇总后重算是 macro 0.862 / micro 0.847,三个数互不相等。

更糟的是 ModelCheckpoint 用 `monitor="val/video_acc"` 选模型。B1_2dcnn 的这个数
全程卡在 0.5785(多数类占比),于是选中了 epoch 10 的多数类预测器;B2_cnn_lstm
选中了 val loss 6.32 的 epoch 52。基线"训崩"有一半是这么来的。

这里改成:每个阶段各持一份 torchmetrics 实例,只 update 不取 batch 值,把**指标
对象**交给 lightning。lightning 会在 epoch 末对整个阶段累积的状态调一次 compute,
得到的才是全集汇总的 macro,也才是 ModelCheckpoint 该看的数。

各阶段必须分开持有:torchmetrics 是有状态的,train/val/test 共用一份会把三个阶段
的样本混进同一份统计里。
"""

from __future__ import annotations

from torch import nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

_STAGES = ("train", "val", "test")


class ClassificationMetrics(nn.Module):
    """按阶段隔离的分类指标集合。

    用法(在 LightningModule 里):

        self.metrics = ClassificationMetrics(num_classes)
        ...
        def validation_step(self, batch, batch_idx):
            ...
            self.metrics.log(self, "val", probs, label, batch_size=b)

    注意 `probs` 与 `label` 必须是同一批样本的一一对应,label 为 long。
    """

    # 全部走 torchmetrics 默认的 average="macro",即平衡准确率。
    # 多数类预测器只得 1/C(三分类 0.333),不是类别占比。
    _FACTORIES = {
        "video_acc": MulticlassAccuracy,
        "video_precision": MulticlassPrecision,
        "video_recall": MulticlassRecall,
        "video_f1_score": MulticlassF1Score,
    }

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        # 键必须扁平成 "<阶段>_<指标>"。按阶段嵌套 ModuleDict 会用 "train" 当键,
        # 而 nn.Module 已经有 train() 方法,ModuleDict 会拒绝: KeyError: attribute
        # 'train' already exists。
        self.metrics = nn.ModuleDict(
            {
                f"{stage}_{name}": factory(num_classes=num_classes)
                for stage in _STAGES
                for name, factory in self._FACTORIES.items()
            }
        )

    def log(self, module, stage: str, probs, label, batch_size=None) -> None:
        """累积这一个 batch,并把指标对象登记给 lightning 做 epoch 级汇总。"""
        if stage not in _STAGES:
            raise ValueError(f"未知阶段 {stage!r},可选 {_STAGES}")

        label = label.long()
        for name in self._FACTORIES:
            metric = self.metrics[f"{stage}_{name}"]
            # 只 update,不取 batch 级返回值 —— 那个值既没用又要多算一次 compute
            metric.update(probs, label)
            module.log(
                f"{stage}/{name}",
                metric,
                on_epoch=True,
                on_step=False,
                batch_size=batch_size,
            )

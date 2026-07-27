#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/skeleton/project/models/make_model.py
Project: /workspace/skeleton/project/models
Created Date: Thursday October 19th 2023
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday February 7th 2026 9:11:51 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2023 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

26-11-2024	Kaixu Chen	remove x3d network.
'''

from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchvideo.models import resnet

class MakeVideoModule(nn.Module):
    '''
    make 3D CNN model from the PytorchVideo lib.

    '''

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_name = hparams.model.model
        self.model_class_num = hparams.model.model_class_num
        self.model_depth = hparams.model.model_depth
        self.transfer_learning = hparams.train.transfer_learning

    def initialize_walk_resnet(self, input_channel:int = 3) -> nn.Module:

        if self.transfer_learning:
            slow = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

            # 和 2D 那两处同一个坑:原来无条件把 blocks[0].conv 换成随机初始化的
            # Conv3d,而调用方传的就是默认 3 通道 —— Kinetics 预训练的 3D stem 被
            # 整个扔掉。B0_3dcnn 是全文最重要的基线,不能带着这个跑。
            _patch_stem_conv3d(slow, input_channel, pretrained=True)
            # change the knetics-400 output 400 to model class num
            slow.blocks[-1].proj = nn.Linear(2048, self.model_class_num)

        else:
            slow = resnet.create_resnet(
                input_channel=input_channel,
                model_depth=self.model_depth,
                model_num_class=self.model_class_num,
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
            )

        return slow

    def __call__(self, *args: Any, **kwds: Any) -> Any:

        if self.model_name == "resnet":
            return self.initialize_walk_resnet()
        else:
            raise KeyError(f"the model name {self.model_name} is not in the model zoo")


class MakeImageModule(nn.Module):
    '''
    the module zoo from the torchvision lib, to make the different 2D model.

    '''

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_name = hparams.model.model
        self.model_class_num = hparams.model.model_class_num
        self.transfer_learning = hparams.train.transfer_learning

    def make_resnet(self, input_channel: int = 3) -> nn.Module:
        model = torch.hub.load(
            'pytorch/vision:v0.10.0', 'resnet50', pretrained=self.transfer_learning
        )
        _patch_resnet_stem(model, input_channel, self.transfer_learning)
        model.fc = nn.Linear(2048, self.model_class_num)

        return model

    def __call__(self, *args: Any, **kwds: Any) -> Any:

        if self.model_name == "resnet":
            return self.make_resnet()
        else:
            raise KeyError(f"the model name {self.model_name} is not in the model zoo")

def _patch_stem_conv3d(model: nn.Module, in_channels: int, pretrained: bool) -> None:
    """slow_r50 的 3D 版本,逻辑同 _patch_resnet_stem。"""
    old_conv = model.blocks[0].conv
    if old_conv.in_channels == in_channels:
        return

    new_conv = nn.Conv3d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(avg_weight.repeat(1, in_channels, 1, 1, 1))
    model.blocks[0].conv = new_conv


def _patch_resnet_stem(model: nn.Module, in_channels: int, pretrained: bool) -> None:
    """只在通道数确实不同时才换掉 stem,换的时候也从预训练权重初始化。

    原来的写法是无条件 `model.conv1 = nn.Conv2d(in_channels, 64, ...)`,而调用方传的
    就是默认的 3 通道 —— 等于把 ImageNet 预训练的第一层卷积**扔掉换成随机初始化**,
    随机特征喂给后面预训练好的 block。B1_2dcnn / B2_cnn_lstm 的验证集准确率整整
    100 个 epoch 钉在 0.333(三分类的随机水平)、训练集却到 1.0,就是这么来的:
    模型靠随机 stem 记住了训练集,学不到任何可迁移的东西。

    clip_align._patch_stem_conv 一直是对的写法,这里对齐它。
    """
    old_conv = model.conv1
    if old_conv.in_channels == in_channels:
        return

    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(avg_weight.repeat(1, in_channels, 1, 1))
    model.conv1 = new_conv


class MakeOriginalTwoStream(nn.Module):
    '''
    from torchvision make resnet 50 network.
    input is single figure.
    '''

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_class_num = hparams.model.model_class_num
        self.transfer_learning = hparams.train.transfer_learning

    def make_resnet(self, input_channel:int = 3):

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

        # from pytorchvision, use resnet 50.
        # weights = ResNet50_Weights.DEFAULT
        # model = resnet50(weights=weights)

        # for the folw model and rgb model 
        model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # change the output 400 to model class num
        model.fc = nn.Linear(2048, self.model_class_num)

        return model
    
class CNNLSTM(nn.Module):
    '''
    the cnn lstm network, use the resnet 50 as the cnn part.
    '''

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_class_num = hparams.model.model_class_num
        self.transfer_learning = hparams.train.transfer_learning

        self.cnn = self.make_cnn()
        # LSTM 
        self.lstm = nn.LSTM(input_size=300, hidden_size=512, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, self.model_class_num)

    def make_cnn(self, input_channel: int = 3):

        model = torch.hub.load(
            'pytorch/vision:v0.10.0', 'resnet50', pretrained=self.transfer_learning
        )
        # 和 MakeImageModule 同一个坑:无条件换掉 conv1 会丢掉 ImageNet 预训练的 stem
        _patch_resnet_stem(model, input_channel, self.transfer_learning)
        # change the output 400 to the lstm input size
        model.fc = nn.Linear(2048, 300)

        return model

    def forward(self, x):

        b, c, t, h, w = x.size()

        # (b, c, t, h, w) -> (b*t, c, h, w),一次前向搞定。
        # 原来是 `for i in range(b)` 逐条视频串行跑 resnet50,而这里的 b 是一条视频的
        # gait 段数(最多 28),白白慢了一个数量级 —— B2_cnn_lstm 是全矩阵最慢的任务。
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        feat = self.cnn(frames).reshape(b, t, -1)

        out, _ = self.lstm(feat)

        # 只取最后一个时间步:LSTM 存在的意义就是把整段聚合完再判类别。
        # 原来对**每个**时间步都出一个预测(再由 trainer 把标签 repeat_interleave 成
        # b*t 个),等于逼着模型在只看到第 1 帧时就定下类别,既不是标准的 CNN-LSTM
        # 基线,也给指标掺进了大量必然错的预测。
        return self.fc(F.relu(out[:, -1]))
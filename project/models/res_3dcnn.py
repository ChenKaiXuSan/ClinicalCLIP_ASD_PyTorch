#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/ClinicalCLIP_ASD_PyTorch/project/models/res_3dcnn.py
Project: /workspace/code/ClinicalCLIP_ASD_PyTorch/project/models
Created Date: Saturday February 7th 2026
Author: Kaixu Chen
-----
Comment:
Implementation of 3D ResNet for video-based gait analysis.
Supports ResNet-18, 34, 50, 101, 152 architectures.

Based on:
"Deep Residual Learning for Image Recognition" (He et al., 2016)
Extended to 3D for spatiotemporal feature learning.

Have a good code time :)
-----
Last Modified: Saturday February 7th 2026 10:28:30 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, List, Optional, Callable


def conv3x3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    padding: int = 1,
) -> nn.Conv3d:
    """3x3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False,
    )


def conv1x1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1x1 convolution."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock3D(nn.Module):
    """
    Basic 3D ResNet block for ResNet-18/34.
    
    Structure:
        conv3x3x3 -> BN -> ReLU -> conv3x3x3 -> BN -> [+residual] -> ReLU
    """
    expansion: int = 1
    
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
            
        # Both conv1 and downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck3D(nn.Module):
    """
    Bottleneck 3D ResNet block for ResNet-50/101/152.
    
    Structure:
        conv1x1x1 -> BN -> ReLU -> conv3x3x3 -> BN -> ReLU -> 
        conv1x1x1 -> BN -> [+residual] -> ReLU
    """
    expansion: int = 4
    
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
            
        # 1x1x1 conv to reduce dimensions
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        
        # 3x3x3 conv for spatial-temporal feature extraction
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        
        # 1x1x1 conv to restore dimensions
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


class ResNet3D(nn.Module):
    """
    3D ResNet backbone for video classification.
    
    Supports multiple architectures: ResNet-18, 34, 50, 101, 152
    """
    
    def __init__(
        self,
        block: Type[BasicBlock3D | Bottleneck3D],
        layers: List[int],
        num_classes: int = 2,
        in_channels: int = 3,
        zero_init_residual: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.5,
    ):
        """
        Args:
            block: Block type (BasicBlock3D or Bottleneck3D)
            layers: Number of blocks in each stage [stage1, stage2, stage3, stage4]
            num_classes: Number of output classes
            in_channels: Number of input channels (3 for RGB, 2 for optical flow)
            zero_init_residual: Zero-initialize residual BN layers
            norm_layer: Normalization layer
            dropout: Dropout rate before final FC layer
        """
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        
        self.inplanes = 64
        self.dropout_rate = dropout
        
        # Stem: Initial convolution layers
        self.conv1 = nn.Conv3d(
            in_channels,
            self.inplanes,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        
        # Residual stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Weight initialization
        self._initialize_weights(zero_init_residual)
        
    def _make_layer(
        self,
        block: Type[BasicBlock3D | Bottleneck3D],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Build a residual stage."""
        norm_layer = self._norm_layer
        downsample = None
        
        # Downsample if stride != 1 or channel mismatch
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            
        layers = []
        # First block (may downsample)
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer,
                )
            )
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, zero_init_residual: bool = False):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu',
                )
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock3D):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            logits: (B, num_classes)
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification layer.
        
        Args:
            x: (B, C, T, H, W)
        Returns:
            features: (B, 512*expansion) for ResNet bottleneck
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


def _resnet3d(
    arch: str,
    block: Type[BasicBlock3D | Bottleneck3D],
    layers: List[int],
    pretrained: bool = False,
    progress: bool = True,
    **kwargs
) -> ResNet3D:
    """Generic 3D ResNet constructor."""
    model = ResNet3D(block, layers, **kwargs)
    
    # TODO: Add pretrained weight loading if available
    if pretrained:
        raise NotImplementedError("Pretrained 3D ResNet weights not implemented yet")
    
    return model


def resnet18_3d(pretrained: bool = False, progress: bool = True, **kwargs) -> ResNet3D:
    """ResNet-18 3D model."""
    return _resnet3d(
        'resnet18_3d',
        BasicBlock3D,
        [2, 2, 2, 2],
        pretrained,
        progress,
        **kwargs
    )


def resnet34_3d(pretrained: bool = False, progress: bool = True, **kwargs) -> ResNet3D:
    """ResNet-34 3D model."""
    return _resnet3d(
        'resnet34_3d',
        BasicBlock3D,
        [3, 4, 6, 3],
        pretrained,
        progress,
        **kwargs
    )


def resnet50_3d(pretrained: bool = False, progress: bool = True, **kwargs) -> ResNet3D:
    """ResNet-50 3D model."""
    return _resnet3d(
        'resnet50_3d',
        Bottleneck3D,
        [3, 4, 6, 3],
        pretrained,
        progress,
        **kwargs
    )


def resnet101_3d(pretrained: bool = False, progress: bool = True, **kwargs) -> ResNet3D:
    """ResNet-101 3D model."""
    return _resnet3d(
        'resnet101_3d',
        Bottleneck3D,
        [3, 4, 23, 3],
        pretrained,
        progress,
        **kwargs
    )


def resnet152_3d(pretrained: bool = False, progress: bool = True, **kwargs) -> ResNet3D:
    """ResNet-152 3D model."""
    return _resnet3d(
        'resnet152_3d',
        Bottleneck3D,
        [3, 8, 36, 3],
        pretrained,
        progress,
        **kwargs
    )


class MakeRes3DCNNModule(nn.Module):
    """
    Factory module for creating 3D ResNet models.
    Compatible with the project's model creation interface.
    """
    
    def __init__(self, hparams) -> None:
        super().__init__()
        
        self.model_class_num = hparams.model.model_class_num
        self.model_depth = getattr(hparams.model, 'model_depth', 50)
        
        # Architecture mapping
        self.arch_dict = {
            18: resnet18_3d,
            34: resnet34_3d,
            50: resnet50_3d,
            101: resnet101_3d,
            152: resnet152_3d,
        }
        
    def initialize_resnet3d(
        self,
        depth: int = 50,
        input_channel: int = 3,
        dropout: float = 0.5,
    ) -> nn.Module:
        """
        Initialize 3D ResNet model.
        
        Args:
            depth: Model depth (18, 34, 50, 101, 152)
            input_channel: Number of input channels (3 for RGB, 2 for optical flow)
            dropout: Dropout rate before classification layer
        
        Returns:
            ResNet3D model
        """
        if depth not in self.arch_dict:
            raise ValueError(
                f"ResNet depth {depth} not supported. "
                f"Choose from {list(self.arch_dict.keys())}"
            )
        
        model_fn = self.arch_dict[depth]
        model = model_fn(
            pretrained=False,
            num_classes=self.model_class_num,
            in_channels=input_channel,
            dropout=dropout,
        )
        
        return model
    
    def __call__(
        self,
        depth: Optional[int] = None,
        input_channel: int = 3,
    ) -> nn.Module:
        """
        Create and return 3D ResNet model.
        
        Args:
            depth: Model depth (uses config default if None)
            input_channel: Number of input channels
        
        Returns:
            3D ResNet model
        """
        if depth is None:
            depth = self.model_depth
            
        return self.initialize_resnet3d(
            depth=depth,
            input_channel=input_channel,
        )
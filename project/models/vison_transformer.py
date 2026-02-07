#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/ClinicalCLIP_ASD_PyTorch/project/models/vison_transformer.py
Project: /workspace/code/ClinicalCLIP_ASD_PyTorch/project/models
Created Date: Saturday February 7th 2026
Author: Kaixu Chen
-----
Comment:
Vision Transformer for Video-based Gait Analysis.
Implements spatiotemporal patch embedding and transformer encoder.

Have a good code time :)
-----
Last Modified: Saturday February 7th 2026 10:23:06 pm
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
from typing import Tuple, Optional
import math


class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding for video data.
    Converts input video (B, C, T, H, W) into sequence of patch embeddings.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        
        # 3D convolution for spatiotemporal patch extraction
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            embeddings: (B, N, D) where N = num_temporal_patches * num_spatial_patches
            num_temporal_patches: number of temporal patches
        """
        B, C, T, H, W = x.shape
        
        # Project to embeddings: (B, D, T', H', W')
        x = self.proj(x)
        
        # Get dimensions after projection
        _, _, T_new, H_new, W_new = x.shape
        num_temporal_patches = T_new
        
        # Flatten spatial and temporal dimensions: (B, D, T'*H'*W')
        x = x.flatten(2)
        
        # Transpose to (B, T'*H'*W', D)
        x = x.transpose(1, 2)
        
        return x, num_temporal_patches


class Attention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention: (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """MLP with GELU activation."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        x = x + self.attn(self.norm1(x))
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VideoVisionTransformer(nn.Module):
    """
    Vision Transformer for Video Classification.
    
    Supports spatiotemporal modeling for gait analysis tasks.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_channels: int = 3,
        num_classes: int = 2,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_cls_token: bool = True,
    ):
        """
        Args:
            img_size: Input image size
            patch_size: Spatial patch size
            tubelet_size: Temporal patch size
            in_channels: Number of input channels (3 for RGB, 2 for optical flow)
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            qkv_bias: Enable bias for QKV projection
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            use_cls_token: Use class token for classification
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        
        # Class token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding (will be added dynamically based on input size)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        # Initialize cls token
        if self.use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize classifier
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
            
    def get_pos_embed(self, num_patches: int) -> torch.Tensor:
        """Generate sinusoidal positional embeddings."""
        if self.use_cls_token:
            num_patches += 1
            
        pos_embed = torch.zeros(1, num_patches, self.embed_dim)
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * 
            (-math.log(10000.0) / self.embed_dim)
        )
        
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)
        
        return pos_embed
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            logits: (B, num_classes)
        """
        B = x.shape[0]
        
        # Patch embedding: (B, N, D)
        x, num_temporal_patches = self.patch_embed(x)
        
        # Add class token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        pos_embed = self.get_pos_embed(x.shape[1] - (1 if self.use_cls_token else 0))
        pos_embed = pos_embed.to(x.device)
        x = x + pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Classification
        if self.use_cls_token:
            x = x[:, 0]  # Use class token
        else:
            x = x.mean(dim=1)  # Global average pooling
            
        x = self.head(x)
        
        return x


class MakeViTModule(nn.Module):
    """
    Factory module for creating Vision Transformer models.
    Compatible with the project's model creation interface.
    """
    
    def __init__(self, hparams) -> None:
        super().__init__()
        
        self.model_class_num = hparams.model.model_class_num
        self.img_size = hparams.data.img_size
        
        # ViT configuration
        self.vit_config = {
            "tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3},
            "small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
            "base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
            "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
        }
        
    def initialize_vit(
        self, 
        model_size: str = "base",
        input_channel: int = 3,
        patch_size: int = 16,
        tubelet_size: int = 2,
    ) -> nn.Module:
        """
        Initialize Vision Transformer model.
        
        Args:
            model_size: Size of the model ("tiny", "small", "base", "large")
            input_channel: Number of input channels (3 for RGB, 2 for optical flow)
            patch_size: Spatial patch size
            tubelet_size: Temporal patch size
        
        Returns:
            VideoVisionTransformer model
        """
        if model_size not in self.vit_config:
            raise ValueError(f"Model size {model_size} not supported. "
                           f"Choose from {list(self.vit_config.keys())}")
        
        config = self.vit_config[model_size]
        
        model = VideoVisionTransformer(
            img_size=self.img_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_channels=input_channel,
            num_classes=self.model_class_num,
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.1,
            attn_drop_rate=0.0,
            use_cls_token=True,
        )
        
        return model
    
    def __call__(self, model_size: str = "base", input_channel: int = 3) -> nn.Module:
        """
        Create and return ViT model.
        
        Args:
            model_size: Size of the model
            input_channel: Number of input channels
        
        Returns:
            Vision Transformer model
        """
        return self.initialize_vit(
            model_size=model_size,
            input_channel=input_channel,
        )

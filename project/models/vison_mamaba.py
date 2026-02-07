#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/ClinicalCLIP_ASD_PyTorch/project/models/vison_mamaba.py
Project: /workspace/code/ClinicalCLIP_ASD_PyTorch/project/models
Created Date: Saturday February 7th 2026
Author: Kaixu Chen
-----
Comment:
Vision Mamba for Video-based Gait Analysis.
Implements state space models (SSM) for efficient spatiotemporal modeling.

Mamba provides linear complexity O(L) compared to Transformer's O(L²),
making it ideal for long video sequences in clinical gait analysis.

Have a good code time :)
-----
Last Modified: Saturday February 7th 2026 10:25:46 pm
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
from einops import rearrange, repeat


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


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) - Core of Mamba.
    
    Implements the selective mechanism that allows the model to 
    filter information based on input content.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor
        self.d_conv = d_conv
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # Depthwise convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # SSM parameters projections (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + self.d_inner)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Trainable SSM parameters
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Activation
        self.act = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            output: (B, L, D)
        """
        B, L, D = x.shape
        
        # Input projection and split
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        
        # Convolution (local context)
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :L]  # Trim to original length
        x = rearrange(x, 'b d l -> b l d')
        
        # Activation
        x = self.act(x)
        
        # SSM computation
        y = self.selective_scan(x)
        
        # Gating mechanism
        y = y * self.act(res)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective scan operation (simplified version).
        
        Args:
            x: (B, L, d_inner)
        Returns:
            y: (B, L, d_inner)
        """
        B, L, D = x.shape
        
        # Generate input-dependent parameters
        x_proj = self.x_proj(x)  # (B, L, d_state + d_state + d_inner)
        delta, B_ssm, C_ssm = x_proj.split(
            [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        
        # Discretization
        delta = F.softplus(delta)  # Ensure positive
        
        # A matrix (stable diagonal matrix)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Discretized A and B
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        deltaB = delta.unsqueeze(-1) * B_ssm.unsqueeze(2)  # (B, L, d_inner, d_state)
        
        # State space recurrence (simplified parallel scan)
        # For efficiency, using a simplified version
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for i in range(L):
            h = deltaA[:, i] * h + deltaB[:, i] * x[:, i].unsqueeze(-1)
            y = torch.einsum('bdn,bn->bd', h, C_ssm[:, i])
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        
        # Skip connection (D parameter)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y


class MambaBlock(nn.Module):
    """
    Mamba block combining SSM with normalization and residual connection.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            output: (B, L, D)
        """
        # Pre-norm with residual
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        x = x + residual
        
        return x


class MLP(nn.Module):
    """Feed-forward network."""
    
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


class MambaLayer(nn.Module):
    """
    Complete Mamba layer with SSM and MLP.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mamba_block = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(d_model)
        self.mlp = MLP(
            in_features=d_model,
            hidden_features=int(d_model * mlp_ratio),
            drop=dropout,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mamba block
        x = self.mamba_block(x)
        
        # MLP with residual
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        x = x + residual
        
        return x


class VideoVisionMamba(nn.Module):
    """
    Vision Mamba for Video Classification.
    
    Uses state space models for efficient spatiotemporal modeling
    with linear complexity in sequence length.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_channels: int = 3,
        num_classes: int = 2,
        embed_dim: int = 768,
        depth: int = 24,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
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
            depth: Number of Mamba layers
            d_state: SSM state dimension
            d_conv: Convolution kernel size in SSM
            expand_factor: Expansion factor in SSM
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            drop_rate: Dropout rate
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
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Mamba layers
        self.layers = nn.ModuleList([
            MambaLayer(
                d_model=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                mlp_ratio=mlp_ratio,
                dropout=drop_rate,
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
        if self.use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
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
        
        # Apply Mamba layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Classification
        if self.use_cls_token:
            x = x[:, 0]  # Use class token
        else:
            x = x.mean(dim=1)  # Global average pooling
            
        x = self.head(x)
        
        return x


class MakeMambaModule(nn.Module):
    """
    Factory module for creating Vision Mamba models.
    Compatible with the project's model creation interface.
    """
    
    def __init__(self, hparams) -> None:
        super().__init__()
        
        self.model_class_num = hparams.model.model_class_num
        self.img_size = hparams.data.img_size
        
        # Mamba configuration
        self.mamba_config = {
            "tiny": {"embed_dim": 192, "depth": 24, "d_state": 8},
            "small": {"embed_dim": 384, "depth": 24, "d_state": 16},
            "base": {"embed_dim": 768, "depth": 24, "d_state": 16},
            "large": {"embed_dim": 1024, "depth": 48, "d_state": 16},
        }
        
    def initialize_mamba(
        self, 
        model_size: str = "base",
        input_channel: int = 3,
        patch_size: int = 16,
        tubelet_size: int = 2,
    ) -> nn.Module:
        """
        Initialize Vision Mamba model.
        
        Args:
            model_size: Size of the model ("tiny", "small", "base", "large")
            input_channel: Number of input channels (3 for RGB, 2 for optical flow)
            patch_size: Spatial patch size
            tubelet_size: Temporal patch size
        
        Returns:
            VideoVisionMamba model
        """
        if model_size not in self.mamba_config:
            raise ValueError(f"Model size {model_size} not supported. "
                           f"Choose from {list(self.mamba_config.keys())}")
        
        config = self.mamba_config[model_size]
        
        model = VideoVisionMamba(
            img_size=self.img_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_channels=input_channel,
            num_classes=self.model_class_num,
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            d_state=config["d_state"],
            d_conv=4,
            expand_factor=2,
            mlp_ratio=4.0,
            drop_rate=0.1,
            use_cls_token=True,
        )
        
        return model
    
    def __call__(self, model_size: str = "base", input_channel: int = 3) -> nn.Module:
        """
        Create and return Mamba model.
        
        Args:
            model_size: Size of the model
            input_channel: Number of input channels
        
        Returns:
            Vision Mamba model
        """
        return self.initialize_mamba(
            model_size=model_size,
            input_channel=input_channel,
        )
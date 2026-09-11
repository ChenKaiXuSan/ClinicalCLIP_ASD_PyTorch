#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: vlm_encoder.py
Project: models
Author: Kaixu Chen
-----
Comment:
用视觉-语言模型(VLM)的视觉塔替换 slow_r50,作为 concept 架构的 token 编码器。

接口与 clip_align.ResNet3DTokenEncoder 完全一致:
    forward(video (B,3,T,H,W), return_raw=False) -> tokens (B, hidden, T', h, w)
    return_raw=True 时额外返回投影前的 (B, token_dim, T', h, w)
后面的 ConceptCrossAttention / 存在性头 / grounding 损失都只依赖这个形状,
所以医生注意力的用法(存在性软标签 + 空间 grounding + 概念 prompt)一处不改。

为什么要换:
  1. M1 用 CLIP ViT-B/32 的文本向量配 slow_r50 的视觉 token,两者不在同一空间,
     概念对比损失几乎惰性(findings.md)。换成同一 VLM 的双塔后概念与 token 天然对齐。
  2. findings.md 确认瓶颈是训练方差(跨种子 0.125)而非架构;冻结视觉塔 + 少量
     可训练参数是直接攻击这一点的假设。
  3. token 分辨率从 7x7 提到 14x14(224)或 24x24(384),grounding 目标图 28x28
     不再被降采样 16 倍。

后端:
  siglip2       transformers 原生(google/siglip2-*),逐帧编码,T' = T。已测试。
  qwen3vl       生成式 VLM(Qwen/Qwen3-VL-*-Instruct)。视频 token 取语言模型隐状态,
                已被前置的文本指令调制 —— 医生关注区域以指令形式进入(vlm_prompts.py)。
  internvideo2  OpenGVLab InternVideo2 stage2 视觉编码器,原生视频。权重是 gated 且
                需要官方仓库代码(INTERNVIDEO2_REPO),见 docs/vlm_backbone.md。未在本机测试。

冻结策略:
  vlm_trainable_blocks = 0  全部冻结,前向在 no_grad 下跑(V0 / V1)
  vlm_trainable_blocks = N  只解冻最后 N 个 transformer block 和末端 LayerNorm(V3)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# 各后端的像素归一化。数据侧只做 /255 + resize,归一化放在编码器内部,
# 这样同一份 dataloader 可以喂 slow_r50(不归一化)也可以喂 VLM。
_NORMALIZE = {
    "siglip2": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "internvideo2": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def _normalize(frames: torch.Tensor, backend: str) -> torch.Tensor:
    mean, std = _NORMALIZE[backend]
    mean = frames.new_tensor(mean).view(1, 3, 1, 1)
    std = frames.new_tensor(std).view(1, 3, 1, 1)
    return (frames - mean) / std


# --------------------------------------------------------------------------- #
# SigLIP 2
# --------------------------------------------------------------------------- #
class _SigLIP2Vision(nn.Module):
    """逐帧编码,输出 post-LayerNorm 的 patch token(head 之前)。"""

    def __init__(self, model_name: str, img_size: Optional[int]) -> None:
        super().__init__()
        from transformers import AutoModel

        full = AutoModel.from_pretrained(model_name)
        self.vision = full.vision_model
        cfg = full.config.vision_config
        self.patch = int(cfg.patch_size)
        self.native_size = int(cfg.image_size)
        self.token_dim = int(cfg.hidden_size)
        self.img_size = int(img_size or self.native_size)
        if self.img_size % self.patch:
            raise ValueError(f"img_size {self.img_size} 必须是 patch {self.patch} 的整数倍")
        self.grid = self.img_size // self.patch
        # 文本塔留给 build_concept_embedding / 零样本脚本用,训练时不加载到 GPU
        del full

    @property
    def blocks(self) -> nn.ModuleList:
        return self.vision.encoder.layers

    @property
    def final_norm(self) -> nn.Module:
        return self.vision.post_layernorm

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames (N,3,H,W) in [0,1] -> (N, L, d), L = grid*grid。"""
        if frames.shape[-1] != self.img_size:
            frames = F.interpolate(
                frames, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False, antialias=True,
            )
        frames = _normalize(frames, "siglip2")
        out = self.vision(
            pixel_values=frames,
            interpolate_pos_encoding=self.img_size != self.native_size,
        )
        return out.last_hidden_state  # (N, L, d),已过 post_layernorm

    def tokens_to_grid(self, tokens: torch.Tensor, b: int, t: int) -> torch.Tensor:
        n, l, d = tokens.shape
        assert l == self.grid * self.grid, (l, self.grid)
        return (
            tokens.view(b, t, self.grid, self.grid, d)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )  # (B, d, T, h, w)


# --------------------------------------------------------------------------- #
# InternVideo2 (stage2 视觉编码器)
# --------------------------------------------------------------------------- #
class _InternVideo2Vision(nn.Module):
    """InternVideo2-Stage2 1B 视觉塔。

    依赖官方仓库 InternVideo/InternVideo2/multi_modality 的代码,通过环境变量
    INTERNVIDEO2_REPO 指定;权重 (*.pt) 由 model_name 指定本地路径。
    输出去掉 cls 之后的 patch token,tubelet=1 时 T' = T,patch 14 @224 -> 16x16。
    """

    def __init__(self, model_name: str, img_size: Optional[int]) -> None:
        super().__init__()
        repo = os.environ.get("INTERNVIDEO2_REPO", "")
        if not repo or not os.path.isdir(repo):
            raise RuntimeError(
                "internvideo2 后端需要官方代码:export INTERNVIDEO2_REPO=/path/to/"
                "InternVideo/InternVideo2/multi_modality(并安装 einops/timm)。"
                "准备步骤见 docs/vlm_backbone.md"
            )
        import sys

        if repo not in sys.path:
            sys.path.insert(0, repo)
        try:
            from models.backbones.internvideo2.internvideo2 import InternVideo2  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"导入 InternVideo2 失败(缺 einops/timm?): {exc}") from exc

        self.img_size = int(img_size or 224)
        self.patch = 14
        self.grid = self.img_size // self.patch
        # 官方 pretrain_internvideo2_1b_patch14_224(config) 的签名随版本变动,这里按
        # stage2 1B 的公开超参显式构建;关闭 flash-attn / fused 算子 / checkpoint,
        # 普通注意力即可(冻结推理不在乎速度)
        self.vision = InternVideo2(
            in_chans=3, patch_size=14, img_size=self.img_size, qkv_bias=False,
            drop_path_rate=0.0, embed_dim=1408, num_heads=16, mlp_ratio=48 / 11,
            init_values=0.1, qk_normalization=True, depth=40, use_flash_attn=False,
            use_fused_rmsnorm=False, use_fused_mlp=False, fused_mlp_heuristic=1,
            attn_pool_num_heads=16, clip_embed_dim=768, layerscale_no_force_fp32=False,
            num_frames=8, tubelet_size=1, sep_pos_embed=False, sep_image_video_pos_embed=True,
            use_checkpoint=False, checkpoint_num=0, clip_return_layer=6, clip_student_return_interval=1,
        )
        state = torch.load(model_name, map_location="cpu")
        state = state.get("module", state.get("model", state))
        missing, unexpected = self.vision.load_state_dict(state, strict=False)
        logger.info("InternVideo2 权重加载: missing=%d unexpected=%d", len(missing), len(unexpected))
        self.token_dim = 1408

    @property
    def blocks(self) -> nn.ModuleList:
        return self.vision.blocks

    @property
    def final_norm(self) -> nn.Module:
        return getattr(self.vision, "clip_projector", nn.Identity())

    def forward_video(self, video: torch.Tensor) -> torch.Tensor:
        """video (B,3,T,H,W) in [0,1] -> (B, d, T, h, w)。"""
        b, c, t, h, w = video.shape
        if h != self.img_size:
            video = F.interpolate(
                video.transpose(1, 2).reshape(b * t, c, h, w),
                size=(self.img_size, self.img_size), mode="bilinear",
                align_corners=False, antialias=True,
            ).reshape(b, t, c, self.img_size, self.img_size).transpose(1, 2)
        frames = _normalize(video.transpose(1, 2).reshape(b * t, c, *video.shape[-2:]), "internvideo2")
        video = frames.reshape(b, t, c, *frames.shape[-2:]).transpose(1, 2)
        x = self.vision(video, mask=None, use_image=False, x_vis_return_idx=-1, x_vis_only=True)
        x = x[:, 1:]  # 去 cls
        return (
            x.view(b, t, self.grid, self.grid, self.token_dim)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )


# --------------------------------------------------------------------------- #
# Qwen3-VL(生成式 VLM):指令调制的视频 token
# --------------------------------------------------------------------------- #
class _Qwen3VLVision(nn.Module):
    """Qwen3-VL 的语言模型隐状态作为视频 token。

    与 SigLIP 的本质区别:视频 token 经过语言模型后**已被前文指令调制**。医生的关注
    区域以文本指令(models/vlm_prompts.py)进入,而不是热图。指令放在视频之前 ——
    因果注意力下视频 token 只能看到前文。

    输出 (B, d_text, t', h', w'),t' = T/2(temporal patch 2),h' = w' = S/32(patch 16 x merge 2)。
    8 帧 224 -> (4, 7, 7);8 帧 448 -> (4, 14, 14)。
    另提供 encode_pooled(回答位置的隐状态,看得到视频与指令)与
    attention_maps(回答位置对视频 token 的注意力,可解释输出)。
    """

    def __init__(self, model_name: str, img_size: Optional[int], prompt: str,
                 layer: int = -1, attn_implementation: str = "sdpa",
                 dtype: str = "bfloat16") -> None:
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        from .vlm_prompts import get_prompt

        # 默认 bf16:8B 权重 16GB,fp32 翻倍;CPU 上 bf16 也能算(登录节点每用户只有 16GB 内存)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype=getattr(torch, dtype), attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
        )
        cfg = self.model.config
        vcfg = cfg.vision_config
        self.patch = int(vcfg.patch_size) * int(vcfg.spatial_merge_size)  # 每个 LLM token 的像素边长
        self.temporal = int(vcfg.temporal_patch_size)
        self.img_size = int(img_size or 224)
        if self.img_size % self.patch:
            raise ValueError(f"img_size {self.img_size} 必须是 {self.patch} 的整数倍")
        self.grid = self.img_size // self.patch
        self.token_dim = int(cfg.text_config.hidden_size)
        self.video_token_id = int(cfg.video_token_id)
        self.layer = int(layer)
        self.prompt = get_prompt(prompt)
        self._chat_cache: dict[str, str] = {}

    # ---- 可训练块(V3 用)----
    @property
    def blocks(self) -> nn.ModuleList:
        return self.model.model.language_model.layers

    @property
    def final_norm(self) -> nn.Module:
        return self.model.model.language_model.norm

    # ---- 输入构造 ----
    def _chat_text(self, prompt: str) -> str:
        if prompt not in self._chat_cache:
            messages = [{
                "role": "user",
                # 文本在前、视频在后:视频 token 才能看到指令
                "content": [{"type": "text", "text": prompt}, {"type": "video"}],
            }]
            self._chat_cache[prompt] = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        return self._chat_cache[prompt]

    def _inputs(self, video: torch.Tensor, prompt: str):
        """video (b,3,T,S,S) in [0,1] -> processor 输出(已放到模型所在设备)。"""
        from transformers.video_utils import VideoMetadata

        b, c, t, h, w = video.shape
        # processor 自己做 rescale(1/255)与 normalize,喂 uint8
        clips = [(video[i].transpose(0, 1) * 255).round().clamp(0, 255).to(torch.uint8).cpu() for i in range(b)]
        # 每段是 1 秒的 gait 周期,T 帧均匀铺满这一秒 -> fps = T,时间戳 0 ~ 1s
        meta = [VideoMetadata(total_num_frames=t, fps=float(t), duration=1.0,
                              frames_indices=list(range(t)), height=h, width=w) for _ in range(b)]
        px = t * self.img_size * self.img_size
        inputs = self.processor(
            text=[self._chat_text(prompt)] * b,
            videos=clips,
            video_metadata=meta,
            do_sample_frames=False,
            size={"shortest_edge": px, "longest_edge": px},
            padding=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        return inputs.to(device)

    @staticmethod
    def _last_positions(attention_mask: torch.Tensor) -> torch.Tensor:
        """每个样本最后一个非 pad 位置(兼容左右 padding)。"""
        idx = torch.arange(attention_mask.shape[1], device=attention_mask.device)
        return (attention_mask * idx).argmax(dim=1)

    def _video_grid(self, inputs, i: int) -> tuple[int, int, int]:
        t, h, w = inputs["video_grid_thw"][i].tolist()
        m = int(self.model.config.vision_config.spatial_merge_size)
        return int(t), int(h // m), int(w // m)

    # ---- 前向 ----
    def forward_video(self, video: torch.Tensor, prompt: Optional[str] = None,
                      return_pooled: bool = False):
        """(b,3,T,S,S) -> tokens (b, d, t', h', w'),可选回答位置的 pooled (b, d)。"""
        prompt = prompt or self.prompt
        inputs = self._inputs(video, prompt)
        out = self.model.model(**inputs, output_hidden_states=self.layer != -1)
        hidden = out.last_hidden_state if self.layer == -1 else out.hidden_states[self.layer]
        mask = inputs["input_ids"] == self.video_token_id
        grids = []
        for i in range(hidden.shape[0]):
            t, h, w = self._video_grid(inputs, i)
            tok = hidden[i][mask[i]]  # (t*h*w, d),按 t, h, w 顺序排列
            assert tok.shape[0] == t * h * w, (tok.shape, t, h, w)
            grids.append(tok.view(t, h, w, -1).permute(3, 0, 1, 2))
        tokens = torch.stack(grids, dim=0).float()  # (b, d, t', h', w')
        if not return_pooled:
            return tokens
        pos = self._last_positions(inputs["attention_mask"])
        pooled = hidden[torch.arange(hidden.shape[0]), pos].float()
        return tokens, pooled

    @torch.no_grad()
    def attention_maps(self, video: torch.Tensor, prompt: str, last_layers: int = 4) -> torch.Tensor:
        """回答位置对视频 token 的注意力,取最后 last_layers 层、所有头平均 -> (b, t', h', w'),和为 1。
        需要 attn_implementation="eager" 才有 attentions。"""
        inputs = self._inputs(video, prompt)
        out = self.model.model(**inputs, output_attentions=True)
        attns = out.attentions[-last_layers:]  # 每层 (b, heads, L, L)
        pos = self._last_positions(inputs["attention_mask"])
        mask = inputs["input_ids"] == self.video_token_id
        maps = []
        for i in range(video.shape[0]):
            rows = torch.stack([a[i, :, pos[i], :] for a in attns], dim=0).float()  # (layers, heads, L)
            row = rows.mean(dim=(0, 1))[mask[i]]
            t, h, w = self._video_grid(inputs, i)
            row = row / row.sum().clamp_min(1e-8)
            maps.append(row.view(t, h, w))
        return torch.stack(maps, dim=0)

    @torch.no_grad()
    def generate(self, video: torch.Tensor, prompt: str, max_new_tokens: int = 128) -> list[str]:
        """自由生成(角色三:解释生成)。video (b,3,T,S,S) -> b 段文本,贪心解码。"""
        inputs = self._inputs(video, prompt)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return [t.strip() for t in self.processor.batch_decode(new_tokens, skip_special_tokens=True)]

    @torch.no_grad()
    def answer_logits(self, video: torch.Tensor, prompt: str, candidates: list[str]) -> torch.Tensor:
        """回答首 token 在候选词上的 logit (b, len(candidates)),零样本诊断用。"""
        inputs = self._inputs(video, prompt)
        logits = self.model(**inputs).logits
        pos = self._last_positions(inputs["attention_mask"])
        step = logits[torch.arange(logits.shape[0]), pos].float()  # (b, vocab)
        tok = self.processor.tokenizer
        cols = []
        for word in candidates:
            ids = {tok.encode(v, add_special_tokens=False)[0] for v in (word, " " + word, word.capitalize())}
            cols.append(step[:, sorted(ids)].max(dim=1).values)
        return torch.stack(cols, dim=1)


# --------------------------------------------------------------------------- #
# 统一封装
# --------------------------------------------------------------------------- #
class VLMTokenEncoder(nn.Module):
    """VLM 视觉塔 -> (B, hidden, T', h, w),接口同 ResNet3DTokenEncoder。"""

    def __init__(
        self,
        backend: str,
        model_name: str,
        hidden_dim: int = 256,
        img_size: Optional[int] = None,
        trainable_blocks: int = 0,
        forward_chunk: int = 64,
        prompt: str = "generic",
        layer: int = -1,
        attn_implementation: str = "sdpa",
        cache_manifest: Optional[str] = None,
        dtype: str = "bfloat16",
    ) -> None:
        super().__init__()
        self.backend = backend
        if cache_manifest:
            # 训练只读离线缓存时不加载视觉塔(Qwen 8B 有 17GB,白占显存)。
            # token_dim / grid 来自抽特征时写的 manifest.json
            import json

            if trainable_blocks > 0:
                raise ValueError("解冻视觉塔(vlm_trainable_blocks>0)不能与 feature_cache_dir 同用")
            with open(cache_manifest) as f:
                mani = json.load(f)
            if mani.get("backend") != backend:
                raise ValueError(f"缓存是 {mani.get('backend')} 抽的,配置却是 {backend}")
            self.tower = None
            self.token_dim = int(mani["token_dim"])
            self.grid = int(mani["grid"])
            self.trainable_blocks = 0
            self.forward_chunk = int(forward_chunk)
            self.video_native = backend in ("internvideo2", "qwen3vl")
            self.token_proj = nn.Conv3d(self.token_dim, hidden_dim, kernel_size=1, bias=False)
            logger.info("VLM %s: 只读缓存模式(manifest=%s),未加载视觉塔", backend, cache_manifest)
            return
        if backend == "siglip2":
            self.tower = _SigLIP2Vision(model_name, img_size)
        elif backend == "internvideo2":
            self.tower = _InternVideo2Vision(model_name, img_size)
        elif backend == "qwen3vl":
            self.tower = _Qwen3VLVision(model_name, img_size, prompt, layer, attn_implementation, dtype)
        else:
            raise ValueError(f"未知的 VLM 后端 {backend},可选 siglip2 / internvideo2 / qwen3vl")
        # 整段视频一次前向的后端(vs. siglip2 逐帧)
        self.video_native = backend in ("internvideo2", "qwen3vl")

        self.token_dim = self.tower.token_dim
        self.grid = self.tower.grid
        self.trainable_blocks = int(trainable_blocks)
        self.forward_chunk = int(forward_chunk)
        self.token_proj = nn.Conv3d(self.token_dim, hidden_dim, kernel_size=1, bias=False)

        self._set_trainable()

    # ---- 冻结控制 ----
    def _set_trainable(self) -> None:
        for p in self.tower.parameters():
            p.requires_grad_(False)
        if self.trainable_blocks > 0:
            blocks = self.tower.blocks
            for blk in list(blocks)[-self.trainable_blocks:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            for p in self.tower.final_norm.parameters():
                p.requires_grad_(True)
        n_train = sum(p.numel() for p in self.tower.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.tower.parameters())
        logger.info(
            "VLM %s: 可训练 %.1fM / 共 %.1fM 参数 (trainable_blocks=%d)",
            self.backend, n_train / 1e6, n_total / 1e6, self.trainable_blocks,
        )

    @property
    def frozen(self) -> bool:
        return self.trainable_blocks <= 0

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen and self.tower is not None:
            # 冻结的塔永远 eval:dropout / stochastic depth 关掉,特征才是确定的
            self.tower.eval()
        return self

    def _require_tower(self) -> None:
        if self.tower is None:
            raise RuntimeError(
                "该编码器是只读缓存模式(feature_cache_dir),没有加载视觉塔;"
                "batch 里缺少 tokens,检查缓存目录是否覆盖了全部视频"
            )

    # ---- 前向 ----
    def encode_raw(self, video: torch.Tensor) -> torch.Tensor:
        """video (B,3,T,H,W) in [0,1] -> 投影前 token (B, token_dim, T', h, w)。

        冻结时在 no_grad 下分块跑,一条视频的全部 gait 段拼成 batch 后帧数可达
        28 段 x 8 帧 = 224 帧,so400m 一次吃不下。
        """
        self._require_tower()
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            if self.video_native:
                outs = [
                    self.tower.forward_video(chunk)
                    for chunk in video.split(max(1, self.forward_chunk // video.shape[2]), dim=0)
                ]
                return torch.cat(outs, dim=0)

            b, c, t, h, w = video.shape
            frames = video.transpose(1, 2).reshape(b * t, c, h, w)
            outs = [self.tower(chunk) for chunk in frames.split(self.forward_chunk, dim=0)]
            tokens = torch.cat(outs, dim=0)
            return self.tower.tokens_to_grid(tokens, b, t)

    @torch.no_grad()
    def encode_with_pooled(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """离线抽特征用:tokens (B,d,t',h',w') + 回答位置的 pooled (B,d)。
        只有 qwen3vl 有真正的回答位置;其它后端 pooled = token 均值。"""
        self._require_tower()
        if self.backend == "qwen3vl":
            toks, pools = [], []
            for chunk in video.split(max(1, self.forward_chunk // video.shape[2]), dim=0):
                t, p = self.tower.forward_video(chunk, return_pooled=True)
                toks.append(t)
                pools.append(p)
            return torch.cat(toks, 0), torch.cat(pools, 0)
        raw = self.encode_raw(video)
        return raw, raw.mean(dim=(2, 3, 4))

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        return_raw: bool = False,
        raw_tokens: Optional[torch.Tensor] = None,
    ):
        """x 是视频,或直接给 raw_tokens(离线缓存)跳过视觉塔。"""
        if raw_tokens is None:
            if x is None:
                raise ValueError("需要 video 或 raw_tokens 之一")
            raw = self.encode_raw(x)
        else:
            raw = raw_tokens
        raw = raw.to(self.token_proj.weight.dtype)
        tokens = self.token_proj(raw)
        return (tokens, raw) if return_raw else tokens


def cache_manifest_path(data_cfg) -> Optional[str]:
    """data.feature_cache_dir 非空时返回其 manifest.json 路径,否则 None。"""
    cache_dir = str(getattr(data_cfg, "feature_cache_dir", "") or "") if data_cfg is not None else ""
    if not cache_dir:
        return None
    path = os.path.join(cache_dir, "manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"特征缓存 {cache_dir} 缺 manifest.json,先跑 scripts/extract_vlm_features.py")
    return path


def build_token_encoder(cfg, hidden_dim: int, data_cfg=None) -> nn.Module:
    """按 model.token_backbone 构建 token 编码器:resnet3d(默认,既有行为)或 vlm。
    data_cfg 给了且 feature_cache_dir 非空时,vlm 走只读缓存模式,不加载视觉塔。"""
    kind = str(getattr(cfg, "token_backbone", "resnet3d"))
    if kind == "resnet3d":
        from .clip_align import ResNet3DTokenEncoder

        return ResNet3DTokenEncoder(
            in_channels=3,
            hidden_dim=hidden_dim,
            backbone_depth=int(getattr(cfg, "clip_backbone_depth", 50)),
            pretrained=bool(getattr(cfg, "clip_backbone_pretrained", True)),
        )
    if kind == "vlm":
        return VLMTokenEncoder(
            backend=str(getattr(cfg, "vlm_backend", "siglip2")),
            model_name=str(getattr(cfg, "vlm_name", "google/siglip2-so400m-patch16-384")),
            hidden_dim=hidden_dim,
            img_size=getattr(cfg, "vlm_img_size", None),
            trainable_blocks=int(getattr(cfg, "vlm_trainable_blocks", 0)),
            forward_chunk=int(getattr(cfg, "vlm_forward_chunk", 64)),
            prompt=str(getattr(cfg, "vlm_prompt", "generic")),
            layer=int(getattr(cfg, "vlm_layer", -1)),
            cache_manifest=cache_manifest_path(data_cfg),
        )
    raise ValueError(f"未知的 token_backbone {kind},可选 resnet3d / vlm")

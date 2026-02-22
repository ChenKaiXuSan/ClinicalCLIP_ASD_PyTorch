#!/bin/bash

# 实验目的：仅保留 CLIP 对齐损失，不使用 map-guided，引导验证“对齐监督本身”的贡献。

set -e

SELECTED_GPU="${1:-${GPU_NUM:-0}}"

cd /workspace/code/ClinicalCLIP_ASD_PyTorch

conda run -p /opt/conda/envs/clip python -m project.main \
  train.experiment=B1_clip_only \
  train.gpu_num="${SELECTED_GPU}" \
  model.map_guided=false \
  loss.clip_weight=1.0 \
  model.lambda_token=0.0

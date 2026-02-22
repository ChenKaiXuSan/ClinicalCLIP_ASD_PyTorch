#!/bin/bash

# 实验目的：关闭 CLIP 对齐损失，仅保留 map-guided 分支，评估空间时序引导本身的效果。

set -e

SELECTED_GPU="${1:-${GPU_NUM:-0}}"

cd /workspace/code/ClinicalCLIP_ASD_PyTorch

conda run -p /opt/conda/envs/clip python -m project.main \
  train.experiment=B2_map_only \
  train.gpu_num="${SELECTED_GPU}" \
  loss.clip_weight=0 \
  model.lambda_token=0.0 \
  model.map_guided=true \
  model.map_guided_type=spatiotemporal

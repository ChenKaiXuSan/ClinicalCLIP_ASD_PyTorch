#!/bin/bash

# 实验目的：在 B3 基础上加入 token-level 监督，评估细粒度语义约束的增益。

set -e

SELECTED_GPU="${1:-${GPU_NUM:-0}}"

cd /workspace/code/ClinicalCLIP_ASD_PyTorch

conda run -p /opt/conda/envs/clip python -m project.main \
  train.experiment=B4_full_token \
  train.gpu_num="${SELECTED_GPU}" \
  loss.clip_weight=1.0 \
  model.lambda_token=0.1 \
  model.map_guided=true \
  model.map_guided_type=spatiotemporal

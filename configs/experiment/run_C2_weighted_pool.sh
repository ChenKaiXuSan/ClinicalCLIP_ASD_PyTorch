#!/bin/bash

# 实验目的：使用 weighted pooling 的 map-guided，评估加权池化策略对性能的影响。

set -e

SELECTED_GPU="${1:-${GPU_NUM:-0}}"

cd /workspace/code/ClinicalCLIP_ASD_PyTorch

conda run -p /opt/conda/envs/clip python -m project.main \
  train.experiment=C2_weighted_pool \
  train.gpu_num="${SELECTED_GPU}" \
  loss.clip_weight=1.0 \
  model.map_guided=true \
  model.map_guided_type=weighted_pool

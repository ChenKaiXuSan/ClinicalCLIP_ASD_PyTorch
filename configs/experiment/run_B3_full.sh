#!/bin/bash

# 实验目的：启用 CLIP 对齐 + map-guided（不含 token loss），作为完整方法主基线。

set -e

SELECTED_GPU="${1:-${GPU_NUM:-0}}"

cd /workspace/code/ClinicalCLIP_ASD_PyTorch

conda run -p /opt/conda/envs/clip python -m project.main \
  train.experiment=B3_full \
  train.gpu_num="${SELECTED_GPU}" \
  loss.clip_weight=1.0 \
  model.lambda_token=0.0 \
  model.map_guided=true \
  model.map_guided_type=spatiotemporal

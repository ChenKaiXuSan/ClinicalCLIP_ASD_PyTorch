#!/bin/bash

# 实验目的：使用 channel gate 形式的 map-guided，对比不同注意力融合方式的表现。

set -e

SELECTED_GPU="${1:-${GPU_NUM:-0}}"

cd /workspace/code/ClinicalCLIP_ASD_PyTorch

conda run -p /opt/conda/envs/clip python -m project.main \
  train.experiment=C1_channel_gate \
  train.gpu_num="${SELECTED_GPU}" \
  loss.clip_weight=1.0 \
  model.map_guided=true \
  model.map_guided_type=channel

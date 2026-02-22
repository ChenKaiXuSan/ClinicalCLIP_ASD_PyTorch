#!/bin/bash

# 实验目的：在时空 gate 中引入 sigmoid 门控，检验对注意力噪声与尺度变化的鲁棒性。

set -e

SELECTED_GPU="${1:-${GPU_NUM:-0}}"

cd /workspace/code/ClinicalCLIP_ASD_PyTorch

conda run -p /opt/conda/envs/clip python -m project.main \
  train.experiment=C3_sigmoid_gate \
  train.gpu_num="${SELECTED_GPU}" \
  loss.clip_weight=1.0 \
  model.map_guided=true \
  model.map_guided_sigmoid_gate=true \
  model.map_guided_type=spatiotemporal

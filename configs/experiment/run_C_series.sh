#!/bin/bash

# 实验目的：顺序运行 C1-C3 map-guided 变体实验，对比不同门控/池化机制。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELECTED_GPU="${1:-${GPU_NUM:-0}}"

bash "${SCRIPT_DIR}/run_C1_channel_gate.sh" "${SELECTED_GPU}"
bash "${SCRIPT_DIR}/run_C2_weighted_pool.sh" "${SELECTED_GPU}"
bash "${SCRIPT_DIR}/run_C3_sigmoid_gate.sh" "${SELECTED_GPU}"

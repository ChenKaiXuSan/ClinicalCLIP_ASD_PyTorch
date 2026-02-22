#!/bin/bash

# 实验目的：顺序运行 B1-B4 消融实验，得到从简化到完整方法的对比结果。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELECTED_GPU="${1:-${GPU_NUM:-0}}"

bash "${SCRIPT_DIR}/run_B1_clip_only.sh" "${SELECTED_GPU}"
bash "${SCRIPT_DIR}/run_B2_map_only.sh" "${SELECTED_GPU}"
bash "${SCRIPT_DIR}/run_B3_full.sh" "${SELECTED_GPU}"
bash "${SCRIPT_DIR}/run_B4_full_token.sh" "${SELECTED_GPU}"

#!/bin/bash

# 实验目的：一次性串行执行 B/C 全套对比实验，便于复现实验总表。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELECTED_GPU="${1:-${GPU_NUM:-0}}"

bash "${SCRIPT_DIR}/run_B_series.sh" "${SELECTED_GPU}"
bash "${SCRIPT_DIR}/run_C_series.sh" "${SELECTED_GPU}"

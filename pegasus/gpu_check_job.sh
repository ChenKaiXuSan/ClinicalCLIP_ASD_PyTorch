#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=00:40:00
#PBS -N cclip_qwencheck
#PBS -o logs/pegasus/qwen_check_out.log
#PBS -e logs/pegasus/qwen_check_err.log

# GPU 上的 Qwen3-VL 精度与耗时检查:bf16 与 fp32 各跑一次 tests/check_qwen_batch.py。
# 输出决定抽特征用哪种精度(bf16 相对差 < 0.01 即可用,速度翻倍且 32B 能放下)。
#   qsub -v CLINICALCLIP_REPO_ROOT=$PWD pegasus/gpu_check_job.sh
#   qsub -v CLINICALCLIP_REPO_ROOT=$PWD,MODEL=Qwen/Qwen3-VL-32B-Instruct,DTYPES=bfloat16 pegasus/gpu_check_job.sh

set -uo pipefail
REPO_ROOT="${CLINICALCLIP_REPO_ROOT:-/work/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
DTYPES="${DTYPES:-bfloat16 float32}"
IMG_SIZE="${IMG_SIZE:-448}"

cd "${REPO_ROOT}"
mkdir -p logs/pegasus
source pegasus/setup_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

for dt in ${DTYPES}; do
    python tests/check_qwen_batch.py sdpa "${dt}" "${MODEL}" "${IMG_SIZE}" 2>&1 | grep -vE "Warning|warn" | tail -3
done
echo "[done] $(date '+%F %T')"

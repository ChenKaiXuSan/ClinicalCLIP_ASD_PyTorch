#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=03:00:00
#PBS -N cclip_qattr
#PBS -o logs/pegasus/qwen_attr_out.log
#PBS -e logs/pegasus/qwen_attr_err.log

# 方案 2:一个属性一个作业(analysis/eval_qwen_attributes.py),8B bf16 全库约 50 分钟。
#   for a in trunk_forward_lean pelvic_retroversion ...; do
#     qsub -N cclip_qa_$a -o logs/pegasus/attr_${a}_out.log -e logs/pegasus/attr_${a}_err.log \
#          -v CLINICALCLIP_REPO_ROOT=$PWD,CLINICALCLIP_DATA_ROOT=$DATA,ATTR=$a pegasus/qwen_attr_job.sh; done

set -uo pipefail
REPO_ROOT="${CLINICALCLIP_REPO_ROOT:-/work/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch}"
DATA_ROOT="${CLINICALCLIP_DATA_ROOT:-/work/SKIING/chenkaixu/data/asd_dataset}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
IMG_SIZE="${IMG_SIZE:-448}"
DTYPE="${DTYPE:-bfloat16}"
ATTR="${ATTR:?需要 ATTR=<属性名>}"
OUT_DIR="${OUT_DIR:-logs/qwen_attributes/$(basename "${MODEL}" | tr 'A-Z' 'a-z')_${IMG_SIZE}}"

cd "${REPO_ROOT}"
mkdir -p logs/pegasus
source pegasus/setup_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python analysis/eval_qwen_attributes.py --root-path "${DATA_ROOT}" --model "${MODEL}" --img-size "${IMG_SIZE}" \
    --dtype "${DTYPE}" --attr "${ATTR}" --out-dir "${OUT_DIR}" ${EXTRA_ARGS:-} 2>&1 | grep --line-buffered -vE "Warning|warn"
echo "[done] $(date '+%F %T')"

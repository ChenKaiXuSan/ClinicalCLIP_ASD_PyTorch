#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=03:00:00
#PBS -N cclip_qexplain
#PBS -o logs/pegasus/qwen_explain_out.log
#PBS -e logs/pegasus/qwen_explain_err.log

# 角色三:M0 分类 + Qwen3-VL 解释生成 + 自动评估(analysis/eval_qwen_explain.py)。
#   qsub -v CLINICALCLIP_REPO_ROOT=$PWD,CKPT=<M0 fold0 ckpt> pegasus/qwen_explain_job.sh

set -uo pipefail
REPO_ROOT="${CLINICALCLIP_REPO_ROOT:-/work/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch}"
DATA_ROOT="${CLINICALCLIP_DATA_ROOT:-/work/SKIING/chenkaixu/data/asd_dataset}"
CKPT="${CKPT:?需要 CKPT=<M0 checkpoint>}"
FOLD="${FOLD:-0}"
LIMIT="${LIMIT:-120}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
OUT_DIR="${OUT_DIR:-logs/qwen_explain/fold${FOLD}}"

cd "${REPO_ROOT}"
mkdir -p logs/pegasus
source pegasus/setup_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python analysis/eval_qwen_explain.py --root-path "${DATA_ROOT}" --ckpt "${CKPT}" --fold "${FOLD}" \
    --limit "${LIMIT}" --model "${MODEL}" --out-dir "${OUT_DIR}" ${EXTRA_ARGS:-} 2>&1 | grep --line-buffered -vE "Warning|warn"
echo "[done] $(date '+%F %T')"

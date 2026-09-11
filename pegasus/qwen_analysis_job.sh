#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=05:00:00
#PBS -N cclip_qwen_ana
#PBS -o logs/pegasus/qwen_analysis_out.log
#PBS -e logs/pegasus/qwen_analysis_err.log

# 不训练的两项:Q2 VLM 注意力 vs 医生区域(eval_qwen_attention.py,eager 注意力),
# Q3 零样本诊断(eval_qwen_zeroshot_diag.py)。结果 json 落在 logs/qwen_analysis/。
#   qsub -v CLINICALCLIP_REPO_ROOT=$PWD pegasus/qwen_analysis_job.sh
#   qsub -v CLINICALCLIP_REPO_ROOT=$PWD,WHICH=diag,FOLDS="0 1 2 3 4" pegasus/qwen_analysis_job.sh

set -uo pipefail
REPO_ROOT="${CLINICALCLIP_REPO_ROOT:-/work/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch}"
DATA_ROOT="${CLINICALCLIP_DATA_ROOT:-/work/SKIING/chenkaixu/data/asd_dataset}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
IMG_SIZE="${IMG_SIZE:-448}"
DTYPE="${DTYPE:-float32}"
WHICH="${WHICH:-attn diag}"
FOLDS="${FOLDS:-0}"; FOLDS="${FOLDS//[,.]/ }"   # qsub -v 里逗号是变量分隔符, 折号用点号: FOLDS=1.2.3.4
LIMIT="${LIMIT:-0}"
VISUAL_PROMPT="${VISUAL_PROMPT:-}"   # box_lumbar: 只对 attn 生效

cd "${REPO_ROOT}"
mkdir -p logs/pegasus logs/qwen_analysis
source pegasus/setup_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
tag="$(basename "${MODEL}" | tr 'A-Z' 'a-z')_${IMG_SIZE}"

for fold in ${FOLDS}; do
    for w in ${WHICH}; do
        case "${w}" in
            attn) script=analysis/eval_qwen_attention.py ;;
            diag) script=analysis/eval_qwen_zeroshot_diag.py ;;
            *) echo "未知 WHICH=${w}" >&2; exit 1 ;;
        esac
        echo "== ${w} fold ${fold} $(date '+%F %T')"
        python "${script}" --root-path "${DATA_ROOT}" --model "${MODEL}" --img-size "${IMG_SIZE}" \
            --dtype "${DTYPE}" --fold "${fold}" --limit "${LIMIT}" $( [[ "${w}" == attn && -n "${VISUAL_PROMPT}" ]] && echo "--visual-prompt ${VISUAL_PROMPT}" ) \
            --out "logs/qwen_analysis/${w}_${tag}_fold${fold}.json" 2>&1 | grep --line-buffered -vE "Warning|warn"
    done
done
echo "[done] $(date '+%F %T')"

#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=02:00:00
#PBS -N cclip_attn
#PBS -o logs/pegasus/attn_alignment_out.log
#PBS -e logs/pegasus/attn_alignment_err.log

# 可解释性对照(不需要训练):一次跑出 uniform / random / center 三个下界,
# 外加 B0_3dcnn 的 Grad-CAM —— 后者是"注意力对齐"这一主张最强的对照。
# 把 M0/M1 日志里的 test/attn_align 和这里的 gradcam 比:不明显更高,主张就不成立。
#
# 前置:B0_3dcnn 至少训完一折(脚本会自动找它的 checkpoint)。
#
# 用法:
#   qsub pegasus/run_attn_alignment.sh                    # fold 0-4 全跑
#   FOLDS=0 qsub pegasus/run_attn_alignment.sh            # 只跑一折
#   CKPT=/path/to/x.ckpt qsub pegasus/run_attn_alignment.sh   # 指定 checkpoint
#   NO_GRADCAM=1 qsub ...                                 # 只要三个下界,不做 Grad-CAM

set -uo pipefail

REPO_ROOT="${CLINICALCLIP_REPO_ROOT:-/work/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch}"
DATA_ROOT="${CLINICALCLIP_DATA_ROOT:-/work/SKIING/chenkaixu/data/asd_dataset}"
FOLDS="${FOLDS:-0 1 2 3 4}"
LIMIT="${LIMIT:-60}"
BASELINE_TAG="${BASELINE_TAG:-B0_3dcnn}"

cd "${REPO_ROOT}"
mkdir -p logs/pegasus
source pegasus/setup_env.sh

for fold in ${FOLDS//,/ }; do
    CKPT_ARG=""
    if [[ "${NO_GRADCAM:-0}" != "1" ]]; then
        ckpt="${CKPT:-}"
        if [[ -z "${ckpt}" ]]; then
            # logs/train/<tag>/<日期>/<时刻>/checkpoint/<fold>/*.ckpt,取最新的一个
            ckpt=$(find "logs/train/${BASELINE_TAG}__f${fold}_s42" \
                        -path "*/checkpoint/${fold}/*.ckpt" ! -name "last.ckpt" \
                        -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
        fi
        if [[ -n "${ckpt}" ]]; then
            echo "fold ${fold}: Grad-CAM checkpoint = ${ckpt}"
            CKPT_ARG="--ckpt ${ckpt}"
        else
            echo "fold ${fold}: 找不到 ${BASELINE_TAG} 的 checkpoint,本折只出三个下界,不做 Grad-CAM" >&2
        fi
    fi

    # shellcheck disable=SC2086
    python analysis/eval_attention_alignment.py \
        --root-path "${DATA_ROOT}" \
        --fold "${fold}" \
        --limit "${LIMIT}" \
        ${CKPT_ARG} \
        2>&1 | tee "logs/pegasus/attn_alignment_fold${fold}.log"
done

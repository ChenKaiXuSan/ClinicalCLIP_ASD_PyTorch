#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=03:00:00
#PBS -N cclip_extract
#PBS -o logs/pegasus/extract_out.log
#PBS -e logs/pegasus/extract_err.log

# 离线抽取冻结 VLM 的视觉 token(scripts/extract_vlm_features.py),V0-V2 训练时直接读缓存。
# 全库约 1900 条视频、约 2800 段;so400m @224 在 H100 上约 1 小时。
#
# 先在登录节点跑 pegasus/prepare_vlm.sh 把权重下到 HF_HOME,再:
#   qsub pegasus/extract_job.sh
#   qsub -v VLM_TAG=siglip2_base_224,MODEL=google/siglip2-base-patch16-224 pegasus/extract_job.sh
#   qsub -v IMG_SIZE=384,VLM_TAG=siglip2_so400m_384 pegasus/extract_job.sh   # 24x24 token,缓存约 3 倍大

set -uo pipefail

REPO_ROOT="${CLINICALCLIP_REPO_ROOT:-/work/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch}"
DATA_ROOT="${CLINICALCLIP_DATA_ROOT:-/work/SKIING/chenkaixu/data/asd_dataset}"
BACKEND="${BACKEND:-siglip2}"
MODEL="${MODEL:-google/siglip2-so400m-patch16-384}"
IMG_SIZE="${IMG_SIZE:-224}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"   # = train.uniform_temporal_subsample_num
VLM_TAG="${VLM_TAG:-siglip2_so400m_224}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/vlm_cache/${VLM_TAG}}"

cd "${REPO_ROOT}"
mkdir -p logs/pegasus
source pegasus/setup_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python scripts/extract_vlm_features.py \
    --root-path "${DATA_ROOT}" \
    --backend "${BACKEND}" --model "${MODEL}" \
    --img-size "${IMG_SIZE}" --num-samples "${NUM_SAMPLES}" \
    --out-dir "${OUT_DIR}" \
    --device cuda:0 --num-workers 8 ${EXTRA_ARGS:-}

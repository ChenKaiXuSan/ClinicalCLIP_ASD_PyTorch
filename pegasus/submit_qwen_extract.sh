#!/bin/bash
# 批量提交 Qwen3-VL 的离线抽特征作业:尺寸 x prompt 各一个 GPU 节点。
#
#   bash pegasus/submit_qwen_extract.sh                       # 8b 两种 prompt
#   SIZES="2b 4b 8b" bash pegasus/submit_qwen_extract.sh      # 尺寸曲线
#   PROMPTS="generic clinical" DTYPE=bfloat16 ...              # GPU 上验证 bf16 合格后
#   DRYRUN=1 ...
#
# 缓存落在 $DATA_ROOT/vlm_cache/qwen3vl_<size>_<img>_<prompt>/,与 matrix.tsv 的 CACHE:<tag> 一致。

set -euo pipefail

REPO_ROOT="${CLINICALCLIP_REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_ROOT="${CLINICALCLIP_DATA_ROOT:-/work/SKIING/chenkaixu/data/asd_dataset}"
SIZES="${SIZES:-8b}"
PROMPTS="${PROMPTS:-generic clinical}"
IMG_SIZE="${IMG_SIZE:-448}"
DTYPE="${DTYPE:-float32}"
CHUNK="${CHUNK:-16}"
ELAPS="${ELAPS:-04:00:00}"
DRYRUN="${DRYRUN:-0}"

declare -A MODEL_OF=(
    [2b]=Qwen/Qwen3-VL-2B-Instruct
    [4b]=Qwen/Qwen3-VL-4B-Instruct
    [8b]=Qwen/Qwen3-VL-8B-Instruct
    [32b]=Qwen/Qwen3-VL-32B-Instruct
)

cd "${REPO_ROOT}"
mkdir -p logs/pegasus

for size in ${SIZES}; do
    model="${MODEL_OF[$size]:-}"
    [[ -n "${model}" ]] || { echo "未知尺寸 ${size}" >&2; exit 1; }
    for prompt in ${PROMPTS}; do
        tag="qwen3vl_${size}_${IMG_SIZE}_${prompt}"
        if [[ -f "${DATA_ROOT}/vlm_cache/${tag}/manifest.json" ]] && \
           [[ "$(ls "${DATA_ROOT}/vlm_cache/${tag}"/*.pt 2>/dev/null | wc -l)" -gt 1800 ]]; then
            echo "[skip] ${tag} 已有完整缓存"
            continue
        fi
        vars="CLINICALCLIP_REPO_ROOT=${REPO_ROOT},CLINICALCLIP_DATA_ROOT=${DATA_ROOT}"
        vars+=",BACKEND=qwen3vl,MODEL=${model},IMG_SIZE=${IMG_SIZE},PROMPT=${prompt},VLM_TAG=${tag},CHUNK=${CHUNK},DTYPE=${DTYPE}"
        cmd=(qsub -N "cclip_x_${size}_${prompt}" -l "elapstim_req=${ELAPS}"
             -o "logs/pegasus/extract_${tag}_out.log" -e "logs/pegasus/extract_${tag}_err.log"
             -v "${vars}" pegasus/extract_job.sh)
        if [[ "${DRYRUN}" == "1" ]]; then
            echo "DRYRUN: ${cmd[*]}"
        else
            "${cmd[@]}"
        fi
    done
done

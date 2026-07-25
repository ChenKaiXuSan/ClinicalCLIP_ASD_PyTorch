#!/bin/bash

# Pegasus 作业的环境引导,被 matrix_job.sh / prepare_*.sh 共同 source。
# 换环境:  CLINICALCLIP_CONDA_ENV=/path/to/env qsub pegasus/matrix_job.sh

CLINICALCLIP_CONDA_ENV="${CLINICALCLIP_CONDA_ENV:-/home/SKIING/chenkaixu/miniconda3/envs/clip}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${USER:-user}-${PBS_JOBID:-manual}}"
# 计算节点无外网,HF 权重由 pegasus/prepare_concepts.sh 在登录节点预下载到这里
export HF_HOME="${HF_HOME:-/work/SKIING/chenkaixu/hf_cache}"
# 一条视频的全部 gait 段拼成一个 batch,段数随视频长度大幅波动,分段缓存能明显减少碎片
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${MPLCONFIGDIR}"

if command -v module >/dev/null 2>&1; then
    module load intelpython/2022.3.1
fi

if [[ -z "${CONDA_PREFIX:-}" || ! -f "${CONDA_PREFIX}/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: module load intelpython 之后仍找不到 conda.sh" >&2
    echo "CONDA_PREFIX=${CONDA_PREFIX:-<unset>}" >&2
    exit 1
fi

# conda 的 shell 函数会引用未定义变量,调用方开了 set -u 时先摘掉
case $- in *u*) _restore_nounset=1 ;; *) _restore_nounset=0 ;; esac
set +u

source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
conda deactivate >/dev/null 2>&1 || true

if [[ ! -x "${CLINICALCLIP_CONDA_ENV}/bin/python" ]]; then
    echo "ERROR: conda 环境里没有可执行的 python: ${CLINICALCLIP_CONDA_ENV}" >&2
    conda info --envs >&2 || true
    exit 1
fi

conda activate "${CLINICALCLIP_CONDA_ENV}" || {
    echo "ERROR: 无法激活 conda 环境: ${CLINICALCLIP_CONDA_ENV}" >&2
    conda info --envs >&2 || true
    exit 1
}

if [[ "${_restore_nounset}" == "1" ]]; then
    set -u
fi
unset _restore_nounset

hash -r

echo "Python: $(python --version 2>&1) @ $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
    echo "nvidia-smi: not found"
fi

python -c 'import torch; print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())'

#!/bin/bash
#PBS -A HP260146
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=02:00:00
#PBS -N cclip_llmm
#PBS -o logs/pegasus/llm_measure_out.log
#PBS -e logs/pegasus/llm_measure_err.log

# LLM 把医生关注区域翻译成测量(analysis/llm_measure_propose.py):三个尺寸的 Qwen3-VL 依次生成,纯文本,不读任何患者数据。
#   qsub -v CLINICALCLIP_REPO_ROOT=$PWD,N=20 pegasus/llm_measure_job.sh
cd "${CLINICALCLIP_REPO_ROOT:-/work/1/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch/.claude/worktrees/vlm}" || exit 1
mkdir -p logs/pegasus
source pegasus/setup_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
N="${N:-20}"
echo "节点 $(hostname)  $(date '+%F %T')"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for M in ${MODELS:-Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen3-VL-4B-Instruct Qwen/Qwen3-VL-2B-Instruct}; do
    echo "== $M  $(date '+%T')"
    python analysis/llm_measure_propose.py --model "$M" --n "$N" --out "${OUT:-logs/llm_measure}" 2>&1 | grep --line-buffered -vE "Warning|warn"
done
echo "[done] $(date '+%F %T')"

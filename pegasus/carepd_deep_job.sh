#!/bin/bash
#PBS -A HP260146
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=06:00:00
#PBS -N cclip_cpd
#PBS -o logs/pegasus/carepd_deep_out.log
#PBS -e logs/pegasus/carepd_deep_err.log

# CARE-PD 训练样本量曲线的端到端侧(analysis/carepd/train_deep.py),一个队列一个作业:
#   qsub -v COHORT=PD-GaM pegasus/carepd_deep_job.sh
cd "${CLINICALCLIP_REPO_ROOT:-/work/1/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch/.claude/worktrees/vlm}" || exit 1
mkdir -p logs/pegasus logs/carepd
source pegasus/setup_env.sh
COHORT="${COHORT:?需要 COHORT}"
echo "节点 $(hostname)  $(date '+%F %T')"; nvidia-smi --query-gpu=name --format=csv,noheader
python analysis/carepd/train_deep.py --cohort "$COHORT" --reps "${REPS:-3}" --epochs "${EPOCHS:-60}" \
    --out "logs/carepd/size_curve_deep_${COHORT}.json" 2>&1 | grep --line-buffered -vE "Warning|warn"
echo "[done] $(date '+%F %T')"

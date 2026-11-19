#!/bin/bash
#PBS -A HP260146
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=04:00:00
#PBS -N cclip_kv
#PBS -o logs/pegasus/koa_train_out.log
#PBS -e logs/pegasus/koa_train_err.log

# KOA-PD-NM 的第二个分支(analysis/koa/train_video.py 或 train_skel.py),一个作业 = 一种输入 × 一个种子的 5 折:
#   qsub -v MODE=masked,SEED=42 pegasus/koa_train_job.sh        MODE: rgb | masked | sil | skel(骨架网络,一次跑完 5 个种子)
cd "${CLINICALCLIP_REPO_ROOT:-/work/1/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch/.claude/worktrees/vlm}" || exit 1
mkdir -p logs/pegasus logs/koa
source pegasus/setup_env.sh
MODE="${MODE:?需要 MODE}"
echo "节点 $(hostname)  $MODE  种子 ${SEED:-全部}  $(date '+%F %T')"
if [ "$MODE" = skel ]; then
    python analysis/koa/train_skel.py 2>&1 | grep --line-buffered -vE "Warning|warn"
else
    python analysis/koa/train_video.py --mode "$MODE" --seed "${SEED:?需要 SEED}" ${EPOCHS:+--epochs $EPOCHS} ${FOLDS:+--folds $FOLDS} 2>&1 | grep --line-buffered -vE "Warning|warn"
fi
echo "[done] $(date '+%F %T')"

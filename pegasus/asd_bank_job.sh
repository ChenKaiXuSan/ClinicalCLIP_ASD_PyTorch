#!/bin/bash
#PBS -A HP260146
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=06:00:00
#PBS -N cclip_bank
#PBS -o logs/pegasus/asd_bank_out.log
#PBS -e logs/pegasus/asd_bank_err.log

# 测量库(logs/measure_bank_3d/bank.pkl)重建之后,重算依赖它的三组数字(只用 CPU,48 核):
#   STEP=p0  ℓ1 逻辑回归在 10 份随机划分上的成绩(全库 / 医生区域内 / 非医生区域)-> logs/geom_probs/p0/repeated_splits_nomiss_auc.json
#   STEP=p1  随机森林 / 梯度提升,同一批划分 -> logs/geom_probs/p1_nomiss/nonlinear_repeated_nomiss.json
#   STEP=llm LLM 提出的测量 -> logs/llm_measure/eval.json
#   qsub -v STEP=p0 pegasus/asd_bank_job.sh
cd "${CLINICALCLIP_REPO_ROOT:-/work/1/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch/.claude/worktrees/vlm}" || exit 1
mkdir -p logs/pegasus
source pegasus/setup_env.sh
DATA="${CLINICALCLIP_DATA_ROOT:-/work/1/SKIING/chenkaixu/data/asd_dataset}"
echo "节点 $(hostname)  STEP=${STEP:?需要 STEP}  $(date '+%F %T')"
case "$STEP" in
    p0)  python analysis/p0_controls.py repeated-splits --data-root "$DATA" --n 10 --exclude-missing --out-dir logs/geom_probs/p0 ;;
    p1)  python analysis/p1_robustness.py nonlinear --data-root "$DATA" --exclude-missing --n-splits 10 --cands 全库 --n-jobs 40 --out-dir logs/geom_probs/p1_nomiss ;;
    llm) python analysis/llm_measure_eval.py --root logs/llm_measure --out logs/llm_measure/eval.json ;;
esac 2>&1 | grep --line-buffered -vE "Warning|warn"
echo "[done] $(date '+%F %T')"

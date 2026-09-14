#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=06:00:00
#PBS -N cclip_nlrep
#PBS -o logs/pegasus/nonlinear_rep_out.log
#PBS -e logs/pegasus/nonlinear_rep_err.log

# P1-6 的换划分版:随机森林 / 梯度提升吃 1272 量测量库(及手挑 5 量), 67 名 3D 完整患者, N 份随机患者划分,
# 报 macro 与 AUC 均值 ± std。只用 CPU(48 核), 放到计算节点是为了不受登录节点 16GB / 会话中断影响。
# 进度逐划分写到 logs/geom_probs/p1_nomiss/nonlinear_rep_progress.log, 结果 json nonlinear_repeated_nomiss.json。
#   qsub -v CLINICALCLIP_REPO_ROOT=$PWD pegasus/nonlinear_rep_job.sh
#   qsub -v CLINICALCLIP_REPO_ROOT=$PWD,N_SPLITS=10,CANDS=全库.手挑 pegasus/nonlinear_rep_job.sh

set -uo pipefail
REPO_ROOT="${CLINICALCLIP_REPO_ROOT:-/work/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch}"
DATA_ROOT="${CLINICALCLIP_DATA_ROOT:-/work/SKIING/chenkaixu/data/asd_dataset}"
N_SPLITS="${N_SPLITS:-10}"
CANDS="${CANDS:-全库.手挑}"; CANDS="${CANDS//./,}"   # qsub -v 里逗号是分隔符, 用点号代替
N_JOBS="${N_JOBS:-40}"

cd "${REPO_ROOT}"
mkdir -p logs/pegasus logs/geom_probs/p1_nomiss
source pegasus/setup_env.sh
export OMP_NUM_THREADS="${N_JOBS}"
echo "== nonlinear repeated splits: N=${N_SPLITS} cands=${CANDS} n_jobs=${N_JOBS} $(date '+%F %T')"
python -u analysis/p1_robustness.py nonlinear --data-root "${DATA_ROOT}" --exclude-missing \
    --n-splits "${N_SPLITS}" --n-jobs "${N_JOBS}" --cands "${CANDS}" --out-dir logs/geom_probs/p1_nomiss \
    2>&1 | grep --line-buffered -vE "Warning|warn" | tee logs/geom_probs/p1_nomiss/nonlinear_rep_progress.log
echo "== done $(date '+%F %T')"

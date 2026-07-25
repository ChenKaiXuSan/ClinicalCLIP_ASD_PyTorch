#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N cclip_matrix
#PBS -o logs/pegasus/matrix_out.log
#PBS -e logs/pegasus/matrix_err.log

# 实验矩阵的数组作业:**一个 sub-request = 一个节点 = 一个配置的一折**。
# 队列的单请求上限就是 24 小时,而 100 epochs 的视频类实验实测 5.3 小时(bf16)/
# 8.3 小时(fp32),一折一节点留足了余量;姿态基线约 0.7 小时。
#
# 不要直接 qsub 这个文件 —— 数组范围和作业清单由提交脚本生成:
#   bash pegasus/submit_matrix.sh
#
# 直接 qsub 时(PBS_SUBREQNO 未定义)只会跑清单的第一行,便于单点调试:
#   qsub -v MATRIX_RUN=/work/.../pegasus/queue/20260725-1200.part0 pegasus/matrix_job.sh

set -uo pipefail

REPO_ROOT="${CLINICALCLIP_REPO_ROOT:-/work/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch}"
cd "${REPO_ROOT}"

if [[ -z "${MATRIX_RUN:-}" ]]; then
    echo "ERROR: 没有 MATRIX_RUN。请用 pegasus/submit_matrix.sh 提交。" >&2
    exit 1
fi

JOBLIST="${MATRIX_RUN}.tsv"
RUNENV="${MATRIX_RUN}.env"
for f in "${JOBLIST}" "${RUNENV}"; do
    [[ -f "$f" ]] || { echo "ERROR: 找不到 $f" >&2; exit 1; }
done

# DATA_ROOT / EPOCHS / PRECISION / NUM_WORKERS / OUT_DIR 由提交时固化在这里,
# 保证同一批作业即使 matrix.tsv 后来被改过也跑的是提交那一刻的设定
# shellcheck disable=SC1090
source "${RUNENV}"

IDX="${PBS_SUBREQNO:-0}"
LINE=$(awk -v n="$((IDX + 1))" 'NR == n' "${JOBLIST}")
if [[ -z "${LINE}" ]]; then
    echo "ERROR: 作业清单 ${JOBLIST} 没有第 $((IDX + 1)) 行" >&2
    exit 1
fi
IFS=$'\t' read -r TAG FOLD SEED ARGS <<< "${LINE}"

mkdir -p "${OUT_DIR}/done" logs/pegasus

DONE_MARK="${OUT_DIR}/done/${TAG}.done"
if [[ -f "${DONE_MARK}" && "${FORCE:-0}" != "1" ]]; then
    echo "[skip] ${TAG} 已完成 (${DONE_MARK});要重跑加 FORCE=1"
    exit 0
fi

RUN_LOG="${OUT_DIR}/${TAG}.log"

{
    echo "===================================================================="
    echo "任务      : ${TAG}   (数组下标 ${IDX})"
    echo "折 / 种子 : fold=${FOLD}  seed=${SEED}"
    echo "覆盖参数  : ${ARGS}"
    echo "epochs    : ${EPOCHS}   precision: ${PRECISION}   workers: ${NUM_WORKERS}"
    echo "节点      : $(hostname)   CPU 核数: $(nproc)"
    echo "开始      : $(date '+%F %T')"
    echo "===================================================================="
} | tee "${RUN_LOG}"

source pegasus/setup_env.sh 2>&1 | tee -a "${RUN_LOG}"

# shellcheck disable=SC2086
python project/main.py ${ARGS} \
    paths.root_path="${DATA_ROOT}" \
    train.experiment="${TAG}" \
    train.folds="[${FOLD}]" \
    train.seed="${SEED}" \
    train.max_epochs="${EPOCHS}" \
    train.precision="${PRECISION}" \
    train.gpu_num=0 \
    data.num_workers="${NUM_WORKERS}" \
    2>&1 | tee -a "${RUN_LOG}"

STATUS=${PIPESTATUS[0]}

if [[ "${STATUS}" == "0" ]]; then
    date '+%F %T' > "${DONE_MARK}"
    echo "[done] ${TAG} 于 $(date '+%F %T')" | tee -a "${RUN_LOG}"
else
    echo "[FAIL] ${TAG} 退出码 ${STATUS};重新提交同一批即可,已完成的会自动跳过" | tee -a "${RUN_LOG}"
fi

exit "${STATUS}"

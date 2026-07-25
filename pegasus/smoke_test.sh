#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=03:00:00
#PBS -N cclip_smoke
#PBS -o logs/pegasus/smoke_out.log
#PBS -e logs/pegasus/smoke_err.log

# 提交全矩阵之前的链路自检:在一个节点上串行把 matrix.tsv 里每个配置都用
# train.fast_dev_run=true 跑一个 batch。目的是让"某个配置根本跑不起来"这件事
# 在 10 分钟内暴露,而不是在 70 个节点跑了半天之后。
#
# fast_dev_run 不写 checkpoint、不写 test_metrics,不会污染正式实验的日志目录。
#
# 用法:
#   qsub pegasus/smoke_test.sh
#   GROUP=main qsub pegasus/smoke_test.sh    # 只自检一部分

set -uo pipefail

REPO_ROOT="${CLINICALCLIP_REPO_ROOT:-/work/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch}"
DATA_ROOT="${CLINICALCLIP_DATA_ROOT:-/work/SKIING/chenkaixu/data/asd_dataset}"
EMB="${EMB:-${DATA_ROOT}/concepts/clip_vit_b32.pt}"
GROUP="${GROUP:-all}"
FOLD="${FOLD:-0}"
PRECISION="${PRECISION:-bf16-mixed}"

cd "${REPO_ROOT}"
mkdir -p logs/pegasus
source pegasus/setup_env.sh

want_group() {
    [[ "${GROUP}" == "all" ]] && return 0
    [[ ",${GROUP}," == *",$1,"* ]]
}

declare -a OK=() BAD=()

while IFS=$'\t' read -r grp name args; do
    [[ -z "${grp:-}" || "${grp}" == \#* ]] && continue
    want_group "${grp}" || continue

    args="${args//EMB/${EMB}}"
    echo
    echo "############ 自检 ${name} ############"
    echo "覆盖参数: ${args}"

    # shellcheck disable=SC2086
    python project/main.py ${args} \
        paths.root_path="${DATA_ROOT}" \
        train.experiment="smoke_${name}" \
        train.folds="[${FOLD}]" \
        train.max_epochs=1 \
        train.precision="${PRECISION}" \
        train.gpu_num=0 \
        train.fast_dev_run=true \
        data.num_workers=4 \
        > "logs/pegasus/smoke_${name}.log" 2>&1

    if [[ $? == 0 ]]; then
        echo "  OK"
        OK+=("${name}")
    else
        echo "  失败 —— 末尾 20 行:"
        tail -20 "logs/pegasus/smoke_${name}.log" | sed 's/^/    /'
        BAD+=("${name}")
    fi
done < pegasus/matrix.tsv

echo
echo "==================== 自检结果 ===================="
echo "通过 ${#OK[@]} 个: ${OK[*]:-无}"
echo "失败 ${#BAD[@]} 个: ${BAD[*]:-无}"
echo "单个配置的完整日志: logs/pegasus/smoke_<name>.log"

(( ${#BAD[@]} == 0 )) || exit 1

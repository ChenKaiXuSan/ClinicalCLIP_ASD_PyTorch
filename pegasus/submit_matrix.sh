#!/bin/bash
# 按 pegasus/matrix.tsv 生成作业清单并提交 PBS 数组作业。
# 一个 sub-request = 一个节点 = 一个配置的一折(队列单请求上限 24 小时)。
#
# 用法:
#   GROUP=all FOLDS=0 bash pegasus/submit_matrix.sh          # ① 单折筛选:14 个节点
#   GROUP=baseline,main bash pegasus/submit_matrix.sh        # ② 五折主表:35 个节点
#   GROUP=ablation,annotator bash pegasus/submit_matrix.sh   # ③ 五折消融:35 个节点
#   GROUP=all bash pegasus/submit_matrix.sh                  # 全矩阵:70 个节点
#   SEEDS=42,1337,2024 GROUP=main bash pegasus/submit_matrix.sh   # ④ 多种子方差
#   DRYRUN=1 ...                                             # 只生成清单不提交
#   FORCE=1 ...                                              # 忽略 done 标记,全部重跑
#   ELAPS=05:00:00 ...                                       # 压缩墙钟请求以塞进维护窗口前
#
# 断点续跑:直接把同一条命令再敲一遍。已经写下 done 标记的任务会被剔除,
# 只有失败/没跑到的会重新排队。

set -euo pipefail

REPO_ROOT="${CLINICALCLIP_REPO_ROOT:-/work/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch}"
DATA_ROOT="${CLINICALCLIP_DATA_ROOT:-/work/SKIING/chenkaixu/data/asd_dataset}"
EMB="${EMB:-${DATA_ROOT}/concepts/clip_vit_b32.pt}"

GROUP="${GROUP:-all}"        # 逗号分隔,对应 matrix.tsv 第一列;all 表示全部
FOLDS="${FOLDS:-0-4}"        # 全库统一 5 折
SEEDS="${SEEDS:-42}"
EPOCHS="${EPOCHS:-100}"      # 统一 100 epochs,不用 early stopping
PRECISION="${PRECISION:-bf16-mixed}"   # 实测比 fp32 快 1.56 倍;整个矩阵必须同一精度
NUM_WORKERS="${NUM_WORKERS:-12}"
# 单个 sub-request 的墙钟上限。H100 实测最长 4 小时 35 分(B2_cnn_lstm),留 30% 余量。
# 别填成队列上限 24:00:00 —— 维护窗口之前放不下,调度器会把作业压在队列里不发,
# 哪怕整个集群空着。窗口紧张时按实测值再压:ELAPS=05:00:00 ...
ELAPS="${ELAPS:-06:00:00}"
EXPECT_FOLD="${EXPECT_FOLD:-5}"        # index_mapping 缓存必须是这个折数
CHUNK="${CHUNK:-150}"        # 队列上限:一个批处理请求最多 150 个 sub-request
DRYRUN="${DRYRUN:-0}"
FORCE="${FORCE:-0}"

cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/logs/pegasus/matrix}"
QUEUE_DIR="${REPO_ROOT}/pegasus/queue"
mkdir -p "${OUT_DIR}/done" "${QUEUE_DIR}" logs/pegasus

# ---- 提交前检查:两个会让整批作业白跑的坑 ----

INDEX_JSON="${DATA_ROOT}/clinical_CLIP_dataset/index_mapping/3/index.json"
if [[ ! -f "${INDEX_JSON}" ]]; then
    echo "ERROR: 找不到交叉验证划分 ${INDEX_JSON}" >&2
    echo "       先跑: bash pegasus/prepare_index.sh" >&2
    exit 1
fi
actual_folds=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "${INDEX_JSON}")
if [[ "${actual_folds}" != "${EXPECT_FOLD}" ]]; then
    echo "ERROR: 划分缓存是 ${actual_folds} 折,矩阵要求 ${EXPECT_FOLD} 折。" >&2
    echo "       cross_validation.py 只认缓存,train.fold 改了也不重新划分。" >&2
    echo "       先跑: bash pegasus/prepare_index.sh  (旧缓存会被备份,不会删)" >&2
    exit 1
fi

expand_folds() {   # 支持 "0-4" / "0,3" / "0"
    local spec=$1 out=""
    for part in ${spec//,/ }; do
        if [[ "${part}" == *-* ]]; then
            out+=" $(seq "${part%-*}" "${part#*-}")"
        else
            out+=" ${part}"
        fi
    done
    echo "${out}"
}

want_group() {
    [[ "${GROUP}" == "all" ]] && return 0
    [[ ",${GROUP}," == *",$1,"* ]]
}

# ---- 读矩阵 ----
declare -a NAMES=() ARGSS=()
needs_emb=0
while IFS=$'\t' read -r grp name args; do
    [[ -z "${grp:-}" || "${grp}" == \#* ]] && continue
    want_group "${grp}" || continue
    [[ "${args}" == *EMB* ]] && needs_emb=1
    NAMES+=("${name}")
    ARGSS+=("${args//EMB/${EMB}}")
done < pegasus/matrix.tsv

if (( ${#NAMES[@]} == 0 )); then
    echo "ERROR: GROUP=${GROUP} 没有匹配到任何实验" >&2
    exit 1
fi

if (( needs_emb == 1 )) && [[ ! -f "${EMB}" ]]; then
    echo "ERROR: M1_concept_cliptext 需要文本概念向量,但 ${EMB} 不存在。" >&2
    echo "       在**登录节点**跑: bash pegasus/prepare_concepts.sh" >&2
    echo "       (只想跑别的实验就换 GROUP,或用 EMB=<路径> 指定已有文件)" >&2
    exit 1
fi

# ---- 构建作业清单:折优先排序 ----
# 先把所有配置的 fold0 排完再进 fold1。第一轮结束就有一份完整的跨配置对比,
# 某个配置有问题也能尽早发现,不必等它把 5 折全排完。
STAMP="$(date +%Y%m%d-%H%M%S)"
ALL_LIST="${QUEUE_DIR}/${STAMP}.all.tsv"
: > "${ALL_LIST}"

skipped=0
for fold in $(expand_folds "${FOLDS}"); do
    for seed in ${SEEDS//,/ }; do
        for i in "${!NAMES[@]}"; do
            tag="${NAMES[$i]}__f${fold}_s${seed}"
            if [[ -f "${OUT_DIR}/done/${tag}.done" && "${FORCE}" != "1" ]]; then
                skipped=$((skipped + 1))
                continue
            fi
            printf '%s\t%s\t%s\t%s\n' "${tag}" "${fold}" "${seed}" "${ARGSS[$i]}" >> "${ALL_LIST}"
        done
    done
done

total=$(wc -l < "${ALL_LIST}")
echo "GROUP=${GROUP}  FOLDS=${FOLDS}  SEEDS=${SEEDS}  EPOCHS=${EPOCHS}  PRECISION=${PRECISION}"
echo "待提交 ${total} 个任务(每个占一个节点,墙钟上限 ${ELAPS});已完成跳过 ${skipped} 个"
if (( total == 0 )); then
    echo "没有需要跑的任务。"
    rm -f "${ALL_LIST}"
    exit 0
fi

# ---- 切块提交:一个批处理请求最多 CHUNK 个 sub-request ----
split -l "${CHUNK}" -d -a 2 "${ALL_LIST}" "${QUEUE_DIR}/${STAMP}.part"
part=0
for chunk in "${QUEUE_DIR}/${STAMP}.part"*; do
    [[ "${chunk}" == *.tsv || "${chunk}" == *.env ]] && continue
    prefix="${QUEUE_DIR}/${STAMP}.part${part}"
    mv "${chunk}" "${prefix}.tsv"

    cat > "${prefix}.env" <<ENV
DATA_ROOT=${DATA_ROOT}
EPOCHS=${EPOCHS}
PRECISION=${PRECISION}
NUM_WORKERS=${NUM_WORKERS}
OUT_DIR=${OUT_DIR}
ENV

    n=$(wc -l < "${prefix}.tsv")
    echo
    echo "--- 第 ${part} 批: ${n} 个任务 -> ${prefix}.tsv"
    awk -F'\t' '{printf "    [%2d] %s\n", NR-1, $1}' "${prefix}.tsv"

    if [[ "${DRYRUN}" == "1" ]]; then
        echo "    DRYRUN: qsub -t 0-$((n - 1)) -l elapstim_req=${ELAPS} -v MATRIX_RUN=${prefix} pegasus/matrix_job.sh"
    else
        qsub -t "0-$((n - 1))" \
            -l "elapstim_req=${ELAPS}" \
            -N "cclip_mx${part}" \
            -v "MATRIX_RUN=${prefix}" \
            -o "logs/pegasus/matrix_${STAMP}_p${part}_out.log" \
            -e "logs/pegasus/matrix_${STAMP}_p${part}_err.log" \
            pegasus/matrix_job.sh
    fi
    part=$((part + 1))
done

rm -f "${ALL_LIST}"

echo
if [[ "${DRYRUN}" == "1" ]]; then
    echo "DRYRUN 结束,没有实际提交。清单留在 ${QUEUE_DIR}/${STAMP}.part*.tsv"
else
    echo "已提交。查看状态: qstat    单个任务日志: ${OUT_DIR}/<tag>.log"
    echo "全部跑完后汇总: python analysis/compare_concept_runs.py --root logs/train"
fi

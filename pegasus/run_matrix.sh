#!/bin/bash
# 按 pegasus/matrix.tsv 执行实验矩阵,双卡各跑一个任务的作业队列。
#
# 每卡只放 1 个任务:batch 是"一条视频的全部 gait 段",最长视频 838 帧 → 28 段,
# 单任务显存峰值可达 31GB,两个任务挤一张 48GB 卡必 OOM。
#
# 用法:
#   GROUP=all FOLDS=0 bash pegasus/run_matrix.sh               # 单折筛选(先跑这个)
#   GROUP=baseline,main bash pegasus/run_matrix.sh              # 五折主表
#   GROUP=ablation,annotator bash pegasus/run_matrix.sh         # 五折消融
#   PRECISION=bf16-mixed ...                                    # 实测快 1.56x, 省 35% 时间
#   SEEDS=42,1337,2024 ...                                      # 多种子报方差
#   DRYRUN=1 ...                                                # 只打印不执行
#
# 全库统一:5 折、100 epochs、不使用 early stopping。
set -u

REPO=/home/kaixu_chen/asd/ClinicalCLIP_ASD_PyTorch
PY=/home/kaixu_chen/miniforge3/envs/asd/bin/python
ROOT=${ROOT:-/mnt/data/xchen/asd_data}
EMB=${EMB:-$ROOT/concepts/clip_vit_b32.pt}

GROUP=${GROUP:-all}          # 逗号分隔;all 表示全部
FOLDS=${FOLDS:-0-4}          # 全库统一 5 折;也可写 "0" 或 "0,3"
SEEDS=${SEEDS:-42}
EPOCHS=${EPOCHS:-100}   # 统一 100 epochs, 不用 early stopping
WORKERS=${WORKERS:-10}
PRECISION=${PRECISION:-bf16-mixed}   # 实测比 fp32 快 1.56x、显存 7.6->4.9GB
GPUS=${GPUS:-0 1}
OUT=${OUT:-$REPO/logs/matrix}
DRYRUN=${DRYRUN:-0}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
cd "$REPO"

expand_folds() {   # 支持 0-9 展开
  local spec=$1 out=""
  for part in ${spec//,/ }; do
    if [[ "$part" == *-* ]]; then
      out+=" $(seq "${part%-*}" "${part#*-}")"
    else
      out+=" $part"
    fi
  done
  echo "$out"
}

want_group() {
  [[ "$GROUP" == "all" ]] && return 0
  [[ ",$GROUP," == *",$1,"* ]]
}

# ---- 构建作业队列(折优先排序)----
# 先把所有配置的 fold0 跑完,再进 fold1。这样第一轮结束就能拿到一份完整的
# 跨配置对比,某个配置有问题也能尽早发现,不必等它把 5 折全跑完。
declare -a NAMES=() ARGSS=()
while IFS=$'\t' read -r grp name args; do
  [[ -z "${grp:-}" || "$grp" == \#* ]] && continue
  want_group "$grp" || continue
  NAMES+=("$name")
  ARGSS+=("${args//EMB/$EMB}")
done < pegasus/matrix.tsv

declare -a JOBS=()
for fold in $(expand_folds "$FOLDS"); do
  for seed in ${SEEDS//,/ }; do
    for i in "${!NAMES[@]}"; do
      JOBS+=("${NAMES[$i]}__f${fold}_s${seed}|${ARGSS[$i]}|$fold|$seed")
    done
  done
done

echo "队列共 ${#JOBS[@]} 个任务 (GROUP=$GROUP FOLDS=$FOLDS SEEDS=$SEEDS EPOCHS=$EPOCHS)"
if [[ "$DRYRUN" == "1" ]]; then
  for j in "${JOBS[@]}"; do echo "  ${j%%|*}"; done
  exit 0
fi

# ---- 双卡作业队列:某张卡空出来就取下一个任务 ----
declare -A GPU_PID=()
idx=0
while (( idx < ${#JOBS[@]} )) || (( ${#GPU_PID[@]} > 0 )); do
  for gpu in $GPUS; do
    pid=${GPU_PID[$gpu]:-}
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      unset "GPU_PID[$gpu]"; pid=""
    fi
    if [[ -z "$pid" ]] && (( idx < ${#JOBS[@]} )); then
      IFS='|' read -r tag args fold seed <<< "${JOBS[$idx]}"
      echo "[$(date +%H:%M:%S)] GPU$gpu ← $tag  ($((idx+1))/${#JOBS[@]})"
      $PY project/main.py $args \
        paths.root_path="$ROOT" \
        train.experiment="$tag" \
        train.folds="[$fold]" \
        train.seed="$seed" \
        train.max_epochs="$EPOCHS" \
        train.precision="$PRECISION" \
        train.gpu_num="$gpu" \
        data.num_workers="$WORKERS" \
        > "$OUT/$tag.log" 2>&1 &
      GPU_PID[$gpu]=$!
      idx=$((idx+1))
      sleep 5
    fi
  done
  sleep 15
done

echo "[$(date +%H:%M:%S)] 矩阵全部完成,共 ${#JOBS[@]} 个任务"

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

# ---- 故障防护 ----
# 2026-07-27 的教训:GPU0 掉出 PCI 总线后,派上去的任务全部秒崩,而调度器只看到
# "卡空了" 就立刻派下一个,几分钟内就能把整个队列烧成失败。加两道闸:
#   1. 派任务前探测该卡,探测不通就把它移出轮转
#   2. 任务在 MIN_RUNTIME 秒内退出算"秒崩",连续 MAX_FAST_FAIL 次就中止全队列
MIN_RUNTIME=${MIN_RUNTIME:-180}
MAX_FAST_FAIL=${MAX_FAST_FAIL:-3}

gpu_alive() {
  # 必须真正初始化 CUDA 才算数:GPU0 掉线后 nvidia-smi -i 1 仍能返回卡名,
  # 但驱动枚举已坏,torch.cuda 看到 0 个设备,派上去的任务照样秒崩。
  nvidia-smi -i "$1" --query-gpu=name --format=csv,noheader >/dev/null 2>&1 || return 1
  CUDA_VISIBLE_DEVICES="$1" timeout 60 "$PY" -c "
import sys, torch
sys.exit(0 if torch.cuda.device_count() > 0 and torch.zeros(8, device='cuda').sum().item() == 0 else 1)
" >/dev/null 2>&1
}

declare -a ACTIVE_GPUS=()
for gpu in $GPUS; do
  if gpu_alive "$gpu"; then
    ACTIVE_GPUS+=("$gpu")
  else
    echo "⚠ GPU$gpu 探测失败,已移出轮转"
  fi
done
if (( ${#ACTIVE_GPUS[@]} == 0 )); then
  echo "✗ 没有可用 GPU,中止。先检查 nvidia-smi 与驱动状态。"
  exit 1
fi
echo "可用 GPU: ${ACTIVE_GPUS[*]}"

# ---- 作业队列:某张卡空出来就取下一个任务 ----
declare -A GPU_PID=() GPU_TAG=() GPU_START=()
declare -a failed=()
fast_fail=0
idx=0
while (( idx < ${#JOBS[@]} )) || (( ${#GPU_PID[@]} > 0 )); do
  for gpu in "${ACTIVE_GPUS[@]}"; do
    pid=${GPU_PID[$gpu]:-}

    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      elapsed=$(( $(date +%s) - ${GPU_START[$gpu]} ))
      if (( elapsed < MIN_RUNTIME )); then
        fast_fail=$((fast_fail+1))
        failed+=("${GPU_TAG[$gpu]}")
        echo "⚠ ${GPU_TAG[$gpu]} 仅运行 ${elapsed}s 就退出(第 $fast_fail 次秒崩)"
        if ! gpu_alive "$gpu"; then
          echo "⚠ GPU$gpu 已失联,移出轮转"
          ACTIVE_GPUS=($(printf '%s\n' "${ACTIVE_GPUS[@]}" | grep -vx "$gpu"))
        fi
        if (( fast_fail >= MAX_FAST_FAIL )); then
          echo "✗ 连续 $fast_fail 次秒崩,中止队列以免空转烧掉剩余任务"
          exit 1
        fi
      else
        fast_fail=0
      fi
      unset "GPU_PID[$gpu]"; pid=""
    fi

    if [[ -z "$pid" ]] && (( idx < ${#JOBS[@]} )); then
      if ! gpu_alive "$gpu"; then
        echo "⚠ GPU$gpu 派发前探测失败,移出轮转"
        ACTIVE_GPUS=($(printf '%s\n' "${ACTIVE_GPUS[@]}" | grep -vx "$gpu"))
        continue
      fi
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
      GPU_TAG[$gpu]=$tag
      GPU_START[$gpu]=$(date +%s)
      idx=$((idx+1))
      sleep 5
    fi
  done

  if (( ${#ACTIVE_GPUS[@]} == 0 )); then
    echo "✗ 所有 GPU 均已失联,中止。已派发 $idx/${#JOBS[@]} 个任务。"
    exit 1
  fi
  sleep 15
done

if (( ${#failed[@]} > 0 )); then
  echo "[$(date +%H:%M:%S)] 队列结束:${#JOBS[@]} 个任务中 ${#failed[@]} 个秒崩"
  printf '  ✗ %s\n' "${failed[@]}"
  exit 1
fi
echo "[$(date +%H:%M:%S)] 矩阵全部完成,共 ${#JOBS[@]} 个任务"

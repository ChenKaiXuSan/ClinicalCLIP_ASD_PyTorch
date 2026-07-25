#!/bin/bash
# 本机 2×2 对照:概念来源(可学习 / CLIP 文本) × 区域消融(正常 / 打乱)
#
# 每张卡同时只跑 1 个任务,分两批。原因:batch 是"一条视频的全部 gait 段",
# 最长的视频有 838 帧 → 28 段,单任务显存峰值可达 31GB,两个任务挤一张 48GB
# 的卡必然 OOM(实测 A4 就是这么挂的)。
set -u

REPO=/home/kaixu_chen/asd/ClinicalCLIP_ASD_PyTorch
PY=/home/kaixu_chen/miniforge3/envs/asd/bin/python
ROOT=/mnt/data/xchen/asd_data
EMB=$ROOT/concepts/clip_vit_b32.pt
OUT=${OUT:-$REPO/logs/run_concept}

EPOCHS=${EPOCHS:-30}
FOLD=${FOLD:-0}
WORKERS=${WORKERS:-10}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT"
cd "$REPO"

launch() {   # 名称 GPU 概念向量 打乱区域
  local NAME=$1 GPU=$2 EMBED=$3 SHUF=$4
  local ARGS=(
    model.backbone=concept
    paths.root_path="$ROOT"
    train.experiment="$NAME"
    train.folds="[$FOLD]"
    train.max_epochs="$EPOCHS"
    train.gpu_num="$GPU"
    model.shuffle_region="$SHUF"
    data.num_workers="$WORKERS"
  )
  [[ "$EMBED" != "none" ]] && ARGS+=(model.concept_text_embedding="$EMBED")
  echo "[$(date +%H:%M:%S)] 启动 $NAME (GPU $GPU, shuffle=$SHUF)"
  "$PY" project/main.py "${ARGS[@]}" > "$OUT/$NAME.log" 2>&1 &
}

echo "=== 第 1 批 ==="
launch A1_learned            0 none  false
launch A3_cliptext           1 "$EMB" false
wait
echo "[$(date +%H:%M:%S)] 第 1 批结束"

echo "=== 第 2 批 ==="
launch A2_learned_shuffled   0 none  true
launch A4_cliptext_shuffled  1 "$EMB" true
wait
echo "[$(date +%H:%M:%S)] 第 2 批结束"

echo "全部完成"

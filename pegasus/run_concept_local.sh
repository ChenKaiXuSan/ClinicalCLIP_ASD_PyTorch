#!/bin/bash
# 本机 2×2 对照:概念来源(可学习 / CLIP 文本) × 区域消融(正常 / 打乱)
# 两张 A6000 各跑 2 个任务。单任务约 5.6GB 显存,48GB 卡放得下。
set -u

REPO=/home/kaixu_chen/asd/ClinicalCLIP_ASD_PyTorch
PY=/home/kaixu_chen/miniforge3/envs/asd/bin/python
ROOT=/mnt/data/xchen/asd_data
EMB=$ROOT/concepts/clip_vit_b32.pt
OUT=${OUT:-$REPO/logs/run_concept}

EPOCHS=${EPOCHS:-30}
FOLD=${FOLD:-0}
WORKERS=${WORKERS:-6}

mkdir -p "$OUT"
cd "$REPO"

# 名称                          GPU  概念向量  打乱区域
JOBS=(
  "A1_learned                    0    none      false"
  "A2_learned_shuffled           0    none      true"
  "A3_cliptext                   1    $EMB      false"
  "A4_cliptext_shuffled          1    $EMB      true"
)

for job in "${JOBS[@]}"; do
  read -r NAME GPU EMBED SHUF <<< "$job"

  ARGS=(
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

  echo "启动 $NAME  (GPU $GPU, shuffle=$SHUF, 概念=$([[ $EMBED == none ]] && echo 可学习 || echo CLIP文本))"
  nohup "$PY" project/main.py "${ARGS[@]}" > "$OUT/$NAME.log" 2>&1 &
  echo "  pid $!"
  sleep 5   # 错开启动,避免同时抢 torch.hub 缓存
done

wait
echo "全部任务结束"

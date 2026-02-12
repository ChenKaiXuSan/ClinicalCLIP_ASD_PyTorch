#!/bin/bash
#PBS -A SSR
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N clip_map_guided_c
#PBS -t 0-2
#PBS -o logs/pegasus/run_map_guided_c.log
#PBS -e logs/pegasus/run_map_guided_c_err.log

set -e

if [[ -n "$PBS_O_WORKDIR" ]]; then
  cd "$PBS_O_WORKDIR"
fi

mkdir -p logs/pegasus

source activate /home/SSR/luoxi/miniconda3/envs/multiview-video-cls

echo "Current working directory: $(pwd)"
echo "Current Python: $(which python)"

ROOT_PATH=""
VIDEO_PATH=""
INFO_PATH=""

EXP_NAMES=(
  "C1_channel_gate"
  "C2_weighted_pool"
  "C3_sigmoid_gate"
)

EXP_ARGS=(
  "model.map_guided=true model.map_guided_type=channel loss.clip_weight=1.0 model.lambda_token=0"
  "model.map_guided=true model.map_guided_type=weighted_pool loss.clip_weight=1.0 model.lambda_token=0"
  "model.map_guided=true model.map_guided_type=spatiotemporal model.map_guided_sigmoid_gate=true loss.clip_weight=1.0 model.lambda_token=0"
)

IDX=${PBS_SUBREQNO:-0}
EXP_NAME=${EXP_NAMES[$IDX]}
EXP_ARG=${EXP_ARGS[$IDX]}

echo "Run index: $IDX"
echo "Experiment: $EXP_NAME"
echo "Overrides: $EXP_ARG"

EXTRA_PATHS=""
if [[ -n "$ROOT_PATH" ]]; then
  EXTRA_PATHS+=" paths.root_path=${ROOT_PATH}"
fi
if [[ -n "$VIDEO_PATH" ]]; then
  EXTRA_PATHS+=" paths.video_path=${VIDEO_PATH}"
fi
if [[ -n "$INFO_PATH" ]]; then
  EXTRA_PATHS+=" paths.info_path=${INFO_PATH}"
fi

python -m project.main \
  model.backbone=clip \
  train.experiment=${EXP_NAME} \
  ${EXP_ARG} \
  ${EXTRA_PATHS}

#!/bin/bash
#PBS -A SSR
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N clip_semantics_d
#PBS -o logs/pegasus/run_semantics_d.log
#PBS -e logs/pegasus/run_semantics_d_err.log

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

EXP_NAME="D_full_for_retrieval"
EXP_ARG="model.map_guided=true model.map_guided_type=spatiotemporal loss.clip_weight=1.0 model.lambda_token=0"

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

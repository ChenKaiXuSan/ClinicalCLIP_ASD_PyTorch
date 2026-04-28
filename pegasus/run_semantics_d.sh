#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N clip_single_compare
#PBS -o logs/pegasus/run_semantics_d.log
#PBS -e logs/pegasus/run_semantics_d_err.log

set -e

if [[ -n "$PBS_O_WORKDIR" ]]; then
  cd "$PBS_O_WORKDIR"
fi

# === 切换到作业提交目录 ===
cd /work/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch

mkdir -p logs/pegasus

source activate /home/SKIING/chenkaixu/miniconda3/envs/clip

echo "Current working directory: $(pwd)"
echo "Current Python: $(which python)"

ROOT_PATH="/work/SKIING/chenkaixu/data/asd_dataset"
VIDEO_PATH="/work/SKIING/chenkaixu/data/asd_dataset/segmentation_dataset_512"
INFO_PATH="/work/SKIING/chenkaixu/data/asd_dataset/clinical_CLIP_dataset"

# Single-experiment selection (same mapping as run_compare.sh)
# You can override by exporting EXP_IDX before qsub, e.g.:
#   qsub -v EXP_IDX=0 pegasus/run_semantics_d.sh
#   qsub -v EXP_IDX=6 pegasus/run_semantics_d.sh
EXP_NAMES=(
  "B1_clip_only"
  "B2_map_only"
  "B3_full"
  "B4_full_token"
  "C1_channel_gate"
  "C2_weighted_pool"
  "C3_sigmoid_gate"
)

EXP_ARGS=(
  "model.map_guided=false loss.clip_weight=1.0 model.lambda_token=0"
  "model.map_guided=true model.map_guided_type=spatiotemporal loss.clip_weight=0 model.lambda_token=0"
  "model.map_guided=true model.map_guided_type=spatiotemporal loss.clip_weight=1.0 model.lambda_token=0"
  "model.map_guided=true model.map_guided_type=spatiotemporal loss.clip_weight=1.0 model.lambda_token=0.1"
  "model.map_guided=true model.map_guided_type=channel loss.clip_weight=1.0 model.lambda_token=0"
  "model.map_guided=true model.map_guided_type=weighted_pool loss.clip_weight=1.0 model.lambda_token=0"
  "model.map_guided=true model.map_guided_type=spatiotemporal model.map_guided_sigmoid_gate=true loss.clip_weight=1.0 model.lambda_token=0"
)

EXP_IDX=${EXP_IDX:-2}

if [[ ${EXP_IDX} -lt 0 || ${EXP_IDX} -ge ${#EXP_NAMES[@]} ]]; then
  echo "Invalid EXP_IDX=${EXP_IDX}. Valid range: 0..$(( ${#EXP_NAMES[@]} - 1 ))"
  exit 1
fi

EXP_NAME=${EXP_NAMES[$EXP_IDX]}
EXP_ARG=${EXP_ARGS[$EXP_IDX]}

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

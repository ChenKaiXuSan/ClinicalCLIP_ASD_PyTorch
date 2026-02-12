#!/bin/bash
#PBS -A SSR
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N clip_compare
#PBS -t 0-6
#PBS -o logs/pegasus/run_compare.log
#PBS -e logs/pegasus/run_compare_err.log

set -e

# Move to submission directory if available
if [[ -n "$PBS_O_WORKDIR" ]]; then
	cd "$PBS_O_WORKDIR"
fi

# === 切换到作业提交目录 ===
cd /work/SSR/share/code/ClinicalCLIP_ASD_PyTorch
mkdir -p logs/pegasus

# Activate conda environment (edit if needed)
source activate /home/SSR/luoxi/miniconda3/envs/clip

echo "Current working directory: $(pwd)"
echo "Current Python: $(which python)"

# Optional overrides (edit for your cluster)
ROOT_PATH="/work/SSR/share/data/asd_dataset"
VIDEO_PATH="/work/SSR/share/data/asd_dataset/segmentation_dataset_512"
INFO_PATH="/work/SSR/share/data/asd_dataset/clinical_CLIP_dataset"

# Experiment mapping (PBS array id -> experiment)
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

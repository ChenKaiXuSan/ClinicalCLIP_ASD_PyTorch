#!/bin/bash
#PBS -A HP260146
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=06:00:00
#PBS -N cclip_koa
#PBS -o logs/pegasus/koa_pose_out.log
#PBS -e logs/pegasus/koa_pose_err.log

# KOA-PD-NM 的预处理,数组作业,每个子作业跑 1/NCHUNK 的视频:
#   STAGE=detect  YOLOv8-pose 找行走者 + 人物居中裁剪(clip 环境)
#   STAGE=sam3d   SAM-3D-Body 3D 关键点(sam_3d_body 环境)
#   qsub -t 0-3 -v STAGE=detect,NCHUNK=4 pegasus/koa_pose_job.sh
cd "${CLINICALCLIP_REPO_ROOT:-/work/1/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch/.claude/worktrees/vlm}" || exit 1
mkdir -p logs/pegasus
KOA="${KOA:-/work/1/SKIING/chenkaixu/data/koa_pd_nm}"
NCHUNK=${NCHUNK:-4}
K=${PBS_SUBREQNO:-0}
exec > "logs/pegasus/koa_${STAGE:?需要 STAGE}_${K}.log" 2>&1      # 数组作业:每个子作业写自己的日志
echo "节点 $(hostname)  阶段 ${STAGE:?需要 STAGE}  子作业 $K / $NCHUNK  $(date '+%F %T')"
if [ "$STAGE" = detect ]; then
    source pegasus/setup_env.sh
    python analysis/koa/detect.py --video-root "$KOA/video" --out "$KOA/proc" --chunk "$K/$NCHUNK" ${LIMIT:+--limit $LIMIT} 2>&1 | grep --line-buffered -vE "Warning|warn"
else
    module load intelpython/2022.3.1
    source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
    conda deactivate
    conda activate /home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body
    export HF_HUB_OFFLINE=1
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    python analysis/koa/run_sam3d.py --video-root "$KOA/video" --proc "$KOA/proc" --chunk "$K/$NCHUNK" ${LIMIT:+--limit $LIMIT} 2>&1 | grep --line-buffered -vE "Warning|warn|^#######"
fi
echo "[done] 子作业 $K  $(date '+%F %T')"

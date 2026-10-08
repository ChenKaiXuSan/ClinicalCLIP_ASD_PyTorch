#!/bin/bash
#PBS -A HP260146
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=04:00:00
#PBS -N cclip_bout
#PBS -o logs/pegasus/bout_out.log
#PBS -e logs/pegasus/bout_err.log

# 设计 B 第一步:在侧视长录像上跑 YOLOv8-pose,逐次经过量躯干角(analysis/bout_trunk.py)。
# 数组作业:清单 logs/bout_trunk/list.tsv 按 NCHUNK 份切,每个子作业跑一份。
#   qsub -t 0-3 -v NCHUNK=4 pegasus/bout_trunk_job.sh
cd "${CLINICALCLIP_REPO_ROOT:-/work/1/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch/.claude/worktrees/vlm}" || exit 1
source pegasus/setup_env.sh
LIST=logs/bout_trunk/list.tsv
NCHUNK=${NCHUNK:-4}
K=${PBS_SUBREQNO:-0}
W=/work/1/SKIING/chenkaixu/pydeps/yolo/${WEIGHTS:-yolov8m-pose.pt}
echo "节点 $(hostname)  子作业 $K / $NCHUNK  权重 $W  $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
n=0
while IFS=$'\t' read -r tag video stride; do
    [ -z "$tag" ] && continue
    if [ $((n % NCHUNK)) -eq "$K" ]; then
        if [ -s "logs/bout_trunk/$tag.json" ]; then
            echo "[skip] $tag"
        else
            echo "[run ] $tag  stride=$stride  $(date '+%T')"
            python analysis/bout_trunk.py --video "$video" --out logs/bout_trunk --weights "$W" \
                --stride "$stride" --imgsz 960 --device cuda:0 --plot --tag "$tag" 2>&1 | grep -vE "Warning|warn"
        fi
    fi
    n=$((n + 1))
done < "$LIST"
echo "[done] 子作业 $K  $(date '+%F %T')"

#!/bin/bash
#PBS -A HP260146
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=04:00:00
#PBS -N cclip_cam
#PBS -o logs/pegasus/attn_gradcam_out.log
#PBS -e logs/pegasus/attn_gradcam_err.log

# 论文里"定位监督的注意力 vs. Grad-CAM"的对照,与主表同一个视频基线、同一批种子:
# 对 <TAG>__f<k>_s<SEED> 的验证集最佳 checkpoint 做 Grad-CAM,与 uniform / random / center 三个参照一起,
# 在该折 test 的前 LIMIT 条视频上算对齐度(analysis/eval_attention_alignment.py,与 2026-09 的口径相同)。
#   qsub -v SEED=42 pegasus/attn_gradcam_job.sh          默认 TAG=B0_3dcnn_e50, LIMIT=60
# 结果:logs/attn_align/<TAG>_s<SEED>_f<k>.log;汇总见 analysis/paper_v2.py。
cd "${CLINICALCLIP_REPO_ROOT:-/work/1/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch/.claude/worktrees/vlm}" || exit 1
mkdir -p logs/pegasus logs/attn_align
source pegasus/setup_env.sh
DATA="${CLINICALCLIP_DATA_ROOT:-/work/1/SKIING/chenkaixu/data/asd_dataset}"
TAG="${TAG:-B0_3dcnn_e50}"; SEED="${SEED:?需要 SEED}"; LIMIT="${LIMIT:-60}"
echo "节点 $(hostname)  $TAG s$SEED  $(date '+%F %T')"
for k in 0 1 2 3 4; do
    run=$(find "logs/train/${TAG}__f${k}_s${SEED}" -type d -path "*/checkpoint/${k}" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    ckpt=$(ls "$run"/*.ckpt 2>/dev/null | grep -v last.ckpt | awk -F- '{print $NF" "$0}' | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -z "$ckpt" ]; then echo "[缺] fold $k 没有 checkpoint"; continue; fi
    echo "fold $k: $ckpt"
    python analysis/eval_attention_alignment.py --root-path "$DATA" --fold "$k" --limit "$LIMIT" --ckpt "$ckpt" 2>&1 \
        | grep --line-buffered -vE "Warning|warn|已评估" > "logs/attn_align/${TAG}_s${SEED}_f${k}.log"
    grep -E "^\s+(uniform|random|center|gradcam)" "logs/attn_align/${TAG}_s${SEED}_f${k}.log"
done
echo "[done] $(date '+%F %T')"

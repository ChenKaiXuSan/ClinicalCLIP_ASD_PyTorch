#!/bin/bash

# t-SNE 生成脚本
# 在适当的环境中运行 t-SNE 可视化

set -e

# 检查是否指定了环保境名称
ENV_NAME=${1:-"clip"}

echo "=== ClinicalCLIP t-SNE 可视化生成 ==="
echo "使用环境: $ENV_NAME"
echo ""

# 激活环境
source /home/SSR/luoxi/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

echo "当前环境: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Python: $(python --version)"
echo ""

# 切换到项目目录
cd /work/SSR/share/code/ClinicalCLIP_ASD_PyTorch

# 创建输出目录
mkdir -p /work/SSR/share/code/ClinicalCLIP_ASD_PyTorch/logs/tsne_results

# 运行脚本
echo "开始生成 t-SNE 可视化..."
echo ""

python analysis/tsne.py \
    --train-root /work/SSR/share/code/ClinicalCLIP_ASD_PyTorch/logs/train \
    --output-root /work/SSR/share/code/ClinicalCLIP_ASD_PyTorch/logs/tsne_results \
    --embed-type both \
    --dpi 300 \
    --perplexity 30 \
    --n-iter 1000 \
    --random-seed 42

echo ""
echo "=== t-SNE 生成完成！==="
echo "输出路径: /work/SSR/share/code/ClinicalCLIP_ASD_PyTorch/logs/tsne_results"
echo ""
echo "生成的文件:"
ls -lh /work/SSR/share/code/ClinicalCLIP_ASD_PyTorch/logs/tsne_results/*.png 2>/dev/null || echo "未找到 PNG 文件"

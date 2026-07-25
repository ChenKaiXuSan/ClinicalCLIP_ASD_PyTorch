#!/bin/bash
# 生成 M1_concept_cliptext 需要的 CLIP 文本概念向量。
#
# 必须在**登录节点**跑:计算节点没有外网,拿不到 HuggingFace 权重。
# 产物 (R x dim, 行序同 med_attn_map.REGIONS) 存到 $DATA_ROOT/concepts/clip_vit_b32.pt,
# matrix_job.sh 里 matrix.tsv 的 EMB 占位符就替换成这个路径。
#
# 用法:
#   bash pegasus/prepare_concepts.sh
#   MODEL=openai/clip-vit-large-patch14 OUT=/path/x.pt bash pegasus/prepare_concepts.sh

set -euo pipefail

REPO_ROOT="${CLINICALCLIP_REPO_ROOT:-/work/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch}"
DATA_ROOT="${CLINICALCLIP_DATA_ROOT:-/work/SKIING/chenkaixu/data/asd_dataset}"
MODEL="${MODEL:-openai/clip-vit-base-patch32}"
OUT="${OUT:-${DATA_ROOT}/concepts/clip_vit_b32.pt}"

cd "${REPO_ROOT}"
source pegasus/setup_env.sh

if ! python -c "import transformers" 2>/dev/null; then
    echo "conda 环境里没有 transformers,正在安装(只影响 ${CLINICALCLIP_CONDA_ENV:-clip} 这个环境)"
    pip install --no-input "transformers>=4.40,<5" || {
        echo "ERROR: transformers 安装失败。没有它就只能跑可学习概念的 M0,M1 要从矩阵里去掉。" >&2
        exit 1
    }
fi

# 权重落到共享 HF_HOME,计算节点后续离线也能复用
python project/utils/build_concept_embedding.py --model "${MODEL}" --output "${OUT}"

echo
echo "完成: ${OUT}"
echo "提交矩阵时会自动带上 model.concept_text_embedding=${OUT}(见 matrix.tsv 的 EMB 占位符)"

#!/bin/bash
# VLM 组(V0-V3)的登录节点准备:下载 SigLIP 2 权重到共享 HF_HOME、编码概念 prompt。
#
# 必须在**登录节点**跑:计算节点没有外网。跑完之后再提交抽特征作业:
#   qsub pegasus/extract_job.sh
# 然后才能提交矩阵:
#   GROUP=vlm bash pegasus/submit_matrix.sh
#
# 用法:
#   bash pegasus/prepare_vlm.sh
#   MODEL=google/siglip2-base-patch16-224 VLM_TAG=siglip2_base_224 bash pegasus/prepare_vlm.sh

set -euo pipefail

REPO_ROOT="${CLINICALCLIP_REPO_ROOT:-/work/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch}"
DATA_ROOT="${CLINICALCLIP_DATA_ROOT:-/work/SKIING/chenkaixu/data/asd_dataset}"
MODEL="${MODEL:-google/siglip2-so400m-patch16-384}"
VLM_TAG="${VLM_TAG:-siglip2_so400m_224}"
OUT="${OUT:-${DATA_ROOT}/concepts/${VLM_TAG}.pt}"

cd "${REPO_ROOT}"
source pegasus/setup_env.sh

python -c "import transformers" 2>/dev/null || {
    echo "ERROR: conda 环境里没有 transformers(环境 clip 应当已有)。" >&2
    exit 1
}

echo "下载 ${MODEL} -> ${HF_HOME}"
python - <<EOF
from huggingface_hub import snapshot_download
p = snapshot_download("${MODEL}", allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
print("权重就绪:", p)
EOF

# 文本塔编码 5 条概念 prompt,行序同 med_attn_map.REGIONS
python project/utils/build_concept_embedding.py --model "${MODEL}" --encoder siglip --output "${OUT}"

echo
echo "完成: ${OUT}"
echo "下一步: qsub -v VLM_TAG=${VLM_TAG},MODEL=${MODEL} pegasus/extract_job.sh"

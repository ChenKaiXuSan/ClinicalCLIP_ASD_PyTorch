#!/bin/bash
# 在**登录节点**把所有 backbone 需要的预训练权重灌进 torch hub 缓存。
# 计算节点没有外网,权重不在 ~/.cache/torch 里的话作业会在建模型那一步直接挂掉:
#
#   B0_3dcnn / B4_clip_old / concept 系列 -> facebookresearch/pytorchvideo slow_r50
#   B1_2dcnn / B2_cnn_lstm                -> pytorch/vision resnet50
#   B3_pose                               -> 不需要预训练权重
#
# 用法:  bash pegasus/prepare_torchhub.sh

set -euo pipefail

REPO_ROOT="${CLINICALCLIP_REPO_ROOT:-/work/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch}"
cd "${REPO_ROOT}"
source pegasus/setup_env.sh

python - <<'PY'
import os
import torch

print("TORCH_HOME:", torch.hub.get_dir())

targets = [
    ("facebookresearch/pytorchvideo", "slow_r50", "B0_3dcnn / B4_clip_old / concept 系列"),
    ("pytorch/vision:v0.10.0", "resnet50", "B1_2dcnn / B2_cnn_lstm"),
]

for repo, entry, used_by in targets:
    print(f"\n=== {repo} :: {entry}   ({used_by}) ===")
    model = torch.hub.load(repo, entry, pretrained=True)
    n = sum(p.numel() for p in model.parameters())
    print(f"OK, 参数量 {n/1e6:.1f}M")

print("\n缓存内容:")
ckpt_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
for f in sorted(os.listdir(ckpt_dir)):
    size = os.path.getsize(os.path.join(ckpt_dir, f)) / 1e6
    print(f"  {f:40s} {size:8.1f} MB")
PY

echo
echo "完成。计算节点会直接命中 $(python -c 'import torch;print(torch.hub.get_dir())') 下的缓存。"

#!/bin/bash
# 在登录节点先把交叉验证划分生成好,再提交矩阵作业。两件事:
#
# 1. 折数对齐。cross_validation.py 只要发现 index_mapping/<class_num>/ 存在就直接加载,
#    train.fold 改了也不会重新划分 —— 缓存是几折,训的就是几折。实验矩阵统一用
#    **5 折**(train 1480 / val 410),折数对不上,train.folds=[0..4] 训出来的东西
#    和 docs/experiment_matrix.md 的表就不是一回事。
# 2. 避免竞态。70 个作业同时启动、缓存又不存在时会一起去写 index.json。
#
# 旧缓存不会被删除,而是移到 index_mapping/<class_num>.bak.<折数>fold/,随时可以换回来。
#
# 用法:
#   bash pegasus/prepare_index.sh          # 需要时重建成 5 折
#   FOLD=8 bash pegasus/prepare_index.sh   # 要别的折数
#   DRYRUN=1 bash pegasus/prepare_index.sh # 只报告现状,不改任何东西

set -euo pipefail

REPO_ROOT="${CLINICALCLIP_REPO_ROOT:-/work/SKIING/chenkaixu/code/ClinicalCLIP_ASD_PyTorch}"
DATA_ROOT="${CLINICALCLIP_DATA_ROOT:-/work/SKIING/chenkaixu/data/asd_dataset}"
FOLD="${FOLD:-5}"
CLASS_NUM="${CLASS_NUM:-3}"
DRYRUN="${DRYRUN:-0}"

cd "${REPO_ROOT}"
source pegasus/setup_env.sh

INDEX_DIR="${DATA_ROOT}/clinical_CLIP_dataset/index_mapping/${CLASS_NUM}"

current_folds=0
has_test=0
if [[ -f "${INDEX_DIR}/index.json" ]]; then
    current_folds=$(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "${INDEX_DIR}/index.json")
    python -c "
import json,sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if d and all('test' in v for v in d.values()) else 1)" "${INDEX_DIR}/index.json" && has_test=1
fi

echo "现有划分: ${INDEX_DIR} -> ${current_folds} 折 (期望 ${FOLD} 折), 三分格式: ${has_test}"

if [[ "${current_folds}" == "${FOLD}" && "${has_test}" == "1" ]]; then
    echo "折数与格式都已对上,无需重建。"
elif [[ "${DRYRUN}" == "1" ]]; then
    echo "DRYRUN: 会把 ${INDEX_DIR} 备份到 ${INDEX_DIR}.bak.${current_folds}fold 并按 ${FOLD} 折重建"
    exit 0
else
    if [[ "${current_folds}" != "0" ]]; then
        BACKUP="${INDEX_DIR}.bak.${current_folds}fold"
        if [[ -e "${BACKUP}" ]]; then
            BACKUP="${BACKUP}.$(date +%Y%m%d%H%M%S)"
        fi
        echo "备份旧划分: ${INDEX_DIR} -> ${BACKUP}"
        mv "${INDEX_DIR}" "${BACKUP}"
    fi

    CLINICALCLIP_DATA_ROOT="${DATA_ROOT}" FOLD="${FOLD}" python - <<'PY'
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "project"))

from hydra import compose, initialize_config_dir

from cross_validation import DefineCrossValidation

data_root = os.environ["CLINICALCLIP_DATA_ROOT"]
with initialize_config_dir(
    config_dir=os.path.join(os.getcwd(), "configs"), version_base=None
):
    config = compose(
        config_name="config",
        overrides=[
            f"paths.root_path={data_root}",
            f"train.fold={os.environ['FOLD']}",
        ],
    )

DefineCrossValidation(config)()
print(f"已重建: {config.paths.index_mapping}")
PY
fi

# 汇报每折规模与患者级泄漏,和 docs/experiment_matrix.md 的表对照
python - "${INDEX_DIR}/index.json" <<'PY'
import collections
import json
import sys
from pathlib import Path


def patients(paths):
    return {Path(p).name.split("-")[0] for p in paths}


split = json.load(open(sys.argv[1]))
leaks = 0
for k in sorted(split, key=int):
    tr, va, te = split[k]["train"], split[k]["val"], split[k]["test"]
    by_class = collections.Counter(Path(p).parent.name for p in te)
    print(
        f"fold {k}: train {len(tr):5d}  val {len(va):4d}  test {len(te):4d}"
        f"   test 分布 {dict(by_class)}"
    )
    ptr, pva, pte = patients(tr), patients(va), patients(te)
    for a, b, name in ((ptr, pva, "train/val"), (ptr, pte, "train/test"), (pva, pte, "val/test")):
        if a & b:
            leaks += 1
            print(f"    !! {name} 患者重叠 {len(a & b)} 人")

print("患者级泄漏:", "无" if leaks == 0 else f"{leaks} 处 —— 必须处理后再用")
PY

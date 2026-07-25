# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

ClinicalCLIP:用 CLIP 式对比学习将步态视频与医生标注注意力图对齐,进行临床疾病三分类(ASD / DHS / LCS_HipOA)。技术栈:PyTorch Lightning + Hydra + pytorchvideo。

## 常用命令

运行环境是 conda env `asd`(`/home/kaixu_chen/miniforge3/envs/asd`),base 环境缺 seaborn 等依赖。

```bash
# 训练全部 10 折(必须以脚本方式运行;project 内部是平铺 import,依赖 sys.path[0]=project/,
# 用 python -m project.main 会 ImportError)
python project/main.py

# CLI 覆盖 Hydra 配置(本机数据路径见下节)
python project/main.py model.backbone=clip paths.root_path=/mnt/data/xchen/asd_data train.experiment=my_exp

# 批量实验:PBS 超算数组作业(路径覆盖写法可参考这些脚本)
pegasus/run_ablation_b.sh / run_compare.sh / run_map_guided_c.sh / run_semantics_d.sh

# 论文指标计算与 embedding t-SNE 分析(用法见 analysis/计算方法说明.md)
python analysis/eval.py ...
analysis/run_tsne.sh
```

仓库无单元测试(原 `tests/` 全部失效,已于清理时删除)。

## 数据路径(易踩坑)

`configs/config.yaml` 默认 `paths.root_path: /workspace/data` 是旧 Docker 容器路径,本机不存在,**每次运行都要覆盖**:

- 本机:`/mnt/data/xchen/asd_data`
- Pegasus 超算:`/work/SKIING/chenkaixu/data/asd_dataset`(pegasus 脚本已内置)

数据根目录下 `clinical_CLIP_dataset/` 的子目录:`json_mix/<疾病>/*.json`(每段视频的元信息)、`video/`(MP4)、`doctor_result/doctor{1,2}.csv`(医生关注区域)、`seg_skeleton_pkl/whole_annotations.pkl`(骨架关键点)、`index_mapping/<class_num>/index.json`(交叉验证划分缓存)。

**缓存陷阱**:`index.json` 存的是生成时环境的绝对路径(当前是 `/workspace/data/...` 前缀)。`cross_validation.py` 只要发现缓存存在就直接加载,换机器必须先 sed 替换前缀或删缓存重建(重建会改变折划分,影响与旧实验的可比性)。

## 架构

流程图见 `docs/clinicalclip_pipeline.drawio`。核心链路:

1. **入口** `project/main.py`:Hydra 加载配置 → `DefineCrossValidation()` 返回 `{fold: {train: [json路径], val: [json路径]}}` → 逐折调用 `train()`,每折独立 fit + test。
2. **交叉验证** `project/cross_validation.py`:`StratifiedGroupKFold(K=10)` 按患者名分组防泄漏;过滤含 "HipOA" 的患者名(FIXME 数据不均衡);`magic_move` 在 train/val 间交换非 ASD 患者;结果缓存到 `index_mapping/`。
3. **数据** `project/dataloader/`:`WalkDataModule` → `whole_video_dataset.LabeledGaitVideoDataset`:读 json 元信息 → PyAV 全帧解码(视频路径按 `json_mix/` → `video/` 前缀重映射,不用 json 里写死的旧绝对路径)→ `MedAttnMap` 把医生 CSV 关注区域映射到 COCO 关键点、结合骨架 pkl 生成逐帧高斯热图 → `move_transform` 按秒切 gait 段、每段 `UniformTemporalSubsample(8)` + `Div255` + `Resize(224)` → `collate_fn` 把一条视频的所有段沿 batch 维拼接(所以 `batch_size=1` 时实际 batch 是段数)。
4. **模型** `project/models/clip_align.py` `VideoAttentionCLIP`:视频编码器 = slow_r50 预训练 ResNet3D,`map_guided_type` 三选一(`spatiotemporal` token 门控+注意力池化 / `channel` MLP 通道门控 / `weighted_pool` 仅加权池化);注意图编码器 = 1 通道 ResNet3D(不预训练);两路 `ClipProjectionHead` 输出归一化 embedding 做 InfoNCE(可学习温度);分类头按 `clip_classifier_source`(默认 video)取特征。
5. **训练** `project/trainer/train_clip_align.py` `CLIPAlignModule`:`loss = CE + clip_weight×InfoNCE + lambda_token×token能量对齐`;test 阶段额外计算 video↔attn 检索 R@1/R@5、对齐相似度 gap/corr,并把 embeddings 存为 `.pt`。
6. **backbone 分发**在 `main.py`:`model.backbone` = `clip` | `3dcnn` | `cnn_lstm` | `2dcnn` 对应 `trainer/` 下四个模块,后三者是基线对比,共用 `models/make_model.py`。
7. **输出**:`logs/train/<experiment>/<日期>/<时刻>/{tensorboard,csv,checkpoint/<fold>,embeddings,test_metrics.txt}`。

## 已知坑与过时文档

- **README 的用法章节是 954577e 重构前的旧内容,勿照搬**:`train.backbone`(实际是 `model.backbone`)、`gait_video_dataset.py`、`TemporalMix`、`scripts/eval.py`、"必须用 `python -m project.main`" 等均已失效或与现状相反。同样过时的 `.github/copilot-instructions.md` 已删除。
- `main.py` 中 `EarlyStopping` 已创建但**没有**加入 `callbacks` 列表,实际未生效,每折跑满 `max_epochs`。
- `train.attn_map=False` 分支不可用:引用不存在的 `dataset_idx['test']` 键,且 `collate_fn` 依赖 `attn_map` 键;保持默认 `True`。
- val/test 的 DataLoader 设了 `drop_last=True`。
- `config.yaml` 的 `model.model: "resnet"` 是残留配置,选择逻辑只看 `model.backbone`。
- `logs/` 约 40 GB,其中 54 个 `.ckpt` 占几乎全部;指标/CSV/TensorBoard/embeddings 仅 64 MB。

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
3. **数据** `project/dataloader/`:采样计划先行——`_plan_frame_index` 先按"每秒一段、每段均匀取 8 帧"算出最终保留的全局帧下标,再只解码这些帧(`_decode_selected` 跳过无用帧的 rgb24 转换)、只为这些帧生成注意力图(`MedAttnMap.build` 用可分离高斯做批量矩阵乘,直接在 224 上生成)。视频路径按 `json_mix/` → `video/` 前缀重映射,不用 json 里写死的旧绝对路径。`collate_fn` 把一条视频的所有段沿 batch 维拼接,所以 `batch_size=1` 时实际 batch 是段数。
   - 改这部分时注意保持采样语义:`_plan_frame_index` 必须与 `UniformTemporalSubsample` 的 `round(linspace(0, L-1, n))` 一致,段长不足时靠重复帧补齐。
   - `whole_video_dataset.LEGACY_ATTN_DIV255`:旧实现让 video 和 attn 共用同一个 Compose,其中 `Div255` 也除在了本就 [0,1] 的高斯图上。模型里 `downsample_attn_to_tokens` 会做 min-max 归一化基本抵消掉,但 `ChannelMapGuidedVideoEncoder` 直接把原始均值喂进 MLP,尺度有影响。目前保持与既有实验一致,重设计实验时可考虑改成 False。
4. **模型** `project/models/clip_align.py` `VideoAttentionCLIP`:视频编码器 = slow_r50 预训练 ResNet3D,`map_guided_type` 三选一(`spatiotemporal` token 门控+注意力池化 / `channel` MLP 通道门控 / `weighted_pool` 仅加权池化);注意图编码器 = 1 通道 ResNet3D(不预训练);两路 `ClipProjectionHead` 输出归一化 embedding 做 InfoNCE(可学习温度);分类头按 `clip_classifier_source`(默认 video)取特征。
5. **训练** `project/trainer/train_clip_align.py` `CLIPAlignModule`:`loss = CE + clip_weight×InfoNCE + lambda_token×token能量对齐`;test 阶段额外计算 video↔attn 检索 R@1/R@5、对齐相似度 gap/corr,并把 embeddings 存为 `.pt`。
6. **backbone 分发**在 `main.py`:`model.backbone` = `clip` | `3dcnn` | `cnn_lstm` | `2dcnn` 对应 `trainer/` 下四个模块,后三者是基线对比,共用 `models/make_model.py`。
7. **输出**:`logs/train/<experiment>/<日期>/<时刻>/{tensorboard,csv,checkpoint/<fold>,embeddings,test_metrics.txt}`。

## 已知坑与过时文档

- **README 的用法章节是 954577e 重构前的旧内容,勿照搬**:`train.backbone`(实际是 `model.backbone`)、`gait_video_dataset.py`、`TemporalMix`、`scripts/eval.py`、"必须用 `python -m project.main`" 等均已失效或与现状相反。同样过时的 `.github/copilot-instructions.md` 已删除。
- 训练**不使用** early stopping(按需求移除),每折固定跑满 `train.max_epochs`。
- `train.attn_map=False` 分支不可用:引用不存在的 `dataset_idx['test']` 键,且 `collate_fn` 依赖 `attn_map` 键;保持默认 `True`。
- `config.yaml` 的 `model.model: "resnet"` 是残留配置,选择逻辑只看 `model.backbone`。
- `logs/` 约 40 GB,其中 54 个 `.ckpt` 占几乎全部;指标/CSV/TensorBoard/embeddings 仅 64 MB。

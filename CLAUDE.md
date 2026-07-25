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
6. **backbone 分发**在 `main.py`:`model.backbone` = `concept` | `clip` | `3dcnn` | `cnn_lstm` | `2dcnn` 对应 `trainer/` 下五个模块,后三者是基线对比,共用 `models/make_model.py`。
7. **输出**:`logs/train/<experiment>/<日期>/<时刻>/{tensorboard,csv,checkpoint/<fold>,embeddings,test_metrics.txt}`。

## 两套架构的区别(重要)

`model.backbone=clip`(`models/clip_align.py`)把医生注意力图当作第二个模态,推理时也要输入,用批内 InfoNCE 对齐。`model.backbone=concept`(`models/clinical_concept.py`,架构图 `docs/clinical_concept_architecture.drawio`)把医生标注降格为训练期监督,推理只输入视频。

改动依据是这三条数据事实(脚本可复现,见 git 历史中的分析):

- 关注区域单独预测疾病的准确率 **66.7%,恰好等于多数类基线**——它不是能独立分类的模态,而是先验;`clip_classifier_source="attn"` 这条路是死的。
- 全库只有 **6 种不同的区域组合**(5 个区域,60% 是 lumbar_pelvis),批内 InfoNCE 会把临床标注完全相同的样本当负例推开,假负例极多。
- 两位医生 **仅 45.7% 一致**,标注该按软目标处理(`presence_for` 返回 0/0.5/1),不该像旧代码那样取并集把 0.5 抬成 1.0。

concept 架构的四项损失见 `models/clinical_concept.py`:分类 CE、区域存在性软 BCE、空间 grounding(KL,按软标签加权)、概念对比(负例是另外 4 个概念而非批内样本)。测试期额外产出 `region_f1` 与 `attn_iou` 两个可解释性指标——这是模型的**预测量**,可与留出的医生标注对分。

`model.shuffle_region=true` 是核心消融:沿区域维度逐样本置换,等价于"同一副骨架、换一个区域"。若指标不掉,说明增益来自姿态渲染而非临床知识,论文主张不成立。注意不能在 batch 维度打乱——`batch_size=1` 时一个 batch 全是同一条视频的 gait 段,区域标签本来就相同。

## 指标口径(容易误读)

`video_acc` / `video_f1_score` 走的是 torchmetrics 默认的 `average="macro"`,即**平衡准确率**(各类召回的均值)。多数类预测器只得 `1/C`(三分类为 0.333),**不是**类别占比。以 fold0 为例,val 集是 ASD 105 / DHS 57 / LCS_HipOA 17,micro 口径的多数类基线是 0.587,macro 口径是 0.333 —— 两个数差很多,论文里必须写明是哪一种,直接写 "accuracy" 会被质疑。

`analysis/compare_concept_runs.py` 会从 `save_helper` 存下的 `best_preds/*_pred.pt` 补算 macro / micro / 逐类召回和两种基线。补算而不是在训练里多记指标,是为了让先后批次跑的实验口径完全一致。

## 显存与并行

batch 是"一条视频的全部 gait 段",最长的视频 838 帧 → 28 段,**单任务显存峰值可达 31GB**,远高于用前几十条视频测出的 5.6GB。48GB 的卡上每卡只能放 1 个任务(实测两个并置必 OOM)。

GPU 实测利用率 86–94%,属算力受限而非数据受限(32 核负载仅 11.8),所以加 worker 或每卡多塞任务都不会提升总吞吐。要提速只能靠 `train.precision=bf16-mixed`,跑全部 10 折时值得开。

## 已知坑与过时文档

- **README 的用法章节是 954577e 重构前的旧内容,勿照搬**:`train.backbone`(实际是 `model.backbone`)、`gait_video_dataset.py`、`TemporalMix`、`scripts/eval.py`、"必须用 `python -m project.main`" 等均已失效或与现状相反。同样过时的 `.github/copilot-instructions.md` 已删除。
- 训练**不使用** early stopping(按需求移除),每折固定跑满 `train.max_epochs`。
- `train.attn_map=False` 分支不可用:引用不存在的 `dataset_idx['test']` 键,且 `collate_fn` 依赖 `attn_map` 键;保持默认 `True`。
- `config.yaml` 的 `model.model: "resnet"` 是残留配置,选择逻辑只看 `model.backbone`。
- `logs/` 约 40 GB,其中 54 个 `.ckpt` 占几乎全部;指标/CSV/TensorBoard/embeddings 仅 64 MB。

# VLM backbone:用冻结的视觉-语言模型替换 slow_r50

分支 `feat/vlm-backbone`。任务、划分、评估协议与主线完全一致(ASD vs non-ASD、
5 折按患者分组、患者级软投票、多种子 + McNemar),**只换 token 编码器**。

## 动机(全部来自 findings.md 的已有结论)

| 已有结论 | 这条线的假设 |
|---|---|
| 10 个配置分类统计等价,跨种子波动 0.125 大于任何配置差 | 冻结视觉塔 + 少量可训练参数,方差应明显更小 |
| M1 的冻结文本概念相对 M0 无优势;5 个 prompt 余弦相似度均值 0.877 | CLIP-B/32 文本配 slow_r50 视觉本就不在同一空间。改用同一 VLM 的双塔后概念与 token 天然对齐 |
| 概念对比损失近乎惰性(梯度量级差两个数量级) | 同上 |
| token 7×7,医生区域图 28×28 被降采样 16 倍 | SigLIP 2 patch16 @224 给 14×14,@384 给 24×24 |

医生注意力的用法**一字不改**:区域存在性软标签、空间 grounding KL、概念 prompt,
全部沿用 `models/clinical_concept.py`。

## 架构

```
视频 (B,3,T,224,224) ∈ [0,1]
   │  models/vlm_encoder.py  VLMTokenEncoder(接口同 ResNet3DTokenEncoder)
   ├─ siglip2:      逐帧 → SigLIP 2 vision tower → post-LN patch token → (B,1152,T,14,14)
   └─ internvideo2: 整段 → InternVideo2 stage2 → 去 cls → (B,1408,T,16,16)   [未在本机测试]
   │
   1×1 Conv3d → (B,256,T',h,w)
   │
ConceptCrossAttention / presence_head / classifier   ← 不变
```

配置(`configs/config.yaml` `model:` 段):

| 字段 | 含义 |
|---|---|
| `token_backbone` | `resnet3d`(默认,既有行为)/ `vlm` |
| `vlm_backend` | `siglip2` / `internvideo2` |
| `vlm_name` | HF 模型名;internvideo2 时为本地 `.pt` 路径 |
| `vlm_img_size` | 送入视觉塔的边长,空则用原生尺寸。dataset 输出 `data.img_size`,编码器内部再缩放 |
| `vlm_trainable_blocks` | 0 全冻结;N 解冻最后 N 个 block + 末端 LayerNorm |
| `vlm_forward_chunk` | 冻结前向的分块帧数(一条视频最多 28 段 × 8 帧 = 224 帧) |
| `data.feature_cache_dir` | 离线特征目录,非空则 dataset 不解码像素、直接返回 tokens |

`model.backbone=vlm_probe`(`trainer/train_vlm_probe.py`):冻结特征 → 均值池化 →
LayerNorm → Linear,无任何先验。

## 离线特征缓存

冻结的塔在 100 epoch × 5 折 × 3 种子里会把同一条视频编码 1500 次,so400m 前向比
slow_r50 训练还贵。`scripts/extract_vlm_features.py` 按训练时完全相同的采样计划
(`_plan_frame_index`,只依赖总帧数与 fps)抽一次,存
`<cache>/<video_name>.pt` = `{"tokens": fp16 (n_chunks, d, T, h, w)}`。
dataset 命中缓存后校验段数,不一致直接报错。

so400m @224:196 token × 1152 × 8 帧 × fp16 ≈ 3.6 MB/段,全库约 2800 段 ≈ 10 GB。
@384 约 3 倍。

## 实验组(pegasus/matrix.tsv `vlm` / `vlm_ft`)

| 名称 | 内容 | 回答 | 对应主线 |
|---|---|---|---|
| V0_vlm_probe | 冻结特征 + 线性头 | VLM 特征够不够;方差是否更小 | B0_3dcnn |
| V1_vlm_concept | concept + 冻结 VLM + 同一 VLM 文本概念 | 同空间概念下先验的收益 | M1 |
| V1b_vlm_concept_learned | 同上,可学习概念 | 文本概念 vs 可学习 | M0 |
| V1c_vlm_no_prior | V1 去掉 grounding + presence | 先验有没有收益 | A4 |
| V3_vlm_ft4 | 解冻最后 4 block,在线编码 | 冻结是不是瓶颈 | —— |
| V2(不训练) | `analysis/eval_vlm_zeroshot.py` | VLM 零样本能否定位医生区域 | attn_align 参照表 |

V0/V1 系列走缓存,单折约 10-20 分钟;V3 与 concept 系列同量级(2.5 小时/折)。

**判读**:V0 若与 B0_3dcnn 患者级统计等价且方差没降,这条线只剩 V2 的零样本
grounding 可讲。V1 与 V1c 的差才是"先验收益",不要拿 V1 和 M0 直接比。

## Pegasus 执行

```bash
# 登录节点:下载权重到 HF_HOME、编码概念 prompt
bash pegasus/prepare_vlm.sh
# GPU 节点:抽特征,约 1 小时
qsub pegasus/extract_job.sh
# 有了 concepts/siglip2_so400m_224.pt 与 vlm_cache/siglip2_so400m_224/manifest.json 之后
GROUP=vlm FOLDS=0 bash pegasus/submit_matrix.sh
GROUP=vlm SEEDS=42,1337,2024 bash pegasus/submit_matrix.sh
```

`VLM_TAG` / `EMB_VLM` / `CACHE` 环境变量可换模型或缓存目录,占位符在提交时替换。

## 生成式 VLM:Qwen3-VL(`vlm_backend=qwen3vl`)

SigLIP 是双塔,prompt 只能编码成向量做相似度。Qwen3-VL 是带语言模型的生成式 VLM,
prompt 是真正的指令:**同一段视频,不同指令得到不同的视频 token**。于是医生的关注
区域可以以**文本指令**进入模型(`models/vlm_prompts.py`),而不是热图 —— 这是与
ClinicalCLIP 主线本质不同的先验注入方式。

```
"Focus on the lumbar spine and pelvis ..."  +  视频 8 帧          ← 指令在前,视频在后
                    │
        Qwen3-VL(冻结, bf16)语言模型
                    │
   视频 token 位置的末层隐状态 → (B, 4096, 4, 14, 14)   [8B @448]   ← 已被指令调制
   回答位置的隐状态             → (B, 4096)  "pooled"               ← 看得到视频与指令
                    │
   concept 交叉注意力 / 探针(不变)
```

三个关键实现点:

- **指令必须在视频之前**。因果注意力下视频 token 只能看到前文;`_chat_text` 把
  text 放在 content 列表的 video 之前。`tests/smoke_qwen.py` 的 [3] 检查 clinical 与
  generic 指令下的 token 确实不同。
- **指令必须患者无关**。测试患者自己的医生标注绝不能进推理输入,否则是泄漏。
  预设 `generic`(只说是步态视频)/ `clinical`(按医生标注频率列出关注部位)/
  `region_<r>`(单区域,注意力分析用)/ `diagnose`(零样本诊断)。
- **预采样帧要带时间戳**。processor 走 `video_metadata`(fps = T,8 帧铺满 1 秒的
  gait 段)+ `do_sample_frames=False`;分辨率用 `size` 锁死为 `T×S×S`,S=448 给
  14×14 token(patch 16 × merge 2 = 32 px/token),temporal patch 2 → t' = 4。

缓存按 (模型, prompt) 分目录,占位符 `CACHE:<tag>`;训练时只读 manifest,
**不加载 8B 模型**(`VLMTokenEncoder(cache_manifest=...)`)。

### 实验组(`pegasus/matrix.tsv` `qwen`)

| 名称 | 内容 | 回答 |
|---|---|---|
| Q0_qwen_probe_generic / Q0b_..._clinical | 回答位置隐状态 + 线性头,只差指令 | **文本先验的干净消融** |
| Q0m / Q0bm | 同上,视频 token 均值 | 池化方式是否影响 |
| Q1_qwen_concept_clinical / Q1g_generic | concept 架构 + 指令调制 token(可学习概念) | 热图先验叠在文本先验上还有没有收益 |
| Q1c | Q1 去掉 grounding + presence | 只剩文本先验 |
| Q2(不训练) | `analysis/eval_qwen_attention.py` | VLM 被要求看某部位时注意力落在哪;与 M0 0.537 / Grad-CAM 0.135 并排 |
| Q3(不训练) | `analysis/eval_qwen_zeroshot_diag.py` | 零样本诊断患者级 AUC;预期接近 0.5,必须报 |

Q1 系列不用文本概念向量:LLM 隐状态与任何文本塔向量都不同空间,用可学习概念。

### 执行

```bash
# 登录节点(已完成):权重在 HF_HOME,Qwen/Qwen3-VL-{2B,8B}-Instruct
# GPU 节点:每种 prompt 一份缓存,8B @448 约 1.5 小时
qsub -v BACKEND=qwen3vl,MODEL=Qwen/Qwen3-VL-8B-Instruct,IMG_SIZE=448,PROMPT=generic,VLM_TAG=qwen3vl_8b_448_generic,CHUNK=16 pegasus/extract_job.sh
qsub -v BACKEND=qwen3vl,MODEL=Qwen/Qwen3-VL-8B-Instruct,IMG_SIZE=448,PROMPT=clinical,VLM_TAG=qwen3vl_8b_448_clinical,CHUNK=16 pegasus/extract_job.sh
GROUP=qwen FOLDS=0 bash pegasus/submit_matrix.sh
# 不训练的两项(GPU 节点,eager 注意力)
python analysis/eval_qwen_attention.py --root-path $DATA --model Qwen/Qwen3-VL-8B-Instruct --img-size 448 --fold 0
python analysis/eval_qwen_zeroshot_diag.py --root-path $DATA --model Qwen/Qwen3-VL-8B-Instruct --img-size 448 --fold 0
```

登录节点每用户内存上限 16GB,只能跑 2B 的冒烟测试(`tests/smoke_qwen.py`,fp32)。

### 精度与尺寸(2026-09-10 实测)

| | CPU(登录节点) | H100 PCIe |
|---|---|---|
| bf16 批量 vs 单段相对差 | 0.17 ~ 0.23(oneDNN bf16 经 28 层累积) | **0.0000** |
| fp32 | 0.0000 | 0.0000,8B @448 峰值显存 34 GB,2 段前向 1.7 s |

GPU 上 bf16 与 fp32 一致,2B/4B/8B 的缓存用 fp32 抽(已在跑),32B 用 bf16(权重 64 GB)。
尺寸曲线(`qwen_scale` 组)只比探针对 generic vs clinical:2B / 4B / 8B / 32B。
全库 1890 条视频约 11000 段(不是之前估的 2800),8B @448 每种 prompt 缓存约 70 GB。

## InternVideo2 后端的准备(未完成)

权重在 HF 上是 gated(`OpenGVLab/InternVideo2-Stage2_1B-224p-f4`,需登录并接受协议),
且不是 transformers 原生格式,要用官方仓库代码加载:

1. `huggingface-cli login`,在模型页接受协议,下载 `InternVideo2-stage2_1b-224p-f4.pt`
2. `git clone https://github.com/OpenGVLab/InternVideo`,`pip install einops timm`
   到 conda 环境 `clip`
3. `export INTERNVIDEO2_REPO=/path/InternVideo/InternVideo2/multi_modality`
4. `model.vlm_backend=internvideo2 model.vlm_name=/path/to/InternVideo2-stage2_1b-224p-f4.pt`

`_InternVideo2Vision` 按 stage2 1B 的公开超参显式构建、关闭 flash-attn,
但本机没有 token 与依赖,**未运行验证**;官方 `InternVideo2` 类的构造签名若与
所用版本不同需要对照调整。文本塔是 BERT-large,概念向量用
`build_concept_embedding.py --encoder bert --model bert-large-uncased`。

## 冒烟测试

```bash
HF_HOME=/work/SKIING/chenkaixu/hf_cache python tests/smoke_vlm.py \
    --root-path /work/SKIING/chenkaixu/data/asd_dataset
```

登录节点 CPU 上用 siglip2-base 约 2-3 分钟,覆盖编码器形状、在线/缓存两条路一致、
探针、真实视频抽特征 → 缓存 → dataset → collate → trainer 一步反传。

## 结果(2026-09-10,单种子 42,患者级 n=79,ASD 54 / non-ASD 25)

全部走 `analysis/patient_level_stats.py` 口径:5 折 test 拼成 79 个独立患者,段级概率按患者均值软投票,
macro = 平衡准确率(多数类基线 0.500),bootstrap 95% 区间,McNemar 精确检验。主线配置取同一种子。

### 主表:8B Qwen 组 vs 主线

| 配置 | backbone | 先验 | macro [95% CI] |
|---|---|---|---|
| B0_3dcnn | slow_r50 端到端 | 无 | **0.720** [0.606, 0.825] |
| A1_no_grounding | slow_r50 concept | 存在性 | 0.707 [0.597, 0.815] |
| M0_concept_learned | slow_r50 concept | 热图 + 存在性 | 0.655 [0.544, 0.764] |
| Q1_qwen_concept_clinical | Qwen3-VL-8B 冻结 token,concept | 指令 + 热图 + 存在性 | 0.615 [0.495, 0.731] |
| Q1g_qwen_concept_generic | 同上 | 热图 + 存在性 | 0.586 [0.466, 0.701] |
| Q0m_qwen_probe_generic_mean | 8B token 均值 + 线性头 | 无 | 0.584 [0.465, 0.701] |
| Q1c_qwen_concept_clinical_no_prior | 8B concept | 仅指令 | 0.575 [0.477, 0.677] |
| Q0bm_qwen_probe_clinical_mean | 8B token 均值 + 线性头 | 仅指令 | 0.573 [0.455, 0.690] |
| A4_no_prior | slow_r50 concept | 无 | 0.573 [0.459, 0.683] |
| Q0_qwen_probe_generic | 8B 回答位置 + 线性头 | 无 | 0.561 [0.491, 0.642] |
| Q0b_qwen_probe_clinical | 8B 回答位置 + 线性头 | 仅指令 | 0.517 [0.425, 0.612] |

McNemar(A 独对 : B 独对,p):

| 比较 | 回答的问题 | 结果 |
|---|---|---|
| Q1 vs Q1c | 热图先验叠在指令上有没有用 | 11 : 16,p = 0.44 |
| Q0b vs Q0 | 文本指令先验(探针) | 4 : 10,p = 0.18(方向为负) |
| Q1 vs Q1g | 文本指令先验(concept) | 6 : 4,p = 0.75 |
| B0 vs Q1g | 冻结 8B token 能否替代端到端 slow_r50 | 24 : 13,p = 0.10 |
| M0 vs Q1 | 同架构换 backbone | 17 : 8,p = 0.11 |

**结论**:
1. 冻结的 Qwen3-VL 特征在患者级上**没有超过**端到端微调的 slow_r50,所有 Qwen 配置都落在
   0.52–0.62,低于 B0 的 0.72,虽未达显著(p ≈ 0.10)但方向一致。
2. **文本指令先验没有收益**:探针上 clinical 反而低于 generic(p = 0.18),concept 上二者等价(p = 0.75)。
3. **训练方差没有降下来**:Q1g 五折段级平衡准确率 0.41 / 0.51 / 0.61 / 0.72 / 0.76,与主线同量级。
   "冻结塔降方差"的假设不成立 —— 方差来自 17 人一折的 test 集,不来自可训练参数量。
4. **grounding 监督在 Qwen token 上同样有效**:Q1 / Q1g 的 attn_align 0.39–0.47(五折),region_ap 0.67–0.95;
   去掉先验的 Q1c 掉到 0.02–0.08(= 均匀下界 0.068),region_ap 0.30–0.69。与主线 M0 0.537 / A4 0.122 同一格局,
   对齐绝对值略低于 slow_r50。

### 尺寸曲线(回答位置探针,generic / clinical)

| 尺寸 | generic | clinical |
|---|---|---|
| 2B | 0.534 | 0.523 |
| 4B | 0.552 | 0.564 |
| 8B | 0.561 | 0.517 |
| 32B(bf16) | 0.461 | 0.458 |

没有随尺寸上升的趋势,32B 低于随机。回答位置的隐状态不是可用的诊断特征;token 均值(Q0m 0.584)略好。

### 不训练的两项(8B)

| 项 | 结果 | 参照 |
|---|---|---|
| Q3 零样本诊断(79 人) | 患者级 AUC **0.444** [0.295, 0.591];阈值 0 下 0 个患者被判 ASD,平衡准确率 0.500 | 随机 0.5 |
| Q2 VLM 自身注意力 vs 医生区域(80 条视频) | attn_align **0.029**,逐区域 0.027–0.040 | 均匀 0.069,M0 0.537,Grad-CAM 0.135 |

通用 VLM 既不能零样本识别 ASD 步态,被要求"只看腰椎骨盆"时注意力也不落在腰椎骨盆上(比均匀还差)。
临床可解释性必须靠显式 grounding 监督,不能指望 VLM 自带。

### 实测开销(H100 PCIe)

| | 抽特征(全库 14075 段 @448,每种 prompt) | 训练(5 折 × 100 epoch,读缓存) |
|---|---|---|
| 2B fp32 | 49 min,43 GB | 每折 ~20 min |
| 4B fp32 | 85–92 min | ~25 min |
| 8B fp32 | 132–153 min,86 GB | ~30 min(35 任务并发时 IO 争抢) |
| 32B bf16 | 54–57 min,106 GB | ~40 min |

缓存目录 `vlm_cache/qwen3vl_{2b,4b,8b,32b}_448_{generic,clinical}` 合计约 550 GB,实验结束后可删。

### 数据备注

`json_mix/DHS/20170926_DHS_lat_V1-0002.json` 与 `0003.json` 内容相同、指向同一个视频文件,
全库 1890 条 json 只有 1889 个不同的 video_name。缓存按 video_name 存,两条 json 共用一份,段数一致,不影响训练。

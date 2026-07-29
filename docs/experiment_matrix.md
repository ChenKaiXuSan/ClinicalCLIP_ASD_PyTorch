# 实验矩阵

任务是 **ASD vs non-ASD 二分类**(`model.model_class_num: 2`)。三分类已放弃,
原因与证据见 [`why_binary.md`](why_binary.md)。

配置在 `pegasus/matrix.tsv`(唯一的真源,本机和超算共用)。执行有两条路:

| 环境 | 脚本 | 并行方式 |
|---|---|---|
| 本机双 A6000 | `pegasus/run_matrix.sh` | 双卡作业队列,某张卡空出来就取下一个任务 |
| Pegasus 超算 | `pegasus/submit_matrix.sh` | PBS 数组作业,**一个节点跑一个配置的一折** |

超算的完整操作步骤见文末「在 Pegasus 上执行」。

## 统一设定

**5 折交叉验证、每个实验 100 epochs、不使用 early stopping。** 配置里 `train.fold: 5`、
`train.max_epochs: 100`,`run_matrix.sh` 的默认值同步为 `FOLDS=0-4 EPOCHS=100`。

每折 train/val/test 三份、按患者分组互不相交,规模见文末「划分协议」。

## 单次耗时(实测)

| 类型 | fp32 | bf16-mixed |
|---|---|---|
| 视频类(concept / clip / 3dcnn / 2dcnn / cnn_lstm) | **8.3 小时** | **5.3 小时** |
| 姿态类(pose) | 约 0.7 小时 | —— |

bf16 实测 GPU 计算快 **1.56 倍**、显存 7.6→4.9 GB。由于 GPU 利用率本就 86–94%
(算力受限),端到端加速大体能兑现。

两张 A6000 各跑 1 个任务(每卡只能放 1 个:最长视频 838 帧 → 28 段,显存峰值可达 31GB)。

## 矩阵内容

### 基线(5 个)—— 确定下界,回答"新架构值不值"

| 名称 | 内容 | 为什么需要 |
|---|---|---|
| `B0_3dcnn` | slow_r50 + 线性头 | **最重要的基线**,没有它整篇文章立不住 |
| `B1_2dcnn` | 2D CNN 逐帧平均 | 弱基线 |
| `B2_cnn_lstm` | CNN + LSTM | 弱基线 |
| `B3_pose` | ST-GCN 纯骨架 | **关键混淆控制**:注意力图本就从骨架渲染,必须回答"只用姿态够不够" |
| `B4_clip_old` | 旧 CLIP 架构 | 与自己的前一版对比;它推理时需要医生标注,不可部署 |

### 主方法(2 个)

| 名称 | 内容 |
|---|---|
| `M0_concept_learned` | concept 架构 + 可学习概念 |
| `M1_concept_cliptext` | concept 架构 + 冻结的 CLIP 文本概念 |

### 消融(5 个)—— 哪个部件在起作用

| 名称 | 内容 | 判读 |
|---|---|---|
| `A0_shuffle_region` | 沿区域维置换 = 同一副骨架换个区域 | **核心消融**。不掉说明增益来自姿态而非临床知识,主张不成立 |
| `A1_no_grounding` | `grounding_weight=0` | 空间监督的贡献 |
| `A2_no_presence` | `presence_weight=0` | 区域预测头的贡献 |
| `A3_no_concept_loss` | `concept_weight=0` | 审阅已证实它对编码器影响很小(`‖∇P‖=5.11` vs `‖∇F‖=0.101`),此项用于证实,**论文里不宜当独立贡献** |
| `A4_no_prior` | grounding 与 presence 同时置零 | **隔离"架构改进"与"临床先验"**。若与 M0 差不多,卖点就不是先验而是架构,故事要改 |

### 标注者分歧(2 个)

两位医生仅 45.7% 一致,`D1_doctor1_only` / `D2_doctor2_only` 与 M0(软标签)对比,验证软标签设计是否真的更好。这也是论文的一个小创新点。

### 可解释性对照(不需训练)

`analysis/eval_attention_alignment.py` 一次跑出四个参照:`uniform` / `random` / `center` 三个下界,加上 `B0_3dcnn` 的 **Grad-CAM**。把 M0/M1 日志里的 `test/attn_align` 与之对比——若不明显高于 Grad-CAM,可解释性主张不成立。这是该主张最强的对照,必须做。

## 建议的执行阶段

| 阶段 | GROUP / FOLDS | 任务数 | 本机双卡耗时 (fp32 / bf16) | Pegasus 节点数 |
|---|---|---|---|---|
| ① 单折筛选 | `GROUP=all FOLDS=0` | 14 | 2.4 天 / 1.5 天 | 14 |
| ② 五折主表 | `GROUP=baseline,main` | 35 | 5.2 天 / 3.3 天 | 35 |
| ③ 五折消融 | `GROUP=ablation,annotator` | 35 | 6.1 天 / 3.9 天 | 35 |
| **全矩阵** | `GROUP=all` | **70** | **11.3 天 / 7.2 天** | **70** |
| ④ 三种子方差(可选) | `GROUP=main SEEDS=42,1337,2024` | 30 | 5.2 天 / 3.3 天 | 30 |

本机跑全矩阵是 **11.3 天**(fp32)或 **7.2 天**(bf16);超算上 70 个节点并发,墙钟时间只受排队影响。
两点建议:

1. **先跑阶段 ①**(单折,全部 14 个配置)。它的作用不是出结论,而是确认每个配置都能跑通、
   100 epochs 是否过拟合、哪些配置值得进全量。直接上全矩阵的风险是某个配置有问题、一周算力白费。
2. **全矩阵建议开 `PRECISION=bf16-mixed`**(超算脚本已是默认)。但整个矩阵必须用同一精度,不能混。

## 在 Pegasus 上执行

节点规格实测:**H100 PCIe 80GB / 48 核 / 单请求上限 24 小时**(队列硬上限,`qstat -Q -f gpu`)。
一折 100 epochs 在 A6000 上是 5.3 小时(bf16),H100 只会更快,所以**一个节点跑一折**留了充足余量。
一个批处理请求最多 150 个 sub-request,70 个任务一次提交得下。

计算节点**有外网**,走预设的 HTTP 代理(`http_proxy=http://10.120.96.1:8080`,DNS 直连不通但代理正常),
所以 torch.hub / HuggingFace 在节点上也能下载。即便如此,权重仍建议在登录节点预热一次
(`prepare_torchhub.sh`),避免 70 个作业同时穿代理下同一份权重。

### 一次性准备

```bash
bash pegasus/prepare_index.sh      # 建 5 折三分划分(旧缓存自动备份, 建完自查患者泄漏)
bash pegasus/prepare_concepts.sh   # M1 需要的 CLIP 文本概念向量(要装 transformers)
bash pegasus/prepare_torchhub.sh   # slow_r50 与 resnet50 预训练权重灌进 torch hub 缓存
qsub  pegasus/smoke_test.sh        # 一个节点上把 14 个配置各跑一个 batch,10 分钟暴露配置问题
```

`prepare_index.sh` 不是可选步骤。`cross_validation.py` 只要发现 `index_mapping/<类别数>/` 存在就直接加载,
`train.fold` 改了也不会重新划分 —— 缓存是几折,训的就是几折。`submit_matrix.sh` 提交前会同时校验
折数和「每折是否有独立 test」,对不上直接拒绝提交;旧格式缓存在加载时也会直接报错。

### 提交

```bash
GROUP=all FOLDS=0 bash pegasus/submit_matrix.sh    # ① 单折筛选,14 个节点
GROUP=baseline,main bash pegasus/submit_matrix.sh  # ② 五折主表,35 个节点
GROUP=all bash pegasus/submit_matrix.sh            # 全矩阵,70 个节点
DRYRUN=1 GROUP=all bash pegasus/submit_matrix.sh   # 只看清单不提交
```

提交脚本把作业清单固化到 `pegasus/queue/<时间戳>.part0.tsv`,数组作业按 `PBS_SUBREQNO` 取自己那一行。
清单和运行参数在提交那一刻就冻结了,之后再改 `matrix.tsv` 不影响已排队的作业。

**断点续跑**:每个任务成功后在 `logs/pegasus/matrix/done/<tag>.done` 落一个标记。
把同一条提交命令再敲一遍,已完成的会被剔除,只有失败/没跑到的重新排队;要强制全部重跑加 `FORCE=1`。

单个任务的实时日志在 `logs/pegasus/matrix/<tag>.log`(tag 形如 `M0_concept_learned_c2__f2_s42`)。

### 可解释性对照

```bash
qsub pegasus/run_attn_alignment.sh   # 需要 B0_3dcnn 至少训完一折(脚本自动找 checkpoint)
```

一次跑出 `uniform` / `random` / `center` 三个下界外加 Grad-CAM,不需要训练,约 2 小时。

## 读结果时必须注意

- `video_acc` / `video_f1_score` 是 **macro 平均(平衡准确率)**,二分类的多数类基线是 **0.5**,段级 micro 基线是 0.640。论文里写 "accuracy" 会被质疑。
- 指标算在**段**上,但有效样本量是**患者**(每折 test 17 人、non-ASD 仅 6 人)。必须报告患者数。
- `test/attn_align` 要和同时输出的 `test/attn_align_uniform` 一起看,两者之差(`attn_align_gain`)才是真正学到的对齐。
- `region_f1_any` / `region_f1_both` 是两种口径,另有免阈值的 `region_ap`。
- 汇总用 `analysis/compare_concept_runs.py`,它会从 `best_preds/*.pt` 补算 macro / micro / 逐类召回和两种基线。

## 划分协议(2026-07 修订)

每折 train/val/test 三份,按患者分组互不相交(实测 0 泄漏):

| fold | train | val | test | test 段数 ASD / non-ASD | test 患者数 |
|---|---|---|---|---|---|
| 0 | 1132 | 379 | 379 | 209 / 170 | 11 / 6 |
| 1 | 1133 | 379 | 378 | 208 / 170 | 10 / 5 |
| 2 | 1134 | 379 | 377 | 209 / 168 | 11 / 6 |
| 3 | 1134 | 379 | 377 | 209 / 168 | 11 / 6 |
| 4 | 1132 | 379 | 379 | 210 / 169 | 11 / 6 |

**non-ASD 侧每折 test 只有 5–6 个患者** —— 一个患者判错,macro 就动 8 个百分点。
任何单折结论都不足以支撑判断,必须看 5 折方差。

外层 `StratifiedGroupKFold(5)` 留出 test,内层 `StratifiedGroupKFold(4)` 把开发集切成
train/val。**val 只用来选 checkpoint,test 只用来报指标。**

修的是两个此前会让所有数字作废的问题:

1. **患者级泄漏**:`magic_move` 给每个非 ASD 患者在 train/val 之间对搬一个片段,导致
   5/5 折、46.8% 的验证样本来自训练见过的患者,且只发生在患者最少的两类,macro 被
   不对称地抬高。已移除。
2. **val 与 test 同批**:`data_loader` 的 test dataset 直接用 `dataset_idx['val']`,
   于是 checkpoint 按 val 选完再在同一批数据上测。现在是独立的第三份划分。

⚠ 这两条修复之前产出的所有结果都不可用,相关日志已删除。

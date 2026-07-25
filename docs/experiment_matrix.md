# 实验矩阵

配置在 `pegasus/matrix.tsv`,执行用 `pegasus/run_matrix.sh`(双卡作业队列,某张卡空出来就取下一个任务)。

## 统一设定

**5 折交叉验证、每个实验 100 epochs、不使用 early stopping。** 配置里 `train.fold: 5`、
`train.max_epochs: 100`,`run_matrix.sh` 的默认值同步为 `FOLDS=0-4 EPOCHS=100`。

5 折划分(按患者分组,`index_mapping/3/index.json`):

| fold | train | val |
|---|---|---|
| 0 | 1480 | 410 |
| 1 | 1510 | 380 |
| 2 | 1479 | 411 |
| 3 | 1506 | 384 |
| 4 | 1510 | 380 |

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

| 阶段 | 命令 | 任务数 | 双卡耗时 (fp32 / bf16) |
|---|---|---|---|
| ① 单折筛选 | `GROUP=all FOLDS=0 bash pegasus/run_matrix.sh` | 14 | 2.4 天 / 1.5 天 |
| ② 五折主表 | `GROUP=baseline,main bash pegasus/run_matrix.sh` | 35 | 5.2 天 / 3.3 天 |
| ③ 五折消融 | `GROUP=ablation,annotator bash pegasus/run_matrix.sh` | 35 | 6.1 天 / 3.9 天 |
| **全矩阵** | `GROUP=all bash pegasus/run_matrix.sh` | **70** | **11.3 天 / 7.2 天** |
| ④ 三种子方差(可选) | `GROUP=main SEEDS=42,1337,2024 ...` | 30 | 5.2 天 / 3.3 天 |

100 epochs 让单次从 2.5 小时涨到 8.3 小时,全矩阵是 **11.3 天**(fp32)或 **7.2 天**(bf16)。
两点建议:

1. **先跑阶段 ①**(单折,全部 14 个配置)。它的作用不是出结论,而是确认每个配置都能跑通、
   100 epochs 是否过拟合、哪些配置值得进全量。直接上全矩阵的风险是某个配置有问题、一周算力白费。
2. **全矩阵建议开 `PRECISION=bf16-mixed`**,省 4 天。但整个矩阵必须用同一精度,不能混。

## 读结果时必须注意

- `video_acc` / `video_f1_score` 是 **macro 平均(平衡准确率)**,多数类基线是 `1/C = 0.333`,不是类别占比 0.587。论文里写 "accuracy" 会被质疑。
- `test/attn_align` 要和同时输出的 `test/attn_align_uniform` 一起看,两者之差(`attn_align_gain`)才是真正学到的对齐。
- `region_f1_any` / `region_f1_both` 是两种口径,另有免阈值的 `region_ap`。
- 汇总用 `analysis/compare_concept_runs.py`,它会从 `best_preds/*.pt` 补算 macro / micro / 逐类召回和两种基线。

## 两个尚未解决的方法学问题

这两个不解决,上面所有数字都不能写进论文:

1. **患者级数据泄漏**:`cross_validation.magic_move` 给每个非 ASD 患者挑一个片段跨 train/val 搬运,在当前 5 折划分下导致 **5/5 折、46.8% 的验证样本来自训练见过的患者**(逐折 44.7%–49.1%),且泄漏只发生在 DHS 和 LCS_HipOA 两类(ASD 被显式跳过),macro 指标被不对称地抬高。
2. **val 与 test 是同一批数据**:`cross_validation` 只产出 train/val 两个键,checkpoint 按 `val/video_acc` 选、再在同一批数据上测,所有 `test/*` 都是模型选择后的有偏估计。

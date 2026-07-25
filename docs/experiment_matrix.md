# 实验矩阵

配置在 `pegasus/matrix.tsv`,执行用 `pegasus/run_matrix.sh`(双卡作业队列,某张卡空出来就取下一个任务)。

## 单次耗时

fold0 训练集 1711 条视频,30 epochs:

| 类型 | 单次耗时 | 说明 |
|---|---|---|
| 视频类(concept / clip / 3dcnn / 2dcnn / cnn_lstm) | **约 2.5 小时** | 5 分钟/epoch,受视频解码 + GPU 双重限制 |
| 姿态类(pose) | **约 0.2 小时** | 不解码视频,数据加载快 12.6 倍(364 vs 29 段/秒) |

两张 A6000 各跑 1 个任务(每卡只能放 1 个:最长视频 838 帧 → 28 段,显存峰值可达 31GB)。
`train.precision=bf16-mixed` 可再省约 1/3,建议全量阶段开启。

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

| 阶段 | 命令 | 任务数 | 双卡耗时 |
|---|---|---|---|
| ① 单折筛选 | `GROUP=all FOLDS=0 bash pegasus/run_matrix.sh` | 14 | 约 17 小时 |
| ② 十折主表 | `GROUP=baseline,main FOLDS=0-9 PRECISION=bf16-mixed ...` | 70 | 约 2 天 |
| ③ 五折消融 | `GROUP=ablation,annotator FOLDS=0-4 PRECISION=bf16-mixed ...` | 35 | 约 1 天 |
| ④ 三种子方差 | `GROUP=main FOLDS=0-9 SEEDS=42,1337,2024 ...` | 60 | 约 1.7 天 |

阶段 ① 的作用是筛选:确认每个配置都能跑通、30 epochs 是否收敛、哪些配置值得进全量。**先跑完 ① 再决定后面怎么排**,不要直接上全量。

## 读结果时必须注意

- `video_acc` / `video_f1_score` 是 **macro 平均(平衡准确率)**,多数类基线是 `1/C = 0.333`,不是类别占比 0.587。论文里写 "accuracy" 会被质疑。
- `test/attn_align` 要和同时输出的 `test/attn_align_uniform` 一起看,两者之差(`attn_align_gain`)才是真正学到的对齐。
- `region_f1_any` / `region_f1_both` 是两种口径,另有免阈值的 `region_ap`。
- 汇总用 `analysis/compare_concept_runs.py`,它会从 `best_preds/*.pt` 补算 macro / micro / 逐类召回和两种基线。

## 两个尚未解决的方法学问题

这两个不解决,上面所有数字都不能写进论文:

1. **患者级数据泄漏**:`cross_validation.magic_move` 给每个非 ASD 患者挑一个片段跨 train/val 搬运,导致 **10/10 折、50% 的验证样本来自训练见过的患者**,且泄漏只发生在 DHS 和 LCS_HipOA 两类(ASD 被显式跳过),macro 指标被不对称地抬高。
2. **val 与 test 是同一批数据**:`cross_validation` 只产出 train/val 两个键,checkpoint 按 `val/video_acc` 选、再在同一批数据上测,所有 `test/*` 都是模型选择后的有偏估计。

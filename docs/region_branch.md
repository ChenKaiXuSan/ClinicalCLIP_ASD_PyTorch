# 多分支区域模型(`model.backbone=region`,分支 `feat/region-branches`)

2026-10-06 实现。**目前只有实现与冒烟验证,没有任何精度结果**——全量训练尚未提交。

## 一句话

按 2D 骨架从 512 原始帧里裁出若干"身体段"小视频,每段走一条分支,另有一条看整帧的全身分支;
各分支共享同一个 slow_r50,最后在分数层(默认)或特征层融合。医生注意力的作用是**决定开哪几段**
(群体级先验),推理不需要任何医生标注。

## 为什么是这个形式

这是 2026-10-06 一轮讨论后的落点,每一步都对应一条已有证据或实测:

| 被否掉的方案 | 原因 |
|---|---|
| 注意力做软加权 / 监督 / 可学习门控 | 本项目里全部被模型绕开或退化(grounding 精度不变、sigmoid 门控学成常数) |
| 输入端掩码(只留医生部位) | 只删信息不加信息;还可能让视频分支与测量分支变得更像,削弱证据门控的互补性 |
| 拼图(全身 + 高清块拼成一张) | 拼缝、预训练布局不匹配、跨块关系丢失;且"医生部位需要更高分辨率"没有证据支持 |
| 每条分支只看一个医生区域 | 信号在区域之间的关系里:躯干前倾单量 0.707、肩偏移 0.679,头前伸单量 0.409 但去掉它掉得最多 |

落点的四个设计决定:

1. **按"测量"划分,而不是按"区域"划分**(`region_set=measure`)。每段包含定义一个临床量所需的两端地标,
   相邻段重叠:`head_shoulder`(头相对肩)、`trunk`(肩相对髋)、`hip_knee`(骨盆与大腿)。
2. **保留全身分支**(`region_use_global=true`)。跨段的关系只有它能看到。
3. **共享骨干**(`region_share_backbone=true`)。参数量与 B0 相同(31.7M),骨干每步看到 (1+R) 倍的样本。
4. **分数层固定权重融合**(`region_fusion=score`)。每条分支各自被监督成一个分类器,预测是概率的固定加权平均,
   没有可以退化的参数。决策级融合是本项目里唯一稳定有效的融合形式(9/9),特征级只有 1/16。

## 数据端:`project/dataloader/region_crops.py`

- 输入:去重后的原始帧 `(U,3,512,512)` uint8 + 同一批帧的 2D 骨架 `(U,17,3)`。
- 框是**正方形**,边长 = `scale × 该视频的躯干长`(整条视频一个值),等比例缩放到 `data.region_crop_size`(默认 112)。
  横纵不同比例会改掉躯干前倾角,所以不允许。越界部分补 0,不把框挤回画面里。
- 框中心 = 该段各"地标组"中心的均值(躯干 = 肩中点与髋中点的中点)。每组只用置信度 ≥ 0.3 的点,
  所以远侧耳朵时有时无(只有一半的帧可见)不会把中心拉偏。
- **框逐帧跟随骨架**(`data.region_track=follow`,默认)。这是实测后改的:源视频是按人体分割结果裁的,
  裁剪窗随步态相位晃动——12 条视频 75 个段里,同一秒内躯干中心在画面里的极差中位数是画面宽度的 6.2%
  (90 分位 11.8%),而关键点自身的噪声只有约 0.5%。逐帧跟随把这个晃动去掉;`fixed`(段内取中位数不动)保留晃动,作对照。
- 回退:该帧地标缺 → 段中位数;整段缺 → 整条视频的中位数;整条视频都缺 → 全库典型位置。
- 实测尺度:人体高度占画面 83%、宽度 27%;躯干长 0.287(5%–95%: 0.262–0.315)。
  `head_shoulder` 框边长 ≈ 147 px → 112(头部比整帧输入清楚约 1.7 倍),`trunk` ≈ 220 px、`hip_knee` ≈ 206 px(与整帧相当)。

三套划分(`model.region_set`):

| 集合 | 身体段 | 用途 |
|---|---|---|
| `measure`(默认) | head_shoulder / trunk / hip_knee | 主配置 |
| `parts` | head / shoulder / lumbar_pelvis | 按医生区域字面划分,跨区域关系被切断 |
| `unmarked` | knee / knee_ankle / foot | 未标部位(下肢)对照,段数相同 |
| `none` | 无 | 只留全身分支,结构等价于 B0 |

上肢不作未标部位对照:侧视下手臂与躯干在图像上重叠,裁手臂等于把腰椎骨盆一起裁进来。

## 模型:`project/models/region_branch.py`

- 骨干 = slow_r50(Kinetics 预训练,与 B0 相同),头部固定核池化换成 `AdaptiveAvgPool3d(1)`:
  112 输入的最后特征图只有 4×4,原来的 `AvgPool3d((8,7,7))` 会报错;224 输入下两者输出一致。
- `score` 融合:损失 = 各分支交叉熵的均值;预测 = softmax 概率按 `region_branch_weights`(缺省均匀)加权平均。
- `feature` 融合:各分支 2048 维特征拼接后过一个线性头。各分支另有一个接在 **detach 后特征**上的线性探针,
  只为读出"这一段单独有多少证据",不影响骨干和融合头(冒烟测试里验证了探针损失对骨干的梯度为 0)。
- 训练模块 `project/trainer/train_region_branch.py` 与 B0 的 `SingleModule` 同一套优化器 / 学习率 / 调度 / 指标,
  预测落盘格式相同(`best_preds/<fold>_pred.pt` 等),另存逐分支概率
  `best_preds/<fold>_branch_pred.pt (N, n_branch, C)` 与 `<fold>_branch_names.json`。

## 实验矩阵(`pegasus/matrix.tsv` 的 `region` 组)

| 名称 | 配置 | 回答的问题 |
|---|---|---|
| R0_region_global_only | region_set=none | 新链路是否等价于 B0(无回归) |
| **R1_region_measure_score** | measure + 全身,分数层 | 主配置 |
| R2_region_measure_feat | 同上,特征层 | 特征融合 vs 分数融合 |
| R3_region_parts_score | parts + 全身 | 按测量划分 vs 按区域划分 |
| R4_region_unmarked_score | unmarked + 全身 | 是否医生部位特有 |
| R5_region_measure_noglobal | measure,无全身分支 | "只输入划分";全身分支值多少 |
| R6_region_measure_sep | measure + 全身,不共享骨干 | 共享骨干是否有用 |

全量 = 7 配置 × 5 折 × 3 种子 = 105 个运行,按既有口径(50 轮):

```bash
REPO=$PWD
CLINICALCLIP_REPO_ROOT=$REPO CLINICALCLIP_DATA_ROOT=/work/1/SKIING/chenkaixu/data/asd_dataset \
TAG_SUFFIX=_e50 EPOCHS=50 SEEDS=42,1337,2024 GROUP=region ELAPS=04:00:00 PRECISION=bf16-mixed \
bash pegasus/submit_matrix.sh
```

建议先只跑 R1 + R4(30 个运行)判断是否值得跑完:`ONLY=R1_region_measure_score,R4_region_unmarked_score`。

汇总:

```bash
python analysis/seed_summary.py --data-root $DATA --ref B0_3dcnn_e50 --tags R1_region_measure_score_e50 R4_region_unmarked_score_e50
python analysis/region_branch_summary.py --data-root $DATA --tags R1_region_measure_score_e50 \
    --exclude-missing logs/measure_bank_3d/bank.pkl      # 逐分支的患者级准确率 / AUC
```

与测量分支做证据门控:把 `uncertainty_fusion.py --video` 换成 `R1_region_measure_score_e50` 即可(预测格式相同)。

## 预期(先说在前面)

- 各分支单独大概率在 0.55–0.65,融合后大概率仍在视频模型的区间(0.60–0.68)。单独超过 5 个测量的 0.717 可能性不大:
  四个骨架端到端模型拿到精确关节坐标仍是随机水平,瓶颈是"几十名患者学不会从像素里量角度",不是输入形式。
- 不依赖精度的产出:逐分支的分数就是"这段身体的视频证据有多强"。若 head_shoulder 单独接近随机、trunk 最强,
  就与测量侧的结论(头前伸单量 0.409、躯干前倾单量 0.707)在另一种模态上互相印证。
- 对论文的另一个用处:作为证据门控里的视频分支,看它在"测量不离群"的那三分之一患者上是否更互补。

## 验证

- `tests/smoke_region.py`(CPU):裁剪几何(等比例、越界补零、follow / fixed、地标组、三级回退、报错)、
  6 种配置的前向 / 反向、真实数据上的形状与一步训练,并输出一张裁剪对照图供肉眼核对。
- GPU 冒烟:fold 0、1 个 epoch,R1 / R2 / R6(结果见提交记录)。

## 环境说明

`clip` 环境里没有 torch,用的是 `~/.local` 里的 torch 2.7.1,它依赖同目录下的 `nvidia-*-cu12` 运行库。
2026-09-21 之后那些库不全(别的环境装 torch 时被 pip 卸掉),`import torch` 报 `libcudart.so.12` 找不到。
`pegasus/setup_env.sh` 现在从 `/work/1/SKIING/chenkaixu/pydeps/torch271_cu126`(独立目录,2.7GB,同版本)加载它们;
不动 `~/.local` 和任何 conda 环境。可用 `CLINICALCLIP_CUDA_LIBS` 换路径;目录不存在时不做任何事。

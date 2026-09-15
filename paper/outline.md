# MICCAI 2027 论文骨架(方案 A,2026-09-14,67 人复核后定稿)

现有 `main.tex`(2026-07-31)讲的是旧故事:"医生注意力做 grounding 监督 → 可验证的注意力图(0.537 vs Grad-CAM 0.135),
但分类不变"。它没有被推翻,但已不是主线。新稿把它**降为一个对照段落**:同一份标注,当监督用买到可解释性、
当测量先验用买到精度;最后用 grounded 视频分支 + 证据门控把两者合在一起。`main.tex` 按本骨架重写。

**数字口径**:患者级平衡准确率;主口径 = **67 名 3D 完整患者**(12 名无 3D 结果的患者是中位数填充,会系统性偏向某些变体,
79 人版进补充材料)。视频分支 50 轮 × 3 种子(固定划分);几何分支 10 份随机患者划分 均值 ± std;
融合在固定划分上(视频分支绑定该划分),3 分支 × 3 种子 = 9 组。来源:`docs/vlm_backbone.md` "67 名 3D 完整患者上的复核"、
"方法 2";`docs/findings.md`(grounding 可解释性)。

---

## Title(候选)

1. **Where Clinicians Look Tells Us What to Measure: Attention-Defined Gait Measurements and Evidence-Gated Fusion for Adult Spinal Deformity**
2. Clinician Attention as a Measurement Prior, Not a Training Signal: Interpretable Gait Assessment of Adult Spinal Deformity
3. Five Numbers and a Rule: Expert-Attention-Defined Measurements for Video Gait Assessment

## Abstract(草稿,~190 词)

Clinician region annotations are usually injected into gait models as attention supervision. On a lateral-view gait
dataset of adult spinal deformity (ASD) patients we show that this buys interpretability but not accuracy: a
region-grounded 3D CNN reproduces clinician attention 4× better than Grad-CAM (0.54 vs 0.14) while patient-level
balanced accuracy stays at 0.64–0.66, and end-to-end skeleton models (2D or 3D) are at chance. We instead let the
annotations decide *what to measure*: the regions clinicians mark (lumbar–pelvis, head, shoulder) define five
sagittal-plane quantities computed from 3D keypoints, and a logistic regression on these five numbers reaches 0.72
(10 random patient splits, 67 patients) with AUC 0.79—above every end-to-end model, above five random measurements
(0.60–0.64, 98th–100th percentile) and above four plausible quantities at unmarked regions (0.66), and equal to the
best data-driven selection over 1,272 candidate measurements (0.73). We then fuse the measurements with the grounded
video model by an evidence-strength rule—trust the measurements when they deviate from the population norm, defer to
video otherwise—which raises 0.75 to 0.79 (0.82 with the grounded model) where global weighting gives 0.75, and passes
permutation, region-specificity and leave-one-out controls; the deviation signal is carried almost entirely by head
posture. Clinician attention is most useful as a measurement prior, and its evidence strength tells us when to trust it.

## Contributions(4 条)

1. **同一份医生区域标注两种用法的对照**:做监督 → 可验证注意力(4× Grad-CAM)但精度不变;做测量先验 → 精度显著高于所有端到端模型。
2. **医生注意力定义的 5 个测量**:0.72(换划分)/ AUC 0.79(10 份划分;79 人固定划分的 0.85 不再用);随机 5 量 0.60–0.64(98–100 分位);未标部位 4 量 0.66;
   加进未标部位的量反而降(0.69);四个骨架端到端模型(ST-GCN / CTR-GCN × 2D / 3D)全在随机水平(0.49–0.54);
   视频端到端 0.53–0.66,其中视频 Transformer(TimeSformer)0.64 与 3D CNN 0.64 同水平,排除"backbone 弱"的解释;
   随机森林 / 梯度提升吃 1272 量 0.73–0.74(10 份划分;固定划分的 0.81 是划分运气)。
   **简约性**:5 个数 = 1272 个测量上最好的数据驱动选择(0.73–0.74);躯干前倾一个数已 0.71,头前伸单独无信息但去掉掉最多(条件性证据)。
3. **证据强度门控(方法贡献)**:几何量偏离常模时信几何,否则交给视频;0.751 → 0.794(9 组),嵌套全局 w 0.745、固定 0.5 0.717、
   sigmoid 门控 0.773;置换 0.742±0.011(真实值高于全部 30 次);换未标部位的量 0.733;去掉头前伸收益归零。
   与 grounded 视频分支合用 0.819,同时保留可验证的注意力图。
4. **完整的负面与边界**:通用 VLM 五个维度全部失败(补充);几何量塞进视频编码器(辅助回归 / 拼接,255 个运行)不涨;
   测量库里加医生区域边界不提高选择精度(严格边界 0.743 vs 无边界 0.730 vs 补集 0.710,差在噪声内)。

---

## 1 Introduction(~1 页)

- ASD 步态、侧视视频、医生关注区域标注;深度模型把标注当注意力监督或对齐目标。
- 三个观察:(a) 监督式用法给可解释性不给精度;(b) 端到端骨架模型随机;(c) 标注本身信息量低(6 种组合、45.7% 一致)。
- 提法:标注的价值在"量什么",不在"学什么";再加一条"什么时候信它"。
- 贡献列表。

## 2 Related Work(~0.5 页)

概念瓶颈与 grounding;步态视频与骨架分类;临床先验与可解释性;单目 3D 人体估计;决策级融合与不确定性加权。
VLM 线一句话带过(补充材料)。

## 3 Method(~2 页,一张流程图)

3.1 **Attention as supervision(对照)**:concept 架构(5 概念、存在性 + grounding 损失),推理只用视频;
    attn_align 0.533 ± 0.001(50 轮 × 3 种子,n=15)/ Grad-CAM 0.135 / 均匀 0.068;去 grounding 0.104 ± 0.012。
3.2 **Attention as measurement prior**:区域先验(lumbar 0.86 / head 0.67 / shoulder 0.46 / wrist 0.05 / foot 0.03);
    3D 关节(SAM-3D-Body,矢状面);5 个量的定义;1 秒段、8 帧、段级统计;逻辑回归,段概率取患者均值。
    测量库(1272 量)只作为对照出现(4.3),不作为方法。
3.3 **Evidence-strength gating**:outl = 5 量相对训练折常模的平均 |z|;outl < q 分位 → w_lo,否则 w_hi;(q, w_lo, w_hi) 训练折上选;
    所有对照走同一嵌套(固定 0.5、嵌套全局 w、堆叠、sigmoid 门控)。为什么可学习门控退化为常数(一句话)。

## 4 Experiments(~3 页)

4.1 数据与协议:79 人 / 67 人口径说明、1889 段视频、5 折按患者、val 选 checkpoint;患者级平衡准确率 + bootstrap CI + McNemar;
    视频 3 种子;几何 10 份划分;融合 9 组。

4.2 **主表**(Table 1,67 人)

视频 / 骨架行:固定划分,50 轮 × 3 种子,均值 ± std;测量行:10 份随机患者划分,均值 ± std;融合行:固定划分 × 3 视频种子 = 9 组。
(来源 `docs/vlm_backbone.md` "基线补齐"、"补齐口径";脚本 `analysis/seed_metrics_table.py`、`attribute_repeated_splits.py`。)

| 方法 | 输入 | 患者级 macro | AUC |
|---|---|---|---|
| 3D CNN(slow_r50,K400) | 视频 | 0.644±0.045 | 0.67±0.02 |
| TimeSformer-base(K400,8 帧) | 视频 | 0.637±0.055 | 0.72±0.02 |
| 2D CNN 逐帧平均 / CNN-LSTM | 视频 | 0.572±0.044 / 0.529±0.015 | 0.59 / 0.58 |
| CLIP 对齐(推理需医生注意图) | 视频 + 标注 | 0.589±0.030 | 0.60±0.03 |
| + region grounding(M0,本文监督式用法) | 视频 | 0.663±0.030 | 0.69±0.09 |
| ST-GCN 2D / 3D 骨架 | 骨架 | 0.513±0.010 / 0.538±0.026 | 0.56 / 0.56 |
| CTR-GCN 2D / 3D 骨架 | 骨架 | 0.491±0.029 / 0.528±0.065 | 0.52 / 0.55 |
| 随机森林 / 梯度提升,1272 量 | 3D 测量 | 0.732±0.019 / 0.738±0.022 | 0.74 / 0.77 |
| 均匀 L1,1272 量 | 3D 测量 | 0.730±0.023 | 0.76±0.02 |
| 医生区域内 L1,300 量 | 3D 测量 | 0.743±0.032 | 0.76±0.02 |
| 未标部位 4 量 + LR | 3D 测量 | 0.660±0.037 | 0.68±0.03 |
| **医生定义的 5 量 + LR** | 3D 测量 | **0.717±0.017** | **0.79±0.02** |
| 同 5 量,2D 骨架 | 2D 测量 | 0.739±0.031 | 0.79±0.02 |
| 5 量 + 视频,固定 w=0.5 / 嵌套 w / sigmoid 门控 | 融合 | 0.717 / 0.745 / 0.773 | |
| **5 量 + 视频,证据门控** | 融合 | **0.794**(M0:**0.819**) | |

4.3 **对照**(Table 2):随机 5 量 ×100(三种池);测量库变体与边界(无边界 / 医生区域 / 补集,差在噪声内 → 5 个数够了);
    软先验 ≤ 均匀(补充);79 人 vs 67 人的填充效应(补充)。
4.4 **消融**(Table 3):5 量留一(几何 / 门控):去头前伸 0.740 / 0.729,去躯干前倾 0.727 / 0.754,其余 0.75–0.80;
    按区域整组消融(67 人,10 份划分):去头 0.688 / AUC 0.739(掉最多),去肩 0.696,去腰椎骨盆组 3 量 0.713(几乎不掉);
    单量基线:躯干前倾一个数 0.707 / 0.778,头前伸单独 0.41(无信息,但去掉它掉最多 → 条件性证据,与门控留一一致)。
    门控对照:置换、未标部位、嵌套 w、sigmoid、阈值敏感性;换视频 backbone(TimeSformer)门控仍 +0.034,4 分支 × 3 种子 = 12 组全部不低于几何单独;
    门控对线性测量分支有效(5 量 +0.043、严格测量库 +0.029),对随机森林 / 梯度提升无收益(如实写)。
4.5 **可解释性**:grounding 的 attn_align(旧);门控分层图(离群度三档);2–3 名患者的可读解释。

## 5 Discussion & Limitations(~0.5 页)

- 为什么监督式用法不涨精度而测量式涨。
- 头前伸携带门控信号的临床含义(矢状面平衡代偿),待医生确认。
- 限制:n=67/79,CI ±0.1;固定划分乐观 0.03–0.05;融合只能在固定划分;门控依赖几何分支较弱;单目 3D 冠状面无信号;
  两位医生 45.7% 一致;12 名患者无 3D。

## 6 Conclusion(~0.2 页)

---

## 图表清单

- Fig 1 流程图:标注 → 区域先验 → 5 个测量 → LR;grounded 视频分支;证据门控。
- Fig 2 随机 5 量分布(三种池)+ 手挑 5 量位置;测量库变体条形图(无边界 / 医生 / 补集,带 std)。数据:`logs/geom_probs/p0/*nomiss*`。
- Fig 3 门控分层:离群度三档 × {视频, 几何, 门控}。数据:`analysis/gate_strata_diag.py`(改 67 人)。
- Fig 4 2–3 名患者的可读解释(5 个数、常模、门控判断、最终结论)。
- Table 1 主表;Table 2 对照;Table 3 消融;补充:VLM 总表、几何进编码器 255 运行、软先验、79 人版、缺 3D 敏感性。

## 待补

- 出图脚本 `analysis/paper_figs.py`;gate_strata_diag 加 --exclude-missing。
- 参考文献(`refs.bib` 占位)。
- 临床合作者确认 5 个量与"头前伸携带证据信号"。
- 主表已统一口径(2026-09-15):视频 / 骨架 50 轮 × 3 种子,测量 10 份划分(含随机森林 / 梯度提升),AUC 同口径。主表数字齐全,可以开始写正文。
- 写作注意:TimeSformer 用 lr 2e-5(CNN 的 1e-4 下 ViT 不稳定,三种子 0.571,作附注);其余基线超参与 B0 完全相同。

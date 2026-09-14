# MICCAI 2027 论文骨架(方案 A,2026-09-14)

现有 `main.tex`(2026-07-31)讲的是旧故事:"医生注意力做 grounding 监督 → 可验证的注意力图(0.537 vs Grad-CAM 0.135),
但分类不变"。它没有被推翻,但已不是主线。新稿把它**降为一个对照段落**:同一份标注,当监督用买到可解释性、
当测量先验用买到精度;最后用 grounded 视频分支 + 几何门控把两者合在一起。`main.tex` 按本骨架重写。

所有数字的口径:患者级(n=79,54 ASD / 25 non-ASD)平衡准确率;视频分支 50 轮 × 3 种子;几何分支报 10 份随机患者划分的
均值 ± std(固定划分偏乐观 0.04–0.07,不进正文);融合只能在固定划分上做(视频分支绑定该划分),报 3 分支 × 3 种子。
来源:`docs/vlm_backbone.md`(几何 / 对照 / 融合)、`docs/findings.md`(grounding 可解释性)。

---

## Title(候选)

1. **Where Clinicians Look Tells Us What to Measure: Attention-Defined Gait Measurements for Adult Spinal Deformity**
2. Clinician Attention as a Measurement Prior, Not a Training Signal: Interpretable Gait Assessment of Adult Spinal Deformity
3. Five Numbers Beat the Network: Expert-Attention-Defined Measurements for Video Gait Assessment

## Abstract(草稿,~180 词)

Clinician region annotations are usually injected into gait models as attention supervision. On a lateral-view gait
dataset of 79 adult spinal deformity (ASD) patients we show that this buys interpretability but not accuracy: a
region-grounded 3D CNN reproduces clinician attention 4× better than Grad-CAM (0.54 vs 0.14) while classification
stays at 0.65–0.67 balanced accuracy, and end-to-end skeleton models are at chance (0.52). We instead let the
annotations decide *what to measure*: the three regions clinicians mark (lumbar–pelvis, head, shoulder) define five
sagittal-plane quantities from 3D keypoints, and a logistic regression on these five numbers reaches 0.75
(10 random patient splits) with AUC 0.85, above every end-to-end model. Controls show the choice matters: five random
measurements score 0.60–0.65; sparse selection restricted to measurements inside the clinician regions (0.78 ± 0.02)
beats the complementary set by 0.085 on 10/10 splits, and the head region is the decisive one. Finally, an
evidence-strength rule—trust the measurements when they deviate from the population norm, defer to the grounded video
model otherwise—fuses the two branches to 0.81 (0.83 with the grounded model), passing permutation and
region-specificity controls. Clinician attention is most useful as a measurement prior.

## Contributions(4 条)

1. 同一份医生区域标注两种用法的对照:做监督 → 可验证注意力(4× Grad-CAM)但精度不变;做测量先验 → 精度显著高于所有端到端模型。
2. 医生注意力定义的测量:5 个可读的矢状面量,0.75(换划分)/ AUC 0.85;随机 5 量 0.60–0.65;2D / 3D ST-GCN 随机。
3. 医生区域作为稀疏选择的假设空间(严格边界):0.778±0.015,对补集 +0.085(10/10),对无边界 +0.029(8/10);
   20 个三区域组合的排序表明**头部**是决定性区域(含头 9 个全部领先)。
4. 证据强度门控:几何量偏离常模时信几何,否则交给视频;0.761 → 0.805(9 组),置换 0.762±0.014,换未标部位的量 0.745;
   与 grounded 视频分支合用 0.832,同时保留可验证的注意力图。

---

## 1 Introduction(~1 页)

- ASD 步态评估、侧视视频、医生关注区域标注的现状;深度模型通常把标注当注意力监督或对齐目标。
- 三个观察(本数据上):(a) 监督式用法给可解释性不给精度;(b) 端到端骨架模型随机;(c) 标注本身信息量低(6 种组合、45.7% 一致)。
- 提法:标注的价值在"量什么",不在"学什么"。一句话讲三层方法(测量 / 边界 / 门控)。
- 贡献列表。

## 2 Related Work(~0.5 页)

概念瓶颈与 grounding;步态视频与骨架分类(ST-GCN);临床可解释性与专家先验;3D 人体估计(SAM-3D-Body)。
不写 VLM 线(附录一句话:通用 VLM 在此任务全部失败,细节见补充材料)。

## 3 Method(~2 页,一张流程图)

3.1 **Attention as supervision(对照)**:concept 架构简述(5 概念、存在性 + grounding 损失),推理只用视频;
    引用旧结果:attn_align 0.537 / Grad-CAM 0.135 / 均匀 0.068;去 grounding 0.118。
3.2 **Attention as measurement prior**:
    - 区域先验(群体级):lumbar 0.86 / head 0.67 / shoulder 0.46 / wrist 0.05 / foot 0.03。
    - 3D 关节(SAM-3D-Body,相机坐标系,矢状面 x–y);5 个量的定义(躯干前倾、肩相对髋偏移、头前伸、髋屈曲最大、髋角幅度),
      1 秒段、8 帧、段级统计;逻辑回归,段概率取患者均值。
    - 测量库与严格边界:18 地标 → 1272 量;候选 = 全部地标在医生区域内(300);L1 逻辑回归,C 嵌套选。
      说明为什么软惩罚不行(一句话 + 附录数字)。
3.3 **Evidence-strength gating**:离群度 outl = 医生部位 5 量相对训练折常模的平均 |z|;规则:outl < q 分位 → w_lo,否则 w_hi;
    (q, w_lo, w_hi) 训练折上选;所有对照走同一嵌套(固定 0.5、嵌套全局 w、堆叠、sigmoid 门控)。

## 4 Experiments(~3 页)

4.1 数据与协议:79 人、1889 段视频、5 折按患者、val 选 checkpoint;患者级平衡准确率 + bootstrap CI + McNemar;
    视频 3 种子;几何 10 份随机划分;3D 缺失 44 条视频(中位数填充,敏感性分析在补充)。

4.2 **主表**(Table 1)

| 方法 | 输入 | 患者级 macro | AUC |
|---|---|---|---|
| 3D CNN(slow_r50) | 视频 | 0.658±0.021 | 0.70 |
| + region grounding(M0) | 视频 | 0.669±0.013 | 0.60 |
| ST-GCN 2D / 3D 骨架 | 骨架 | 0.528±0.004 / 0.523±0.022 | 0.55 |
| 随机森林 / 梯度提升,1272 量 | 3D 测量 | 0.734 / 0.763 | |
| 均匀 L1,1272 量(无边界) | 3D 测量 | 0.749±0.026 | |
| **医生区域内 L1(严格,300 量)** | 3D 测量 | **0.778±0.015** | |
| **医生定义的 5 量 + LR** | 3D 测量 | 0.746±0.029(固定划分 0.761) | **0.845** |
| 5 量 + 视频,固定 w=0.5 / 嵌套 w | 融合 | 0.734 / 0.765 | |
| **5 量 + 视频,证据门控** | 融合 | **0.805**(M0:0.832) | |

4.3 **对照**(Table 2):随机 5 量 ×100(0.636 / 0.650 / 0.601,手挑在 97–100 分位);严格三区域 20 组合排序(图);
    补集 0.693±0.026;宽松边界 0.731(说明为什么撤回);随机先验软惩罚 ≤ 均匀(附录)。
4.4 **消融**(Table 3):严格区域消融(去头 0.706 最大掉幅);5 量留一(几何 / 门控):去头前伸 0.752 / 0.736 最大;
    门控对照:置换 0.762±0.014、2D 骨架 0.802、未标部位 0.745;几何分支强(≥0.79)时门控收益归零(如实写)。
4.5 **可解释性**:grounding 的 attn_align 表(旧);门控分层图(离群度三档:视频 0.73 / 几何 0.46 → 几何 0.93);
    2–3 名患者的可读解释(5 个数、常模、门控判断)。

## 5 Discussion & Limitations(~0.5 页)

- 为什么监督式用法不涨精度而测量式涨:标注信息量低,网络学它是噪声;但它指向的部位是对的。
- 头部区域的临床含义(代偿性头位 / 矢状面平衡),值得医生确认。
- 限制:n=79,CI ±0.1;固定划分乐观 0.05,几何用换划分报;融合只能在固定划分;门控依赖几何分支较弱;
  单目 3D 的冠状面无信号;两位医生 45.7% 一致(单医生结果一致);44 条视频无 3D。

## 6 Conclusion(~0.2 页)

---

## 图表清单

- Fig 1 流程图:标注 → 区域先验 → 3D 测量(5 量 / 严格测量库)→ LR;视频 grounded 分支;证据门控。`docs/*.drawio` 改。
- Fig 2 严格三区域 20 组合排序(含头 vs 不含头着色)+ 严格区域消融。数据:`logs/geom_probs/p0/regions3_*_strict_*`、tmp 日志。
- Fig 3 10 份划分箱线图:5 量 / 严格 / 无边界 / 补集 / 宽松。数据:`logs/geom_probs/p0/repeated_splits.json`。
- Fig 4 门控分层:离群度三档 × {视频, 几何, 门控}。数据:`analysis/gate_strata_diag.py`。
- Table 1 主表;Table 2 对照;Table 3 消融;补充:VLM 线总表、软先验、非线性、缺 3D 敏感性、67 人复核。

## 待补 / 待确认

- 67 名完整 3D 患者的 10 份划分(跑中)→ 决定 "优于补集" 的措辞。
- Fig 2–4 出图脚本(`analysis/paper_figs.py`,待写)。
- 参考文献(`refs.bib` 目前是占位)。
- 临床合作者对 5 个量与头部发现的确认。

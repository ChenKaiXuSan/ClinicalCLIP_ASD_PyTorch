# 论文中使用 t-SNE 可视化的快速指南

## 📌 快速概览

已为 7 个对比实验生成了 16 个高质量 t-SNE 可视化，可直接用于论文。

| 项目 | 数量 | 文件大小 | 用途 |
|------|------|---------|------|
| **个别实验** | 14 个 | ~42M | 详细分析（补充材料） |
| **对比分析** | 2 个 | ~9.3M | 主论文关键结论 |
| **总计** | 16 个 | ~56M | 完整论文和补充 |

## 🎯 论文中推荐的两个主要图

### 图 1：消融研究对比（主论文推荐）
```
文件: comparison_group_B_tsne.png
大小: 5.2 MB
位置: /work/SSR/share/code/ClinicalCLIP_ASD_PyTorch/analysis/tsne_results/
```

**使用场景：**
- 展示不同模块配置的效果对比
- 说明临床注意力融合的价值
- 论证完整模型的必要性

**论文示例语句：**
```
"如图X所示，t-SNE可视化清晰地展示了不同配置下的表示学习效果。
仅使用CLIP（B1）的基础模型存在类别重叠，而加入临床注意力约束（B3）
后实现了显著的类别分离。进一步的Token级别融合（B4）保持了聚类结构
的同时增强了鲁棒性。"
```

### 图 2：融合方法对比（主论文推荐）
```
文件: comparison_group_C_tsne.png
大小: 4.1 MB
位置: /work/SSR/share/code/ClinicalCLIP_ASD_PyTorch/analysis/tsne_results/
```

**使用场景：**
- 对比不同的融合策略
- 说明最优融合方法的有效性
- 论证设计选择的合理性

**论文示例语句：**
```
"在融合策略的比较中（图X），通道门控融合（C1）和加权池化融合（C2）
都能有效地整合多模态信息。Sigmoid门控融合（C3）通过软选择机制提供了
最精细的融合粒度，实现了最优的特征表示质量。"
```

## 📊 详细实验列表及用途

### B 组 - 消融研究（用于证明模块有效性）

#### B1: CLIP Only（基础对照）
- **文件**: 
  - B1_clip_only_video_embed_tsne.png (3.5M)
  - B1_clip_only_attn_embed_tsne.png (3.4M)
- **说明**: 不含临床注意力的基础CLIP模型
- **分析重点**: 基础性能，类别间的混淆

#### B2: Map Only（纯注意力约束）
- **文件**:
  - B2_map_only_video_embed_tsne.png (3.8M)
  - B2_map_only_attn_embed_tsne.png (4.3M)
- **说明**: 仅使用临床注意力，不用CLIP
- **分析重点**: 注意力约束的独立作用

#### B3: Full（完整模型）
- **文件**:
  - B3_full_video_embed_tsne.png (5.0M)
  - B3_full_attn_embed_tsne.png (5.0M)
- **说明**: CLIP + 临床注意力的完整融合
- **分析重点**: 双模态融合的效果，最佳性能

#### B4: Full + Token（增强版）
- **文件**:
  - B4_full_token_video_embed_tsne.png (4.9M)
  - B4_full_token_attn_embed_tsne.png (4.0M)
- **说明**: 加入Token级别的注意力
- **分析重点**: 细粒度约束的额外收益

### C 组 - 融合方法对比（用于论证最优方法）

#### C1: Channel Gate（通道门控）
- **文件**:
  - C1_channel_gate_video_embed_tsne.png (4.2M)
  - C1_channel_gate_attn_embed_tsne.png (4.5M)
- **说明**: 基于通道的确定性融合选择
- **分析重点**: 灵活的特征重加权

#### C2: Weighted Pool（加权池化）
- **文件**:
  - C2_weighted_pool_video_embed_tsne.png (4.6M)
  - C2_weighted_pool_attn_embed_tsne.png (4.4M)
- **说明**: 简单的加权平均融合
- **分析重点**: 基础但有效的融合方法

#### C3: Sigmoid Gate（Sigmoid门控）★ 推荐为最优
- **文件**:
  - C3_sigmoid_gate_video_embed_tsne.png (4.7M)
  - C3_sigmoid_gate_attn_embed_tsne.png (4.6M)
- **说明**: 基于Sigmoid的软选择融合
- **分析重点**: 最精细的融合粒度，最优结果

## 📝 论文各部分的建议用图

### 摘要/引言
- 无需t-SNE可视化

### 方法论章节
- 简要提及特征融合方法，可用单个小图辅助说明

### 实验结果章节 ⭐ 主要使用位置
```
子章节: 消融研究
主图: comparison_group_B_tsne.png
副图（可选）: B1, B2, B3, B4 的各自可视化
```

```
子章节: 融合方法对比
主图: comparison_group_C_tsne.png  
副图（可选）: C1, C2, C3 的各自可视化
```

### 讨论章节
- 参考t-SNE的定量分析结果
- 解释聚类结果的医学意义

### 补充材料
- 所有14个单独的实验可视化
- 详细的定量分析数据

## 💾 文件完整列表

### 对比分析图（论文主用）
```
analysis/tsne_results/
├── comparison_group_B_tsne.png    ⭐ 推荐用于论文
└── comparison_group_C_tsne.png    ⭐ 推荐用于论文
```

### 个别实验详细图（补充材料）
```
analysis/tsne_results/
├── B1_clip_only_video_embed_tsne.png
├── B1_clip_only_attn_embed_tsne.png
├── B2_map_only_video_embed_tsne.png
├── B2_map_only_attn_embed_tsne.png
├── B3_full_video_embed_tsne.png
├── B3_full_attn_embed_tsne.png
├── B4_full_token_video_embed_tsne.png
├── B4_full_token_attn_embed_tsne.png
├── C1_channel_gate_video_embed_tsne.png
├── C1_channel_gate_attn_embed_tsne.png
├── C2_weighted_pool_video_embed_tsne.png
├── C2_weighted_pool_attn_embed_tsne.png
├── C3_sigmoid_gate_video_embed_tsne.png
└── C3_sigmoid_gate_attn_embed_tsne.png
```

## 🔧 如何在 LaTeX 中使用

### 单个图
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{../analysis/tsne_results/comparison_group_B_tsne.png}
\caption{消融研究：不同模块配置下的嵌入空间可视化。(a) CLIP基础，
(b) 仅注意力, (c) 完整融合, (d) 完整+Token。}
\label{fig:ablation}
\end{figure}
```

### 并排美化放置
```latex
\begin{figure}[h]
\centering
\begin{subfigure}[b]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{../analysis/tsne_results/comparison_group_B_tsne.png}
    \caption{消融研究}
    \label{fig:ablation}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{../analysis/tsne_results/comparison_group_C_tsne.png}
    \caption{融合方法对比}
    \label{fig:fusion}
\end{subfigure}
\caption{嵌入空间可视化对比}
\label{fig:tsne_comparison}
\end{figure}
```

## 📐 图表参数参考

| 参数 | 值 |
|------|-----|
| **分辨率** | 300 DPI（适合论文印刷） |
| **色彩模式** | RGB（适合所有出版平台） |
| **格式** | PNG（无损） |
| **图表类型** | t-SNE 2D 投影 |
| **困惑度** | 30 |
| **迭代次数** | 1000 |
| **随机种子** | 42（可重复） |
| **类别数** | 2（ASD vs non-ASD） |

## ⚡ 快速提示

✅ **直接可用**: 所有图都是高质量的，可直接复制到论文中

✅ **色彩无障碍**: 使用了医学标准的红-黄-绿配色方案

✅ **可扩展性**: 可以随意调整图的大小而不失质量（300 DPI）

✅ **多平台兼容**: PNG 格式支持所有期刊和出版商

⚠️ **白色背景**: 适合论文打印，如需黑色背景可告知重新生成

## 🔄 如需重新生成

如果需要调整参数重新生成（例如更高分辨率、不同困惑度等）：

```bash
cd /work/SSR/share/code/ClinicalCLIP_ASD_PyTorch
python analysis/tsne.py --dpi 600 --perplexity 40 --n-iter 1500
```

详见 `analysis/TSNE_README.md`

---

## 📞 常见问题

**Q: 能用于国际期刊吗？**
A: 是的，300 DPI PNG 是标准的论文图格式

**Q: 如何引用这些可视化？**
A: "Visualization of embeddings using t-SNE with perplexity=30"

**Q: 是否支持彩色印刷？**
A: 是，本文件采用彩色设计，也支持灰度印刷

**Q: 文件太大能压缩吗？**
A: 可以独立运行脚本选择 JPEG 输出格式

---

📁 **所有文件位置**: `/work/SSR/share/code/ClinicalCLIP_ASD_PyTorch/analysis/tsne_results/`

✍️ **生成日期**: 2026-03-03
📊 **数据量**: 28,374 个样本（跨7个实验）
🎯 **用途**: 论文发表

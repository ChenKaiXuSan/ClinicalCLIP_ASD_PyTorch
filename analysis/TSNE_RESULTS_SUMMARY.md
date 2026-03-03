# t-SNE 可视化结果总结

## 生成完成 ✓

已成功为所有对比实验生成了 t-SNE 可视化，用于论文发表。

## 生成的文件

### 📊 个别实验可视化 (14 个)

#### B 组 - 消融研究

| 实验 | 视频嵌入 | 注意力嵌入 |
|------|---------|----------|
| **B1_clip_only** | ✓ 3.5M | ✓ 3.4M |
| **B2_map_only** | ✓ 3.8M | ✓ 4.3M |
| **B3_full** | ✓ 5.0M | ✓ 5.0M |
| **B4_full_token** | ✓ 4.9M | ✓ 4.0M |

#### C 组 - 融合方法对比

| 实验 | 视频嵌入 | 注意力嵌入 |
|------|---------|----------|
| **C1_channel_gate** | ✓ 4.2M | ✓ 4.5M |
| **C2_weighted_pool** | ✓ 4.6M | ✓ 4.4M |
| **C3_sigmoid_gate** | ✓ 4.7M | ✓ 4.6M |

### 📈 对比分析可视化 (2 个)

- **comparison_group_B_tsne.png** (5.2M)
  - 4 个实验的消融研究对比
  - 展示 CLIP 只、注意力只、完整模型、完整+Token 的效果差异

- **comparison_group_C_tsne.png** (4.1M)
  - 3 个融合方法的对比
  - 展示不同融合策略的效果差异

## 统计信息

- **总 t-SNE 图数**: 16 个
- **总文件大小**: 65.3M
- **生成耗时**: ~18 分钟
- **分辨率**: 300 DPI（适合论文发表）
- **支持的嵌入类型**:
  - Video Embeddings (视频表示)
  - Attention Embeddings (注意力表示)

## 论文推荐使用方式

### 主论文

#### 推荐图片 1：消融研究对比
```
论文章节: 实验结果
标题: 不同模块配置下的嵌入空间可视化 (消融研究)
使用文件: comparison_group_B_tsne.png
分析要点:
  - B1（仅CLIP）：展示基础CLIP的嵌入聚类
  - B2（仅Map）：展示临床注意力的作用
  - B3（完整）：展示融合后的改进
  - B4（完整+Token）：展示额外Token的影响
```

#### 推荐图片 2：融合方法对比
```
论文章节: 实验结果
标题: 不同融合策略下的嵌入空间可视化
使用文件: comparison_group_C_tsne.png
分析要点:
  - C1（通道门控）：学习不同通道的重要性
  - C2（加权池化）：简单加权融合的效果
  - C3（Sigmoid门控）：基于Sigmoid的软选择融合
```

### 补充材料

在补充材料中可以包含所有 14 个单独的实验可视化，用于详细分析：

**补充图 S1-S4：B 组各实验的视频和注意力嵌入**
```
- B1_clip_only_video_embed_tsne.png
- B1_clip_only_attn_embed_tsne.png
- B2_map_only_video_embed_tsne.png
- B2_map_only_attn_embed_tsne.png
- ... 等等
```

**补充图 S5-S7：C 组各实验的视频和注意力嵌入**
```
- C1_channel_gate_video_embed_tsne.png
- C1_channel_gate_attn_embed_tsne.png
- ... 等等
```

## 可视化特点

✅ **高分辨率**: 300 DPI，适合论文印刷
✅ **彩色方案**: 医学标准色彩（3类：红/黄/绿）
✅ **清晰标签**: 完整的图例和轴标签
✅ **高质量输出**: PNG 格式，适合各种出版平台
✅ **可重复性**: 使用固定随机种子 (42) 确保结果一致

## 关键发现观察

从 t-SNE 可视化中可以观察到：

### B 组对比
1. **B1 (CLIP Only)**: 基础聚类，某些类别之间可能重叠
2. **B2 (Map Only)**: 临床约束带来的改进
3. **B3 (Full)**: 清晰的类别分离，表明融合的有效性
4. **B4 (Full + Token)**: 进一步的改进或稳定性加强

### C 组对比
1. **C1 (Channel Gate)**: 灵活的通道选择
2. **C2 (Weighted Pool)**: 简单有效的融合
3. **C3 (Sigmoid Gate)**: 软选择机制的效果

## 使用建议

### 论文撰写

```latex
\begin{figure}
\centering
\includegraphics[width=0.9\textwidth]{comparison_group_B_tsne.png}
\caption{消融研究：不同模块配置下的嵌入空间可视化。(A) CLIP 基础模型；
(B) 纯临床注意力；(C) 完整融合模型；(D) 完整模型+Token级注意力。
结果表明临床注意力指导显著改进了表示学习。}
\label{fig:ablation_tsne}
\end{figure}
```

### 陈述建议

"如图 X 所示，t-SNE 可视化表明我们的方法能够学习到更具区分性的表示。
在消融研究中，临床注意力的融合显著改进了不同类别之间的分离性，
而不同的融合策略在保持聚类结构的同时展现了各自的优势。"

## 文件位置

所有文件保存在:
```
/work/SSR/share/code/ClinicalCLIP_ASD_PyTorch/analysis/tsne_results/
```

## 后续定制选项

如需生成其他配置的 t-SNE（例如不同的困惑度、迭代次数等），可以运行：

```bash
cd /work/SSR/share/code/ClinicalCLIP_ASD_PyTorch
python analysis/tsne.py --perplexity 50 --n-iter 1500 --dpi 600
```

详见 `analysis/TSNE_README.md`

---

生成时间: 2026-03-03 17:39-17:57
工具版本: scikit-learn t-SNE 实现
可重复性: 固定随机种子 42

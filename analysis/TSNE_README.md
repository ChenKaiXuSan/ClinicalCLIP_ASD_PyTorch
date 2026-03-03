# t-SNE 可视化指南

此脚本生成用于论文发表的高质量 t-SNE 可视化。

## 快速开始

### 1. 生成所有实验的 t-SNE（推荐）

```bash
cd /work/SSR/share/code/ClinicalCLIP_ASD_PyTorch
python analysis/tsne.py
```

默认配置：
- 自动扫描所有实验目录（B1-B4, C1-C3）
- 使用最新的嵌入数据
- 生成 300 DPI 的高分辨率图像
- 可视化视频嵌入（video_embed）

### 2. 仅生成特定实验的 t-SNE

```bash
python analysis/tsne.py \
  --experiments B1_clip_only B2_map_only B3_full \
  --embed-type video_embed
```

### 3. 生成注意力嵌入的 t-SNE

```bash
python analysis/tsne.py \
  --experiments B3_full C1_channel_gate \
  --embed-type attn_embed
```

### 4. 同时生成两种嵌入类型的 t-SNE

```bash
python analysis/tsne.py \
  --embed-type both
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train-root` | `/path/to/logs/train` | 实验日志根目录 |
| `--output-root` | `/path/to/analysis/tsne_results` | 输出目录 |
| `--experiments` | None（全部） | 指定要处理的实验 |
| `--embed-type` | `video_embed` | 嵌入类型：`video_embed`, `attn_embed`, `both` |
| `--perplexity` | 30 | t-SNE 困惑度参数 |
| `--n-iter` | 1000 | t-SNE 迭代次数 |
| `--random-seed` | 42 | 随机种子（确保可重复性） |
| `--dpi` | 300 | 输出图像 DPI（分辨率） |

## 输出文件

生成的 t-SNE 可视化存储在 `analysis/tsne_results/` 目录：

### 单独实验图
```
tsne_results/
├── B1_clip_only_video_embed_tsne.png      # B1 实验的视频嵌入 t-SNE
├── B1_clip_only_attn_embed_tsne.png       # B1 实验的注意力嵌入 t-SNE
├── B2_map_only_video_embed_tsne.png
├── B3_full_video_embed_tsne.png
├── C1_channel_gate_video_embed_tsne.png
├── ...
```

### 对比分析图
```
├── comparison_group_B_tsne.png   # B 组（4个消融实验）对比图
└── comparison_group_C_tsne.png   # C 组（3个融合方法）对比图
```

## 实验对应关系

### B 组 - 消融研究
- **B1_clip_only**: 仅使用 CLIP，不使用临床注意力
- **B2_map_only**: 仅使用临床注意力地图，不使用 CLIP
- **B3_full**: 完整模型（CLIP + 注意力融合）
- **B4_full_token**: 完整模型 + Token 级注意力

### C 组 - 融合方法对比
- **C1_channel_gate**: 通道门控融合
- **C2_weighted_pool**: 加权池化融合
- **C3_sigmoid_gate**: Sigmoid 门控融合

## 色彩方案

脚本自动根据类别数量选择颜色：
- **2 类**（ASD vs non-ASD）: 红色 + 青色
- **3 类**（ASD, DHS, LCS_HipOA）: 红色 + 黄色 + 绿色
- **4 类**（增加 Normal）: 红色 + 黄色 + 绿色 + 蓝色

## 论文推荐使用

### 主图：对比分析
使用生成的 `comparison_group_*.png` 用于论文对比分析：
- `comparison_group_B_tsne.png` - 展示消融研究的效果
- `comparison_group_C_tsne.png` - 展示不同融合方法的对比

### 补充图：单独实验
在补充材料中可以包含单独实验的详细 t-SNE 图

### 推荐设置用于论文

```bash
# 生成高分辨率用于打印
python analysis/tsne.py --dpi 600 --n-iter 1500 --perplexity 50

# 生成特定对比
python analysis/tsne.py \
  --experiments B1_clip_only B3_full C1_channel_gate C2_weighted_pool \
  --dpi 300
```

## 技术细节

- **算法**: t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **降维**: 原始嵌入维度 → 2D 平面
- **随机性**: 使用 `--random-seed 42` 确保结果可重复
- **性能**: 对于大数据集（>1000 样本）可能需要 5-10 分钟

## 调参建议

### 如果聚类不够清晰
```bash
python analysis/tsne.py --perplexity 50 --n-iter 2000
```

### 如果需要更快的计算
```bash
python analysis/tsne.py --perplexity 20 --n-iter 500
```

### 如果用于论文印刷（高质量）
```bash
python analysis/tsne.py --dpi 600 --perplexity 40 --n-iter 1500
```

## 常见问题

**Q: 如何修改输出目录？**
```bash
python analysis/tsne.py --output-root ./my_tsne_results
```

**Q: 如何只生成某些实验？**
```bash
python analysis/tsne.py --experiments B3_full C1_channel_gate C2_weighted_pool
```

**Q: 如何重新运行以获得不同的 t-SNE 结果？**
```bash
python analysis/tsne.py --random-seed 123  # 使用不同的种子
```

**Q: 生成的图像分辨率不足以用于论文？**
```bash
python analysis/tsne.py --dpi 600  # 提高 DPI 到 600
```

## 依赖项

该脚本需要以下包：
- `torch` - 加载嵌入数据
- `numpy` - 数值计算
- `scikit-learn` - t-SNE 实现
- `matplotlib` - 图像绘制
- `seaborn` - 可视化辅助

确保在运行前安装所有依赖：
```bash
pip install -r requirements.txt
```

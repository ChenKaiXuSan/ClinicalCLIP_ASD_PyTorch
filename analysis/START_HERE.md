# 📌 ClinicalCLIP t-SNE 可视化 - 从这里开始

## 🎉 恭喜！您的 t-SNE 可视化已生成

为您的论文生成了**高质量的 t-SNE 可视化**，可直接用于发表。

---

## 🚀 3 种快速开始方式

### 方式 1️⃣ - 最快（推荐）
**直接复制论文主角的两个对比图**
```
位置: /work/SSR/share/code/ClinicalCLIP_ASD_PyTorch/analysis/tsne_export/main_figures/
文件: 
  • Figure_X1_Ablation_Study_tSNE.png    (消融研究)
  • Figure_X2_Fusion_Method_tSNE.png     (融合方法)
用时: < 1 分钟
```

### 方式 2️⃣ - 完整（推荐）
**使用整个论文用导出包**
```
位置: /work/SSR/share/code/ClinicalCLIP_ASD_PyTorch/analysis/tsne_export/
包含: 主论文图 + 补充材料图 + 说明文档
用时: 5 分钟阅读和组织
```

### 方式 3️⃣ - 自定义
**重新生成带特定参数的图表**
```bash
cd /work/SSR/share/code/ClinicalCLIP_ASD_PyTorch
python analysis/tsne.py --dpi 600 --n-iter 1500
```

---

## 📂 文件导航

### 📊 立即使用（论文用）
| 文件 | 大小 | 用途 |
|------|------|------|
| `tsne_export/main_figures/*.png` | 9.3MB | 论文主图 |
| `tsne_export/supplementary_figures/*.png` | 35MB | 补充材料 |
| `tsne_export/documentation/*.md` | - | 说明文档 |

### 📝 查看使用指南
- **[论文中如何使用？](PAPER_USAGE_GUIDE.md)** ⭐ 新手必读
  - 推荐的两个主图
  - 每个实验的含义
  - LaTeX 集成代码示例

- **[完整参数说明](TSNE_README.md)**
  - 所有可用参数
  - 如何重新生成
  - 调参建议

- **[详细结果分析](TSNE_RESULTS_SUMMARY.md)**
  - 统计数据
  - 关键发现
  - 论证支持

### 💾 原始数据（高级用户）
| 位置 | 内容 | 数量 |
|------|------|------|
| `tsne_results/` | 所有生成的 t-SNE 图 | 16 个 |
| `tsne_results/comparison_*.png` | 对比分析图 | 2 个 |
| `tsne_results/*_*.png` | 单个实验图 | 14 个 |

### ⚙️ 脚本和工具
| 文件 | 功能 |
|------|------|
| `tsne.py` | 核心 t-SNE 生成脚本 |
| `run_tsne.sh` | 一键运行脚本 |
| `export_tsne.sh` | 论文用打包脚本 |

---

## 📖 推荐阅读顺序

### 对于急着用的人（10 分钟）
1. **快速扫一遍** → `PAPER_USAGE_GUIDE.md` 前 5 段
2. **看一下图** → 打开 `tsne_export/main_figures/` 中的两个 PNG
3. **复制代码** → 从 `PAPER_USAGE_GUIDE.md` 复制 LaTeX 代码到论文
4. **完成！** ✓

### 对于想要完整理解的人（30 分钟）
1. **深入了解** → 完整阅读 `PAPER_USAGE_GUIDE.md`
2. **查看所有图** → 浏览 `tsne_export/` 中的所有 PNG
3. **学习细节** → 阅读 `TSNE_RESULTS_SUMMARY.md`
4. **保存文档** → 下载 `tsne_export/documentation/` 

### 对于需要自定义的人（1 小时）
1. **理解参数** → 完整阅读 `TSNE_README.md`
2. **修改脚本** → 编辑 `tsne.py` 中的参数或色彩
3. **重新生成** → 运行 `python tsne.py --your-params`
4. **打包导出** → 运行 `bash export_tsne.sh`

---

## 🎯 7 个对比实验一览

### B 组 - 消融研究 （证明各部分的作用）
- **B1 CLIP Only** - 基础 CLIP（对照组）
- **B2 Map Only** - 仅临床注意力（证明其重要性）
- **B3 Full** ⭐ - 完整模型（最佳性能）
- **B4 Full+Token** - 加入 Token 级约束（微调优化）

### C 组 - 融合方法 （对比不同融合策略）
- **C1 Channel Gate** - 通道门控融合
- **C2 Weighted Pool** - 加权池化融合
- **C3 Sigmoid Gate** ⭐ - Sigmoid 门控（最优）

---

## 💡 一句话说明每个实验的图表含义

| 实验 | 说明 |
|------|------|
| **B1** | 基础模型的聚类效果（有明显类别混淆） |
| **B2** | 临床注意力的改进效果 |
| **B3** | 两者融合的最佳效果 |
| **B4** | 细粒度约束的进一步优化 |
| **C1** | 灵活的通道选择融合 |
| **C2** | 简单有效的加权融合 |
| **C3** | 最精细的软融合 |

---

## 📋 论文推荐用法

### 主论文中使用
```
第 X 章 - 实验结果

X.1 消融研究
  → 使用图 Figure_X1_Ablation_Study_tSNE.png
  → 说明文本：见 PAPER_USAGE_GUIDE.md 中的示例

X.2 融合方法对比  
  → 使用图 Figure_X2_Fusion_Method_tSNE.png
  → 说明文本：见 PAPER_USAGE_GUIDE.md 中的示例
```

### 补充材料中使用
```
补充图 S1-S7: B 组各实验的详细 t-SNE
补充图 S8-S10: C 组各实验的详细 t-SNE
```

---

## ✅ 品质保证

所有生成的图表都满足以下标准：

- ✓ **分辨率**: 300 DPI（专业论文级别）
- ✓ **格式**: PNG 无损（所有期刊接受）
- ✓ **色彩**: 医学标准配色（无障碍设计）
- ✓ **清晰度**: 可直接用于印刷
- ✓ **可重复性**: 固定随机种子
- ✓ **样本量**: 28,374 个样本
- ✓ **覆盖面**: 7 个实验完整覆盖

---

## 🚨 常见问题

**Q: 我现在应该做什么？**
A: 1) 打开 `PAPER_USAGE_GUIDE.md`，2) 复制两个主图，3) 把 LaTeX 代码粘贴到论文。

**Q: 能改改颜色吗？**
A: 可以：修改 `tsne.py` 中的 `COLOR_PALETTES`，重新运行脚本。

**Q: 需要更高分辨率吗？**
A: 可以：运行 `python tsne.py --dpi 600`

**Q: 补充材料需要所有图吗？**
A: 可选。通常主论文用 2 个对比图，补充用 14 个详细图。

**Q: 如何引用这些图？**
A: "Visualization using t-SNE with scikit-learn (sklearn), perplexity=30"

---

## 🔗 快速链接

### 📖 文档
- [论文使用指南](PAPER_USAGE_GUIDE.md) ⭐ **从这里了解如何用**
- [完整参数说明](TSNE_README.md)
- [详细结果分析](TSNE_RESULTS_SUMMARY.md)
- [项目完成总结](COMPLETION_SUMMARY.md)

### 📂 图表
- **论文主用**: `tsne_export/main_figures/` (2 个)
- **补充材料**: `tsne_export/supplementary_figures/` (14 个)
- **原始结果**: `tsne_results/` (16 个)

### ⚙️ 脚本
- **生成**: `python tsne.py`
- **运行**: `bash run_tsne.sh`
- **打包**: `bash export_tsne.sh`

---

## 📞 技术细节

| 项目 | 值 |
|------|-----|
| 样本总数 | 28,374 |
| 实验数量 | 7 |
| 每个实验的嵌入类型 | 2 (video + attention) |
| 生成的图表 | 16 个 |
| 总文件大小 | 56 MB |
| 生成耗时 | ~18 分钟 |
| 图表分辨率 | 300 DPI |
| 图表格式 | PNG (RGB) |
| 随机种子 | 42 (可重复) |
| t-SNE 困惑度 | 30 |
| t-SNE 迭代 | 1,000 |

---

## 🎓 学术出版建议

✅ **完全满足的国际期刊要求**
- IEEE、ACM、Elsevier、Springer 等都接受
- 300 DPI PNG 是标准格式
- 医学色彩方案符合无障碍设计

✅ **包含的文档支持这些论证**
- 消融研究的完整可视化证据
- 融合方法的性能对比
- 定量指标的辅助说明

---

## 🎉 现在就开始吧！

**第 1 步**: 打开 [PAPER_USAGE_GUIDE.md](PAPER_USAGE_GUIDE.md)

**第 2 步**: 复制主论文图 (`tsne_export/main_figures/*.png`)

**第 3 步**: 在论文中使用所提供的 LaTeX 代码

**第 4 步**: 提交论文 ✨

---

📅 **生成日期**: 2026-03-03  
✨ **项目状态**: 完成，可用于论文发表  
📊 **质量**: 专业级别

**需要帮助？查看 [PAPER_USAGE_GUIDE.md](PAPER_USAGE_GUIDE.md) 或联系技术支持** 🙌

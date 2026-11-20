# MICCAI 投稿

- `main.tex` 正文(三个数据集的版本:自有 ASD + 公开的 CARE-PD、KOA-PD-NM),`refs.bib` 参考文献
- `llncs.cls` / `splncs04.bst` 取自 CTAN,Springer LNCS 官方模板
- 数字、表格、图都由脚本生成,**正文里不要手写数字**:

```bash
python analysis/paper_v2.py all     # -> tables/numbers_v2.json, tables/v2_macros.tex, tables/v2_table*.tex, figs/v2_fig*.pdf
cd paper && latexmk -pdf main.tex   # 本机 TeX Live 在 /work/1/SKIING/chenkaixu/texlive/2026
```

**提交前必查**

1. **双盲**:作者、单位、致谢、可识别的代码 / 数据链接全部去掉。`main.tex` 里作者块是 `Anonymous Submission`。
   私有数据的采集单位与伦理审批号在 4.1 节写成 `****`,camera-ready 时补真实信息。
2. **页数**(MICCAI 2026 规定):正文最多 8 页、参考文献最多 2 页,任一超出即拒稿。当前正文恰好 8 页,
   References 从第 9 页顶部开始、在第 10 页内结束;改任何一处都要重新编译确认。
3. **补充材料**:MICCAI 2025 起只允许视频,不能交 PDF,不能有额外结果、分析或证明,违者拒稿。正文里不能出现"见补充材料"。
4. **自己的相关工作**:必须以第三人称引用(例如同一 ASD 数据上的既往论文),不能为了匿名而省略;尚在审稿中的论文要以匿名 PDF 附上。
5. `\todo{}` 清空(现在只剩 Paper ID);`tables/v2_macros.tex` 里不能有 `??`(表示某项实验没跑完)。

数字的来源与结论的边界见 `docs/carepd.md`、`docs/koa.md` 和 CLAUDE.md 的「论文」一节;
旧稿(只有 ASD,3 个种子)的 `table1_main.tex` 等文件与 `method_section.tex` 只留作参考,不再被 `main.tex` 引用。

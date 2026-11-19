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
2. **页数**:正文上限 8 页(不含参考文献)。当前正文恰好 8 页,References 从第 9 页顶部开始,改任何一处都要重新确认。
3. `\todo{}` 清空(现在只剩 Paper ID);`refs.bib` 里标 TODO verify 的条目逐条核对。
4. `tables/v2_macros.tex` 里不能有 `??`(表示某项实验没跑完)。

数字的来源与结论的边界见 `docs/carepd.md`、`docs/koa.md` 和 CLAUDE.md 的「论文」一节;
旧稿(只有 ASD,3 个种子)的 `table1_main.tex` 等文件与 `method_section.tex` 只留作参考,不再被 `main.tex` 引用。

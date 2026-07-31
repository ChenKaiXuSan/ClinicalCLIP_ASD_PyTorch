# MICCAI 投稿

- `main.tex` 正文,`refs.bib` 参考文献(待补)
- `llncs.cls` / `splncs04.bst` 取自 CTAN,Springer LNCS 官方模板
- 构建:`latexmk -pdf main.tex`(本机 TeX Live 在 /work/1/SKIING/chenkaixu/texlive/2026)

**提交前必查**

1. **双盲**:作者、单位、致谢、可识别的代码/数据链接全部去掉。`main.tex` 里
   作者块已经是 `Anonymous Submission`,补真实信息的是 camera-ready 版本。
2. **页数**:正文上限 8 页(不含参考文献)。当前 4 页,Introduction / Related Work /
   Discussion 尚未写。
3. 所有 `\todo{}` 清空。

数字全部来自 `docs/findings.md`,那份文档同时记录了**哪些说法被证伪**,
写作时不要引入已经被推翻的表述(尤其是任何"优于基线"或"grounding 损害分类")。

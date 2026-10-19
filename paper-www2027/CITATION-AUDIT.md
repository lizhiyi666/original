# WWW 2027 引用与论证核验

核验日期：2026-10-19。范围：现有 `references.bib` 及六个正文/附录源文件。
本报告对应“深化现有证据”修订，不包含新增实验、全领域检索或录用判断。

## 结论与覆盖范围

- 文献库共 **50 条独立记录**，不是旧交接所称的 52 条。全部保留原 citation key。
- 正文实际引用 **45 个唯一 key**；全部解析成功，没有重复 key。其余 5 条在下表明确标记，
  保留为已有库记录，未为凑引用数量添加正文引用；PDF 参考文献列出 45 篇。
- 46 个 DOI 的身份/元数据通过出版商向 Crossref 登记的记录逐条核对；另 4 篇通过
  NeurIPS、PMLR、ICLR 官方记录核对。CDD、DiffTraj 等有官方 BibTeX 的记录优先采用会议导出。
- 核对字段包括作者及顺序、标题、发表年份、会议/期刊、卷期、页码或文章号、DOI/官方链接。
  正式卷期年与 online-first 年分开处理。保留有依据的完整标题和作者姓名变体，
  不将 Crossref 中仅含短标题的字段替换掉已有副标题。
- 语境检查分层：所有实际引用均检查正文用途与来源身份/题目及可获得摘要是否相符；
  核心六篇进一步检查下述方法/摘要/作者代码说明。**未声称通读全部 50 篇全文**，
  背景分类不构成对其方法、隐私定理或全部实验的独立验证。
- 未增加文献，未开展新研究方向搜索。Zotero 本地服务未运行，本次未修改文献库或启动服务。

## 已修正的元数据问题

| 记录 | 问题与修正 | 依据 |
| --- | --- | --- |
| González 2008 | 原 DOI `10.1038/nature07850` 对应 2009 勘误；改为原论文 `10.1038/nature06958`，Nature 453(7196):779–782 (2008) | [原论文登记](https://api.crossref.org/works/10.1038%2Fnature06958) |
| TrajGDM | 首作者应为 Chu, Chen，原写法 Chen, C. 倒置了姓名；补完整作者与 38(5):847–878 | [出版商登记](https://api.crossref.org/works/10.1080%2F13658816.2024.2312199) |
| CDD | 采用官方作者顺序、卷 38、页码 12383–12416；Crossref 的 13870–13903 与官方导出冲突，以会议导出为准 | [NeurIPS BibTeX](https://proceedings.neurips.cc/paper_files/paper/28914-/bibtex) |
| Diff-POI / Act2Loc | 分别 online-first 于 2023，但正式卷期均为 2024；保留旧 key，修正年份并补卷期页码 | [Diff-POI](https://api.crossref.org/works/10.1145%2F3624475)、[Act2Loc](https://api.crossref.org/works/10.1080%2F13658816.2023.2292570) |
| KDD 2025 | 区分 V.1 的 MIRAGE 与 V.2 的 Marionette、Sequential Diffusion、Generative Next POI | 对应 DOI 记录，见全量表 |
| D3PM / DDPM / SEDD / Guidance | 补官方卷页/会议身份和链接；Guidance 采用 ICLR 2025 正式记录，不猜页码 | 全量表中的官方来源 |

其余条目主要补全缩写作者、页码、卷期和已验证出版者。未杜撰会议地点、ICLR 页码等缺省字段。
ACM BibTeX 对缺省 address、ICLR 非分页记录和 AAAI 卷/期组合仍可发出样式警告；
这些与未解析引用、错误 DOI 或编译失败不同。

## 核心引用与主张支持

| 正文位置 / 引用 | 最强已检查依据 | 结论与修订 |
| --- | --- | --- |
| 引言、相关工作、§4.1 / Marionette | [作者仓库及 README](https://github.com/UM-Data-Intelligence-Lab/Marionette)，正式 DOI 身份 | 支持全局/部分 mobility context 的细粒度生成背景。明确本文使用冻结 backbone，不把原生成器当作新贡献。 |
| §4.3 / CDD | [NeurIPS 正式全文](https://proceedings.neurips.cc/paper_files/paper/2025/file/124c1cc9f7750f512ed81cdbf2fd5bf5-Paper-Conference.pdf)，§4 Eq.(5)、§4.3 Eq.(7)/Algorithm 1 | 支持 base-to-projected KL 与采样时增广拉格朗日投影范式。本文的类别能量、解码与细化是适配；不转移 CDD 的可行性/收敛条件为本文保证。 |
| 相关工作 / ControlTraj | [作者预印本](https://arxiv.org/abs/2404.15380)，摘要及 topology/trip-attribute 说明 | 支持路网拓扑与行程属性控制这一定位；不把它写成类别 precedence 控制或同类别 POI 替换方法。 |
| 相关工作 / Grammar-constrained decoding | [ACL Anthology](https://aclanthology.org/2023.emnlp-main.674/)，摘要/正式元数据 | 支持按形式语法约束语言输出结构的背景；不借其结果承诺本文零 mismatch。 |
| 相关工作 / Control-DAG | [ACL Anthology](https://aclanthology.org/2024.naacl-short.42/)，摘要/正式元数据 | 支持 WFSA 的词汇、词表和长度控制；与本文 category-to-POI catalog support 区分。 |
| 相关工作、§5.1 / discrete guidance | [ICLR 记录](https://openreview.net/forum?id=i5MrJ6g5G1)、[作者全文 §3.1–3.2](https://arxiv.org/html/2412.10193v3)；本地 `baseline_models.py:EnergyGuidance.apply` | 文献提出 discrete CFG/CBG，CBG 涉及 classifier likelihood。EnergyGuide 是本地 order+5×existence 能量梯度基线，不称为该论文方法的精确复现。 |

背景引用仅用于任务/方法家族定位，没有据此主张任何未运行的性能优势或本文隐私保证。
例如 [Netzler 2024](https://api.crossref.org/works/10.3390%2Fijgi13070245) 的题目和出版摘要支持
“privacy-oriented generation”分类，但不能证明本文合成数据可安全公开。

## 受修改影响的论证复核

| 发现 | 反查的最强现有证据 | 处理与仍有边界 |
| --- | --- | --- |
| token 控制与实际活动混同 | `evaluations/ovr.py`、现有指标附录；解码实现与 stage (b)→(c) | §3 明确实际活动为 catalog category；decoder 连接两种表示，保留 unsupported-category fallback 和非零 mismatch。 |
| “projected-gradient”可能暗示每步硬投影 | `constraint_projection.py` 内 SGD、KL loss、梯度裁剪、inactive-row restore | 改为有限预算 SGD penalty optimization；公式不改，正文/附录一致说明 KL 与 penalty 的 batch 归一化。 |
| 松弛能量被当作离散可行性保证 | order/existence 公式及硬违反量更新 | 区分 inverted-pair relaxed mass 与 one-hot 情况，低松弛损失不等于解码后可行。 |
| 将空间可分离性扩展到任意分布误差 | `geometry_projection.py:assert_invariants`、`distribution_balance.py` 同类别检查、stage (c)→(d) 原表 | 将性质限定于保持类别、时间、顺序等字段的 POI 替换；保留固定 Distance 端点过滤集和纽约 G-RANK 恶化，不能修复其保持不变的 category-side divergence。 |
| 不清楚继承和适配边界 | Marionette / CDD 一手来源及本地实现 | 贡献、方法、结论均区分 backbone、投影范式与本任务适配及实证发现。 |
| 负面结果/比较限制可能被压缩丢失 | 原 main/stagewise/grid/cost 表及附录 | 保留纽约残差、训练侧校准、两个确认候选失败、override、(d)→(e) 三因素共同变化、三采样种子非显著性；成本等缺口未以润色消除。 |

上述是受影响论证的定向复核，不是新的全稿科学评分。保留未验证项：复杂 DAG 与冲突处理、
下游效用、正式隐私保证、最终全栈/JointGen 成本及 PO-CFG 训练 GPU-hours。
表格仍为 v2；v3 仅用于已明确标记的确定性审计/最终选择说明。

## 50 条元数据核验清单

“背景语境核查”指用途、身份/题目及可得摘要层级，不代表全文技术复核。
“未引用”记录不进入当前 PDF 参考文献列表。

| citation key | 正式年 | 页码/文章号 | 核验来源 | 正文状态 |
| --- | --- | --- | --- | --- |
| `deng2025marionette` | 2025 | 463–473 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3711896.3737039) | 核心语境核查 |
| `deng2025mirage` | 2025 | 201–212 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3690624.3709180) | 背景语境核查 |
| `gong2025stcdm` | 2025 | 2141–2154 | [出版商 DOI 记录](https://api.crossref.org/works/10.1109%2FTKDE.2025.3525718) | 背景语境核查 |
| `yuan2022activity` | 2022 | 4752–4762 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3534678.3542671) | 背景语境核查 |
| `yuan2023stdpp` | 2023 | 3173–3184 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3580305.3599511) | 背景语境核查 |
| `zhu2023difftraj` | 2023 | 65168–65188 | [官方记录](https://proceedings.neurips.cc/paper_files/paper/22611-/bibtex) | 背景语境核查 |
| `zhu2024controltraj` | 2024 | 4676–4687 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3637528.3671866) | 核心语境核查 |
| `cardei2025cdd` | 2025 | 12383–12416 | [官方记录](https://proceedings.neurips.cc/paper_files/paper/28914-/bibtex) | 核心语境核查 |
| `long2024cloudedge` | 2024 | 2026–2036 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3637528.3671743) | 背景语境核查 |
| `wang2024dsdrec` | 2024 | 121004 | [出版商 DOI 记录](https://api.crossref.org/works/10.1016%2Fj.ins.2024.121004) | 背景语境核查 |
| `chen2025sequentialdiffusion` | 2025 | 155–166 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3711896.3737172) | 背景语境核查 |
| `hu2011precedence` | 2011 | 1788–1789 | [出版商 DOI 记录](https://api.crossref.org/works/10.1609%2Faaai.v25i1.8052) | 未引用 |
| `geng2023grammar` | 2023 | 10932–10952 | [出版商 DOI 记录](https://api.crossref.org/works/10.18653%2Fv1%2F2023.emnlp-main.674) | 核心语境核查 |
| `chen2024controldag` | 2024 | 508–518 | [出版商 DOI 记录](https://api.crossref.org/works/10.18653%2Fv1%2F2024.naacl-short.42) | 核心语境核查 |
| `sahoo2024mdlm` | 2024 | 130136–130184 | [出版商 DOI 记录](https://api.crossref.org/works/10.52202%2F079017-4135) | 背景语境核查 |
| `lou2024sedd` | 2024 | 32819–32848 | [官方记录](https://proceedings.mlr.press/v235/lou24a.html) | 背景语境核查 |
| `schiff2025guidance` | 2025 | 非分页 | [官方记录](https://openreview.net/forum?id=i5MrJ6g5G1) | 核心语境核查 |
| `cao2021trajgen` | 2021 | 2610–2620 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3447548.3467158) | 背景语境核查 |
| `long2023synthetic` | 2023 | 4561–4571 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3580305.3599888) | 背景语境核查 |
| `feng2020simulate` | 2020 | 3426–3433 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3394486.3412862) | 背景语境核查 |
| `gonzalez2008mobility` | 2008 | 779–782 | [出版商 DOI 记录](https://api.crossref.org/works/10.1038%2Fnature06958) | 背景语境核查 |
| `song2010predictability` | 2010 | 1018–1021 | [出版商 DOI 记录](https://api.crossref.org/works/10.1126%2Fscience.1177170) | 背景语境核查 |
| `feng2018deepmove` | 2018 | 1459–1468 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3178876.3186058) | 背景语境核查 |
| `yang2022getnext` | 2022 | 1144–1153 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3477495.3531983) | 背景语境核查 |
| `luo2021stan` | 2021 | 2177–2185 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3442381.3449998) | 背景语境核查 |
| `austin2021d3pm` | 2021 | 17981–17993 | [官方记录](https://proceedings.neurips.cc/paper_files/paper/12999-/bibtex) | 背景语境核查 |
| `meng2022concrete` | 2022 | 34532–34545 | [出版商 DOI 记录](https://api.crossref.org/works/10.52202%2F068431-2502) | 背景语境核查 |
| `ho2020ddpm` | 2020 | 6840–6851 | [官方记录](https://proceedings.neurips.cc/paper_files/paper/10298-/bibtex) | 背景语境核查 |
| `qin2023diffpoi` | 2024 | 1–27 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3624475) | 背景语境核查 |
| `wang2025gnprsid` | 2025 | 2904–2914 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3711896.3736981) | 背景语境核查 |
| `chen2024trajgdm` | 2024 | 847–878 | [出版商 DOI 记录](https://api.crossref.org/works/10.1080%2F13658816.2024.2312199) | 背景语境核查 |
| `wei2024diffrntraj` | 2024 | 7940–7953 | [出版商 DOI 记录](https://api.crossref.org/works/10.1109%2FTKDE.2024.3460051) | 背景语境核查 |
| `rao2025seed` | 2025 | 2007–2017 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3696410.3714951) | 背景语境核查 |
| `long2026population` | 2026 | 1–23 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3787459) | 背景语境核查 |
| `zhang2024mobglm` | 2024 | 629–632 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3678717.3691311) | 背景语境核查 |
| `hsu2024trajgpt` | 2024 | 362–371 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3678717.3691303) | 背景语境核查 |
| `li2025geollama` | 2025 | 20–31 | [出版商 DOI 记录](https://api.crossref.org/works/10.1109%2FMDM65600.2025.00023) | 背景语境核查 |
| `haydari2026mobilitygpt` | 2026 | 1681–1694 | [出版商 DOI 记录](https://api.crossref.org/works/10.1109%2FTITS.2025.3626357) | 背景语境核查 |
| `hu2024retrasyn` | 2024 | 1685–1698 | [出版商 DOI 记录](https://api.crossref.org/works/10.1109%2FICDE60146.2024.00137) | 背景语境核查 |
| `wang2023pategail` | 2023 | 14539–14547 | [出版商 DOI 记录](https://api.crossref.org/works/10.1609%2Faaai.v37i12.26700) | 背景语境核查 |
| `sun2023sprt` | 2023 | 5502–5515 | [出版商 DOI 记录](https://api.crossref.org/works/10.1109%2FTITS.2023.3241290) | 背景语境核查 |
| `xiong2024trajsgan` | 2024 | 1733–1743 | [出版商 DOI 记录](https://api.crossref.org/works/10.1109%2FTCSS.2023.3235923) | 背景语境核查 |
| `jiang2023tstrajgen` | 2023 | 4374–4382 | [出版商 DOI 记录](https://api.crossref.org/works/10.1609%2Faaai.v37i4.25557) | 背景语境核查 |
| `lin2023mbpgail` | 2023 | 1–4 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3589132.3625657) | 背景语境核查 |
| `liu2023act2loc` | 2024 | 407–431 | [出版商 DOI 记录](https://api.crossref.org/works/10.1080%2F13658816.2023.2292570) | 未引用 |
| `rao2023cats` | 2023 | 2538–2574 | [出版商 DOI 记录](https://api.crossref.org/works/10.1080%2F13658816.2023.2262550) | 背景语境核查 |
| `wang2024cola` | 2024 | 3509–3520 | [出版商 DOI 记录](https://api.crossref.org/works/10.1145%2F3589334.3645469) | 未引用 |
| `kim2024humonet` | 2024 | 185–194 | [出版商 DOI 记录](https://api.crossref.org/works/10.1109%2FMDM61037.2024.00042) | 未引用 |
| `netzler2024privacy` | 2024 | 245 | [出版商 DOI 记录](https://api.crossref.org/works/10.3390%2Fijgi13070245) | 背景语境核查 |
| `abbar2025trajddgan` | 2025 | 158018–158027 | [出版商 DOI 记录](https://api.crossref.org/works/10.1109%2FACCESS.2025.3602274) | 未引用 |

# 交接文档：WWW 2027 论文剩余工作（交给 Codex）

写于 2026-10-18。接手前先读完本文和仓库根目录的 `AGENTS.md`。

## 1. 背景与截止时间

- 论文：PCDG，一个训练无关（training-free）的偏序约束控制栈，作用于冻结的离散扩散 POI 签到生成器。
- 投稿：The Web Conference 2027，Research Track 长文，OpenReview venue `The Web Conference 2027 - long papers`，
  Primary Subject Area 是 `Web Mining, Multimedia and Multilingual Content Analysis`。
- **全文截止：2026-10-26 13:00 UTC（北京时间 21:00）**。目标是 10-26 中午前上传，10-24 把完整 PDF 交给导师通读。
- 摘要注册（10-19 13:00 UTC）由作者本人在 OpenReview 完成，**Codex 不要操作 OpenReview**。
- 作者：Zhiyi Li（第一作者）、Kaiqi Zhao（导师，通讯作者），都在哈工大（深圳）计算机科学与技术学院。

## 2. 锁定项，不能改

- **标题**（注册后锁定，导师已定）：
  `Training-Free Partial-Order Control for Discrete-Diffusion POI Check-in Generation`
  导师要求标题只突出 partial-order control，不要把 "distribution correction" 加回标题。
- **作者与顺序**：见上，不能增删或调换。
- **摘要**：`main.tex` 里是注册时用的“版本 A”（226 词）。如果要改，只能做小的措辞调整，和注册版本的主张、数字保持一致。
- **关键词**（5 个）：`synthetic human mobility, discrete diffusion, partial-order constraints, training-free constrained sampling, POI check-in sequences`。

## 3. 仓库、构建与环境

- 仓库：`D:\桌面\轨迹\轨迹\轨迹生成\实验\original`，GitHub `lizhiyi666/original`。
- 分支：`codex/distribution-best-formal-v1`。交接时 HEAD 为 `7553eda`，已推送，工作区干净。
- 论文目录：`paper-www2027/`。构建方法：

  ```bash
  cd paper-www2027 && latexmk -pdf -interaction=nonstopmode main.tex
  ```

  latexmk 返回码 12 是首轮的严格警告，PDF 仍然正确。判断是否正常要看 `main.log`：`^! ` 错误数和 undefined reference 数都要为 0。
  `latexmkrc` 处理了中文路径下 BibTeX 的问题，不要删。
- Python：`D:\Anaconda\envs\Marionette\python.exe`。系统的 `python` 是 3.14，没有项目依赖。
- 统计 overfull 时注意：在 Bash 单行里写 `\\hbox` 的正则容易被 shell 吞掉反斜杠，结果显示 0 个也不可信。
  可以用 `Overfull .hbox` 这种写法。目前有 12 处，最差 15.5pt，都在原有正文里，不影响投稿。

## 4. Git 规则（来自 AGENTS.md，必须遵守）

- 每组逻辑修改前后都要提交并推送到 GitHub，推送后核对 `git rev-parse HEAD` 与 `origin/codex/distribution-best-formal-v1` 一致。只完成本地 commit 不算已提交到 GitHub。
- 只暂存本任务相关的文件，不用 `git add .`。不提交数据集、权重、缓存、`main.aux/.log/.bbl` 之类的编译中间文件。`main.pdf` 已被跟踪，可以提交。
- 推送失败时保留本地提交并报告，不强推、不 reset。github.com 443 端口偶尔连不上，可以 `timeout 90 git push ...` 重试几次。
- **不能改动 `experiment_runs/` 下的任何封存产物，也不能重跑实验。** 论文里的所有数字都来自已封存的结果。

## 5. 当前状态

编译正常：0 错误，0 undefined reference，正文到第 8 页，总共 11 页（上限：正文 8 页，含参考文献和附录 12 页）。

| 文件 | 状态 |
|---|---|
| `main.tex` | 标题、作者、摘要、关键词已定稿。CCS 块是旧的，需要重新生成 |
| `sections/intro.tex` | 已改为三阶段主线，4 条贡献 |
| `sections/related.tex` | 还是旧稿，没有覆盖新增的两个阶段（见 T8） |
| `sections/problem.tex` | 已完成 |
| `sections/method.tex` | §4.1–4.8 已完成。缺 Fig.1 |
| `sections/experiments.tex` | 数字已全部按 eval v2 填好。缺 Fig.3、§5.1 计数和成本数字 |
| `sections/conclusion.tex` | 局限 (i)–(x)、结论、附录 A–E。附录里还有 TODO |
| `tables/method-comparison.tex` | 定稿，两张 7 行表 |
| `tables/ablation.tex` | 定稿，8 行，放在附录 C |
| `tables/stagewise.tex` | 定稿，阶段 (a)–(e) |
| `figures/` | **空的** |

剩下 9 处 `\todo`，可以用 `grep -n '\\todo{' paper-www2027/sections/*.tex` 列出。

## 6. 必须遵守的写作边界

这些是已经和用户确认过的事实，改写时不能弱化或拔高：

- **纽约有两项指标仍差于无约束 backbone**：CategoryTransition 0.0183 对 0.0070，G-RANK 0.1914 对 0.1612。纽约的 CategoryTransition、Category、G-RANK 三项仍是 PO-CFG 最好。不能写 "no fidelity loss"、"without any trade-off" 这类说法。伊斯坦布尔六项全部优于 backbone。
- **超参数选择**：ρ/λ/β 是在训练侧校准集上选的（backbone 见过这些数据，不是独立验证集）。进入三种子确认的两组配置**都没有通过全部预设门槛**。测试集指标从未参与选择。不能写成 "pre-registered and passed"。
- **(d)→(e) 改了三个因素**：ρ、β，以及同类别候选池的构造。(d) 的候选池是原 POI + top-15 + 最近 16，(e) 是原 POI + top-11 + 最近 12 + 训练集访问最多的 8 个。不能把这一步的差异归因于单个因素。
- **评估版本**：所有表格都用 evaluation v2。v3（确定性 G-RANK）只在附录里作为交叉验证，最大偏移 0.0032，不改变任何结论。不能把 v3 的数字放进 v2 的行里。
- **Radius 定义**是 `sqrt(mean(distance_to_centroid))`，不是标准的回转半径。
- **误差棒**是固定 checkpoint 下三个采样种子的样本标准差，不代表显著性检验，也不是置信区间。
- **命名对照**：`PCDG-P` 对应中文稿里的 "PCDG"（只有投影），`PCDG-PD` 对应 "方案A（类别一致解码）"，`PCDG`（full）对应 "分布校正版（最佳候选）"。
- 没有实现的东西只能写进 future work：`po_encoding` 全局条件、类别到 POI 的硬 logit 过滤、复杂偏序 DAG、自适应调度、训练期偏序损失。
- Backbone 是 Marionette（前人工作），投影机制改编自 CDD，引用时都要说明，不能当作本文贡献。

## 7. 任务清单（按优先级）

### T1　腾出正文版面（先做，否则图放不进去）

- 正文已经满 8 页，三张图大约需要 0.7–1 页。
- 做法：把 `tables/method-comparison.tex` 的两张表（JSD 和约束）合并成一张宽表；把 `tables/stagewise.tex` 的长表注压缩进 caption。如果还不够，再把 Fig.2 并进 Fig.1。
- 不能删正文中的“残差说明”和“选择协议”段落。
- 验收：插入三张图的占位（可以先用 `\rule`）后，`sec:conclusion` 之前的内容仍在第 8 页以内。

### T2　Fig.3：取舍平面图（核心图，可以完全用脚本生成）

- 内容：横轴 Unsat（%），纵轴 Distance JSD。两个子图，分别是两座城市。每个方法一个点，再用虚线把阶段 (a)→(b)→(c)→(d)→(e) 连起来。
- 数据：只能用 `tables/method-comparison.tex` 和 `tables/stagewise.tex` 里已有的三种子均值，并在脚本中注明来源。
- 输出：脚本放 `figure_build/`，矢量 PDF 输出到 `paper-www2027/figures/tradeoff.pdf`。替换 `experiments.tex:167` 的 `\todo`。

### T3　Fig.1：四阶段框架图

- 现有脚本 `figure_build/generate_method_main_v2.py` 画的是旧版单阶段框架，输出到仓库根目录的 `figures/method_main`，不是 `paper-www2027/figures`。
- 要扩展成四个阶段：投影 → 类别一致解码 → 同类别几何细化 → 分布校正（ρ/λ/β）。图里要明确标出几何细化“约束不变”。
- 必须画和不要画的内容，参考 `method_innovation_map.md` 的 §11–§15。
- 替换 `method.tex:154` 的 `\todo`。

### T4　附录 D：指标定义

- 替换 `conclusion.tex:130` 的 `\todo`，同时删掉 `experiments.tex:50` 指向这里的 `\todo`。
- 照代码写，不要凭印象：
  - 六项 JSD 指标：`evaluations/statistical_metrics.py`（`Get_Statistical_Metrics`、`grank_jsd`、`radius` 等）。
  - 约束指标：`tools/ablation_common.py` 第 95–125 行附近，以及 `evaluations/ovr.py`。
- 重点说明：`strict_ovr` 是在 POI 的**实际类别**（`poi_category[POI]`）上计算的，`token_strict_ovr` 才是在类别 token 上计算的。§4.6 的论证依赖这个区别。

### T5　附录 B：18 组网格表

- 数据：`experiment_runs/ny-ist-distribution-balanced-v1-20261012/screen-selection.json` 和 `confirm-selection.json`。
- 替换 `conclusion.tex:120` 的 `\todo`。如果版面不够，可以只列 18 组的最差比值和平均比值。
- **还有一处需要核对**：`ny-ist-distribution-best-formal-v1-20261013/selected-candidate.json` 里的 `selection_source` 指向 `...-eval-v3-20261013`，说明最终排序用的是校准集上的 v3 重新评估结果。附录 B 目前没写评估版本，要查清楚后补一句。如果 v2 和 v3 下的排序结果不同，要报告给用户，不能自己选一个写进去。

### T6　§5.1 数据集计数

- 来自 `ny-ist-distribution-best-formal-v1-20261013/manifest.json` 的 `profiles`：纽约 train 3160 / test 2108，伊斯坦布尔 train 7035 / test 4914。
- OOD 的定义去 `docs/newyork_ood_experiment.md` 查证后再写。替换 `experiments.tex:18` 的 `\todo`。

### T7　附录 E 和 §5.5：成本

- 只能写实际测过的数字。来源：`experiment_runs/pcdg-geo-v1-tol10-20261010/performance-qualification.json` 和同目录下的 `speed/`。几何细化固定为 200 步，资格条件是中位空间采样时间不超过 backbone 的 1.10 倍。
- 文档里已经说明：离线细化的耗时不等于在线采样总耗时，旧几何实验的速度比例也不能套用到 A+Geo。缺少可比计时的阶段，就在文中如实写“未测量”。PO-CFG 的重训成本如果没有日志，同样这样处理。
- 替换 `conclusion.tex:149` 和 `experiments.tex:175` 的 `\todo`。

### T8　Related Work 对齐新主线

- 现在的 `related.tex` 没有涉及类别一致解码、后处理的空间修正和分布匹配正则这几个方向。
- 只能引用 `references.bib` 里已有的条目，或者经过核实的真实文献。**不要编造引用。** 新增的条目要能在 DBLP 或出版商页面上查到。

### T9　投稿前收尾

- `conclusion.tex:74` 的 `\eta` 核对可以留到 camera-ready，但要把 `\todo` 去掉，改成注释。
- 把 `main.tex` 里的 `\todo` 宏改成空操作，确认 PDF 里没有红字残留。
- 用 ACM 官方工具 <https://dl.acm.org/ccs> 重新生成 CCS 块，不要手改 `concept_id`。
- 两个作者邮箱目前是 `TODO-confirm@hit.edu.cn` 占位符，而且没有 ORCID。**不要猜，向用户要真实值。** 由于 `anonymous` 选项，评审版 PDF 不会显示这些信息，所以投稿不受影响，但要在交付说明里提醒用户。

## 8. 最终验收清单

- [ ] 0 错误，0 undefined reference，0 处 `\todo` 残留
- [ ] 正文不超过 8 页，总页数不超过 12 页
- [ ] 匿名检查：`pdftotext main.pdf - | grep -i "Zhiyi\|Harbin\|hit.edu"` 应该没有结果。参考文献里出现 Kaiqi Zhao 是正常的（GETNext 那篇文献的作者）。PDF 元数据里也不能有真实姓名
- [ ] 标题与注册版本完全一致
- [ ] 全文数字和表格一致，表格数字和第 6 节的事实一致
- [ ] 如果附代码链接，要用匿名仓库
- [ ] 每组修改都已推送，并核对过远程哈希

## 9. 参考资料

- `paper-www2027/README.md`：各章节状态和诚信边界
- `paper-www2027/REGISTRATION.md`：注册时填写的内容
- `docs/distribution-best-formal-v1.md`：最终结果的来源和审计记录
- `docs/category-consistent-decoding-v2.md`、`docs/pcdg-geo-v1.md`、`docs/distribution-balanced-v1.md`：三个阶段的协议
- `g(⋅)设计方案.md` 第 726 行起：中文版的结果总表

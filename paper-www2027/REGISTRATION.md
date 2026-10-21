# WWW 2027 摘要注册填写内容

更新：2026-10-21。本文用于作者核对已注册内容，不代表已经进入或核验投稿系统。
标题、作者顺序、五关键词及当前摘要继续锁定，以 `main.tex` 和作者实际注册记录为准。

## 截止时间（官网已核实，portal 由作者核对）

| 事项 | 官方 CFP（AoE 日终） | 北京时间（UTC+8） |
|---|---|---|
| Abstract Registration | 2026-10-18 | 2026-10-19 20:00 前（已过） |
| Submission Deadline（全文） | 2026-10-25 | 2026-10-26 20:00 前 |

[官方 Research Track 页面](https://www2027.thewebconf.org/research-track-papers/)
于 2026-10-21 重新读取。旧记录的 OpenReview 13:00 UTC / 北京时间 21:00 比官网晚一小时，
本轮未进入 portal，不能确认该记录。按更早的官网时间安排，由作者核对实际系统截止时间。

投稿入口：OpenReview venue `The Web Conference 2027 - long papers`
（Research Track 长文；**不要**用 `Conference_Short_Papers`）。

---

## Title

```text
Training-Free Partial-Order Control for Discrete-Diffusion POI Check-in Generation
```

82 字符。按导师意见（2026-10-14）去掉 "Distribution Correction"，标题只强调 partial-order control；
分布保真相关机制保留在摘要和正文中。

## Authors

按此顺序填写，顺序即作者顺序：

| # | Name | Affiliation | Country | Role |
|---|---|---|---|---|
| 1 | Zhiyi Li | School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen | China | 第一作者 |
| 2 | Kaiqi Zhao | School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen | China | 通讯作者 |

单位英文名（两人相同，可直接复制）：

```text
School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen
```

两位作者的真实邮箱已按作者提供的信息写入 `main.tex`；旧占位符说明已失效。
请作者核对 portal 与源文件是否一致。ORCID 尚未提供，由本人确认，不猜测或代填。

## Abstract

当前长摘要沿用已授权的 `eliminates residual` → `reduces residual` 单处修正；
其余内容保持原样，本轮没有重新改写摘要。作者需确认 portal 中保存的实际版本。
下列为便于核对的纯文本／TeX 摘录，正式排版以 `main.tex` 为准。

```text
Location-based social networks (LBSNs) are a core source of web-scale human-mobility data, yet sharing raw check-in traces is privacy-sensitive, motivating realistic synthetic generation. Beyond statistical realism, many downstream web applications need control over the semantic order of activities—e.g., leisure before dining—a sample-level, cross-position requirement that single-step conditioning cannot express. Enforcing such constraints is commonly assumed to cost distributional fidelity; we show this trade-off is largely separable. We present PCDG, a training-free control stack for a frozen conditional joint temporal + category/POI discrete-diffusion generator. PCDG compiles category partial-order constraints into differentiable energies, projects the category-position logits through a KL-preserving augmented-Lagrangian optimization in the late reverse steps, reduces residual category-POI inconsistency via a category-consistent decoder, and restores spatial realism through a distribution-aware same-category geometry refinement that leaves every constraint metric unchanged by construction. Across two cities under an out-of-distribution split and three sampling seeds, PCDG cuts the strict ordering-violation rate from 92.89\% to 4.04\% on Istanbul and from 89.45\% to 11.28\% on New York, raising pairwise ordering coverage by 6.5x and 4.2x, while improving all six fidelity divergences over the unconstrained backbone on Istanbul and four of six on New York. On Istanbul it surpasses a fully retrained partial-order conditional model on all five constraint metrics and five of six fidelity divergences, without retraining anything. Stage-wise ablations localize where fidelity is lost and recovered across the three-stage stack.
```

### 历史短版（非当前锁定摘要，不用于替换）

```text
Location-based social networks are a core source of web-scale human-mobility data, yet sharing raw check-in traces is privacy-sensitive, motivating synthetic generation. Many web applications additionally need control over the semantic order of activities—leisure before dining—a sample-level, cross-position requirement that single-step conditioning cannot express. Enforcing such constraints is commonly assumed to cost distributional fidelity; we show the trade-off is largely separable. We present PCDG, a training-free control stack for a frozen joint temporal + category/POI discrete-diffusion generator: an augmented-Lagrangian projection of the category-position logits, category-consistent POI decoding, and a constraint-neutral, distribution-aware geometry refinement. Across two cities under an out-of-distribution split and three seeds, PCDG cuts strict ordering violations from 92.89% to 4.04% and from 89.45% to 11.28%, while improving all six fidelity divergences over the unconstrained backbone on one city and four of six on the other, surpassing a fully retrained conditional model without any retraining.
```

## Keywords

表单限 5 个，与 `main.tex` 的 `\keywords` 已同步：

```text
synthetic human mobility, discrete diffusion, partial-order constraints, training-free constrained sampling, POI check-in sequences
```

五关键词与当前 `main.tex` 一致；不替换或新增。标题不包含 distribution correction。

## Track

Research Track — Web Mining, Multimedia and Multilingual Content Analysis

## CCS

五项 CCS 已于 2026-10-18 从官方工具导出并接入 `main.tex`，本轮一致性复核通过。
无需重复生成或手改编号。

---

## 全文提交前由作者确认

1. 已完成摘要注册；实际题名、作者顺序、邮箱、摘要和所选 track 与锁定内容一致。
2. 阅读 `REVIEW-AND-FIXES.md` 中“摘要锁定未改”“证据不足”和“待作者确认”项。
3. 确认数据许可、伦理要求、作者责任、AI 使用披露（如适用）、COI 与无并行投稿。
4. 核对 portal 截止时间，上传经验证的匿名 `main.pdf`，不要上传含作者身份的源包。

框架图、权衡图、指标附录、校准网格、成本表和 train/test 计数均已交付；
旧“待补图／TODO”的说明不再适用。当前构建状态见 `README.md` 和交接记录。

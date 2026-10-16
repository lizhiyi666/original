# WWW 2027 摘要注册填写内容（截止 2026-10-18 AoE）

注册会**锁定标题与作者列表**。以下内容可直接粘贴进表单。
正文全文截止 2026-10-25。

---

## Title

```text
Training-Free Partial-Order Control and Distribution Correction for Discrete-Diffusion POI Check-in Generation
```

109 字符。

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

> **需要你自己填的两项**：两位作者的真实 e-mail，以及（表单若要求）ORCID。
> `main.tex` 里目前是 `TODO-confirm@hit.edu.cn` 占位符 —— 我没有这两个真实值，
> 没有替你猜。ORCID 必须从 orcid.org 本人账号查，填错会把工作归到他人名下。

## Abstract

254 words。若表单限 250 words，用后面的短版。

```text
Location-based social networks (LBSNs) are a core source of web-scale human-mobility data, yet sharing raw check-in traces is privacy-sensitive, motivating realistic synthetic generation. Beyond statistical realism, many downstream web applications need control over the semantic order of activities—e.g., leisure before dining—a sample-level, cross-position requirement that single-step conditioning cannot express. Enforcing such constraints during sampling is commonly assumed to cost distributional fidelity; we show that this trade-off is largely separable, and we remove most of it. We present PCDG, a training-free control stack for a frozen conditional joint temporal + category/POI discrete-diffusion generator. From a sample-level category partial-order matrix, PCDG compiles differentiable order-violation and existence energies and, in the late reverse steps, projects only the category-position logits through a KL-preserving augmented-Lagrangian optimization; a category-consistent POI decoder then eliminates residual category-POI inconsistency, and a distribution-aware same-category geometry refinement—which leaves every constraint metric unchanged by construction—restores spatial realism by matching population-level travel-length and visitation statistics. Across two cities under an out-of-distribution split and three sampling seeds, PCDG cuts the strict ordering-violation rate from 92.89% to 4.04% on Istanbul and from 89.45% to 11.28% on New York, raising pairwise ordering coverage by 6.5x and 4.2x, while improving all six fidelity divergences over the unconstrained backbone on Istanbul and four of six on New York. On Istanbul it surpasses a fully retrained partial-order conditional model on all five constraint metrics and five of six fidelity divergences, without retraining anything. Stage-wise ablations localize where fidelity is lost and recovered, and we report the two New York statistics that remain above the backbone.
```

### 短版（150 words，仅在表单硬限 250 以下时使用）

```text
Location-based social networks are a core source of web-scale human-mobility data, yet sharing raw check-in traces is privacy-sensitive, motivating synthetic generation. Many web applications additionally need control over the semantic order of activities—leisure before dining—a sample-level, cross-position requirement that single-step conditioning cannot express. Enforcing such constraints is commonly assumed to cost distributional fidelity; we show the trade-off is largely separable and remove most of it. We present PCDG, a training-free control stack for a frozen joint temporal + category/POI discrete-diffusion generator: an augmented-Lagrangian projection of the category-position logits, category-consistent POI decoding, and a constraint-neutral, distribution-aware geometry refinement. Across two cities under an out-of-distribution split and three seeds, PCDG cuts strict ordering violations from 92.89% to 4.04% and from 89.45% to 11.28%, while improving all six fidelity divergences over the unconstrained backbone on one city and four of six on the other, surpassing a fully retrained conditional model without any retraining.
```

## Keywords

```text
synthetic human mobility, controllable generation, discrete diffusion, training-free constrained sampling, partial-order constraints, distribution correction, POI check-in sequences, LBSN
```

## Track

Research Track — Web Mining, Multimedia and Multilingual Content Analysis

## CCS

摘要注册阶段**不需要** CCS。全文截止前用官方工具 <https://dl.acm.org/ccs> 重新生成整块，
不要手改 `concept_id`。

---

## 注册后、全文截止前（10-25）必须完成

1. `sections/method.tex` 需要 Fig.1（四阶段框架）与 Fig.2（单个投影步）。
2. Fig.3：Unsat × Distance JSD 平面，每方法一个点 + 阶段 (a)–(e) 的虚线路径。
3. 三个 `\todo`：`app:metrics` 指标定义、`app:cost` 成本数字、`app:grid` 的 18 组网格表。
4. §5.1 的 OOD train/test 计数。
5. 页数已满（正文到 p8，总 11/12）。插图前需要腾位置，见 `README.md`。

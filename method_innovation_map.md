# Method Innovation Map

本文件用于后续硕士论文方法主图设计。判断优先级为：**代码真实性 > 开题报告/论文叙事 > Marionette 视觉参考**。Marionette 只提供视觉组织启发，不用于推断本项目的算法机制。

标记含义：

- `[Confirmed]`：代码中有明确、可追踪的实现证据。
- `[Strongly Supported]`：报告与代码方向一致，但部分实现或配置仍需谨慎表述。
- `[Uncertain]`：存在代码路径或论文主张，但接入、默认状态或语义仍无法稳定确认。
- `[Not Found]`：在当前仓库代码中未找到。

## 1. Research Problem

### 1.1 论文问题定义

开题报告将问题定义为：面向包含连续到达时间、活动类别和 POI 的完整打卡序列，在保持统计真实性、时空合理性和多样性的同时，满足样本级活动类别偏序关系，例如 `A ≺ B`。报告进一步提出复杂多组偏序约束下的真实性、约束满足和采样效率协调问题。

### 1.2 代码能够支持的问题范围

`[Confirmed]` 当前代码可以支持以下较窄且明确的问题：

```text
给定带有 time / context / po_matrix 的样本条件，
先生成时间事件，再对 category + POI token 做联合离散扩散采样，
并在 reverse diffusion 的部分步骤中对 category logits 做偏序投影，
最后导出 POI 序列和 GPS。
```

`[Not Found]` 当前代码不能证明以下更强的问题已经实现：

- 约束矩阵作为普通 Transformer 全局条件 embedding 输入；
- category 与 POI 的硬一致性候选筛选；
- 复杂偏序 DAG 的传递闭包、冗余删除、冲突/环路检测；
- 基于当前满足程度的自适应约束强度和施加时机；
- 完整可用的 classifier-free guidance 双分支；
- 训练期偏序损失已经稳定有效。

## 2. Code-grounded Method Pipeline

下表重建从数据到输出的真实调用链。`基础/新增` 是相对于原 Marionette/Add-Thin/离散扩散骨架的分类，不代表所有代码都由本论文首次提出。

| 阶段 | 模块与代码证据 | 输入 | 输出 | 作用 | 分类 |
|---|---|---|---|---|---|
| Input | `datamodule.py:load_sequences`, `Sequence.__init__` | pkl 中的 `arrival_times`, `marks`, `checkins`, conditions, `poi_gps`, 可选 `po_matrix`/`po_encoding` | `Sequence` 列表 | 读取时间、category、POI 和条件 | 基础数据接口 |
| Preprocessing | `tools/prepare_newyork_po1.py:_parse_source`, `_filter_pois`, `_build_daily_sequences`, `prepare_dataset` | 原始签到记录 | 时间排序序列、category/POI 词表、条件字段、GPS、POI-category 映射 | 本地时间转换、去重、长尾过滤、token 化 | `[Strongly Supported]` 数据增强/新增管线 |
| Relation construction | `tools/prepare_newyork_po1.py:_make_po_matrix`；补充脚本 `tools/make_Istanbul_withPO.py:get_full_partial_order_matrix` | 每条序列的 category token | `po_matrix` | 若 A 的末次位置严格早于 B 的首次位置，置 `M[A,B]=1` | 核心创新的约束表示 |
| Representation | `datamodule.py:Batch.from_sequence_list` | `Sequence` 列表 | `checkin_sequences`、`category_mask`、`poi_mask`、`mask`、`tau` | 拼接 `[START]+category+[SEP]+POI+[END]`，建立位置 mask | 基础/必要表示 |
| Condition | `add_thin/diffusion/model.py:AddThin.compute_emb`; `discrete_diffusion/diffusion_transformer.py:ConditionEmbeddingModel.forward` | time、tau、condition1..6、indicator | 时间分支 embedding、离散扩散 `cond_emb` | 编码时间和上下文条件并通过 cross-attention 使用 | 基础方法 |
| Temporal forward | `add_thin/diffusion/model.py:AddThin.noise`; `add_thin/processes/hpp.py:generate_hpp` | 原始时间 Batch、Add-Thin step | thinned events + HPP events `x_n` | 时间点过程的加噪 | 基础方法 |
| Temporal reverse | `AddThin.sample`, `sample_posterior`, `sample_x_0` | HPP 初始事件和条件 | 生成时间事件 Batch | 逐步还原事件时间/事件集合 | 基础方法 |
| Discrete forward | `discrete_diffusion/diffusion_transformer.py:alpha_schedule`, `q_pred`, `q_sample`, `log_sample_categorical` | clean token、t、category/POI mask | noisy token `x_t` | 对 category 和 POI 使用不同离散转移并采样 | 基础方法 |
| Denoiser | `discrete_diffusion/conditional_attention.py:Transformer`; `DiffusionTransformer.predict_start` | `x_t`, `cond_emb`, t、token type | `p(x_0|x_t)` logits | token/position/time/type embedding + Transformer decoder + condition cross-attention | 基础方法 |
| Reverse discrete sampling | `DiffusionTransformer.p_pred`, `p_sample`, `sample_fast` | 初始 mask token、条件、约束 | 逐步去噪 token | 从高噪声状态倒序得到 category+POI token | 基础方法，约束插入点在此 |
| Constraint parse | `constraint_projection.py:parse_po_matrix_to_constraints`, `_compile_constraints`, `compile_batched_constraints` | `batch.po_matrix` | `(A,B)` 约束集合、`W_A/W_B` | 将矩阵边编译成概率计算需要的矩阵 | 核心新增机制 |
| Constraint energy | `ConstraintProjection.compute_constraint_violation_optimized` | model log-prob、category mask、`W_A/W_B` | order violation、existence violation | 计算“前 B 后 A”逆序能量和类别存在性惩罚 | 核心新增机制 |
| Logit projection | `ConstraintProjection.project_with_matrices` | 原始 model logits、约束能量 | 投影后 logits | KL 保真 + 增广拉格朗日优化，使用 softmax/Gumbel-softmax | 核心新增机制 |
| Constraint-aware sampling | `DiffusionTransformer.p_sample` | `p_pred` logits、投影开关、频率/最后 K 步 | 采样分布 | 满足条件时只改 category positions 的 logits，然后全序列 Gumbel-max 采样 | 核心新增机制 |
| POI extraction | `datamodule.Batch.to_seq_list` | 生成 token Batch、`gps_dict` | `checkins`, `marks`, `arrival_times`, `gps` 等字典 | 用 `poi_mask` 提取 POI，用 GPS 字典反序列化 | 基础输出接口 |
| Output | `sample.py:simulation` 保存逻辑 | `to_seq_list` 结果 | `data/<name>/*_generated_part*.pkl` | 保存生成序列及 `t_max` | 基础实验基础设施 |

### 2.1 真实数据流

```text
PKL / raw check-ins
→ Sequence / Batch
→ [time, category, POI, context, masks, optional po_matrix]
→ Add-Thin temporal generation
→ category+POI discrete forward/reverse diffusion
→ p_sample: optional category-logit projection
→ joint Gumbel-max sampling
→ Batch.to_seq_list
→ generated POI sequence + GPS
```

## 3. Thesis-grounded Method Description

### 3.1 研究目标

报告 1.2.1 和 3.1.1 的核心目标是：

1. 将 POI、活动类别和时间建模为联合生成对象；
2. 将“类别 A 先于类别 B”形式化为样本级偏序约束；
3. 将约束与离散扩散 reverse sampling 结合；
4. 在约束满足和生成真实性之间进行权衡。

`[Strongly Supported]` 1–4 均能在当前代码中找到相应主链，最强证据是 `DiffusionTransformer.p_sample` + `ConstraintProjection`。

### 3.2 技术路线与方法描述

报告 3.2.1/3.2.2 的已实现叙事是：

```text
时空语义表示与偏序形式化
→ 条件时间点过程 + category/POI 联合离散扩散
→ 由偏序矩阵得到约束集合
→ 计算顺序/存在性惩罚
→ reverse diffusion 后期 category-logit KL-ALM 投影
→ category 与 POI 联合采样
```

`[Confirmed]` 这条主线与代码调用链吻合。

### 3.3 报告中的创新陈述与 Claim → Evidence

| 论文 Claim | 代码 Evidence | 判定 |
|---|---|---|
| “在 POI 打卡序列生成中引入活动类别偏序约束” | `sample_fast` 读取 `batch.po_matrix`；`p_sample` 调 `ConstraintProjection` | `[Confirmed]` |
| “依据类别首次/末次出现位置构造偏序矩阵” | `tools/prepare_newyork_po1.py:_make_po_matrix` | `[Confirmed]` |
| “将偏序关系转成可微偏序违反能量” | `compute_constraint_violation_optimized` 的 soft probabilities、prefix cumulative、order violation | `[Confirmed]` |
| “引入存在性惩罚，避免删除类别规避约束” | `viol_exist_A/B`、`projection_existence_weight` | `[Confirmed]` |
| “KL 保真项 + 增广拉格朗日 + Gumbel-Softmax 投影” | `project_with_matrices` 的 KL、lambda/mu 外层更新、soft/Gumbel path | `[Confirmed]` |
| “投影只作用于类别位置 logits” | `p_sample` 传入 `batch.category_mask`；投影器只用 category slice | `[Confirmed]` |
| “在 reverse 后期按频率施加” | `projection_frequency` 与 `projection_last_k_steps` 条件 | `[Confirmed]` |
| “投影后类别与 POI 共同采样” | `p_sample` 将投影后 `model_log_prob` 送入全词表 `log_sample_categorical` | `[Strongly Supported]`；共同采样存在，但没有 POI-category 硬一致性筛选 |
| “训练期偏序辅助损失” | `PartialOrderLoss` + `DensityEstimation.step` 接口 | `[Uncertain]`；全词表 logits 与 category-only loss 语义存在维度不匹配风险 |
| “偏序低维编码作为全局条件” | `po_encoding` 在数据结构中存在，但 `ConditionEmbeddingModel.forward` 未读取 | `[Not Found]` 作为当前 denoiser 条件 |
| “复杂偏序 DAG 的传递/冗余/冲突检测” | 当前仅矩阵逐边解析和固定投影参数 | `[Not Found]` |
| “自适应约束强度/时机” | 当前只有固定 `projection_frequency`、`projection_last_k_steps` 和 ALM 内部固定规则 | `[Not Found]` |
| “CFG baseline 完整可用” | `sample.py` 有参数名，但没有完整无条件/有条件双分支组合 | `[Not Found]` |

### 3.4 报告与代码的配置冲突

`[Uncertain]` 报告 5.1/7.1 将训练期偏序辅助损失描述为“当前默认关闭/拟研究”，但仓库的 `config/train.yaml` 和 `config/task/density.yaml` 都出现 `po_loss_weight: 0.2`。这说明“论文叙事中的默认状态”和“当前仓库配置文件”并不完全一致。由于 `PartialOrderLoss` 还存在 logits 维度接入疑点，主图应把训练期辅助损失标为探索项或省略，而不能画成稳定核心路径。

## 4. Three-source Consistency Analysis

Marionette 列只表示视觉组织启发：分区、层级、主流程、局部反向过程放大、弱化基础模块并突出一个贡献。它不提供本项目的算法证据。

| 主要模块 | 开题报告/论文描述 | 代码真实实现 | Marionette 视觉启发 | 三源结论 |
|---|---|---|---|---|
| 输入与表示 | 时间、活动类别、POI、上下文条件 | `Sequence`/`Batch` 及 token layout | 用一个输入面板集中表达语义对象 | `[Strongly Supported]`，视觉可借鉴，内容以代码为准 |
| 条件编码 | 多类上下文经 Transformer 编码并注入生成 | `ConditionEmbeddingModel` + `Transformer` cross-attention；Add-Thin 另有 temporal condition encoder | 条件从上方/侧方汇入主模型 | `[Confirmed]` |
| 时间生成 | 条件时间点过程 | `AddThin` + HPP/Thin | 基础生成分支用低饱和色弱化 | `[Confirmed]` |
| category/POI 离散扩散 | 独立转移、x0 参数化、联合反向采样 | `alpha_schedule`、`q_pred/q_sample`、`p_pred/p_sample` | 用一个核心生成面板表达多步反向过程 | `[Confirmed]` |
| 偏序矩阵 | 首末位置定义严格“全部先于” | `_make_po_matrix` | 约束输入可做侧边数据对象 | `[Confirmed]` |
| 偏序能量 | 逆序概率积 + 存在性 | `compute_constraint_violation_optimized` | 用局部放大框解释机制 | `[Confirmed]` |
| 可微投影 | KL + ALM + Gumbel，category-only | `project_with_matrices` 在 `p_sample` 中执行 | 创新用唯一强调色贯穿能量→投影→采样 | `[Confirmed]` |
| POI 候选/可行集 | 报告语义容易让读者联想到可行 POI 控制 | 没有 candidate filtering 或 category→POI restriction | 不应从视觉参考推断候选筛选 | `[Not Found]` |
| 训练偏序损失 | 报告称拟研究并实现 | 有接口但维度/配置状态不稳定 | 不应作为主面板核心色 | `[Uncertain]` |
| 复杂 DAG 自适应 | 作为研究问题二计划 | 没有对应算法链 | 不应画成完成模块 | `[Not Found]` |
| 输出与评价 | 完整 POI 序列、约束和 JSD 评价 | `to_seq_list`、`evaluations`、采样脚本 | 输出面板简洁，评价可置图外 | `[Strongly Supported]` |

### 4.1 三者可以一致的部分

- 用分区和局部 inset 组织“输入—生成—约束—输出”；
- 把已有生成骨架和论文核心约束分成不同视觉层级；
- 用单一强调色突出 category-order constraint；
- 在 reverse sampling 局部展开“预测 → 干预 → 采样”。

### 4.2 只存在于代码或只适合代码审计的部分

- `W_A/W_B` 编译、tensor reshape、debug 输出、GPU 分块、checkpoint 和 pkl 字段校验；
- `ConstraintClassifier` 独立训练脚本和 `classifier_guidance_injection.py` 未接入函数；
- `Batch.to_seq_list` 的 GPS 缺失删除。

### 4.3 只存在于论文叙事或尚未实现的部分

- `po_encoding` 作为 denoiser 全局条件；
- 复杂偏序 DAG 自适应处理；
- 完整 CFG；
- 训练期偏序损失的稳定、已验证贡献；
- category→POI 硬可行候选集。

## 5. Existing Components

以下模块有代码实现，并且主要承担通用生成骨架或数据基础，不应包装为论文核心创新。

### Category A：Problem / Input

- 时间事件 `arrival_times` / `time`；
- category/marks 序列；
- POI/checkins token 序列；
- 六类事件/序列上下文条件；
- `poi_gps`、`poi_category` 元数据；
- 可选样本级 `po_matrix`。

### Category B：Baseline / Existing Components

- `datamodule.Sequence`、`Batch` 的变长事件批处理；
- `AddThin` 时间点过程及 HPP/Thin；
- `PointClassifier`、`MixtureIntensity` 时间 backbone；
- 离散 `alpha_schedule`、category/POI 转移矩阵；
- `q_pred`、`q_sample`、`q_posterior`；
- `conditional_attention.Transformer` denoiser；
- token/position/time/type embedding 与条件 cross-attention；
- 普通 Gumbel-max 离散采样；
- `Batch.to_seq_list` 输出反序列化。

### Category D：Implementation Details

- `index_to_log_onehot`、`log_onehot_to_index`、`extract`、log-space 工具；
- padding、mask、tensor reshape、batch collation；
- WandB、Hydra、checkpoint、optimizer、manual optimization；
- `rank/world_size` 分块采样；
- shape/debug print、异常处理；
- GPS 字符串解析与缺失项删除；
- SVD scaler 的存储细节。

## 6. My Core Contribution

### 核心创新定位

`[Confirmed]` 相对于已有条件时空联合生成和 category/POI 离散扩散骨架，本项目实际新增的核心机制是：

> 将每条序列的类别偏序矩阵转化为可微的顺序/存在性能量，并在离散扩散 reverse sampling 的后期，以 KL 保真的增广拉格朗日投影修改 category-position logits，再与 POI logits 共同采样。

它是一个**采样期、类别位置上的结构化控制层**，不是新的 denoiser，也不是独立的 POI 候选生成器。

### 为什么属于论文贡献

1. 报告将“类别偏序约束下的 POI 序列生成”定义为核心研究问题一；
2. 报告将“偏序违反能量 + 存在性惩罚 + KL/ALM/Gumbel 投影”明确列为 PCDG 核心；
3. 代码在 `p_sample` 的模型分布和最终离散采样之间提供了实际插入点；
4. 投影只作用于 category positions，保留了联合 category/POI 采样接口，形成与普通无约束 diffusion 的可辨识差异。

### 不应被包装成核心创新的内容

- Add-Thin 时间过程；
- Transformer denoiser；
- 普通离散扩散 q/p 过程；
- 普通 Gumbel-max 采样；
- SVD `po_encoding`（当前未进入模型）；
- PostSwap、EnergyGuide、CFG（对照/未完成路径）；
- 任何普通 batching 或工程设施。

## 7. Code Evidence of the Contribution

### 7.1 约束构造

`tools/prepare_newyork_po1.py:_make_po_matrix`：

```text
first[A] = A 的首次位置
last[A]  = A 的末次位置
M[A,B] = 1  iff last[A] < first[B]
```

这对应报告中“全部 A 出现位置严格早于全部 B 出现位置”的严格偏序定义。

### 7.2 约束进入 reverse process

`DiffusionTransformer.sample_fast`：

```text
batch.po_matrix
→ parse_po_matrix_to_constraints
→ po_constraints
→ p_sample(..., po_constraints, diffusion_index)
```

### 7.3 约束能量

`ConstraintProjection.compute_constraint_violation_optimized`：

- 将 `[B,V,L]` logits 转成 `[B,L,V]`；
- 可选 Gumbel-softmax，否则 softmax；
- 用 `category_mask` 只保留 category positions；
- 取 category vocabulary slice；
- 用 `W_A/W_B` 得到每条约束的 A/B 概率；
- 用 B 的 prefix cumulative 与当前 A 概率计算 order violation；
- 对 A/B 总概率不足 1 计算 existence violation。

### 7.4 约束投影

`ConstraintProjection.project_with_matrices`：

```text
y_model = original logits
y = trainable projected logits
loss = KL(log_softmax(y), log_softmax(y_model))
     + ALM(order_violation)
     + existence_weight * ALM(existence_violation)
```

通过内层梯度更新和外层 `lambda/mu` 更新生成投影结果。

### 7.5 采样位置与作用范围

`DiffusionTransformer.p_sample`：

```text
model_log_prob, log_x_recon = p_pred(...)
→ optional projection(model_log_prob, category_mask)
→ optional energy guidance baseline
→ log_sample_categorical(model_log_prob)
```

`[Confirmed]` 当前实际修改的是**采样分布的 logits**，不是 candidate set，不是 transition matrix，也不是 POI token mask。

## 8. Position of the Contribution in the Generation Pipeline

### Existing Method

```text
Input time/context + POI/category training sequences
        ↓
Batch tokenization and condition encoding
        ↓
Forward diffusion q(x_t | x_0)
        ↓
Transformer predicts p(x_0 | x_t)
        ↓
Posterior p(x_{t-1} | x_t) and ordinary sampling
        ↓
Generated category + POI sequence
```

### My Method

```text
Input time/context + sample-level po_matrix
        ↓
Batch tokenization and condition encoding
        ↓
Forward diffusion q(x_t | x_0)
        ↓
Transformer predicts joint category/POI logits
        ↓
Reverse posterior distribution
        ↓
┌─────────────────────────────────────────────────────┐
│ MY CONTRIBUTION: category-order constrained sampling │
│ po_matrix → constraint set → order/existence energy  │
│ → KL-preserving ALM projection on category logits    │
└─────────────────────────────────────────────────────┘
        ↓
Joint Gumbel-max sampling of projected category + POI
        ↓
Generated POI sequence
```

明确结论：**Constraint is applied during inference / reverse generation.**

训练期另有一条候选路径：`DensityEstimation.step` → `PartialOrderLoss`。但由于全词表 logits 接入 category-only 偏序损失存在维度/语义不匹配风险，且报告将其列为拟研究项，因此不能把它与主采样约束画成同等确定的路径。

## 9. Actual Constraint Mechanism

### 9.1 实际改变什么

`[Confirmed]` 约束改变的是：

- reverse step 中模型预测的 `model_log_prob` / logits；
- 仅在 `category_mask` 标记的 category positions 生效；
- 通过 KL 项尽量保持靠近原模型分布；
- 经过投影后的全序列分布再执行 Gumbel-max 离散采样。

### 9.2 实际没有改变什么

`[Not Found]` 当前没有证据表明约束：

- 改变 POI candidate set；
- 生成可行 POI 集合；
- 直接修改 category→POI transition probability；
- 对非法 POI 建立显式 `-inf` mask；
- 逐步过滤或拒绝不满足约束的 POI token；
- 在 `Transformer` 内部增加偏序 attention；
- 将 `po_encoding` 作为普通 condition embedding。

### 9.3 Guidance 的准确定位

`[Confirmed]` `p_sample` 中确实存在可选 energy guidance 分支，但它是 `sample.py --baseline energy_guidance` 的对照路径，使用约束违规能量梯度，不应与 PCDG 的 ALM 投影混称为同一机制。

`[Not Found]` `ConstraintClassifier`、`train_classifier.py`、`classifier_guidance_injection.py` 没有被当前 `sample.py` 的 energy guidance 主链加载或调用；不能把它们画成当前核心分类器引导。

## 10. Training vs Inference Role

| 约束路径 | 代码位置 | 阶段 | 当前状态 | 主图处理 |
|---|---|---|---|---|
| `PartialOrderLoss` | `tasks.py:DensityEstimation.step`, `partial_order_loss.py:PartialOrderLoss.forward` | training | `[Uncertain]` 有配置和接口，但全词表 logits 与 category-only loss 维度语义存在风险；报告也称拟研究/当前默认关闭 | 主图默认不画；可在训练细节图或虚线“探索项”表示 |
| ALM projection | `constraint_projection.py`, `DiffusionTransformer.p_sample` | inference / reverse generation | `[Confirmed]` 主约束路径 | 主图必须突出 |
| Energy guidance | `p_sample` + `sample.py --baseline energy_guidance` | inference | `[Confirmed]` 可选基线，不是主 PCDG | 放实验对照图 |
| PostSwap | `baseline_posthoc_swap.py`, `sample.py` | post-processing | `[Confirmed]` 生成后重排基线 | 不放 PCDG 主图 |
| CFG | `sample.py` 参数名 | inference | `[Not Found]` 完整双分支未实现 | 不放主图 |

结论：当前可稳定宣称的核心约束作用发生在**推理期的 reverse generation**；训练期偏序损失只能作为待修正/探索分支。

## 11. Main Figure Must-show Elements

主图应让答辩委员只看一张图就能回答以下问题。

| 必答问题 | 主图位置 | 细节需求 | 强调色 | Panel |
|---|---|---|---|---|
| 1. 解决什么问题？ | 图题或左上角一句短问题陈述：`POI sequence generation under category partial-order constraints` | 不需要公式 | 偏序强调色可用于关键词 | (a) |
| 2. 输入是什么？ | 左侧 Input/Representation 框 | 展示时间、category、POI、context、`M` 五类对象 | 输入低饱和色 | (a) |
| 3. 基础生成框架是什么？ | 中央 Joint Generator | 标出 Add-Thin 时间分支和 category/POI discrete diffusion | 基础色 | (a) |
| 4. 条件信息是什么？ | generator 上方或侧方 Condition Encoder | 只写 time/context conditions；不要把 `po_encoding` 画成已接入条件 | 基础色 | (a) |
| 5. 如何生成 POI 序列？ | 中央到右侧 reverse sampling | 表达联合 category/POI token denoising 和最终解码 | 基础色 | (a) |
| 6. 创新在哪里？ | reverse sampling 前的彩色高亮插入框 | `M → energy → category-logit projection` | 唯一强调色 | (a)+(b) |
| 7. 偏序如何进入？ | Panel (b) 左侧 `po_matrix` 输入约束解析器 | 展示 A≺B 和 order/existence energy | 唯一强调色 | (b) |
| 8. 约束影响什么？ | Panel (b) 中央 `category logits only` 标签 | 明确“不改变 candidate set；POI logits 不直接投影” | 强调色 | (b) |
| 9. 输出是什么？ | 右侧 Output 框 | 展示生成 `category + POI + time` 序列，POI 可映射 GPS | 输出低饱和色 | (a) |

### 11.1 主图逻辑主干

```text
Input event sequence / context
→ Conditional joint temporal + discrete generator
→ Reverse denoising logits
→ [Category-order constrained projection]
→ Joint category/POI sampling
→ Generated POI sequence
```

### 11.2 必须用文字明确的限定

主图中应直接放置短标签：

- `Inference-time / reverse diffusion`；
- `Category positions only`；
- `KL-preserving ALM projection`；
- `Joint sampling with POI logits`；
- `No candidate filtering`（可放 Panel (b) 的灰色注释，防止读者误解）。

## 12. Detail Figure Must-show Elements

Panel (b) 或局部放大图应展开以下真实机制：

1. `po_matrix` 中一条 `A→B` 边；
2. `parse_po_matrix_to_constraints` 产生 `([A],[B])`；
3. category logits 经 softmax/Gumbel-softmax；
4. `category_mask` 选出 category positions；
5. `P_A(i)`、`P_B(i)` 与 B-prefix 累积；
6. `order violation + existence violation`；
7. `KL(original || projected) + ALM penalty`；
8. category logits 投影后的输出；
9. POI logits 沿原模型分布保留；
10. 两者共同进入 `log_sample_categorical`。

可以在细节图边缘注明代码证据：`ConstraintProjection.project_with_matrices`、`DiffusionTransformer.p_sample`。不需要在主图放所有函数名。

## 13. Information Hierarchy

### Level 1：读者第一眼必须看到

- 输入 POI event representation；
- Conditional joint diffusion framework；
- Category-order constrained reverse generation；
- Generated POI sequence；
- 唯一的核心创新色链条：`M → energy → category-logit projection`。

### Level 2：仔细阅读后理解

- time Add-Thin 与 category/POI discrete diffusion 的分工；
- condition encoder 的输入；
- category/POI token layout；
- reverse 后期、按频率施加；
- KL 保真、ALM、Gumbel-softmax；
- 约束只作用于 category positions，POI 与 category 一起采样。

### Level 3：正文或细节图才需要

- 具体 class/function 名；
- `W_A/W_B` 张量和维度；
- `lambda/mu` 更新、梯度裁剪、temperature；
- `alpha_schedule` 数组；
- tensor transpose、padding token、debug 参数；
- SVD、GPS 清理、并行采样、checkpoint。

Level 1 必须最突出；Level 2 以短标签或局部放大呈现；Level 3 默认不进入方法主图。

## 14. Do NOT Draw

以下内容会导致科研失真，方法主图禁止绘制：

- `[Not Found]` 基于偏序的 POI candidate filtering；
- `[Not Found]` 非法 POI 的硬 mask 或 transition restriction；
- `[Not Found]` category→POI 的显式一致性投影；
- `[Not Found]` 将 `po_encoding` 输入 Condition Encoder 的箭头；
- `[Not Found]` 复杂偏序 DAG、传递闭包、冗余消除、环路检测；
- `[Not Found]` 基于当前满足程度的自适应投影强度/时机；
- `[Not Found]` 完整 CFG 双分支；
- `[Uncertain]` 把 `PartialOrderLoss` 画成已验证、必经的训练主损失；
- `[Not Found]` 把 `ConstraintClassifier` 画成当前 energy guidance 主分类器；
- 把 PostSwap 画成 reverse diffusion 内部步骤；
- 把 energy guidance baseline 画成 PCDG 的 ALM 投影本体；
- 把 `po_matrix` 画成普通 token embedding 条件；
- 把普通 Transformer、Add-Thin、Gumbel-max、padding、GPS 解析包装成论文创新；
- 从 Marionette 借来的算法模块、变量、图标、面板标题或具体箭头路径；
- 任何只有论文声称、但当前代码无法确认的概率重加权、可行集构造或损失项。

## 15. Recommended Figure Layout

### Panel (a)：Overall Framework

采用紧凑的左到右主流程，但避免复制 Marionette 的三条横向大带：

```text
[Input event representation]
          →
[Condition encoding + temporal branch]
          →
[Joint category/POI discrete diffusion]
          →
[Reverse denoising logits]
          →
[Generated POI sequence]
```

具体组织：

- **左侧**：一个窄而信息密度适中的 Input 框，包含 `time`、`category`、`POI`、`context` 和侧边输入 `po_matrix`；
- **中间**：大容器 `Conditional Joint Generator`，内部并列放时间 Add-Thin 与 category/POI discrete diffusion；
- **中右侧**：用强调色边框标出 `Reverse-step control` 插入点，箭头从 model logits 进入 category-only projection；
- **右侧**：Output 框展示带时间、类别、POI 的生成序列，并用一句小字说明 GPS 可由 `poi_gps` 映射；
- **上方条件线**：从 context/time encoder 短距离汇入生成器，避免跨图长曲线；
- **主数据流**：粗实线；reverse iteration 用短虚线/局部反向时间标识。

### Panel (b)：Core Constraint Mechanism

Panel (b) 只放大一个 reverse step：

```text
[Sample-level po_matrix]
          ↓
[A ≺ B constraint parsing]
          ↓
[Order + existence energy]
          ↓
[KL-preserving ALM projection]
          ↓
[Category logits only]
          ┘
[POI logits unchanged]
          ↓
[Joint category/POI sampling]
```

建议用一条强调色粗线串起约束链，用灰色线表示原始 denoiser logits 和 POI logits。Panel (b) 必须给出“projection occurs before joint sampling”的空间顺序。

### 15.1 视觉原则

- 基础模块低饱和蓝/灰/绿；偏序机制唯一高饱和强调色；
- 容器、核心模型、机制细节使用三级框层级；
- 主图最多保留一条主流程和一条约束侧链；
- `category positions only` 用文字 + mask 小图双重编码；
- 约束矩阵、能量、投影和采样可编号 `①–④`，正文再展开。

## 16. Method Innovation Map

```text
Problem
  ↓
POI event sequence with time / category / POI / context
  ↓
Existing conditional temporal + category/POI discrete diffusion
  ↓
Reverse denoising predicts joint category/POI logits
  ↓
┌─────────────────────────────────────────────────────────────┐
│                    My Core Contribution                     │
│                                                             │
│  Sample-level category partial-order matrix M                │
│          ↓                                                  │
│  Parse A ≺ B constraints                                    │
│          ↓                                                  │
│  Differentiable order violation + existence penalty          │
│          ↓                                                  │
│  KL-preserving augmented-Lagrangian projection               │
│          ↓                                                  │
│  Modify category-position logits only                        │
└─────────────────────────────────────────────────────────────┘
  ↓
Joint Gumbel-max sampling of projected category + POI logits
  ↓
Generated POI sequence with time / category / POI / GPS
```

### Core Contribution

`[Confirmed]` 在已有条件时空联合离散扩散的 reverse sampling 中，加入面向样本级类别偏序的可微约束投影：用偏序顺序/存在性能量驱动 KL 保真的增广拉格朗日优化，只修改 category-position logits，再与 POI logits 联合采样。

### Technical Position

`[Confirmed]` 创新位于 `DiffusionTransformer.p_sample` 的 `p_pred` 与 `log_sample_categorical` 之间，且只在满足 `use_constraint_projection`、频率和后期步数条件时执行。

### Mechanism

`po_matrix → parse constraints → category soft/Gumbel probabilities → order/existence violations → KL-ALM projection → projected category logits → joint category/POI sample`。

### Evidence

- `tools/prepare_newyork_po1.py:_make_po_matrix`；
- `discrete_diffusion/diffusion_transformer.py:DiffusionTransformer.sample_fast`；
- `discrete_diffusion/diffusion_transformer.py:DiffusionTransformer.p_sample`；
- `constraint_projection.py:parse_po_matrix_to_constraints`；
- `constraint_projection.py:ConstraintProjection.compute_constraint_violation_optimized`；
- `constraint_projection.py:ConstraintProjection.project_with_matrices`。

### Thesis Claim

对应开题报告：

- 3.1.1“类别偏序约束下的 POI 打卡序列生成”；
- 3.2.2“采样期约束投影”；
- 4.1.1 中 PCDG 的核心方法目标；
- 5.1 中“类别偏序约束投影的实现”。

### Main Figure Representation

用唯一强调色把 `M → order/existence energy → category-logit KL-ALM projection → joint sampling` 画成主创新链。必须标出 `inference/reverse diffusion`、`category positions only`，并将投影插入点放在联合采样之前。

### Detail Figure Representation

Panel (b) 展开一条 `A ≺ B` 关系、B-prefix 逆序能量、existence penalty、KL+ALM 目标和 projected category logits；用灰色旁路表示 POI logits 未被直接投影。

## 17. Evidence and Uncertainty

### 17.1 高置信事实

- `[Confirmed]` category/POI token layout 和独立 mask；
- `[Confirmed]` discrete forward/reverse diffusion；
- `[Confirmed]` Transformer denoiser 和 condition cross-attention；
- `[Confirmed]` sample-level `po_matrix` 解析；
- `[Confirmed]` order/existence energy；
- `[Confirmed]` KL-ALM projection；
- `[Confirmed]` projection 只作用于 category positions；
- `[Confirmed]` projection 位于 reverse sampling、联合采样之前；
- `[Confirmed]` 最终 POI 序列由 `Batch.to_seq_list` 得到。

### 17.2 需要谨慎表述的事实

- `[Uncertain]` 训练期 `PartialOrderLoss` 是否在当前配置下真正稳定执行：配置有 `po_loss_weight=0.2`，但 logits 维度与 loss 期望不匹配，且报告称其为拟研究/默认关闭；
- `[Strongly Supported]` 报告中的“投影后 category 与 POI 共同采样”：代码确实在同一个 `p_sample` 中对全序列采样，但没有 category→POI 硬一致性机制；
- `[Uncertain]` OOD opposite split 是否属于当前训练运行时必经步骤：脚本存在，但 `DataModule` 读取已生成 pkl，不在训练主调用链中自动执行；
- `[Not Found]` `po_encoding` 作为扩散条件；
- `[Not Found]` 完整 CFG、复杂 DAG 自适应和 candidate filtering。

## Final Conclusion

相对于已有条件时间点过程与 category/POI 联合离散扩散，本方法真正新增的是采样期类别偏序控制层：由样本级偏序矩阵得到顺序与存在性能量，在 reverse diffusion 后期用 KL 保真的增广拉格朗日投影修改 category logits，再与 POI logits 联合采样。主图应把这条“矩阵→能量→类别 logits 投影→联合采样”链作为唯一核心创新，用强调色放在 reverse step 的采样前；不要画成 POI 候选过滤、硬 transition restriction、全局 `po_encoding` 条件或已完成的复杂 DAG 自适应模块。


# 开题报告—代码交叉验证：科研方法图语义

本文档以 `report.pdf` 的开题报告叙事为 A，以仓库当前代码为 B。所有判断均以报告文本和实际函数调用为依据，不把“拟研究”当成“已实现”。

## A. 论文叙事层面的方法流程

开题报告将问题定义为：给定真实用户移动行为的时间、活动类别和 POI 语义，在完整 POI 打卡序列生成中满足样本级活动类别偏序关系（例如 A 先于 B），并协调真实性、多样性、时空合理性与约束满足。

报告叙事流程为：

```text
真实签到数据
→ 时空语义表示（t, category, POI + 条件）
→ 条件时空联合生成
   ├─ 条件时间点过程生成连续到达时间
   └─ 类别/POI 联合离散扩散生成访问内容
→ 从类别首次/末次位置构造样本级偏序矩阵 M
→ 将 M 解析成偏序集合，并转成可微偏序能量 + 存在性惩罚
→ 逆扩散后期、每步联合采样前，仅对类别 logits 做 KL 保真增广拉格朗日投影
→ 类别（投影后）与 POI 共同采样
→ 得到满足偏序的完整 POI 序列
→ 以 OVR/Coverage/Unsat、JSD、下游任务和效率评价
```

报告还提出第二个递进问题：将多组约束组织为 DAG，做传递/冗余/冲突检测，并根据扩散阶段和当前满足程度自适应调节约束强度、施加时机和计算开销。报告明确说明研究问题二在开题阶段主要是问题定义和思路论证，不是已完成算法。

## B. 代码真实实现流程

### B1. 数据与样本表示

入口是 `datamodule.DataModule.prepare_data` → `datamodule.load_sequences` → `datamodule.Sequence`。数据来自 pkl 中的 `arrival_times`、`marks`、`checkins`、六类 condition、indicator、`poi_gps`，可选 `po_matrix` 和 `po_encoding`。

PO1 构造链在 `tools/prepare_newyork_po1.py`：`_parse_source` → `_filter_pois` → `_build_daily_sequences` → `prepare_dataset` → `_make_po_matrix` → `_materialize_sequences`。它完成本地时间、去重、POI 频次过滤、9 类 category 词表、POI 词表、`poi_category`/`poi_gps`、每条序列的偏序矩阵和 SVD 编码。

### B2. Batch 与离散 token 布局

`datamodule.Batch.from_sequence_list` 形成：

```text
[START=0] + category_1..category_n + [SEP=1] + poi_1..poi_n + [END=2]
```

padding token 为 3；`category_mask` 与 `poi_mask` 分别标出两段，`mask` 标出时间事件位置，`tau` 在 `Sequence.__init__` 中由时间差得到。

### B3. 时间分支

`tasks.DensityEstimation.step` 调用 `add_thin.diffusion.model.AddThin.forward`。该函数通过 `get_n` 采样 Add-Thin 步数，`noise` 先 `Batch.thin` 再由 `add_thin.processes.hpp.generate_hpp` 加入 HPP 事件；`compute_emb` 编码事件时间、间隔、六类事件条件和序列级 indicator；随后由 `PointClassifier` 和 `MixtureIntensity` 计算时间分支损失。

采样时 `sample.py:simulation` 调用 `task.tpp_model.sample`，其 `AddThin.sample` 从 `generate_hpp` 的噪声事件开始，通过 `sample_posterior` → `sample_x_0` 逐步还原时间事件。

### B4. 离散扩散 forward

`configs.instantiate_model` 实例化 `discrete_diffusion.diffusion_transformer.DiffusionTransformer`。其 `alpha_schedule` 为 category 和 POI 建立不同的转移/累积概率 buffer。

- `q_pred_one_timestep`：用 `category_mask`/`poi_mask` 分别计算 category 与 POI 的单步转移；
- `q_pred`：计算 q(x_t|x_0)；
- `q_sample`：调用 `q_pred` 后以 `log_sample_categorical` 的 Gumbel-max 采样离散 x_t；
- `training_losses`：`sample_time` → `index_to_log_onehot` → `q_sample` → `condition_encoder` → `predict_start` → 交叉熵（忽略 padding token 3）。

### B5. 反向去噪与核心生成模型

`discrete_diffusion.conditional_attention.Transformer` 使用 token embedding、位置 embedding、扩散时间 embedding、token type embedding（特殊/category/POI），再通过 `Decoder` 的 self-attention 和对条件的 cross-attention 输出 logits。

`DiffusionTransformer.predict_start` 得到 p(x_0|x_t)；`p_pred` 在 x0 参数化下调用 `q_posterior` 得到下一步分布；`sample_fast` 从 `[START, category-mask, SEP, poi-mask, END]` 开始，倒序循环 `p_sample`，最后 `log_onehot_to_index` 得到 token。

采样总链为：

```text
sample.py:simulation
→ AddThin.sample（时间）
→ Batch.mask_check
→ DiffusionTransformer.sample_fast（category/POI 离散反向扩散）
→ Batch.to_seq_list(gps_dict)
→ 保存 generated_part*.pkl
```

### B6. 代码中的偏序约束

采样约束主链是：

```text
Batch.po_matrix
→ DiffusionTransformer.sample_fast
→ parse_po_matrix_to_constraints
→ ConstraintProjection._compile_constraints / compile_batched_constraints
→ compute_constraint_violation_optimized
→ project_with_matrices
→ p_sample 中替换 model_log_prob
→ log_sample_categorical 联合采样
```

`constraint_projection.ConstraintProjection` 计算两类能量：

- 顺序违规：B 的前缀概率与当前位置 A 概率的乘积累加；
- 存在性违规：受约束 A/B 的总概率低于 1 时的 ReLU 惩罚。

`project_with_matrices` 以 KL 项保持接近原模型分布，以增广拉格朗日内外迭代优化 logits，并可用 Gumbel-Softmax 松弛、梯度裁剪和 `mu` 上限。

`DiffusionTransformer.p_sample` 仅在 `use_constraint_projection=True`、存在 `po_constraints`、满足 `projection_frequency` 且位于 `projection_last_k_steps` 时触发；投影使用 `batch.category_mask`，因此只作用于 category 位置，POI logits 不直接投影。

另有两个对照/实验路径：

- `sample.py --baseline energy_guidance`：在 `p_sample` 中用偏序违规能量梯度做 `model_log_prob - guidance_scale * grad`；这是能量引导，不是独立分类器。
- `sample.py --baseline posthoc_swap` → `baseline_posthoc_swap.apply_posthoc_swap`：生成完成后按类别约束做稳定拓扑重排。

### B7. 生成后的 POI 序列

`Batch.to_seq_list` 用 `poi_mask` 提取 POI token，用 `category_mask` 提取 category，用 `gps_dict` 映射 GPS；缺少 GPS 的 POI 会被删除并同步删除时间、类别和条件，最终输出 `arrival_times`、`marks`、`checkins`、`gps`、condition 字段。

## C. 两者一致的部分

| 报告叙事 | 代码证据 | 一致性判断 |
|---|---|---|
| POI 打卡序列是时间、类别、POI 的离散事件序列 | `Sequence`、`Batch.from_sequence_list` | 一致 |
| 连续时间与离散访问内容联合生成 | `AddThin` + `DiffusionTransformer`，由 `DensityEstimation.step` 联合训练 | 一致（实现为两个子模型/损失分支） |
| category 与 POI 联合逆扩散 | `sample_fast` 对整条 `[category 段 + POI 段]` 循环 `p_sample` | 一致 |
| category/POI 使用不同离散转移 | `alpha_schedule`、`q_pred`、`q_pred_one_timestep` | 一致 |
| token 类型、位置、扩散时间和条件 cross-attention | `conditional_attention.Transformer` | 一致 |
| 按类别首次/末次位置定义严格“全部先于” | `tools/prepare_newyork_po1.py:_make_po_matrix` | 一致 |
| 偏序矩阵解析为约束集合 | `parse_po_matrix_to_constraints` | 一致 |
| 偏序能量 + 存在性惩罚 | `compute_constraint_violation_optimized` | 一致 |
| KL + 增广拉格朗日 + Gumbel 松弛 | `ConstraintProjection.project_with_matrices` | 一致 |
| 逆扩散后期、按频率、采样前投影且只改类别 | `DiffusionTransformer.p_sample` | 一致 |
| 约束与真实性之间存在权衡 | `projection_existence_weight`、KL 项、报告表 5-1 的 JSD 变化 | 机制和现象一致 |

## D. 两者不一致的部分

### D1. 报告把 M 描述为联合分布条件，但代码未把 `po_encoding` 注入 denoiser

报告公式写成带 `M` 的联合条件分布，并提出低维偏序编码可作为全局条件。代码虽然在 `Sequence`/`Batch` 中保存 `po_matrix`、`po_encoding`，但 `DiffusionTransformer.ConditionEmbeddingModel.forward` 只读取 `time` 和 `condition1..6`；`Transformer.forward` 也未读取 `po_encoding`。当前 M 的实际作用点是采样阶段投影输入，而不是生成器的普通条件 embedding。

### D2. 报告说“投影后的类别分布与 POI 分布共同采样”，代码的投影接口确实如此，但 POI 与 category 的语义对齐并未在投影中显式约束

代码在同一 `p_sample` 中采样两类 token，但 `ConstraintProjection` 只看 category positions 的概率，不检查生成 POI 是否属于生成 category。POI/category 的训练集对应关系在数据校验 `_validate_loaded_dataset` 中检查，生成时没有候选 POI 过滤或 category→POI transition restriction。

### D3. 报告将训练期偏序辅助损失写成拟研究项；代码已有接口但不是稳定主路径

`DensityEstimation.step` 在 `po_loss_weight>0`、`svd_components` 存在且 `batch.po_matrix` 非空时调用 `PartialOrderLoss`。但 `PartialOrderLoss.forward` 注释期望 category-only logits，而 `training_losses` 返回的 `category_logits` 实际包含扩散词表维度；代码没有显式切片 category token 范围。报告也明确说该路径“仍需修正与验证、当前默认关闭”，因此不能把它画成已经验证的核心模块。

### D4. 报告中的 CFG 尚未形成完整实现

`sample.py` 暴露 `--baseline cfg` 和 `cond_dropout_rate` 参数，但 `DiffusionTransformer` 中没有已完成的无条件/有条件双分支 CFG 采样组合；报告 5.1/7.1 也明确承认 CFG 尚不完整。方法主图不应把 CFG 当作 PCDG 内部组件。

### D5. 独立 ConstraintClassifier 不是当前 energy guidance 主链

`constraint_classifier.py:ConstraintClassifier`、`train_classifier.py` 和 `classifier_guidance_injection.py:apply_classifier_guidance` 构成独立实验代码；当前 `sample.py --baseline energy_guidance` 直接调用 `ConstraintProjection.compute_constraint_violation_optimized` 的能量梯度，没有加载该分类器。因此报告若把“分类器引导”与当前 PCDG 主方法合并，会与代码不符。

### D6. 报告中的复杂偏序 DAG、自适应约束尚未实现

当前代码只有矩阵逐边解析、矩阵编译、固定 `projection_frequency`、固定 `projection_last_k_steps` 和固定超参数更新；没有独立的传递闭包、冗余消除、冲突/环路检测或基于当前满足程度的自适应策略。这些只能写入“后续研究问题二”，不能画成当前已完成模块。

### D7. 报告中的数据/OOD叙事比当前主调用链更宽

代码存在 `tools/make_opposite_split.py` 等数据处理脚本，能够支持相反顺序划分思路；但 `DataModule` 主流程只是读取已经生成的 pkl。方法主图应画“输入 pkl + 已生成的偏序矩阵”，不要暗示训练运行时自动完成 OOD 重划分。

## E. 方法主图必须出现的模块

主图应突出论文真正的科学主线，而不是堆叠工程细节。建议至少出现以下节点，并在节点旁标注实际代码对象：

1. **POI 轨迹输入与时空语义表示**：`Sequence` / `Batch.from_sequence_list`；包括 arrival time、category、POI、条件和 mask。
2. **条件时空联合生成基础**：`AddThin`（时间） + `DiffusionTransformer`（category/POI）。
3. **类别/POI 联合离散扩散**：`q_sample` / `q_pred`、`predict_start`、`p_pred`、`sample_fast`。
4. **样本级类别偏序矩阵构建**：`_make_po_matrix`；明确 A≺B 的“全部先于”定义。
5. **偏序约束可微化**：`parse_po_matrix_to_constraints` + `compute_constraint_violation_optimized`；包含 order energy 与 existence penalty。
6. **采样期类别 logits 投影**：`ConstraintProjection.project_with_matrices`；标出 KL 保真、增广拉格朗日、Gumbel-Softmax。
7. **约束位置与时机**：`p_sample` 中“逆扩散后期、按频率、联合采样前、仅 category positions”。
8. **联合采样与最终 POI 序列**：`log_sample_categorical` → `Batch.to_seq_list`。
9. **保真—约束评价闭环**：约束满足指标与 JSD/效率指标；代码侧对应 `evaluations` 和采样评估脚本。

主图最适合强调的创新箭头是：

```text
样本级类别偏序矩阵 M
→ 可微顺序/存在性能量
→ 逆扩散后期 category-logit KL-ALM 投影
→ category 与 POI 联合采样
```

这条箭头应视觉上成为主干，时间点过程和普通 Transformer 作为生成基础并列支撑。

## F. 可以隐藏到模块细节图中的模块

- `alpha_schedule` 的具体数组构造、log-space 数值裁剪；
- `index_to_log_onehot`、`log_onehot_to_index`、`extract`、`log_add_exp` 等张量工具；
- Transformer 内部的 `EncoderLayer`、`DecoderLayer`、`MultiHeadAttention`、SiLU 和位置 embedding；
- `Batch.thin`、HPP 采样的事件增删细节；
- `condition1..6_indicator` 的 24 窗口映射；
- `ConstraintProjection` 的 W_A/W_B 编译、lambda/mu 更新、梯度裁剪、debug 输出；
- 并行采样的 `rank/world_size` 分块与结果合并；
- GPS 字符串解析、缺失 GPS 删除；
- `po_encoding` 的 StandardScaler + TruncatedSVD 细节（前提是在主图中明确它目前未接入 denoiser）；
- PostSwap、EnergyGuide、CFG 等对照方法，应放在实验设置/基线图，不应挤入 PCDG 主干。

## G. 不应该出现在方法主图中的实现细节

以下内容既不是论文核心方法，也会误导读者：

- argparse 的投影参数、debug flag、WandB 路径、GPU rank；
- `print` 调试语句、shape 打印、异常处理和 checkpoint 加载；
- `ConstraintClassifier` 的独立训练脚本（除非专门画分类器基线图）；
- `classifier_guidance_injection.py` 的未接入辅助函数；
- `po_encoding` 作为“已使用条件”的箭头；当前代码没有这条路径；
- candidate filtering、非法 POI transition mask、类别到 POI 的硬一致性筛选；仓库中没有这些机制；
- 复杂偏序 DAG、传递闭包、冲突检测、自适应强度/时机；当前只是研究问题二的计划；
- 把 PostSwap 画成逆扩散内部步骤；它发生在 `sample_fast` 和 `to_seq_list` 之后；
- 把 CFG 画成已完成的 PCDG 组件；报告和代码均表明它尚未完整。

## “类别偏序约束”在论文中的核心呈现建议

论文主方法应把类别偏序约束定位为“对已有条件时空联合离散扩散的采样期结构化控制层”，而不是另一个独立生成器，也不是普通输入条件。最准确的叙述是：

> 对每条参考/给定序列构造样本级类别偏序矩阵 M；将每条 A≺B 转换为可微的逆序能量和存在性惩罚；在离散扩散 reverse step 中，仅对 category positions 的 logits 进行 KL 保真的增广拉格朗日投影；然后与未投影的 POI logits 一起完成下一步联合采样。

这样既能体现创新性，也与当前代码一致。需要避免三种过度表述：

1. 不要说 M 已作为 Transformer 的全局条件 embedding（`po_encoding` 尚未接入）；
2. 不要说约束保证 POI 与 category 的硬一致性（没有 candidate filtering/transition restriction）；
3. 不要说训练期偏序损失已经稳定贡献主方法（当前是条件触发的探索接口，存在维度接入疑点）。

## 最终交叉验证结论

当前最稳妥的论文方法主线是：

```text
条件时空联合生成基础（已有 Marionette/Add-Thin/离散扩散骨架）
→ 样本级类别偏序矩阵
→ 偏序违反 + 存在性能量
→ 逆扩散后期 category-logit 可微 KL-ALM 投影（核心创新）
→ category/POI 联合离散采样
→ POI 序列输出与约束/保真评价
```

研究问题二（复杂偏序 DAG 与自适应施加）和训练期偏序辅助损失，应在论文中明确标为“后续/拟研究或探索项”，不要与当前已经打通的采样期 PCDG 主路径混写。


# 科研方法图语义图（严格对应当前代码）

本文档只按当前仓库中可执行代码的调用关系描述，不把论文中常见但代码不存在的模块补进来。

## 总体语义图

```text
Input
  → Condition
  → Preprocessing
  → Diffusion
  → Reverse Generation
  → Constraint Mechanism
  → Output
```

## Input

### 数据文件与入口

- 训练入口：`train.py:main`。通过 `configs.instantiate_datamodule`、`configs.instantiate_model`、`configs.instantiate_task` 组装训练对象。
- 采样入口：`sample.py:simulation`。先调用 `evaluate_utils.get_task` 加载 checkpoint 和数据，再遍历 `datamodule.test_dataloader()`。
- 数据文件：`data/<name>/<name>_train.pkl` 与 `<name>_test.pkl`，由 `datamodule.load_sequences` 读取；必要字段包括 `sequences`、`t_max`、`num_marks`、`num_pois`、`poi_gps`。

### 单条样本的真实字段

`datamodule.Sequence.__init__` 接收：

- `time`/`arrival_times`：事件到达时间；
- `checkins`：POI token 序列；
- `category`/`marks`：与每个 POI 对齐的 category token；
- `condition1..condition6`：事件级条件；
- `condition1_indicator..condition6_indicator`：24 个时间窗口的条件指示；
- `tmax`、可选 `po_matrix`、可选 `po_encoding`。

批处理由 `datamodule.Batch.from_sequence_list` 完成，输出 `Batch`，其中离散扩散真正使用的目标张量是 `checkin_sequences`。

## Condition

条件不是 POI/category 的单独输入 embedding，而是时间和六类上下文条件：

1. `add_thin/diffusion/model.py:AddThin.compute_emb`
   - 事件时间 `x_n.time` 与间隔 `x_n.tau` 经 `time_encoder` 编码；
   - `condition1..condition6` 经 `event_condition_encoder` 编码，拼成 `event_level_cond_emb`；
   - `condition1_indicator..condition6_indicator` 经 `AddThin.seq_condition_encoder`（其内部 `ConditionEmbeddingModel.forward`）编码，取最后位置为 `seq_level_cond_emb`。
2. `discrete_diffusion/diffusion_transformer.py:ConditionEmbeddingModel.forward`
   - 离散扩散 Transformer 使用 `batch.time`、`batch.condition1..condition6`；
   - 这些 embedding 拼接后经 `input_up_proj`、位置 embedding 和 3 层 Transformer encoder，得到 `cond_emb`。
3. POI 和 category 的关系不通过 `ConditionEmbeddingModel` 输入；它们作为待生成 token / 约束矩阵使用。`po_encoding` 在 `Sequence`、`Batch` 中被保存和传递，但当前 `DiffusionTransformer` 没有读取它。

## Preprocessing

### 原始数据到 pkl（PO1 数据构造）

`tools/prepare_newyork_po1.py` 是当前最完整的 PO1 预处理链：

1. `_parse_source`：解析 UTC、时区、本地时间，按用户和时间排序、去重，映射 `Root_Category`，生成 `Local_Date`。
2. `_filter_pois`：按最小 POI 频次过滤。
3. `_build_daily_sequences`：按用户/日期构造日序列，生成时间、原始 venue、类别名和六个条件值。
4. `prepare_dataset`：建立 category vocabulary（token 从 4 开始）、POI vocabulary（token 从 13 开始），生成 `poi_gps` 与 `poi_category`，再划分 train/test。
5. `_make_po_matrix`：根据类别首次/末次出现位置生成类别偏序矩阵；若类别 A 的最后位置早于 B 的首次位置，则 `po_matrix[A,B]=1`。
6. `_fit_po_encoder` / `_materialize_sequences`：训练集偏序矩阵做 StandardScaler + TruncatedSVD，保存 `po_encoding`；同时生成 `arrival_times`、`marks`、`checkins`、条件字段和 `po_matrix`。

旧/补充脚本 `tools/make_Istanbul_withPO.py` 也实现 `get_full_partial_order_matrix`、`encode_po_matrix`，但当前 `DataModule` 只按 pkl 字段读取，不能据此推断它必经主链。

### Batch 内部布局

`Batch.from_sequence_list` 把每条样本拼为：

```text
[START=0] + category_1..category_n + [SEP=1] + poi_1..poi_n + [END=2]
```

并建立：

- `category_mask`：只标记 category 段；
- `poi_mask`：只标记 POI 段；
- padding token 为 3；
- `mask`：时间事件有效位；`tau` 由 `Sequence` 的 `torch.diff(time)` 计算。

## Diffusion

### 时间/事件 Add-Thin（已有基础模块）

`add_thin/diffusion/model.py:AddThin` 是时间点过程扩散：

- `noise`：按 `alpha_cumprod[n]` 对真实事件 `thin`，再用 `add_thin/processes/hpp.py:generate_hpp` 叠加齐次 Poisson 事件，得到 `x_n`。
- `forward`：随机采样步数 `get_n`，调用 `noise`，再由 `compute_emb`、`classifier_model`、`intensity_model` 计算时间分支训练量。

### 离散 token forward process

`discrete_diffusion/diffusion_transformer.py:DiffusionTransformer`：

- `alpha_schedule` 生成 category/POI 不同的转移和累积概率 buffer（`log_at/log_bt/log_ct` 及其 cumulative 版本）。
- `q_pred_one_timestep` 实现 q(x_t|x_{t-1})：category 位置使用 category 转移，POI 位置使用 POI 转移，并通过 `category_mask`/`poi_mask` 分流。
- `q_pred` 实现 q(x_t|x_0)。
- `q_sample` 调用 `q_pred` 后用 `log_sample_categorical`（Gumbel-max）采样离散 `x_t`。
- `training_losses`：`sample_time` 随机步数，`index_to_log_onehot` 将目标 token 转 log-one-hot，`q_sample` 加噪，随后预测 x0 并做交叉熵。

## Reverse Generation

### 核心生成模型（已有基础模块）

`discrete_diffusion/conditional_attention.py:Transformer` 是离散 token 的核心 denoiser：

- token embedding + positional embedding + diffusion timestep embedding；
- `token_type_layer` 区分 category/POI（由 `category_mask`、`poi_mask` 得到）；
- `Decoder`（self-attention + cross-attention）以 `cond_emb` 为条件；
- `output_layer` 输出 `num_classes-2` 个非 mask 类别的 logits。

`DiffusionTransformer.predict_start` 调用该 Transformer 得到 `p(x_0|x_t)`；`p_pred` 在 x0 参数化下通过 `q_posterior` 得到 `p(x_{t-1}|x_t)`。

### 反向采样链

`DiffusionTransformer.sample_fast`：

1. 根据 `unpadded_length` 构造初始序列 `[START, category-mask..., SEP, poi-mask..., END]`，其余为 padding；
2. 从 `batch.po_matrix` 调用 `parse_po_matrix_to_constraints`（支持共享矩阵或逐样本矩阵）；
3. 从 `num_timesteps-1` 到 0 循环调用 `p_sample`；
4. `p_sample` 执行 `p_pred` →（可选约束/ guidance）→ `log_sample_categorical`；
5. 最终 `log_onehot_to_index` 得到 token 序列，并重建 category/POI mask。

`sample.py:simulation` 的完整生成顺序是：

```text
task.tpp_model.sample(...)       # Add-Thin 生成时间序列
→ time_samples.mask_check()
→ task.discrete_diffusion.sample_fast(time_samples, ...)
→ Batch.to_seq_list(gps_dict)
```

## Constraint Mechanism

### 1. 训练阶段：偏序损失（论文候选创新，但当前默认未必生效）

- `tasks.DensityEstimation.step` 在 `po_loss_weight>0`、`svd_components` 存在且 `batch.po_matrix` 非空时，调用 `self.po_loss_fn`。
- `partial_order_loss.py:PartialOrderLoss.forward`：对预测概率计算未来累积概率，构造软邻接矩阵 `pred_adj`，归一化后与 `target_matrix` 做 MSE。
- `configs.instantiate_task` 从配置读取 `po_loss_weight`；`config/train.yaml` 默认值为 0.2，并在 epoch 50–100 做 warmup。
- 重要事实：`DensityEstimation.step` 直接使用 `DiffusionTransformer.training_losses` 返回的 `category_logits`；该返回值实际由 `predict_start` 产生并包含全 token 类别维度，而 `PartialOrderLoss` 的注释期望 category-only logits，代码中没有显式切片到 category token 范围。因此这条训练约束链存在潜在维度/语义不匹配，不能把它描述成已验证有效的 category-only 训练约束。

### 2. 采样阶段：ALM/梯度投影（当前代码中真正执行的主约束）

`constraint_projection.py:ConstraintProjection`：

- `parse_po_matrix_to_constraints` 将矩阵中的每个 `A→B` 转为 `([A],[B])`；
- `_compile_constraints` / `compile_batched_constraints` 将约束编译为 `W_A`、`W_B`；
- `compute_constraint_violation_optimized` 对 category 位置的 softmax/Gumbel-softmax 概率计算顺序违规（B 的前缀概率与 A 的当前位置相乘）和存在性违规；
- `project_with_matrices` 以 KL 保持原模型分布，同时用增广拉格朗日罚项迭代优化 logits；使用 `projection_tau`、`lambda`、`mu`、内外迭代、`projection_existence_weight` 等参数；
- `DiffusionTransformer.p_sample` 仅当 `use_constraint_projection`、有 `po_constraints`、步数满足 `projection_frequency` 且位于 `projection_last_k_steps` 时调用投影，且投影只通过 `category_mask` 作用于 category 位置。

### 3. 采样阶段：energy-based guidance baseline（可选基线）

`DiffusionTransformer.p_sample` 在 `use_guidance_baseline=True` 时，以 `compute_constraint_violation_optimized` 的能量梯度更新：

```text
model_log_prob ← model_log_prob - guidance_scale * ∇ violation
```

`sample.py` 中 `--baseline energy_guidance` 打开该路径；通过 `guidance_last_k_steps` 和 `guidance_frequency` 限制执行步数。它不是独立训练的约束分类器路径。

### 4. 后处理基线（采样后，不是扩散内约束）

`sample.py --baseline posthoc_swap` 调用 `baseline_posthoc_swap.apply_posthoc_swap`：

- `get_eval_cats` 用 `poi_category` 把生成 POI 映射到评估类别；
- `extract_constraints_in_eval_space` 或 `extract_constraints_from_test_seq` 得到边；
- `fix_single_sequence` 用稳定拓扑排序重排受约束类别的位置块；
- 该方法发生在 `sample_fast` 和 `to_seq_list` 之后，不改变反向扩散分布。

### 5. 实际存在与不存在的机制

- 存在：`category_mask`/`poi_mask` 的位置隔离；Gumbel-max 离散采样；约束矩阵编译；投影 logits 优化；energy guidance；后处理类别块重排。
- 未找到：基于约束的 POI candidate filtering、逐步 transition restriction、对非法 POI token 的显式 mask、把 `po_encoding` 注入 denoiser 的实现。
- `classifier_guidance_injection.py:apply_classifier_guidance` 和 `constraint_classifier.py:ConstraintClassifier` 是独立实验代码；`sample.py` 的 `energy_guidance` 实际使用的是投影器违规能量梯度，并未加载或调用该分类器。

## Output

### 扩散 token 输出

`DiffusionTransformer.sample_fast` 返回新的 `datamodule.Batch`，其中 `checkin_sequences` 是生成后的完整 token 布局，`category_mask`/`poi_mask` 指示两类位置。

### 最终 POI 序列输出

`datamodule.Batch.to_seq_list(gps_dict)`：

1. 用 `poi_mask` 提取 POI token；用 `category_mask` 提取 category token；用 `mask` 提取生成时间和条件；
2. 通过 `gps_dict` 把 POI token 转为 `[lat, lon]`；找不到 GPS 的 POI 会被删除，并同步删除对应时间、类别和条件位置；
3. 返回字典列表：`arrival_times`、`marks`、`checkins`、`gps`、`condition1..condition6` 及各 indicator。

`sample.py` 最后将列表保存为 `./data/<data_name>/<data_name>_<RUN_ID>_generated_part<rank>.pkl` 的 `sequences` 字段（另含 `t_max`）。

## 已有基础模块 vs 论文创新模块

### 已有基础模块（原 Marionette / Add-Thin / 离散扩散骨架）

- `add_thin/diffusion/model.py:AddThin` 的时间点过程 Add-Thin forward/reverse；
- `add_thin/processes/hpp.py:generate_hpp`；
- `add_thin/backbones` 中的时间 classifier/intensity backbone；
- `discrete_diffusion/conditional_attention.py:Transformer`；
- `DiffusionTransformer` 的 alpha schedule、q/p 过程、交叉熵训练和 Gumbel 离散采样；
- `datamodule.Sequence/Batch` 的时间、条件、padding 和 token 拼接；
- `Batch.to_seq_list` 的 POI/GPS 序列反序列化。

### 论文目标对应的创新/新增模块（按代码标注与调用判断）

- PO1 数据构造：`tools/prepare_newyork_po1.py` 中的 `po_matrix`、`po_encoding`、`poi_category` 元数据；
- 训练约束候选：`partial_order_loss.py:PartialOrderLoss` 及 `DensityEstimation.step` 的加权/warmup 接入；
- 采样约束主路径：`constraint_projection.py:ConstraintProjection` 的偏序解析、违规度、增广拉格朗日投影；
- 采样控制：`DiffusionTransformer.sample_fast/p_sample` 对约束矩阵、频率和最后 K 步的接入；
- 对照基线：`baseline_posthoc_swap.py` 的后处理拓扑重排，以及 `p_sample` 中的 energy guidance 分支。

## 约束阶段结论（直接回答问题 8）

当前代码同时放置了训练阶段和采样阶段的约束接口，但语义不同：

- 训练阶段：只有在配置、SVD 元数据和 `po_matrix` 条件同时满足时才加入 `PartialOrderLoss`；且当前 logits 维度接入存在疑点，不能假定已稳定实现。
- 采样阶段：`ConstraintProjection` 是明确实现并由 `sample.py --use_constraint_projection` 打开的反向采样约束；`energy_guidance` 是可选采样基线；`posthoc_swap` 是采样后处理。


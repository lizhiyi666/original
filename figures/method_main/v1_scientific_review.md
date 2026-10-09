# V1 Scientific Review

审查对象：`figures/method_main/method_main_v1.drawio` 及其导出 `method_main_v1.png`。

审查依据：`method_innovation_map.md`、`code/code_semantic_graph.md`、`thesis_method_graph.md`、`visual/visual_contract.md`、当前 Draw.io XML 和 PNG。科研判断遵循：代码真实性 > 论文叙事 > 视觉参考。

## 1. Overall Verdict

**Major Revision**

当前图已经把核心机制的主要元素画出，尤其是 Panel (b) 的 `po_matrix → order/existence energy → KL-preserving ALM projection → category logits → joint sampling` 链条基本成立；但 Panel (a) 中 `po_matrix` 的来源被画错，Training / Inference 边界没有表达，reverse diffusion 的轨迹和约束插入点也没有形成严格的数据流。这些问题会影响答辩委员对方法定位的准确理解，因此不宜直接作为论文终稿。

审查结论不是“Scientific Rework Required”，因为核心创新没有被完全画错，且 Panel (b) 已提供可修正的真实机制骨架；但必须先修复 P0 科研语义问题。

## 2. Critical Scientific Errors

### C1. `po_matrix` 被错误地画成 Condition Representation 的输出

**图中现象**：Panel (a) 的橙色虚线从 `Condition Representation` 指向 `Category-order Constraint-aware Generation`，文字标注为 `sample-level po_matrix`。

**代码事实**：`DiffusionTransformer.ConditionEmbeddingModel.forward` 只读取 `batch.time` 和 `batch.condition1..condition6`，不读取 `po_matrix` 或 `po_encoding`。`sample_fast` 才直接读取 `batch.po_matrix`，解析后传入 `p_sample`。因此 `po_matrix` 不是当前条件编码器的输出，也不是普通 condition embedding。

**严重性**：Critical Scientific Error。该箭头会让读者误以为偏序矩阵进入生成器的条件表示，与代码和 `method_innovation_map.md` 的关键结论相冲突。

**P0 修复方向**：将 `po_matrix` 从 Input / Constraint Specification 侧直接连到 reverse-step constraint mechanism；不要从 `Condition Representation` 发出该箭头。若要表达时间/上下文条件，单独保留灰色或蓝色条件线。

## 3. Major Issues

### M1. Training / Inference 边界未表达

代码中 `AddThin` 和 `DiffusionTransformer.training_losses` 用于训练；`sample.py:simulation`、`AddThin.sample`、`DiffusionTransformer.sample_fast/p_sample` 用于生成。当前图没有 Training / Inference 标签，读者无法判断类别偏序投影何时生效。

这会弱化一个论文核心事实：**Constraint is applied during inference / reverse generation.** 建议在 Panel (a) 增加一条轻量阶段标识，或将训练和推理分成两个紧凑 lane；至少要在高亮约束框中保留 `Inference / Reverse diffusion`。

### M2. Panel (a) 的 reverse diffusion 轨迹是悬空的

`Forward diffusion` 下方的 `x₀ → xₜ → x_T` 与 `Reverse denoising` 下方的 `x_T → … → x₀` 没有连接到 `Category / POI Discrete Diffusion` 或高亮约束框。它们看起来像独立装饰，而不是实际模型路径。

**影响**：Ambiguous Connector / disconnected semantic object。审稿人无法确认约束是在 `x_t` 的 reverse logits 上，还是在一个独立的后处理阶段。

**P1 修复方向**：把离散扩散的 forward / reverse 放进同一个明确容器，或将 reverse step 直接连接到“model logits → constraint projection → joint sampling”的局部展开。

### M3. Panel (a) 中“离散扩散 → Constraint-aware Generation”的箭头过于模块化

代码中的投影插入位置是 `p_pred` 生成 model log-prob 与 `log_sample_categorical` 之间的 `p_sample` 内部，而不是一个完整独立的后续生成器。当前粗箭头从 `Conditional Joint Generator` 整体指向一个大号 `Constraint-aware Generation` 框，容易被解读为“先完成普通生成，再交给约束生成器”。

**P1 修复方向**：把高亮模块放在 reverse diffusion loop 内，用“model logits → category-only projection → joint sampling”的插入点表达；Panel (a) 可以保留高层框，但必须通过编号/callout 与 Panel (b) 的 reverse-step 展开建立严格对应。

### M4. Input 的“Historical POI event sequence”与推理输入语义混合

当前 `Input` 框展示完整的 `time · category · POI` 事件序列。训练时这确实是 `Batch.checkin_sequences` 的数据来源；但 `sample.py` 推理时，生成内容并不是把历史 POI token 直接喂给离散 denoiser，而是由时间/上下文 Batch、序列长度和 `po_matrix` 初始化 mask token 后从噪声开始采样。

**影响**：容易让读者误认为这是历史序列条件生成或自回归 continuation。

**P1 修复方向**：将 Input 改为“Training event sequences + generation conditions”或拆成 `Observed training sequence` 与 `Generation-time conditions / po_matrix` 两类对象，并明确生成 token 从 mask/noise state 开始。

### M5. `Category logits′` 的“位置”还不够明确

Panel (b) 通过 `Category logits′` 与 `POI logits (unchanged)` 已经接近正确，但没有把“category positions only”与 `category_mask` 的位置选择联系起来。读者可能把它理解为只投影 category vocabulary，而非只投影序列中的 category positions。

**P1 修复方向**：在 `Category logits′` 框旁增加短标签：`category positions selected by category_mask`；在 logits 小图中用位置条区分 category positions 与 POI positions。

## 4. Minor Issues

### m1. Panel (b) 没有明确标出它是 Panel (a) 高亮框的局部展开

两面板颜色和标题相近，能够暗示对应关系，但缺少编号、放大标记或 callout 连接。建议在 Panel (a) 高亮框和 Panel (b) 左上角加同一个 `①` 或 “zoom-in” 标签。

### m2. `No candidate filtering / no explicit POI transition restriction` 是有价值的防误读说明，但不宜占据与算法模块同等视觉重量

它不是生成模块，而是科学边界说明。建议放到 Panel (b) 底部小号注释或图注中。

### m3. Panel (a) 的 `Generated POI Sequence` 只写 POI，未提示同步输出时间和 category

代码 `Batch.to_seq_list` 返回 `arrival_times`、`marks`、`checkins`、`gps` 和条件。若主图只强调 POI 研究对象可以保留当前标题，但建议加小号副标签 `with time / category / GPS`，避免误以为模型只生成 POI 而不生成时间。

### m4. `Forward diffusion` 与 `Reverse denoising` 标签在同一基线附近，容易造成时间方向混淆

局部反向箭头本身合理，但应明确标注 `x_T → … → x_0` 是 reverse sampling time axis，而非整张图的主流程方向。

## 5. Innovation Representation Review

### 已正确表达的部分

- `[Confirmed]` Panel (b) 明确包含 `Sample-level po_matrix`。
- `[Confirmed]` `A ≺ B` 和 `M[A,B]=1` 提供了关系语义。
- `[Confirmed]` `Order + Existence Energy` 同时表达顺序违规和存在性惩罚。
- `[Confirmed]` `KL-preserving ALM Projection` 不是 generic projection，也没有被画成 candidate filtering。
- `[Confirmed]` 投影后分成 `Category logits′` 和 `POI logits (unchanged)`，方向基本正确。
- `[Confirmed]` 两类 logits 最终进入 `Joint Gumbel-max Sampling`。

### 仍需强化的部分

- `[Major]` Panel (a) 必须显示投影位于 reverse loop 内，而不是普通生成器之后。
- `[Major]` `po_matrix` 必须绕过 Condition Representation，直接作为 constraint specification 输入。
- `[Minor]` Panel (a) 与 Panel (b) 需要明确的 zoom/callout 对应。

### 没有出现的误导（这是优点）

当前图没有把约束画成 POI candidate filtering、硬 transition restriction 或 POI logits projection；这与代码事实一致。

## 6. Diffusion / Inference Review

### Overall Framework 检查

| 检查项 | 结论 | 说明 |
|---|---|---|
| Input 是否明确 | 部分明确 | 研究对象清楚，但训练输入与推理条件混合 |
| 条件信息是否明确 | 部分明确 | 有 `Condition Representation`，但未说明六类 context/time 的实际范围 |
| Diffusion 是否明确 | 基本明确 | 有 `Category / POI Discrete Diffusion` 与 forward/reverse 标签 |
| Forward / Reverse 是否区分 | 是，但连接不足 | 标签存在，轨迹未连接到核心模块 |
| constraint 插入位置 | 高层上正确，细节上不够严格 | Panel (b) 暗示采样前投影，Panel (a) 像独立后置模块 |
| output 是否明确 | 基本明确 | POI 序列清晰，时间/category/GPS 可再补充 |

### Inference 位置结论

当前图没有把约束画到训练损失上，因此没有直接产生“训练约束”的错误；但由于缺少 Training / Inference 分界，属于 Major Issue。V2 必须明确：

```text
Training: learn temporal + discrete diffusion model
Inference: reverse diffusion
           → category-order projection
           → joint sampling
```

## 7. Constraint Mechanism Review

Panel (b) 的链条审查：

```text
po_matrix
  → Constraint Parsing
  → Order + Existence Energy
  → KL-preserving ALM Projection
  → Category logits′
  → Joint Gumbel-max Sampling
```

### 完整性

- `po_matrix`：存在，且显示 `M[A,B]=1; A≺B`。
- `order/existence energy`：存在，并有两条文字说明。
- `KL-preserving ALM projection`：存在，且术语准确。
- `category-position logits`：目前写作 `Category logits′`，语义接近但需明确“positions only”。
- `constrained sampling`：`Joint Gumbel-max Sampling` 存在，但建议明确是“projected category + unchanged POI logits”。

### 机制是否被错误画成 generic projection / filtering

没有。Panel (b) 的投影框写明 KL-preserving、ALM，并明确 POI logits unchanged；没有 candidate filtering 框。这一点通过审查。

## 8. Logit-level Review

### 正确点

- `ALM Projection → Category logits′` 的箭头方向正确。
- `POI logits (unchanged)` 作为并行输入进入 joint sampling，符合 `p_sample` 中投影只作用于 category positions 的实际路径。
- 没有 `Constraint → POI candidate filtering` 的错误箭头。

### 需要修正的精度

当前 `Category logits′` 可能被理解为“只保留 category vocabulary 的 logits”。代码实际是对完整 model log-prob 张量，在 `category_mask` 标记的序列位置计算并投影类别相关概率；因此建议写成：

```text
Category-position logits′
(selected by category_mask)
```

这样能同时表达位置维度和类别概率切片。

## 9. Panel Correspondence Review

### 一致之处

- Panel (a) 与 Panel (b) 使用相同的 terracotta 强调色。
- 两者都使用 `Category-order` 标题词。
- Panel (b) 详细展开了 Panel (a) 想表达的核心创新。

### 不足之处

- 没有显式 callout、编号或连接线证明 Panel (b) 是 Panel (a) 中高亮框的放大。
- Panel (a) 的高亮框名为 `Category-order Constraint-aware Generation`，Panel (b) 标题为 `Category-order Constraint Mechanism`，语义相关但不完全同名。
- Panel (a) 的高亮框被画成一个后续大模块，Panel (b) 则显示它是 reverse step 内部操作；这种空间组织不一致会造成 Major Issue。

建议 V2 统一为同一短标题，例如 `Category-order Constrained Reverse Sampling`，并在 Panel (a) 中标 `①`，Panel (b) 以 `① Detail` 对应。

## 10. Visual Hierarchy Review

### 现有层级

- Panel 标题和核心高亮框最醒目。
- terracotta 是唯一强调色，能够定位 contribution。
- 基础模块使用蓝/灰/绿，视觉权重低于约束模块。

### 可能的视觉竞争

- Panel (a) 的 `Conditional Joint Generator` 外框、`Category-order` 框和 Panel (b) 大标题都较重；创新仍然可见，但高亮框作为“后置模块”而非“采样内插入点”的空间关系削弱了其科学中心性。
- Panel (b) 右侧 logits 分支尺寸较小，容易在缩小到论文单栏时丢失 `Category logits′` 与 `POI logits (unchanged)` 的区别。

结论：视觉层级方向正确，但需要通过 reverse-step 插入点和 callout 强化创新，而非单纯增加颜色或字号。

## 11. Unsupported / Ambiguous Modules

### Unsupported modules

未发现必须删除的明显 unsupported algorithm module：

- `[Not Found]` candidate filtering：未画成模块；
- `[Not Found]` explicit POI transition restriction：未画成模块；
- `[Not Found]` full CFG：未画成模块；
- `[Not Found]` `po_encoding` condition injection：未画成模块；
- `[Not Found]` complex DAG adaptive controller：未画成模块。

### Ambiguous / misleading elements

1. `[Critical]` `sample-level po_matrix` 从 `Condition Representation` 发出：来源错误。
2. `[Major]` `Constraint-aware Generation` 位于整个 generator 之后：可能被读成 post-hoc module。
3. `[Major]` forward/reverse token strip 与核心 diffusion 模块脱节：语义关系不明确。
4. `[Major]` Input 未区分训练事件数据和推理条件：可能被读成历史序列条件生成。
5. `[Minor]` `Category logits′` 未明确 `category_mask` 选择的是序列位置。

## 12. Required Changes

### P0 — 必须修改

1. 修正 `po_matrix` 来源箭头：从 Input / Constraint Specification 直接进入 reverse constraint mechanism，不从 Condition Representation 输出。
2. 在 Panel (a) 或图注中明确 `Constraint is applied during inference / reverse generation`。
3. 将约束高亮框视觉上嵌入 reverse diffusion sampling loop，或用明确 callout 表示其位于 `model logits` 与 `joint sampling` 之间，避免被理解为后处理生成器。

### P1 — 强烈建议修改

1. 增加 Training / Inference 阶段标识，至少给出“Training: learn diffusion model / Inference: constrained reverse sampling”。
2. 将 Input 改为同时区分 `training event sequence` 与 `generation-time conditions + po_matrix`。
3. 把 `Forward diffusion` / `Reverse denoising` 轨迹连接到 `Category / POI Discrete Diffusion` 或 reverse-step inset。
4. 将 `Category logits′` 改为 `Category-position logits′ (category_mask)`。
5. 在 Panel (a)/(b) 之间加 `①`/`① Detail` 或简洁 callout，证明局部放大关系。
6. 统一 Panel (a) 与 Panel (b) 的核心模块命名，避免 `Constraint-aware Generation` 与 `Constraint Mechanism` 看起来像两个不同模块。

### P2 — 可选优化

1. 将“无 candidate filtering / 无 explicit POI transition restriction”下沉到图注或 Panel (b) 小号边界说明。
2. 给输出加 `time / category / GPS` 小号副标签。
3. 在 reverse 时间轴旁加 `local reverse-step axis`，避免与全图左到右主流程混淆。
4. 论文单栏缩放后复查右侧 logits 分支的可读性。

## 13. Recommended V2 Structure

### Panel (a) — Overall Framework

```text
Training data / generation conditions
  ├─ event time + context conditions
  ├─ category / POI token representation
  └─ sample-level po_matrix  ───────────────┐
                                             │ constraint specification
Condition representation                      │
  ↓                                          │
Temporal Add-Thin + Category/POI Discrete Diffusion
  ↓
Forward q(x_t | x_0)  [training]
  ↓
Reverse denoising p(x_{t-1} | x_t)  [inference]
  ↓
model category/POI logits
  ↓
① Category-order constrained reverse sampling
  ↓
joint Gumbel-max sampling
  ↓
Generated POI sequence (time / category / POI / GPS)
```

Panel (a) 的关键是：`po_matrix` 作为侧输入进入 ①，而不是进入 Condition Representation；① 必须位于 reverse denoising 和 joint sampling 之间。

### Panel (b) — Category-order Constraint Mechanism

```text
① Sample-level po_matrix
   M[A,B]=1  ⇔  A ≺ B
        ↓
Constraint parsing
        ↓
Category-position probabilities
   (category_mask; softmax / Gumbel-softmax)
        ↓
Order violation + existence violation
        ↓
KL-preserving augmented-Lagrangian projection
        ↓
Category-position logits′
        ├───────────────┐
POI logits (unchanged) ┘
        ↓
Joint Gumbel-max sampling
        ↓
Next reverse state x_{t-1}
```

Panel (b) 不应添加 candidate set、hard POI filtering、transition restriction 或全局 `po_encoding` condition。

## 14. Final Scientific Checklist

- [x] Input
- [x] Condition
- [x] Diffusion
- [ ] Reverse Diffusion 与约束插入点形成严格连接
- [x] `po_matrix`
- [x] order/existence energy
- [x] KL-preserving ALM projection
- [ ] category-position logits（需补 `category_mask` 位置语义）
- [x] constrained sampling
- [x] Generated POI Sequence

### 答辩委员会视角最终回答

**目前不能仅凭 V1 图完全准确理解本文核心创新。**

委员会成员可以识别“POI 离散扩散 + 类别偏序约束 + ALM 投影”这一大意，也能从 Panel (b) 看出约束没有被画成 candidate filtering；但他们可能误解 `po_matrix` 是条件编码器输出、误解约束是普通生成器之后的后处理，并无法确认约束只在 inference/reverse sampling 生效。修正 P0 和关键 P1 后，图才具备论文方法主图所需的科研可信度。


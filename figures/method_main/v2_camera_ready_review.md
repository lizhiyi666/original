# V2 Camera-ready Review

审查对象：`figures/method_main/method_main_v2.drawio` 及其 PNG、SVG、PDF 导出。

审查原则：代码真实性和已完成的科研结构保持不变；本轮只判断论文定稿可用性，不修改图形、算法或方法机制。

## 1. Overall Verdict

**Camera-ready with Minor Revision**。

这张图可以作为硕士论文正文中的方法主图，但应以双栏跨栏宽度（约 170–180 mm）排版。当前不存在影响方法理解的 P0 科研错误；仍有两项定稿级 P1 问题：单栏缩放不可读，以及 Draw.io 可编辑源没有完整保存静态图中的反向循环内部节点和部分辅助标注。

## 2. Publication Readability

按导出 PNG 的 2771 × 1822 像素、约 1.52:1 比例估算，PDF 页面为 907.2 × 596.5 pt（约 320 × 211 mm），因此论文插入时会发生明显缩放。

在 170–180 mm 宽度下，Panel 标题和主要模块标题可读，核心约束链仍可定位；9.5–11 pt 的源标注缩放后约为 5–6 pt，适合屏幕阅读但不适合高质量打印正文。8 pt token/位置标签属于辅助信息，不应承担核心语义。

## 3. Single-column Readability

不通过。若缩放到 80–90 mm 单栏宽度，16 pt 模块标题约降至 4–5 pt，`po_matrix` 说明、`category_mask`、`POI logits (unchanged)`、`x_t/x_{t-1}` 和底部边界说明均会过小。该版本不适合单栏放置；不建议通过继续缩小整图解决。

## 4. Double-column Readability

基本通过。推荐宽度 170–180 mm 时：

- Panel (a) 的 TRAINING / INFERENCE 分区、Reverse Diffusion Loop 和输出框清晰；
- Panel (b) 的 `po_matrix → energy → ALM projection → category-position logits′ → Joint Sampling` 主链可读；
- `category_mask` 的 sequence-position 语义和 POI logits unchanged 仍能辨认；
- 底部 “No candidate filtering / No explicit POI transition restriction” 应视为脚注式边界说明，而不是主流程节点。

打印时建议使用 300 dpi 以上栅格预览或直接采用 SVG/PDF。

## 5. Typography

生成脚本中的字号层级约为：Panel 标题 22–24 pt，主要模块 14–16 pt，次级说明 9.5–13 pt，最小 token/位置标签 8 pt。该层级关系正确，但次级说明在 180 mm 成图时偏小，单栏时不可接受。

数学和符号文本（`A ≺ B`、`M[A,B]=1`、`x_T`、`x_{t-1}`）在双栏下可辨认；它们不应被解释为训练损失公式。标题、模块标题和注释之间的层级差足够明显。

PDF 使用嵌入式 DejaVuSans Type 3 字体，未发现缺失字体或位图嵌入；正式投稿前应确认学校模板是否接受 Type 3 字体。

## 6. Information Density

Panel (a) 信息密度较高但仍可扫描：训练和推理 lane、约束侧输入、reverse loop 和输出构成明确主线。Panel (b) 约束链较密，但每个节点承担不同语义，没有重复模块。辅助说明较多，尤其是 mask 示意和底部边界说明；它们没有压过核心创新色链，但缩小时会成为最先丢失的信息。

## 7. Innovation Visibility

双栏缩放后仍可第一眼看到 terracotta 强调链，并能顺序识别：

`po_matrix → Order + Existence Energy → KL-preserving ALM Projection → Category-position logits′ → Joint Sampling`。

Panel (a) 的 `① Category-order projection` 位于 reverse loop 内，Panel (b) 的 `① Detail` 提供局部展开，因此不会被误解为生成后的 post-processing。创新应继续以“采样期结构化控制层”表述，而不是独立生成器或普通条件编码器。

## 8. Panel (a) Review

- TRAINING 与 INFERENCE / GENERATION 已明确分开；
- `po_matrix` 以独立 constraint specification 进入 reverse step，没有连接到 Condition Representation；
- `x_t → Denoising → Category-position logits → projection → x_{t-1}` 的局部方向清楚；
- 输出明确为 Generated POI Sequence，并附带 time/category/POI/GPS 说明；
- 约束框位于 reverse loop 上方并用虚线指向投影点，语义上是采样期输入而非普通 condition。

## 9. Panel (b) Review

核心链完整且方向正确。ALM Projection 的输出明确标为 `Category-position logits′`，并与 `POI logits (unchanged)` 分支共同进入 Joint Sampling。`Category-position Mask` 明确写有 `sequence positions only`，位置条示意不会被合理地解释为 POI 候选集合。底部边界说明进一步排除了 candidate filtering 和显式 POI transition restriction。

## 10. Terminology Consistency

图中 `po_matrix`、`Order + Existence Energy`、`KL-preserving ALM Projection`、`Category-position logits′`、`POI logits (unchanged)`、`Joint Sampling` 和 `Reverse Diffusion Loop` 与 `method_innovation_map.md`、`thesis_method_graph.md` 的稳定术语一致。

`Category-position logits`（投影前）与 `Category-position logits′`（投影后）层次区分合理。图中没有使用未实现的 `po_encoding` condition、完整 CFG、candidate filtering 或复杂 DAG 模块。

## 11. Arrow / Connector Review

静态图中的主箭头均按左到右或 `x_t → x_{t-1}` 方向，约束虚线仅用于 specification/callout，未形成循环误导。Panel (b) 的 mask 虚线是辅助依赖，不与主链竞争。未发现 connector crossing、错误后处理箭头或将 POI logits 投影的错误路径。

## 12. Color / Visual Hierarchy Review

基础模块采用低饱和蓝、灰、绿；类别偏序创新统一采用 terracotta；输出和 joint sampling 使用基础蓝色。颜色不承担唯一语义，箭头和文本仍能独立表达方向。Panel (a)/(b) 的核心创新颜色一致，Level 3 的 mask、位置标签和负边界说明没有压过 Level 1/2。

## 13. Export Quality

- PNG：存在且尺寸为 2771 × 1822；视觉检查未发现裁切、溢出或断裂字符。
- SVG：可解析，未发现 `<image>` 位图嵌入；文字和线条保持矢量。
- PDF：单页、矢量输出，无 `pdfimages` 位图对象；字体已嵌入。
- Unicode：箭头、偏序符号和下标在 PNG/SVG/PDF 中均正常显示。

## 14. Editability

Draw.io XML 可解析，35 个 cell、14 条 connector 的端点均有效，节点没有越出 1600 × 1050 画布，外层 panel、主要模块和连接器均为原生 cell。

但静态导出中的 `x_t`、`Denoising`、类别 logits 内部步骤、mask 位置 token、图例和部分注释没有作为独立 Draw.io cell 保存；它们在 XML 中不是可单独编辑对象。因此“可编辑源与最终导出完全同构”尚未达到 camera-ready 的最佳标准。

## 15. Recommended Minor Revisions

### P0

无。当前没有发现会改变论文方法理解的科学错误。

### P1

1. 正文中将本图固定为双栏跨栏图，推荐物理宽度 170–180 mm；不要以 80–90 mm 单栏宽度使用当前版本。
2. 在最终归档前，使 Draw.io 源补齐静态图中的 reverse-loop 内部节点和关键辅助标注，保证主要视觉对象均可独立编辑。

### P2

1. 若学校打印模板要求更大的最小字号，可将底部边界说明移入 Figure caption，将 token/位置标签作为纯示意弱化。
2. 投稿前确认模板对嵌入式 Type 3 字体的兼容性；必要时在不改变版式的情况下改用嵌入 TrueType 字体。
3. 进行一次真实打印尺寸校样，重点检查 `category_mask`、`POI logits (unchanged)` 和 `x_{t-1}`。

## 16. Recommended Final Size

- Canvas ratio：约 1.52:1，保持当前横向 landscape 比例。
- Physical width：170–180 mm（双栏跨栏）。
- Recommended minimum font size：正文成图后不低于 7 pt；核心模块标题建议 8.5–10 pt；8 pt 以下仅用于非关键示意标签。
- Figure placement：放置在方法章节中介绍 reverse diffusion 与类别偏序约束之后，使用跨双栏位置；caption 紧跟图下，不建议再压缩到单栏。

## 17. Final Checklist

- [x] Scientific correctness
- [x] Training / Inference clarity
- [x] Reverse diffusion clarity
- [x] po_matrix path
- [x] ALM projection
- [x] Category-position logits
- [x] category_mask = sequence positions
- [x] POI logits unchanged
- [x] Joint sampling
- [x] Panel correspondence
- [x] Typography hierarchy
- [ ] Single-column readability
- [x] Vector export
- [ ] Complete Draw.io editability parity with static export

## 18. Final Verdict

这张图可以进入硕士论文正文，但应作为双栏跨栏方法主图使用。唯一需要在正式归档前处理的定稿问题是：将 Draw.io 可编辑源补齐到与静态导出一致；同时不能把当前版本压缩为单栏宽度，否则关键注释和符号会失去可读性。该结论不要求改变算法、约束机制或 Panel (a)/(b) 的整体结构。

建议 caption 强调：条件时空与离散扩散生成、样本级类别偏序约束、以及 inference-time reverse diffusion 中的 category-logit KL-preserving ALM projection 和 joint sampling；不应声称 candidate filtering、POI transition restriction 或训练期偏序损失已构成稳定主路径。

# WWW 2027 投稿前复核与修正

日期：2026-10-21。对象：Research Track 长文，现有 `paper-www2027/`。
基线：`9b05681`；中断前修订检查点：`160b20a`。当前 PDF 为 12 页。
范围：按已批准方案通读正文、附录和实际 PDF，修正文稿与实现/证据间的表述问题，完成分页、保护检查及交付。
沿用 [既有引用审计](CITATION-AUDIT.md)，没有重新声称通读全部文献或开展全面新颖性搜索。
本报告由同一执行者整合 Reviewer、Submission Checker、Integrity Auditor 和 Writer 的定向检查，不代表独立多人评审或录用判断。

## 1. 已修正并验证

| ID / 严重程度 | 位置与发现 | 证据及处理 | 验证结果 |
| --- | --- | --- | --- |
| F01 / 高 | 恢复构建为 13 页；结论末两行孤立在 p8，附录另起 p9 | 精简 §4.5、局限性与结论的重复说明；不删实验、不缩小字号/边距、不改图表 | 正文止 p7，参考文献 p8–9，附录 p10–12；全部页面目视通过 |
| F02 / 中 | §4.1 原文容易将投影位置理解为预测 x0 的原始输出 | `discrete_diffusion/diffusion_transformer.py:p_pred` 先构造 `q_posterior`；`p_sample` 对 `model_log_prob` 投影。明确 z 是 reverse-step log-probabilities | §4.1、§4.5 与附录 A 术语一致；只改描述 |
| F03 / 中 | §4.6 Eq.(9) 未明确受控步解码使用投影后分布 | 同文件 `p_sample` / `log_sample_categorical`：投影后分布传入一次 Gumbel 抽样，support masking 后复用同一噪声。定义带波浪号 z；区分 sampled category token 与实际 POI category | 公式与调用链一致；保留中间 mask、最终 unsupported-category fallback 及非零 mismatch |
| F04 / 中 | §5.3 将 token KL 的效应写成 Full 中独立边际效应 | 表 3 的 No KL 同时关闭两项 KL；No Distance KL 只关闭距离项。二者比较仅识别距离正则关闭时的 token anchor 效应 | 已限定归因条件；所有数字、SD、粗体及表格源文件未改 |
| F05 / 中 | 附录 B 未完整给出门槛的参照和比较方式 | `tools/distribution_study_common.py:failures/select`：约束参照 projection Full；JSD 参照 JointGen；city-wise seed means；绝对容差 1e-12；拒绝 undefined。正文补齐筛选/确认门槛、方向和数据池 | 原两候选确认失败及 override 仍明确；没有把校准写成独立验证或改 v2/v3 口径 |
| F06 / 中 | p1 已解释 Web 用途，但未点明所投 track | 当前官方规则要求第一页同时说明 Web 与 track 相关性；在现有 Relevance to the Web 段补 Web Mining, Multimedia and Multilingual Content Analysis | 在实际 p1 可见；没有把未测下游效用写成结果 |
| F07 / 低 | 注册说明含旧摘要措辞、占位邮箱说明和过期待办 | `main.tex`、既有作者信息及当前官方时间；更新 `REGISTRATION.md`，旧短摘要明确标为历史 | 当前锁定摘要与注册长摘要说明一致；实际 portal 内容仍待作者核对 |
| F08 / 低 | 分页与交接记录未跟随恢复修订更新 | 更新正式 PDF、README 和 HANDOFF；说明旧 `balance=false` 在当前 acmart 中被忽略，实际版式由输出验收 | 当前来源、页数、PDF 指纹、问题报告和构建命令一致 |
| F09 / 低 | p4 公式（10）过长，编号自动落到公式下方单独一行 | 用户授权后，使用 `equation` + `aligned` 在加号处分两行；数学内容、编号、字号和边距保持不变 | 编号在右侧垂直居中；编译及完整性检查通过，锁定测试 8/8；仍为 12 页，与 `58d66fe` 逐页像素比较仅 p4 变化 |

上述修订未改变方法机制、实验结果、随机种子、训练/采样任务或评估器。
标题、当前摘要、五关键词及作者顺序相对 `9b05681` 原样保留；历史 `eliminates` → `reduces` 授权修正没有重新执行。

## 2. 摘要锁定未改

| ID / 严重程度 | 原文与风险 | 最强反查证据 / 当前处理 |
| --- | --- | --- |
| L01 / 中 | “pairwise ordering coverage” 容易被读成正确排序比例 | §5.1 和附录 D 明确 pair coverage 只要求两个类别均出现；排序由 OVR/Unsat 衡量。6.5×/4.2×对应出现覆盖率；摘要按锁定要求不改 |
| L02 / 中 | “trade-off is largely separable” 与 “restores spatial realism” 可能被读成通用保真保证 | §4.7、§5.4 和结论限定为保持实际类别/时间/顺序的空间替换；纽约 CategoryTransition、G-RANK 仍较差。同类别细化的不变性不是任意分布误差可消除的证明 |
| L03 / 中 | “Across two cities under an out-of-distribution split” 没有展示两城 OOD 来源证据强弱不同 | §5.1 和局限性保留 NY 的偏好反转定义、IST 历史划分来源不全；摘要不增删限定语 |
| L04 / 低 | “joint temporal + category/POI discrete-diffusion generator” 可能被理解为时间也采用离散扩散 | §4.1 已区分 arrival-time diffusion point process 与 category/POI discrete branch；本轮不改锁定摘要 |

这些是作者决定是否另行授权修改的明确位置，不代表已解决。若作者决定维持摘要，应同时理解正文中的相应边界。

## 3. 证据不足与保留的科学边界

以下均已反查正文/附录，不将已有披露重复列为“遗漏”，也不以润色替代新证据。

| ID / 程度 | 已有证据及反查位置 | 未建立的结论 / 完成条件 |
| --- | --- | --- |
| E01 / 高 | 表 1、表 5，§5.2/5.4；纽约最终 CategoryTransition 0.0183 vs JointGen 0.0070，G-RANK 0.1914 vs 0.1612 | 不能声称两城全面保真改善。当前保留负面结果；进一步提升需要另行授权研究，非本轮任务 |
| E02 / 高 | §5.4、附录 B；校准池被 backbone 见过，两候选确认均未过全部分布门槛，正式采样采用 override | 不支持独立验证选优或“全部质量门槛已通过”；当前选择过程已透明披露 |
| E03 / 中 | §5.1、局限性；NY 3160/2108，IST 7035/4914；既有 manifest 与 split 说明 | IST 原始 split 构造及完整训练 provenance 未恢复。需要原始记录，不从 OOD 文件名推断具体划分保证 |
| E04 / 中 | §5.5、附录 E、表 6；3.35%–3.98% 来自旧 geometry 相对 projection Full 的资格测试 | 最终全栈/JointGen 端到端开销与 PO-CFG 训练 GPU-hours 未测；不能将旧比例转用于最终全栈 |
| E05 / 中 | 表 1/3/4 的三采样种子 sample SD；§5.1/5.2、局限性 | 没有训练种子稳定性、置信区间或显著性检验；粗体仅比较未舍入均值 |
| E06 / 中 | §4.7、§5.4、表 5；(c)→(d) 保持实际类别及 Distance 端点过滤集；(d)→(e) 同时改 rho、beta、候选池 | 不能把后一步收益单因子归于 beta；跨方法 Distance 的可用子集也不保证相同 |
| E07 / 中 | 引言与局限性明确 downstream utility、复杂 DAG、冲突处理及正式隐私保证未评估 | 不支持下游推荐收益、通用约束可行性或安全发布保证。数据相似性不等于隐私保护 |
| E08 / 中 | 引言/相关工作区分 Marionette backbone、CDD 投影适配与解码/细化；沿用 CITATION-AUDIT.md 的六篇核心语境核查 | 支持已有贡献定位，但本轮没有完整领域新颖性检索或外部竞争方法复现；不作“首次”或录用概率断言 |

度量口径已在附录 D 保留：Unsat 是关系微平均而非失败轨迹比例；Radius 单位为 sqrt(km)，不是标准回转半径；Distance 使用端点类别过滤；v2 论文表与 v3 确定性审计不可混填。

## 4. 待作者确认

1. 实际已完成摘要注册；portal 的标题、作者顺序、邮箱、摘要、长文 track 与锁定材料一致。未进入投稿系统验证。
2. 是否授权处理 L01–L04。当前严格保留摘要，未自行修正这些术语或宽泛表述。
3. 数据使用/再分发许可、适用伦理或知情同意要求、作者责任与 AI 使用披露、COI、无并行投稿等事实。论文不能代替作者事实声明。
4. 核对实际系统截止时间并自行上传。官网全文截止为 2026-10-25 AoE 日终，即北京时间 10-26 20:00 前；旧 portal 21:00 记录未核实，按较早时间安排。
5. ORCID 尚未提供，不猜测、不代填。匿名 PDF 不包含源文件内的作者身份；不要把带身份的源包当作匿名附件上传。

## 5. 验证与交付记录

- 官方规则：[WWW 2027 Research Track](https://www2027.thewebconf.org/research-track-papers/)，2026-10-21 直接读取：长文正文最多 8 页、含附录/参考文献最多 12 页，第一页说明 Web/track 相关性，double-blind，ACM 双栏。官网允许 LLM 文字重述，作者承担全文责任。
- 实际构建：沿用 TeX Live、原 `latexmkrc` 与既有 build 目录；`latexmk` 退出 0。正文结束 p7，参考文献 p8–9，附录 p10–12，PDF 12 页。
- 完整性：`figure_build/verify_paper.py --build-dir ccfa-workfiles/writing/www2027/build` 通过；检查器的附录页序门槛更新为参考文献后从 p10 开始。8 项锁定内容测试全部通过。
- 保护检查：`main.tex`、全部表文件、全部图文件、引用库及科研/封存目录相对基线无差异；正文修改没有引入新结果。原图稿未重新生成。
- PDF：12 页逐页渲染并目视检查；无裁切、重叠、丢字或孤立结论页。字体嵌入，无 Type 3，文本/元数据/链接匿名，无 TODO、undefined 引用或 overfull。
- 非阻断提示：underfull、review reference-block 提示、34 条原有 BibTeX 样式警告及未使用的 `balance=false` 选项。没有捏造元数据消除警告。
- prose checker：本轮改动的 method/conclusion（含附录）已运行；剩余提示是 precision-term/分号密度启发项，经语境检查保留实际技术条件与公式叙述。
- 正式输出：`main.pdf`；SHA256 `06b821125d36e8f14c674fc0c051d0ce6575f80d9ee10dbd7cdd3df19372e5cf`。公式（10）调整后重新目视检查 p4/p5，其余页面与已验收版本像素一致；调整前 PDF 可从 `58d66fe` 恢复。缓存、日志和预览不进入 Git 提交。

本轮完成条件是范围内文字/分页问题修正、构建保护检查和交付同步；E/L/A 类作者或科学问题不因编译成功自动关闭。

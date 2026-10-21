# WWW 2027 论文接续状态

更新：2026-10-21。原 T1–T9 任务说明保存在提交 `c828598`；本文记录证据修订、Figure 1 和投稿前复核交付。
后续先读本文及根目录 `AGENTS.md`，不要重复运行已完成实验。

## 1. 当前交付和未闭环项

T1–T9 和本轮已授权修订均已完成，包含官方 CCS 导出、摘要白名单、50 条文献元数据核验、
正文定向改写、编译、匿名、字体与逐页检查。科学边界见第 6 节，不把编译通过当作科学问题解决。
Figure 1 重设计、图注及无障碍描述已完成。第 8 节保留 2026-10-20 的历史图稿交付；
当前正式 `main.pdf` 的投稿前修订、验收与指纹以第 9 节为准。
统一问题报告为 `REVIEW-AND-FIXES.md`，按已修正、摘要锁定、证据不足和作者确认分类。

1. **摘要事实冲突已按用户授权修正。** 仅将 `eliminates residual` 改为 `reduces residual`，
   与六份正式输出的 **0.0034%–0.0219%** mismatch 一致。代码最终类别无支持时
   回退到任意合法 catalog POI。其他摘要内容继续锁定，零 mismatch 仍不成立。
   另须注意 “pairwise ordering coverage” 对应两个类别均出现，不表示排序正确，正文已区分。
2. **CCS 已闭环。** 2026-10-18 从 <https://dl.acm.org/ccs> 正常浏览器页面逐层选择五项，
   使用 Generate Code 取得官方 CCSXML 和 `\ccsdesc`，已替换 `main.tex` 旧块。
   Location based services / Machine learning 为 500，Social networks / Data mining 为 300，
   Neural networks 为 100。五项编号、名称及权重均与用户提供内容一致。
   新提供的六关键词未替换锁定的五关键词；标题和作者顺序未改变。
3. **不操作 WWW 投稿系统。** 注册、核对 portal 时间及上传由作者本人完成。
   本轮仅为文献核验只读访问公开 OpenReview 论文记录，未进行投稿操作。
4. **引用核验已交付。** 文献库 50 条、正文实际引用 45 条，5 条未引用记录保留。
   46 个 DOI 与 4 个无 DOI 的官方记录已核验；核心六篇做定向语境核查。
   背景文献为身份/题目/可得摘要层级，不声称通读 50 篇全文。详见 `CITATION-AUDIT.md`。

## 2. 锁定内容和作者

- 标题：`Training-Free Partial-Order Control for Discrete-Diffusion POI Check-in Generation`。
  导师要求不要加回 “distribution correction”。
- 标题、五关键词、作者顺序与 `c828598` 完全一致；摘要只允许上述单处授权替换。
- 第一作者 Zhiyi Li：`25s151146@stu.hit.edu.cn`。
- 通讯作者 Kaiqi Zhao：`zhaokaiqi@hit.edu.cn`。
- 两人均为哈工大（深圳）计算机科学与技术学院；邮箱已填，ORCID 未提供，不猜。
- 保留 `anonymous,review`。源文件有身份信息，不要将源包作为匿名附件公开提交。

## 3. 截止时间：保留一小时差异

[官方 Research Track 页面](https://www2027.thewebconf.org/research-track-papers/)
于 2026-10-21 重新核实：摘要 10-18 AoE 日终、全文 10-25 AoE 日终，分别对应北京时间
**10-19 / 10-26 接近 20:00**，不是 21:00。旧交接写 OpenReview 为 13:00 UTC / 21:00 北京时间，
本任务未进入 OpenReview，不能确认 portal。按更早官网时间安排，作者自行核对差异。
长文正文最多 8 页、含参考文献及附录最多 12 页；第一页须说明 Web/track 相关性。

## 4. 环境和 Git 规则

- 仓库：`D:\桌面\轨迹\轨迹\轨迹生成\实验\original`。
- 分支：`codex/distribution-best-formal-v1`；远程 `lizhiyi666/original`。
- 历史检查点：`c828598` → `fca940c` → `faccb4f`；本轮基线为 `5efc4cc`。
  摘要与测试组 `d1c088e`、证据/引用/正文组 `e5abed7` 均已推送并核对远程完整 SHA。
  Figure 1 基线 `4389ee0`；恢复检查点 `b205989`、正式矢量图 `f56b5e5`、论文图注集成
  `09b462d0900c3b9da86d300e66ee913aec8de8bb` 已推送。最后一项于 2026-10-20 重新核对远程 SHA 一致。
  最终交接核验为本文件所在的后续提交，查 `git log`。每组前后仍需提交、推送并核对 SHA。
- 只暂存本任务文件，不 `git add .`；不提交数据、权重、缓存、日志、字节码、依赖目录。
- 直连曾超时；现有代理 `git -c http.proxy=http://127.0.0.1:7897 push origin
  HEAD:refs/heads/codex/distribution-best-formal-v1` 可用。相同 `-c` 配合 `ls-remote`
  核验分支。未修改全局配置，不强推、不 reset；前置推送失败就停止新修改。
- 作图 Python：`D:\Anaconda\envs\Marionette\python.exe`。
- 论文目录内构建：`latexmk -pdf -interaction=nonstopmode main.tex`。
- **当前真实退出 0**，12 页，结论 p7，附录 A 从 p8 开始；零错误、未解析引用、TODO 调用、
  overfull。第一页 Web 相关性、PDF 字体嵌入、文本/元数据/链接匿名检查及逐页视觉检查通过。
- 原稿 `amssymb` 造成 `\Bbbk` 重定义，已移除重复导入。**返回码 12 不应被称为无害**；
  只搜索 `^!` 会漏 file-line-style 错误。保留 `latexmkrc` 的中文路径 BibTeX 修复。
- 保留原 ACM 双栏、字体和边距；仅将现有主表/权衡图提前排队，避免独立浮动页。
  `acmart` 原生 `balance=false` 关闭可选末页列平衡，消除参考文献尾页的 overfull vbox；
  尾页按普通双栏流排，未缩小字号或图表。
- 剩余非致命提示：review 的 reference block 隐藏、underfull vbox，以及 34 条 BibTeX
  样式提示，主要是可选 address、ICLR 非分页记录和 AAAI 卷期组合；不为消警告编造字段。
  旧“52 条、未全量审计”已更正为 50 条元数据核验，语境覆盖边界见审计报告。

## 5. T1–T9 执行记录

| 任务 | 结果与依据 |
| --- | --- |
| T1 | 主比较两表合成上下共享 caption 面板；stage-wise 移入附录 C；概要压缩。三张原表数据行/精度/SD/粗体保持不变 |
| T2 | `figure_build/generate_tradeoff.py` → `figures/tradeoff.pdf/.svg`；仅用已有 v2 均值，等量程线性轴，不移动测量坐标 |
| T3 | `figure_build/generate_method_main_v2.py` → `figures/framework.pdf/.svg`；逆循环总览、logit 投影、同 Gumbel 类别条件 POI 解码、同坐标系细化四面板；标注 fallback，权重仍是目标输入 |
| T4 | 附录 D 指标定义来自 `evaluations/statistical_metrics.py`、`evaluations/ovr.py`、`tools/ablation_common.py`；补分母、过滤、catalog/token、历史直方图边界 |
| T5 | `generate_paper_evidence.py` 生成 `tables/calibration-grid.tex`，18 行 v2；SHA 校验 v3 selection source，顺序与失败门槛均一致 |
| T6 | manifest 的 NY 3160/2108、IST 7035/4914；`tools/make_opposite_split.py` 说明 NY 是偏好反转，非完整配置绝对不相交；IST 来源缺口保留 |
| T7 | `pcdg-geo-v1-tol10-20261010/speed/confirm-r2-b0.1-summary.json` → `tables/cost.tex`；64 条 batch，3.35%–3.98%，分母 projection Full |
| T8 | 相关工作对齐投影/解码/细化；核验五篇主要引用；移除无证据的“此前无人处理”和“其他均需重训”；CDD 采用会议版作者顺序，补部分页码 |
| T9 | 邮箱、TODO 空操作、锁定块/匿名/字体/页数/日志检查完成；官方五项 CCS 导出已接入，XML/LaTeX 一致性检查通过 |

另修正文 token KL 方向：实现 `F.kl_div(log_p, log_q, log_target=True)` 是
**base→projected**。附录写明 token KL 除以完整 batch 大小，约束 penalty 按关系/样本求和。
未修改科学实现或实验结果。

图按最终正文尺寸检查，SVG 可编辑文字，固定 ID 盐值并去时间元数据；同环境可稳定再生。
临时渲染只放 `ccfa-workfiles/writing/www2027/build/`，不提交。
`figure_build/verify_paper.py` 使用 pypdf 与 Poppler，构建后运行，检查锁定块、原表数据行、
日志、页数、匿名/字体和受保护路径差异；可用应用自带文档 Python。
CCS 检查对照 2026-10-18 官方导出的五个 ID/名称/权重，并验证恰好五条对应 `\ccsdesc`。
错误 ID、错误权重、重复 `\ccsdesc`、缺失 XML 的内存负例均被拒绝。
CCS 接入后 `latexmk` 退出 0，仍为 11 页、正文结束 p7、附录 p8；第一页和 p7/p8 接口重新目视核验。
官方导出截图保存在本地 `ccfa-workfiles/writing/www2027/build/ccs-official-export.png`，不提交缓存。
以上 CCS 接入的 11 页状态为历史记录。本轮当前 PDF 为 12 页；摘要事实冲突已按授权修正，
但完整性通过仍不等于科学审稿通过。

### 本轮“深化现有证据”修订

- 按 Reviewer 的证据核查收紧论证，Writer 配合 Humanization 改写，删除重复背景和重复
  几何超参数叙述；完整设置仍在附录 A，公式含义和实现未改。
- §3 区分 category token 和 catalog POI category；§4 说明解码衔接两者，同类别替换的
  不变字段只支持受限空间可分离性，不推出所有分布误差可消除。
- 区分 Marionette backbone、CDD 投影范式及本任务适配；修正“projected-gradient”为
  SGD penalty optimization，明确松弛 inverted-pair mass 不等于离散可行性保证。
- EnergyGuide 是本地 order+5×existence 梯度基线，不是 Schiff 等 discrete CBG 的精确复现。
- 引用修正包括 González 原论文 DOI、TrajGDM 首作者、Diff-POI/Act2Loc 正式卷期年、
  CDD 官方页码，以及完整作者/卷页信息。50 个 citation key 保持不变。
- 没有新增文献、结果表或图形重绘。没有更改科学代码、种子、评估协议、训练/采样任务或封存文件。

## 6. 科学边界与来源

- 同类别细化保持实际类别、时间、顺序和条件等字段；它不能修复保持不变的类别分布误差，
  也不保证候选池中存在匹配所有参考分布的解。decoder 仍有 unsupported-category fallback。
- NY CategoryTransition 0.0183 vs JointGen 0.0070、G-RANK 0.1914 vs 0.1612；PO-CFG 在这两项
  及 Category 最好。IST 六项优于 JointGen、五项优于 PO-CFG。
- 三个固定 checkpoint 采样种子的样本 SD，不是 CI、显著性或训练种子稳定性。
- 主结果/消融/网格表为 v2；最终候选用 v3 重评估排序。18 组初筛、两组确认顺序一致，
  因此未触发换候选的作者决策。两组确认均未通过全部分布门槛，all-gates 排名为空；
  正式采样经授权 override，保留约束门槛、容许分布失败。不能说全部门槛通过。
- v2：`experiment_runs/ny-ist-distribution-balanced-v1-20261012/{screen-selection,confirm-selection}.json`。
  v3：`experiment_runs/ny-ist-distribution-balanced-v1-eval-v3-20261013-linux/summary.json`。
  最终：`experiment_runs/ny-ist-distribution-best-formal-v1-20261013/selected-candidate.json`。
- 训练侧校准数据 backbone 见过，非独立验证；无测试指标进入选择/候选构造/优化。
- v2 已按 POI 身份对齐；v3 使 top1000 频数并列排序确定化。0.0032 最大偏移只适用于
  18 组正式审计指标，不能推广到全部历史方法；不把 v3 单元混入 v2 行。
- (d)→(e) 同时改 rho、beta、候选池。(d) original+top15+nearest16；
  (e) original+top11+nearest12+train-most-visited8。不能单因素归因。
- (c)→(d) 保持实际类别、事件顺序、条件、约束指标及 Distance 端点过滤集。
  但不同方法的 Distance 子集不保证一致。
- Radius=`sqrt(mean(Haversine distance to arithmetic lat/lon centroid))`，单位 sqrt(km)，
  不是标准回转半径。Unsat 是关系微平均，不是失败轨迹比例；pair coverage 不要求排序正确。
- 旧几何计时分母是 projection Full，不是 JointGen；不含时间生成、加载、传输，不能套于
  PCDG-PD+Geo 或最终全栈。最终全栈成本、PO-CFG GPU-hours 未测。
- 旧交接“未实现类别→POI 硬过滤”不正确：已有 hard support masking，附带 fallback。
  复杂 DAG 验证、自适应调度、正式隐私保证、下游效用没有已完成实验证据。
- 不修改 `experiment_runs/`，不重训、不重采样、不重启运行。

## 7. 后续最短路径

作者通读当前稿及 `CITATION-AUDIT.md`，确认接受第 6 节保留的证据边界，再自行核对注册信息
和投稿期限并上传 PDF。另须处理 `REVIEW-AND-FIXES.md` 中的锁定摘要及作者确认事项。
当前摘要严格锁定，后续任何摘要修改需作者另行授权。不要顺带扩展实验、切换评估版本
或改锁定标题；任何后续新修改继续遵守 GitHub 检查点规则。

核验的五篇主要来源：
[Marionette](https://github.com/UM-Data-Intelligence-Lab/Marionette)、
[CDD](https://proceedings.neurips.cc/paper_files/paper/2025/hash/124c1cc9f7750f512ed81cdbf2fd5bf5-Abstract-Conference.html)、
[ControlTraj](https://arxiv.org/html/2404.15380v1)、
[Grammar-constrained decoding](https://aclanthology.org/2023.emnlp-main.674/)、
[Control-DAG](https://aclanthology.org/2024.naacl-short.42/)。本轮另核验
[Discrete Guidance](https://openreview.net/forum?id=i5MrJ6g5G1) 的 §3.1–3.2 方法语境。
全量记录见 `CITATION-AUDIT.md`；本次是受影响论证的定向核查，不是全稿录用判断。

## 8. Figure 1 最终交付核验（2026-10-20）

- 用户已批准跳过不可用的图像概念稿，直接重建可编辑矢量图；没有生成新的实验结果。
  四面板为逆循环总览、投影、类别条件解码和同类别细化。相同 A/B/C 事件贯穿后两面板，
  示例是 schematic；实线数据流、虚线条件/目标输入。图注保留中间步 mask 与最终 fallback 的边界。
- 既有 LaTeX 编译脚本以 TeX Live 和现有 build 目录核验，退出 0，目标已是最新；
  12 页，正文 p1–7，附录从 p8。Figure 1 在 p3，p3/p7/p8 已重新渲染并目视检查；
  未改变的页面沿用上一轮检查。未缩小正文、边距、表格或图内字号，未放宽检查器。
- `figure_build/verify_paper.py`：通过。标题/关键词/作者顺序、官方 CCS、摘要白名单、
  原表数值/精度/SD/粗体、匿名、字体、引用、无溢出和受保护路径检查通过。
- `python -B -m unittest discover -s figure_build -p test_verify_paper.py -v`：8/8 通过。
  授权替换通过；旧 overclaim、其他摘要词句改动、摘要缺失、标题/关键词及作者重排均被拒绝。
- `test_generate_method_main_v2.py`：5/5 通过，覆盖 7 × 3.25 英寸、Times New Roman、
  最小 8 pt、相同目录/实际类别、条件虚线、必要标签、live text、唯一语义 ID 和无嵌入位图。
  图形测试使用原 Marionette 作图环境；摘要/PDF 检查用带 `pypdf` 的文档环境。
- 与 `4389ee0` 比较，§4.1 起的方法正文及公式完全不变；全部表格、其他图稿、
  科学实现和封存实验未改。只更新 framework 源码/测试/图稿、方法开头/图注/描述、PDF 和交接文档。
  v2 表格与 v3 审计描述继续分离。
- 正式导出：`paper-www2027/figures/framework.svg`（live text 与独立矢量对象）及
  `framework.pdf`（嵌入字体的矢量图）；预览为 `ccfa-workfiles/writing/www2027/build/framework.png`。
  图形源码为 `figure_build/generate_method_main_v2.py`，未重新生成已验收的图稿。
- 最新构建已同步到正式 `paper-www2027/main.pdf`，两份 SHA256 相同：
  `77903bfcac8cd4fdd5ee693713dfb7b702d3ad0a497e714d60c74febc8e625a9`。
  SVG SHA256：`303404bd24088487fbf0c7d3b1698a9a985fce2589ef2d09cc1bc77da95b613a`；
  图形 PDF SHA256：`42e8722a72e9ac4edbb938613d62f8b440f53917744bb371560bc1609ec88a50`。
  旧论文 PDF 保留在 `4389ee0` 等 Git 历史中；本地未跟踪缓存不批量提交或清理。

## 9. 投稿前复核与分页修正（2026-10-21，当前交付）

- 接续用户批准的方案，基线 `9b05681`；恢复修改检查点 `160b20a` 已实时核对与 GitHub 一致。
  没有重复启动实验、修改科学代码、评估协议、种子、checkpoint 或封存结果。
- 定向复核支持原位修正：第一页明确 Web Mining track；区分 clean-token 预测与
  reverse-step log-probabilities，解码公式明确使用投影后的分布；校准门槛补齐参照、
  方向、未舍入均值、容差及数据池；token-KL 归因限定在 distance-KL 关闭条件下。
  `REGISTRATION.md` 已更新真实信息及作者核对事项，未操作投稿系统。
- 两行结论原本单独落到 p8，加上附录强制换页造成 13 页。精简 §4.5、局限性和结论后，
  当前为 **12 页，正文 p1–7，附录 p8–10，参考文献 p11–12**。保留所有实质边界；
  没有缩小字号、边距或表图，没有改动已验收图稿或放宽检查器。
- 使用现有 TeX Live、`latexmkrc` 和 `ccfa-workfiles/writing/www2027/build/` 构建，
  `latexmk` 退出 0。`verify_paper.py --build-dir ccfa-workfiles/writing/www2027/build`
  通过，锁定内容回归 **8/8** 通过。标题/摘要/作者/关键词、全部表文件、图文件、
  引用库及科学实现另与 `9b05681` 检查差异。
- 12 页均重新渲染并目视检查，图表、公式、引用、页边界无裁切或重叠。
  字体嵌入、匿名文本/元数据/链接、无 TODO、无未解析引用与无 overfull 均通过。
  保留可解释的 underfull、参考文献样式和 review 模式提示。
  当前 `acmart` 2.16 将历史 `balance=false` 报为未使用选项；不把该选项当作实际生效的依据。
- 正式 `paper-www2027/main.pdf` 与验收构建一致，SHA256：
  `06b821125d36e8f14c674fc0c051d0ce6575f80d9ee10dbd7cdd3df19372e5cf`。
  上一正式 PDF 可从 `9b05681` 恢复。临时渲染、缓存和日志不提交。
- `REVIEW-AND-FIXES.md` 是本轮统一报告。保留的核心风险包括摘要术语与宽泛表述、
  训练侧校准、纽约残差、Istanbul 来源缺口、全栈成本、下游效用及隐私证据不足。
  文稿/构建修正完成不等于这些科学风险消失；注册核对、作者事实确认和上传仍由作者完成。
- 随后按用户授权调整公式（10）：以 `58d66fe` 为已同步检查点，改用 `equation` + `aligned`
  分两行，编号在右侧垂直居中；数学内容、编号、字号和边距不变。重新编译退出 0，完整性检查
  及 8 项锁定测试通过，仍为 12 页。重新目视检查 p4/p5；与检查点逐页像素比较仅 p4 变化。
  调整前 PDF 可从 `58d66fe` 恢复；上列 SHA256 已更新为此次正式输出。

## 10. Figure 1（b）箭头调整（2026-10-21，当前交付）

- 接续用户要求调整（b）箭头的任务。基线 `637ac79` 已核对与 GitHub 一致；恢复的
  源码及图稿修改经检查后提交并推送为 `2a8a19b`，随后完成正式构建和交接更新。
- 将投影 token 列左移 6 pt，为右侧目标及训练输入留出独立通道；类别目标与训练输入的
  纵向虚线分别位于 x=133 和 x=143，训练输入在标题下方接入类别目标线。
  标签、字号、条件虚线含义和科学拓扑不变。像素比较确认图形变化仅在（b），其余面板一致。
- 修正上次 PowerShell 将输出目录选项拆开的编译调用，使用既有 LaTeX 编译脚本和
  既有 build 目录，TeX Live 构建退出 0。完整性检查通过；图形测试 5/5、锁定内容测试
  8/8 通过。12 页，正文止于 p7，附录从 p8；全部页面文本不变，仅 p3 图形像素变化。
  已目视检查 p3 的箭头、标签、图注和正文；成功重建版本与检查版本逐页渲染一致。
- 正式 `paper-www2027/main.pdf` 已与验收构建同步，SHA256：
  `b3e593cd8ade79471c8064d2c4d1f522c4decda8feffed72fdffce5d44f7aa0b`。
  此值取代前文旧交付指纹；此前版本保留在 Git 历史中。
  图形 PDF SHA256：`bb47aefbf954c0a9959994a5b90b9086407f1a3d28c7d22e39c47c51aaa4a846`；
  SVG SHA256：`e3d60d99de40ad739a45aa4ff3ea55ee7d24c6405d009c7ae3f01dd1243928ea`。
  可编辑源码及 SVG 沿用现有路径；本次采用直接修正方案，未生成 PPT。
  编译缓存、预览和日志留在既有工作目录，不提交；实验及封存产物未变。

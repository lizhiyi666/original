# WWW 2027 论文接续状态

更新：2026-10-18。原 T1–T9 任务说明保存在提交 `c828598`；本文记录执行结果。
后续先读本文及根目录 `AGENTS.md`，不要重复运行已完成实验。

## 1. 当前交付和未闭环项

T1–T8 已完成；T9 的邮箱、TODO、编译、匿名、字体、页数检查已完成，**官方 CCS 导出仍未取得**。
当前是供导师通读的完整稿，不是无条件的投稿就绪声明。

1. **摘要事实冲突保留待导师决定。** 摘要写 decoder “eliminates residual category–POI
   inconsistency”，六份正式输出却有 **0.0034%–0.0219%** mismatch。代码最终类别无支持时
   回退到任意合法 catalog POI。正文、框架图、局限已更正；用户选择暂不改锁定摘要。
   另须注意 “pairwise ordering coverage” 对应两个类别均出现，不表示排序正确，正文已区分。
2. **CCS 工具受阻。** <https://dl.acm.org/ccs> 持续显示安全验证。未绕过、未编造编号。
   `main.tex` 旧块已标注未核验，需作者给出官方导出后替换。
3. **不操作 OpenReview。** 注册、核对 portal 时间及上传由作者本人完成。

## 2. 锁定内容和作者

- 标题：`Training-Free Partial-Order Control for Discrete-Diffusion POI Check-in Generation`。
  导师要求不要加回 “distribution correction”。
- 标题、摘要、五关键词、作者顺序与 `c828598` 完全一致。
- 第一作者 Zhiyi Li：`25s151146@stu.hit.edu.cn`。
- 通讯作者 Kaiqi Zhao：`zhaokaiqi@hit.edu.cn`。
- 两人均为哈工大（深圳）计算机科学与技术学院；邮箱已填，ORCID 未提供，不猜。
- 保留 `anonymous,review`。源文件有身份信息，不要将源包作为匿名附件公开提交。

## 3. 截止时间：保留一小时差异

[官方 Research Track 页面](https://www2027.thewebconf.org/research-track-papers/)
于 2026-10-18 核实：摘要 10-18 AoE 日终、全文 10-25 AoE 日终，分别对应北京时间
**10-19 / 10-26 接近 20:00**，不是 21:00。旧交接写 OpenReview 为 13:00 UTC / 21:00 北京时间，
本任务未进入 OpenReview，不能确认 portal。按更早官网时间安排，作者自行核对差异。
长文正文最多 8 页、含参考文献及附录最多 12 页；第一页须说明 Web/track 相关性。

## 4. 环境和 Git 规则

- 仓库：`D:\桌面\轨迹\轨迹\轨迹生成\实验\original`。
- 分支：`codex/distribution-best-formal-v1`；远程 `lizhiyi666/original`。
- 修改前 `c828598`，T1 后 `fca940c`，矢量图/概要压缩后 `faccb4f`；最终正文/核验修订
  为后续提交，查 `git log`。每组修改前后仍需提交、推送并核对远程 SHA。
- 只暂存本任务文件，不 `git add .`；不提交数据、权重、缓存、日志、字节码、依赖目录。
- 直连曾超时；现有代理 `git -c http.proxy=http://127.0.0.1:7897 push origin
  HEAD:refs/heads/codex/distribution-best-formal-v1` 可用。相同 `-c` 配合 `ls-remote`
  核验分支。未修改全局配置，不强推、不 reset；前置推送失败就停止新修改。
- 作图 Python：`D:\Anaconda\envs\Marionette\python.exe`。
- 论文目录内构建：`latexmk -pdf -interaction=nonstopmode main.tex`。
- **当前真实退出 0**，11 页，结论 p7，附录 A 从 p8 开始；零错误、未解析引用、TODO 调用、
  overfull。第一页 Web 相关性、PDF 字体嵌入、文本/元数据/链接匿名检查及逐页视觉检查通过。
- 原稿 `amssymb` 造成 `\Bbbk` 重定义，已移除重复导入。**返回码 12 不应被称为无害**；
  只搜索 `^!` 会漏 file-line-style 错误。保留 `latexmkrc` 的中文路径 BibTeX 修复。
- 剩余非致命提示：review 的 reference block 隐藏、末页列平衡、underfull vbox、部分旧 BibTeX
  缺 publisher/address/pages。未进行 52 条文献全量审计。

## 5. T1–T9 执行记录

| 任务 | 结果与依据 |
| --- | --- |
| T1 | 主比较两表合成上下共享 caption 面板；stage-wise 移入附录 C；概要压缩。三张原表数据行/精度/SD/粗体保持不变 |
| T2 | `figure_build/generate_tradeoff.py` → `figures/tradeoff.pdf/.svg`；仅用已有 v2 均值，等量程线性轴，不移动测量坐标 |
| T3 | `figure_build/generate_method_main_v2.py` → `figures/framework.pdf/.svg`；框架+单步细节合图；标注 fallback；权重是目标输入而非第四串行操作 |
| T4 | 附录 D 指标定义来自 `evaluations/statistical_metrics.py`、`evaluations/ovr.py`、`tools/ablation_common.py`；补分母、过滤、catalog/token、历史直方图边界 |
| T5 | `generate_paper_evidence.py` 生成 `tables/calibration-grid.tex`，18 行 v2；SHA 校验 v3 selection source，顺序与失败门槛均一致 |
| T6 | manifest 的 NY 3160/2108、IST 7035/4914；`tools/make_opposite_split.py` 说明 NY 是偏好反转，非完整配置绝对不相交；IST 来源缺口保留 |
| T7 | `pcdg-geo-v1-tol10-20261010/speed/confirm-r2-b0.1-summary.json` → `tables/cost.tex`；64 条 batch，3.35%–3.98%，分母 projection Full |
| T8 | 相关工作对齐投影/解码/细化；核验五篇主要引用；移除无证据的“此前无人处理”和“其他均需重训”；CDD 采用会议版作者顺序，补部分页码 |
| T9 | 邮箱、TODO 空操作、锁定块/匿名/字体/页数/日志检查完成；官方 CCS 导出未完成 |

另修正文 token KL 方向：实现 `F.kl_div(log_p, log_q, log_target=True)` 是
**base→projected**。附录写明 token KL 除以完整 batch 大小，约束 penalty 按关系/样本求和。
未修改科学实现或实验结果。

图按最终正文尺寸检查，SVG 可编辑文字，固定 ID 盐值并去时间元数据；同环境可稳定再生。
临时渲染只放 `ccfa-workfiles/writing/www2027/build/`，不提交。
`figure_build/verify_paper.py` 使用 pypdf 与 Poppler，构建后运行，检查锁定块、原表数据行、
日志、页数、匿名/字体和受保护路径差异；可用应用自带文档 Python。通过不等于 CCS/科学审稿通过。

## 6. 科学边界与来源

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

作者/导师决定摘要非零残差措辞 → 作者提供官方 CCS 导出 → 修改相应源块 →
重建及完整性检查 → 检查受影响页面 → 提交/推送/核对 SHA → 作者自行提交。
不要顺带扩展实验、切换评估版本或改锁定标题。

核验的五篇主要来源：
[Marionette](https://github.com/UM-Data-Intelligence-Lab/Marionette)、
[CDD](https://proceedings.neurips.cc/paper_files/paper/2025/hash/124c1cc9f7750f512ed81cdbf2fd5bf5-Abstract-Conference.html)、
[ControlTraj](https://arxiv.org/html/2404.15380v1)、
[Grammar-constrained decoding](https://aclanthology.org/2023.emnlp-main.674/)、
[Control-DAG](https://aclanthology.org/2024.naacl-short.42/)。本次是受影响论证的定向核查，不是全稿录用判断。

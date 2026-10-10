# Claude Code 交接文档：PCDG 两城市轨迹实验

更新日期：2026-10-10（Asia/Shanghai）
仓库：D:/桌面/轨迹/轨迹/轨迹生成/实验/original
当前分支：codex/pcdg-geo-tol10
当前提交：b607a69d1754df3cfb0c639585b8b7d6bc335722
远程：origin/codex/pcdg-geo-tol10，当前已与本地同步。

## 可以直接交给 Claude Code 的开场指令

你正在接手一个已经完成多轮审计的 PCDG 轨迹生成项目。先阅读本文件、AGENTS.md、docs/handoff-20261010.md、docs/category-consistent-decoding-v2.md、docs/pcdg-geo-tol10.md 和 docs/two-city-distance-v2-run.md，再执行只读状态检查。不要根据历史 PID 自动 kill、重启或 resume 任何实验，不要覆盖旧结果，不要改变已有实验的参数、种子、batch、代码快照或封存指纹。

当前优先级是：

1. 先用 python -B tools/fetch_two_city_distance_v2.py --verify-only 复核本地 66/66 successor 交付。
2. 再按 docs/category-consistent-decoding-v2.md 运行独立的时间缓存审计：python -B tools/audit_category_decoding_cache.py；只允许新建 experiment_runs/ny-category-cache-audit-v1-20261010/，不得修改旧的 ny-category-sampled-v2-20261010。
3. 审计完成后再整理论文表格和实验结论；不要把诊断数据写成正式性能结论，也不要把 10% 几何验收规则描述成原 5% 协议成功。

若状态、哈希、输入指纹或服务器进程与本文不一致，先停下来保留现场并报告，不要通过放宽检查、缩小 batch、改 seed、改参数或重新采样来“修复”。

## 当前总览

| 项目 | 当前状态 | 结论/下一步 |
|---|---|---|
| NewYork OOD 基础训练与历史采样 | 已完成，结果封存 | 仅作固定 checkpoint 对照；不要重训 |
| pcdg-ablation-v1 | 已完成 | 历史 PCDG 消融结果保留 |
| two-city-v1 | 旧目录有过运行记录 | 不要操作；它不是当前 successor |
| two-city-distance-v2-batched-20261009 | 66/66，服务器审计通过，本地交付审计通过 | 先执行 --verify-only，再用于论文汇总 |
| pcdg-geo-v1-20261010 | stopped at quality gate | 原 5% 协议未通过，全部文件保留 |
| pcdg-geo-v1-tol10-20261010 | complete，18/18，本地交付审计通过 | 10% 是探索性修订，不是 5% 协议成功 |
| ny-category-sampled-v2-20261010 | failed（原长度配对审计失败） | 必须保留 failed；不可重采样或改状态 |
| ny-category-cache-audit-v1-20261010 | 工具和测试已提交，审计可执行 | 只读独立缓存审计，不能覆盖源失败状态 |

## Git 与修改前提

- 每组逻辑修改前后都要检查分支、远程、工作区，并精确提交本任务文件后推送 GitHub。不要提交数据集 PKL、模型权重、实验缓存、临时传输目录、依赖目录、日志或密钥。
- 本次交接文档写入前，类别缓存审计相关改动已提交并推送：b607a69 Add independent category cache audit。
- 相关前序提交：9169bae（实际类别耦合）、13ccf2c（128 条诊断）、ab5e048（诊断下载交付）、654829e（记录真实结果和保留 failed）、a73d1f2/d01103d（10% 几何修订与部署记录）。
- 目前工作区在交接文档修改前是干净的；新增本文后应只暂存本文，保留任何后来出现的其他任务改动。

## 最近半月时间线

### 2026-09-30 至 2026-10-03：NewYork OOD 与旧基线

- 固化 W&B 认证，建立受保护的 NewYork OOD 训练/采样流程。
- 修复 singleton sampling，并增加 emptyfix-v1：保留空轨迹、屏蔽 padding 约束、保持全局索引和 seed，不删除或填充空记录。
- 完成 train-only projection calibration、正式 OOD 重采样、空输入验证、baseline suite 和固定预算 PCDG ablation。
- 旧结果、旧 manifest、checkpoint 和历史失败 fixture 都是只读锚点。

### 2026-10-04 至 2026-10-08：两城市结果、评估 v2、距离 KL

- 两城市旧实验完成了 60/60 unique results，其中 39 份新结果、21 份复用；base_models_retrained=false，empty-fixture-v1 recovery audit 通过，原始文件保持不变。
- 评估升级为 v2：Category 为 24 个小时 JSD 的等权平均；Interval 被 CategoryTransition 替换；所有结果必须带 evaluation_version=2。
- 新增 train-aligned cumulative-route Distance KL：训练集参考、Haversine 路程、软直方图、独立随机流、legacy|batched 后端和距离诊断。
- batched 后端只允许在新 run/output tag 中显式启用；默认旧路径仍是 legacy。已有旧结果不能用新代码恢复续跑。
- 创建两城市 distance-v2 独立调度器，固定两城市、三 seed、66 份结果、batch 64、FP32、TF32 关闭、10x50 投影预算；严禁重训、调参或复用旧生成轨迹。

### 2026-10-08 至 2026-10-09：distance-v2 successor

- successor 使用独立目录和新 manifest；明确记录实现切换：35 份 inherited legacy results + 31 份 new batched results。
- 服务器审计确认：66/66、231726 条轨迹、12 个时间缓存、输入和旧父目录未变、paired spatial RNG、distance switches、全数据集指标重算、全部哈希有效。
- 本地目录 experiment_runs/two-city-distance-v2-batched-20261009/ 已下载，local-delivery-audit.json 为 passed，verified_files=1453。
- 报告明确说明混合实现版本，不能把跨后端效率差异解释成纯方法/消融差异。均值和标准差是三个固定采样 seed 的均值与样本标准差，不是多次训练的不确定性。

### 2026-10-10：几何精修与类别一致解码

- 实现 same_category_v1 末端几何精修：仅在同一实际类别的合法 POI 中选择，冻结类别、时间、长度、条件和偏序，独立噪声流，默认关闭。
- 原 5% 几何运行 pcdg-geo-v1-20261010 在质量门停止；不能报告为失败后的补跑成功，也不能改写。
- 查看训练侧确认结果后，另开 pcdg-geo-v1-tol10-20261010，把 DailyLoc/G-RANK 相对退化上限改为 10%。选择配置为 distance_w=1.0, radius_w=2.0, prior_w=0.1, steps=200；四个在线速度比值约 1.033–1.040，均不超过 10%；18 份正式结果和本地 2349 文件交付审计通过。
- NewYork 数据集构建修复：保留完整数据集级 SVD/category 元数据；训练 3160、测试 2108，36 类方向全部翻转。NewYork OOD 是激进偏序设置，不能与 Istanbul 的绝对数值直接等读。
- 方案 A 类别一致 POI 解码先后经历两版：旧 argmax 版本被真实 51%/49% 反例否定；当前 sampled-category-poi-v2 先采样实际类别，再复用同一 Gumbel 噪声约束 POI。开关默认关闭，不改旧结果。
- 128 条真实 NewYork 对照已生成，但原“导出长度相同”检查失败：开启组保留了 3 个旧导出器会删除的词表外 POI 事件，开启组事件数 877、关闭组 874。状态必须继续为 failed。
- 诊断观察：实际类别-POI 不一致率 0.175057 -> 0，严格 OVR 0.270085 -> 0.022472，但原端点筛选 Distance 0.622042 -> 0.508524 不能单独解释为改善；无端点筛选 Distance 0.246411 -> 0.271266、Radius 0.290133 -> 0.344307，因此不是正式质量结论。
- 本提交新增 tools/category_cache_audit.py、tools/audit_category_decoding_cache.py 和 tests/test_category_cache_audit.py。10 项测试已通过；审计通过只表示缓存对齐和完整性通过，scientific_quality_passed 必须保持空值，源状态仍是 failed。

## 关键结果和科学解释

### 两城市 distance-v2 successor

- 结果文件：experiment_runs/two-city-distance-v2-batched-20261009/report.md、registry.json、audit.json、local-delivery-audit.json。
- Istanbul 的 PCDG-Geo/Full 分布指标明显改善 Distance/Radius；NewYork 也改善 Distance/Radius，但 G-RANK 仍差，类别-POI mismatch 仍是瓶颈。
- 报告中的 PCDG-Geo 是几何精修后的 Full；纯 PCDG 与 Full 的命名在方法表和消融表中有明确限制。消融 Full 使用 PCDG-Geo，而其他消融行未加几何精修，因此 Distance/Radius/DailyLoc/G-RANK 的差异不能纯归因于某一个偏序组件。
- 必须披露：Istanbul 历史基础训练 batch=512，NewYork=64；Istanbul 历史训练数据没有完整可回溯指纹；CFG scale=1；标准差只反映固定 checkpoint 的 sampling variation。

### PCDG-Geo tol10

- 本地状态：experiment_runs/pcdg-geo-v1-tol10-20261010/status.json 为 complete，quality_goals_met=true；local-delivery-audit.json 为 passed，18 个 formal results，2349 个文件。
- 配置：same_category_v1、200 steps、8 paths、top-k 32、temperature 1、learning rate 0.05、noise interval 20、prior 0.1、radius 2.0。
- 这是在训练侧工程校准/确认后放宽其他 JSD 门槛的探索性验收；不能称为未接触测试基准的独立泛化验证。

### NewYork 根因判断

1. 基础空间模型地板较高：训练序列 3160、POI 3751，历史 batch=64；JointGen 的 Distance/Radius 已明显高于 Istanbul。
2. 类别 token 到实际 POI 的落地脱节：token 顺序约 10% 违反，映射到 POI 后严格违反约 38.5%，category_poi_mismatch_rate 约 0.21。
3. PCDG 以约束满足换分布保真；几何精修能修回 Distance/Radius，但不能修类别-POI mismatch，G-RANK 仍可能变差。

## 接手后的操作顺序

### 1. 先做本地只读复核

~~~powershell
Set-Location 'D:/桌面/轨迹/轨迹/轨迹生成/实验/original'
git status --short --branch
git fetch origin
git rev-parse HEAD
git rev-parse origin/codex/pcdg-geo-tol10
& 'D:/Anaconda/envs/Marionette/python.exe' -B tools/fetch_two_city_distance_v2.py --run-id two-city-distance-v2-batched-20261009 --verify-only
& 'D:/Anaconda/envs/Marionette/python.exe' -B tools/fetch_geometry_study.py --run-id pcdg-geo-v1-tol10-20261010 --verify-only
~~~

若 two-city-distance-v2-batched-20261009/status.json 仍显示 server-complete-awaiting-local-verification，以 --verify-only 的结果为准，但不要手改 status；需要修改状态时应使用仓库已有交付工具或另开审计提交。

### 2. 执行独立类别缓存审计

~~~powershell
& 'D:/Anaconda/envs/Marionette/python.exe' -B tools/audit_category_decoding_cache.py
& 'D:/Anaconda/envs/Marionette/python.exe' -B tools/audit_category_decoding_cache.py --verify-only
~~~

审计工具会通过 SSH alias pcdg 读取原失败运行的只读快照，并在本地 CPU 验证：原 manifest、开启组 payload、关闭组有序子集、开启组完整缓存事件、六类条件、合法 POI/GPS、两批随机流、10x500 预算、父封存文件和代码/输入哈希。它拒绝路径逃逸、符号链接、冲突覆盖和源状态篡改。服务器解释器必须是 /root/anaconda3/envs/pcdg-exp/bin/python，远端 plain python 不可用。

预期是新审计目录 experiment_runs/ny-category-cache-audit-v1-20261010/ 通过，而源目录 experiment_runs/ny-category-sampled-v2-20261010/ 仍记录 failed 和 Generated length or record fields changed。即使独立审计通过，也不能重采样、修补旧 payload 或改写科学质量结论。

### 3. 更新论文和统一报告

- 论文工作目录：paper-www2027/。paper-www2027/README.md 说明当前是 WWW 2027 working draft，实验数字仍需从 distance-v2 刷新。
- 需要把表格切换到 evaluation v2，保留 CategoryTransition，补齐 No Distance KL 行，填充 experiments.tex 中的 TODO，补 Fig.1/2/3。
- 论文必须说明：PCDG/PCDG-Geo 的约束与分布保真 trade-off、固定 checkpoint 三 seed 均值±样本标准差、CFG scale=1、两城市历史 batch 差异、Istanbul 历史数据 provenance 限制、10% 几何验收是探索性修订。
- 不把 128 条类别解码诊断当成三 seed 正式实验，不把 failure 变成 success，不用测试指标选择新参数。

## 不能做的事

- 不重复启动、自动重启、重训或重新采样已完成结果。
- 不修改运行中的实验协议、参数、种子、batch、精度、checkpoint、封存 manifest、delivery inventory 或旧输出。
- 不覆盖已有不同内容的文件；下载和合并必须使用隔离 staging、SHA-256 比对和 no-overwrite 逻辑。
- 不把服务器完成当成本地交付完成；必须等待传输退出、检查 staging、逐文件校验并确认 local-delivery-audit.json。
- 不把历史 two-city-v1、旧 pcdg-geo-v1、旧 ny-category-sampled-v2 当成可以 resume 的当前任务。
- 不使用当前代码恢复没有对应实现版本/manifest 的旧实验；旧实验只能在其封存代码下只读审计。

## 常用环境与入口

- Windows Python：D:/Anaconda/envs/Marionette/python.exe。
- 服务器 SSH：pcdg。
- 服务器 Python：/root/anaconda3/envs/pcdg-exp/bin/python。
- 全量本机回归：& 'D:/Anaconda/envs/Marionette/python.exe' -B -m unittest discover -s tests -v。Windows 预期有 1 个 Linux 文件锁测试跳过；其余测试必须通过。
- 当前类别缓存审计回归：& 'D:/Anaconda/envs/Marionette/python.exe' -B -m unittest tests.test_category_cache_audit -v，已验证 10/10 通过。
- 两城市文档：docs/two-city-distance-v2-run.md、docs/distance-kl-v2.md、docs/distance-batched-successor-run.md。
- 几何文档：docs/pcdg-geo-v1.md、docs/pcdg-geo-tol10.md。
- 类别解码文档：docs/category-consistent-decoding-v2.md。

## 论文时间窗口

paper-www2027/README.md 记录的硬截止日期为：abstract registration 2026-10-18，full paper 2026-10-25（正文不超过 8 页，总计不超过 12 页）。当前优先补齐可追溯实验数字和限制说明，再做排版压缩与图稿更新。


# G-RANK v3 跨平台复算交付（2026-10-13）

## 结果

技术修复、已有结果复算和新版本本地交付均完成；原实验质量目标仍未达标。

- Windows 和 Linux 分别对同一批 **56 份封存 payload、64 组指标**完成 CPU 复算，进程均正常退出。
- 64/64 组跨平台对照通过，top-1000 POI 成员及计数指纹完全一致；全部标量指标的最大绝对差为 **2.220446049250313e-16**，低于原定 `1e-12` 容差。
- 两端的六项分布指标、五项约束、门槛判定和排序统一使用 v3；除 G-RANK、totalJSD、版本号外，复算指标与 v2 封存值均在 `1e-12` 内一致。
- 原来源的 1,188 个文件在运行前后通过哈希校验；本地收尾再次确认来源指纹一致。Linux 结果传输实际退出后，7 个文件逐一与服务器 SHA-256 对照通过。
- 没有重训、重采样、改种子/batch、追加搜索、补跑确认或启动正式测试；新增正式轨迹为 **0**。
- v2 默认接口及历史定义保持不变，旧结果与上次交付失败记录保留。此次是单独的 v3 交付，不把旧 v2 复算失败改成通过。

## 修复与验证

规则预先固定为：整数访问次数降序；并列时整数 POI ID 升序；分别截取 top-1000 后，按 POI ID 升序对齐并集计算历史 JSD。保留 GPS 长度筛选、JSD epsilon、其他指标公式及严格门槛。不是按测试结果挑选排序方式。

- 新模块：`evaluations/grank_v3.py`，评价版本 3，修订 `grank-count-desc-poi-id-asc-v3`。
- 新命令：`tools/reevaluate_distribution_v3.py`。旧脚本不修改，新脚本仅接受独立新输出目录；已有目录（包括半成品）拒绝覆盖或自动续跑。
- Windows 完整回归：**280 项测试，1 项平台相关跳过，其余通过**。
- 新增定向测试：Windows 和 Linux 均 **14/14 通过**。覆盖并列截断、身份对齐、超大整数 ID、输入顺序独立、旧指标/RNG 保持、来源/receipt/索引核对、拒绝覆盖和对照差异检测。
- 真实数据验证：两端各 64 组完整复算；独立跨平台对照通过。并非仅靠合成测试判断完成。

## 质量结论没有改变

v3 筛选前两名仍为 `r0.5-c0-p1` 和 `r0.5-c0-p0.1`，没有更换候选，也没有补跑候选。两组在 New York 和 Istanbul 的 CategoryTransition、Category、G-RANK 均仍高于同版 JointGen 上限；Distance、Radius、DailyLoc 及五项约束门槛保持通过。确认通过者仍为 0，正式测试没有启动。

这是训练侧工程校准的回顾性复核，基础模型见过训练数据，不是新增独立泛化验证。结论只限本次候选范围；不把三种子均值的改善解释为统计显著。

### 确认阶段 G-RANK 对照

每城三个固定种子 `135398/135399/135400`，每种子 512 条。表中 v2 为原服务器封存均值，v3 为新版本均值 ± 样本标准差；低者更好。v3 候选必须与 v3 JointGen 比较，不能跨版本混用阈值。

| 城市 | 方法/候选 | v2 封存均值 | v3 均值 ± 样本标准差 |
|---|---|---:|---:|
| New York | JointGen | 0.214152 | 0.225153 ± 0.012406 |
| New York | PCDG | 0.230578 | 0.239113 ± 0.009284 |
| New York | r0.5-c0-p1 | 0.231029 | 0.236071 ± 0.005223 |
| New York | r0.5-c0-p0.1 | 0.246257 | 0.251778 ± 0.005769 |
| Istanbul | JointGen | 0.223529 | 0.214832 ± 0.009283 |
| Istanbul | PCDG | 0.246212 | 0.234643 ± 0.003185 |
| Istanbul | r0.5-c0-p1 | 0.245601 | 0.236077 ± 0.004767 |
| Istanbul | r0.5-c0-p0.1 | 0.250866 | 0.239562 ± 0.001793 |

全部指标、各城/各配置均值与标准差、逐种子配对差值、旧/新排名见新目录的 `summary.json`；每份来源的旧/新指标及 top-1000 指纹见 `records.json`。原始轨迹不复制改写，仍由来源文件 SHA-256 绑定。

## 交付路径

工作区：`D:\桌面\轨迹\轨迹\轨迹生成\实验\original`。

- Windows 新结果：`experiment_runs/ny-ist-distribution-balanced-v1-eval-v3-20261013`。
- Linux 新结果的本地副本：`experiment_runs/ny-ist-distribution-balanced-v1-eval-v3-20261013-linux`。
- 跨平台交付回执：Windows 新结果目录内的 `cross-platform-audit.json`。
- 服务器新结果：`/root/experiments/pcdg/ny-ist-distribution-balanced-v1-eval-v3-20261013`。
- 服务器独立代码目录：`/root/experiments/pcdg/grank-evaluation-v3-code-20261013`。
- 原 v2 本地来源：`experiment_runs/ny-ist-distribution-balanced-v1-20261012`，保持原样。

每个平台自己的 `audit.json` 只证明该平台复算和完整性，仍标记 `cross_platform_verified=false`；另附的 `cross-platform-audit.json` 才证明 Windows/Linux 对照通过。没有改写任何已封存单平台审计。

## GitHub 与指纹

- 分支：`codex/grank-evaluation-v3`。
- 修改前检查点：`3a0b06d6d2a47cae2d005626078faf16670adb5c`，已推送并核对远程。
- 两端实际运行的 v3 实现提交：`62a2db17ac198a29865b9ddd60236a555307da23`，部署前已推送并核对远程。
- 原实验实现提交仍为 `95977661aaa0ccc4756a9d88149bc380a1e0d5ea`。本报告之后的文档提交不改变实际运行代码提交。
- 部署代码包两端 SHA-256：`93c3df666614650464541699351b88a8bdaaced210b9a4abf1c1c79f57c7c7e8`。
- 原来源 1,188 文件的集合指纹：`f4b342d1ce98a249cd5d00234e0c082abde2495d3370b62af3c230564a2d7792`。
- Linux 单平台 audit SHA-256：`c956c643ca7e8201f056e28f3e4ae2f56d79898e08cb9574476d6312457d93e2`。

| Windows 交付文件 | SHA-256 |
|---|---|
| audit.json | `1082d4f7a770756873abb6cb02660a14f0250df4c9ec9ea32db7f1e97bec7fad` |
| manifest.json | `cbb2bbff4c8a77276f83bec61595b0028de9569b54eec3cdfec8c220ac4492e2` |
| records.json | `7a9d32ce995370f5486e510bb1adff1d0f4a8feeb4d58c2e835b9cb91ae44e93` |
| summary.json | `fff65f56c580128775f9930f258c315b7427be1432798a5454fd31bb6814b246` |
| cross-platform-audit.json | `b66df449568113b74ffb59ee7eb0dfca5d2782eef498a6fd60da7c9e2305428e` |

数据、结果和传输暂存仍处于 Git 忽略目录；仅提交实现、测试、说明和这份报告。原静默跟进保持暂停，本次没有新建监控或实验。

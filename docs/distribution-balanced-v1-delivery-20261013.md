# 分布校正 v1：校准停止与本地交付阻塞

核验日期：2026-10-13（Asia/Shanghai）。运行标识：`ny-ist-distribution-balanced-v1-20261012`。

## 结论与边界

- 服务器有限校准技术完成：`state=stopped`、`phase=sealed`、`audit.state=passed`，原因是 `no-confirmed-candidate-meets-all-gates`，不是运行崩溃。
- 18 组筛选完成，前两组完成两城三种子确认；两组均未同时通过全部门槛。正式测试没有启动，正式结果为 **0**，不是 6/6。
- 两个确认候选在两城的 Distance、Radius、DailyLoc 均值满足 JointGen 上限，五项约束均值均不差于原始 PCDG；CategoryTransition、Category、G-RANK 均未满足 JointGen 上限。
- 压缩包传输已退出，服务器/本地压缩包 SHA-256 一致，本地 1,188 个交付文件通过哈希验证。
- **本地交付验收尚未通过**：CPU 复算在首个基线的 G-RANK 处失败，下载程序退出码为 1，未产生 `local-delivery-audit.json`。不能把文件完整性通过描述为指标复现通过。
- 原自动跟进 `automation` 已暂停，原聊天保留。没有重启实验、追加搜索、修改种子/batch/评价器、重训、重采或覆盖封存文件。

这是训练侧工程校准，基础模型见过这些训练数据，不是独立泛化验证。门槛按每城三种子未舍入均值比较，容差仅 `1e-12`。下表只是显示舍入，不改变判定，也不表示统计显著。

## 确认配置与实际指标

两个候选共用 `rho=0.5`、`lambda=0`，分别为 `beta=1`（`r0.5-c0-p1`）和 `beta=0.1`（`r0.5-c0-p0.1`）。种子为 `135398/135399/135400`，每城每种子 512 条确认记录。没有通过者，也没有冻结正式测试配置。

以下为服务器封存指标的均值 ± 样本标准差（n=3）；不是已通过本地全部复算的指标。

### New York：六项分布指标，越低越好

| 指标 | JointGen 上限 | beta=1 | beta=0.1 | 两候选均值判定 |
|---|---:|---:|---:|---|
| Distance | 0.573494 ± 0.076556 | 0.127557 ± 0.071592 | 0.126078 ± 0.039717 | 通过 |
| Radius | 0.147158 ± 0.014006 | 0.072002 ± 0.004452 | 0.076261 ± 0.000743 | 通过 |
| CategoryTransition | 0.020832 ± 0.000494 | 0.025788 ± 0.001350 | 0.025788 ± 0.001350 | 未通过 |
| DailyLoc | 0.055990 ± 0.004999 | 0.043500 ± 0.005150 | 0.046338 ± 0.003780 | 通过 |
| Category | 0.032921 ± 0.000726 | 0.033628 ± 0.000748 | 0.033628 ± 0.000748 | 未通过 |
| G-RANK | 0.214152 ± 0.012289 | 0.231029 ± 0.004239 | 0.246257 ± 0.004576 | 未通过 |

### Istanbul：六项分布指标，越低越好

| 指标 | JointGen 上限 | beta=1 | beta=0.1 | 两候选均值判定 |
|---|---:|---:|---:|---|
| Distance | 0.343586 ± 0.015223 | 0.139216 ± 0.014302 | 0.142832 ± 0.013144 | 通过 |
| Radius | 0.060395 ± 0.008044 | 0.052160 ± 0.007895 | 0.048937 ± 0.006679 | 通过 |
| CategoryTransition | 0.026995 ± 0.002740 | 0.039489 ± 0.000295 | 0.039489 ± 0.000295 | 未通过 |
| DailyLoc | 0.034712 ± 0.003284 | 0.024989 ± 0.005166 | 0.029925 ± 0.008494 | 通过 |
| Category | 0.061335 ± 0.005179 | 0.080615 ± 0.005020 | 0.080615 ± 0.005020 | 未通过 |
| G-RANK | 0.223529 ± 0.008747 | 0.245601 ± 0.006355 | 0.250866 ± 0.001890 | 未通过 |

### 五项约束指标

两个 beta 候选的约束指标相同，均通过三种子均值门槛；前三项越低越好，后两项越高越好。这里的 PCDG 是原始 `full`/Off，不是 A 或 A+Geo。

| 指标 | New York PCDG | New York 候选 | Istanbul PCDG | Istanbul 候选 |
|---|---:|---:|---:|---:|
| strict_ovr | 0.404136 ± 0.024603 | 0.176915 ± 0.015474 | 0.186273 ± 0.012936 | 0.017321 ± 0.007209 |
| ovr_skip | 0.151880 ± 0.015356 | 0.078497 ± 0.008716 | 0.057735 ± 0.004803 | 0.005384 ± 0.002219 |
| Unsat | 0.469901 ± 0.024196 | 0.241539 ± 0.015368 | 0.217469 ± 0.006427 | 0.032680 ± 0.005222 |
| pair_coverage | 0.631327 ± 0.028160 | 0.816113 ± 0.015050 | 0.842543 ± 0.004019 | 0.974747 ± 0.003602 |
| category_coverage | 0.790471 ± 0.017516 | 0.901646 ± 0.008521 | 0.921783 ± 0.005252 | 0.989066 ± 0.002225 |

### 未通过指标的逐种子差值

差值为候选减同种子 JointGen；正数表示更差。CategoryTransition 与 Category 对两个 beta 相同。

| 城市 | 种子 | CategoryTransition 差值 | Category 差值 | G-RANK 差值 beta=1 | G-RANK 差值 beta=0.1 |
|---|---:|---:|---:|---:|---:|
| New York | 135398 | +0.005895 | -0.000154 | +0.013503 | +0.031896 |
| New York | 135399 | +0.004070 | +0.001637 | +0.033730 | +0.043140 |
| New York | 135400 | +0.004901 | +0.000637 | +0.003395 | +0.021279 |
| Istanbul | 135398 | +0.015503 | +0.020097 | +0.019760 | +0.018996 |
| Istanbul | 135399 | +0.011497 | +0.023080 | +0.024690 | +0.036955 |
| Istanbul | 135400 | +0.010482 | +0.014664 | +0.021766 | +0.026058 |

全部 11 项指标的逐种子差值、样本标准差及原始结果保存在交付目录 `summary.json` 的 `confirm.results.<配置>.paired_deltas`、`aggregate` 和 `cities` 中。筛选阶段全部 18 组结果及排序保存在 `screen-selection.json`；确认失败清单保存在 `confirm-selection.json`。这些文件均已通过哈希校验。

无首尾类别筛选的 Distance 以及原指标入选数量保留在每份结果的 `diagnostics` 中；原评价版本和历史 Radius 公式未改动。未通过仅说明本次预注册候选范围未达标，不证明目标不可能。

## CPU 复算阻塞：G-RANK 并列截断的平台差异

首个失败对象：`baselines/screen/Istanbul_PO1_OOD/seed-135398.pkl` 的 `full`。

| 项目 | 值 |
|---|---|
| 服务器封存 G-RANK | 0.2418136520485205 |
| 服务器只读 CPU 重算 | 0.2418136520485205 |
| Windows 本地 CPU 重算 | 0.24207485317424154 |
| 本地减封存值 | 0.0002612011257210334 |
| 本地只读诊断采用服务器 top-1000 成员集合 | 0.2418136520485205，差值 0.0 |

评价器 `evaluations/statistical_metrics.py` 的 `grank_jsd` 使用 `np.argsort(-prob)`，没有指定并列时的稳定排序或 POI ID 次序，然后截取 top-1000。两端 NumPy 均为 1.26.4，评价器代码 SHA-256 相同，POI 频次表的 SHA-256 也相同，但默认排序在此次 Windows/Linux 运行中选择了不同的并列成员：

| 分布 | 事件数 | 不同 POI 数 | 截断频次 | 该频次并列数 | 选入并列数 | 两端各自独有的入选 POI 数 |
|---|---:|---:|---:|---:|---:|---:|
| generated | 3106 | 1138 | 1 | 616 | 478 | 110 |
| reference | 3171 | 1260 | 1 | 716 | 456 | 250 |

本地使用服务器入选集合的独立诊断精确复现了封存值，确认差异来自这次 top-1000 成员选择，不是传输损坏。未改写函数或产物，也未把该诊断替代正式复算。只定位了首个失败对象，**没有宣称其余全部指标均完成复算**。

环境：服务器 Python 3.10.21、NumPy 1.26.4、PyTorch 2.0.0+cu117（Linux）；本地 Marionette Python 3.10.19、NumPy 1.26.4、PyTorch 2.0.0（Windows）。本报告不把具体底层 SIMD/排序实现差异当成已进一步验证的原因。

修复并列规则会影响评价口径及已有排序/指标，须先获用户确认，并使用独立提交和输出目录保留旧结果；不能擅自改封存值、放宽 `1e-12` 容差或进入正式测试。

## 审计、位置与指纹

- 实验源代码 GitHub 分支：`codex/distribution-balanced-v1`。
- 实验实际运行提交：`95977661aaa0ccc4756a9d88149bc380a1e0d5ea`；本次修改报告之前，已再次核对本地 HEAD 与 GitHub 远程分支均为该值。本次只增加这份报告，实验仍绑定原提交。
- 服务器工作区：`/root/experiments/pcdg/ny-ist-distribution-balanced-v1-20261012`。
- 服务器交付目录：`/root/experiments/pcdg/ny-ist-distribution-balanced-v1-20261012/experiment_runs/ny-ist-distribution-balanced-v1-20261012`。
- 本地交付目录：`D:\桌面\轨迹\轨迹\轨迹生成\实验\original\experiment_runs\ny-ist-distribution-balanced-v1-20261012`。
- 本地保留压缩包：`tmp/distribution-balance-fetch-wa7qaohk/delivery.tar.gz`，49,899,845 字节。
- 服务器保留压缩包：`/tmp/distribution-balance-fetch-blvw0r5e/delivery.tar.gz`，49,899,845 字节。
- 已确认退出的控制器/worker PID：`24042 / 24054 / 24055 / 24229 / 24230`；未发现本任务仍在运行的实验进程。
- 服务器审计：`originals_unchanged=true`、`no_training=true`、`no_temporal_resampling=true`、`quality_goals_met=false`。
- 本地只读哈希复核：`state=passed`、`verified_files=1188`、`cpu_recomputed=false`。其中审计清单包含 1,186 个文件，另有 `audit.json` 和 `status.json`；传输包哈希覆盖全部成员。

| 对象 | SHA-256 |
|---|---|
| 两端 delivery.tar.gz | `a3758b4b2e6df51adbae6ea6a474fb40669b04f8b46da6b4f139e8dc9f3e27f2` |
| summary.json | `a945ed09f58f4a99b9bcc70a957cfca70d9496176a7ed2f83c64a6b196632aa3` |
| audit.json | `5930bb8745e22e0617e9301d331bad9557aecab090d4b90caf7f4af4a29f8b09` |
| manifest.json | `6003c8412bec77e4dd9f4ae80cb12a24dd61f5537a44be5365fc28931b95bb61` |
| status.json | `903cae49a3bf7a6480803164f5a7cd9a7e8b1a1609279b20e24e9c17ec968eb0` |
| 两端 evaluations/statistical_metrics.py | `fb4a46e2263dac968165d57944131e98f2740030048beceeb4f5ca5183a2be70` |

数据、缓存、压缩包及封存原始报告留在忽略目录，不纳入 Git。后续报告提交不得被误认为重新运行实验的源代码提交。

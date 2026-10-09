# distance-v2 批量后端接续运行

2026-10-09 用户明确授权：先暂停服务器任务，切换新代码后继续。

## 路径与不可变边界

- 旧代码：`/root/experiments/pcdg/two-city-distance-v2`；旧结果：该目录的 `experiment_runs/two-city-distance-v2`。
- 新代码：`/root/experiments/pcdg/two-city-distance-v2-batched-20261009`。
- 接续结果：新代码目录的 `experiment_runs/two-city-distance-v2-batched-20261009`。
- 新入口：`tools/launch-distance-batched-successor.sh`；控制器：`tools/resume_distance_batched.py`。
- SSH 别名 `pcdg`；解释器 `/root/anaconda3/envs/pcdg-exp/bin/python`。
- 旧目录不覆盖、不改 manifest、不回写进度或报告。新目录内 `inherited/legacy` 保存旧目录除 W&B 活动文件和锁文件外的逐字节副本；`parent-snapshot.json` 记录全部 SHA-256。

暂停前已完成 35/66：NewYork 33 份，Istanbul 2 份；已封存时间缓存 8 份。原控制器 PID917、worker PID8334/8335 仅用于迁移时核验，不可在后续未验证进程身份时盲目使用。

旧 Istanbul Full 两个分片暂停时各完成 768/2457、704/2457，只有内存结果和 progress.json，没有可恢复的中间 payload/checkpoint。接续程序重新计算这两个未完成分片，不重算已有 35 份完整结果，不重新生成已完成时间缓存。

## 接续语义

目标仍为两城市、三种子、66 个唯一结果和 231726 条记录。保持 batch=64、FP32、TF32 关闭、每次投影 10×50、8 路径/top-32/32 分箱以及 80% 显存上限，不训练、不调参、不改种子。

这是**明确标注版本切换的接续实验**，不是全程同一后端。新 registry 的每条结果包含后端、实现版本、原始 manifest 和来源；旧结果保持 `legacy` 标识与原始字节，后续结果标记 `batched`。报告和表格明确警示混合实现，跨版本效率不能直接解释为方法/消融差异。

启动前先做本机回归、服务器回归与 GTX/RTX 正确性验证；新控制器会重新审计全部继承结果的文件、分片、指标和随机流，并在两城市训练缓存上验证真实模型的正常/空输入、Gumbel/确定性、双 GPU Full 重复性。任何不通过都不得启动正式接续采样。

## 启动、恢复、监控

首次预检（只生成验证结果，不开始剩余正式采样）：

```bash
bash tools/launch-distance-batched-successor.sh --preflight-only
```

预检通过后正式接续：

```bash
bash tools/launch-distance-batched-successor.sh --resume
```

每次使用新控制器日志文件名。启动前核验旧控制器/worker 已停止，且新目录没有正在运行的控制器或 worker；禁止双重启动。`pipeline.lock` 仍使用非阻塞独占锁。不要用旧入口恢复原版任务。

只读关注新 `status.json`、`registry.json`、`inheritance-audit.json`、两城市 `batched-preflight/*/receipt.json`、worker 的 `progress.json`、真实进程命令和 GPU 占用。

健康运行保持安静。显式可恢复的异常退出，仅在无残留 worker 且源码、输入、snapshot、manifest 指纹一致时使用新入口 `--resume`；仅跟踪同步失败时使用 `--resume --sync-only`。OOM、非有限值、错误指纹、科学协议异常不得绕过，也不得缩小 batch 或自动调整参数。

## 完成与本地交付

新 `status.json` 为 `server-complete`、`audit.json` 为 passed，且 66/66、231726、原目录/快照完整性与全部版本来源通过后，再执行：

```powershell
& 'D:/Anaconda/envs/Marionette/python.exe' -B tools/fetch_two_city_distance_v2.py --run-id two-city-distance-v2-batched-20261009
& 'D:/Anaconda/envs/Marionette/python.exe' -B tools/fetch_two_city_distance_v2.py --run-id two-city-distance-v2-batched-20261009 --verify-only
```

接续交付包包含继承副本，因此不依赖旧实验达到 server-complete。原始 metadata 中的服务器路径不重写，审计通过 canonical mapping 指向经过 SHA-256 校验的副本。仅本地 `local-delivery-audit.json` 验证通过后，才能宣布完整任务完成。

本文件是运行约定；实际部署哈希、PID、验证结果及启动状态由本次迁移完成后的执行记录补充。

## 跨平台部署注意

从 Windows 生成 Linux 代码包时使用 `git -c core.autocrlf=false archive`，并检查 `.sh` 不含 CR 字节。仓库 `.gitattributes` 另将 `*.sh` 固定为 LF。部署包只包含跟踪的源码、配置、脚本、测试和必要说明，不包含数据集、权重、outputs、依赖或密钥。

本次第一份代码包受 `core.autocrlf=true` 影响，脚本在 Python 启动前报 `pipefail` 选项错误；失败代码目录已保留为 `two-city-distance-v2-batched-20261009-deployment-crlf`，失败日志 01/02 未删除。修复包与原包的 113 个文件在统一换行后逐字节一致，仅修复跨平台换行；原始实验代码和结果未修改。

Linux 修复包 SHA-256：`5cbb34c6cb873971a8a43153035ec5bcb7281fa0126b0ea0b2a9e8d21ea94356`；部署代码提交：`785f7c76f083cd72b81572b8471978f3fdfc2c19`。真实模型预检日志为 `/root/experiments/pcdg/two-city-distance-v2-batched-20261009-preflight-03.log`。

## 已验证的切换结果

- 本机回归 136 项通过、1 项 Linux 专用测试跳过；服务器回归 137 项全部通过。RTX 3090 上 batch=64 的扩展正确性测试 10 项全部通过。
- RTX 3090 合成基准（2 次预热、5 次测量、同卡同输入）完整投影含距离几何初始化：legacy 21.233 秒、batched 1.761 秒，耗时减少 91.7%。该数字不是完整真实模型采样加速倍数。
- 两城市真实模型预检均通过：正常/空轨迹、确定性/Gumbel、相同 Full 输入跨 GPU 输出一致，均保持每批 64 条、5000 次优化更新。正常 Full 预检空间采样为 NewYork 约 27.86 秒/批、Istanbul 约 25.47 秒/批。
- 867 个旧目录文件及其副本 SHA-256 已复核；继承结果 35 份，已完成时间缓存 8 份。旧 source/manifest/结果未改写。
- 正式接续控制器 PID9424，初始 worker PID9509/9510；仅作本次启动证据，后续必须核验真实命令。日志：`/root/experiments/pcdg/two-city-distance-v2-batched-20261009-controller-01.log`。
- 新 worker 已产生真实批次进度后，才结束旧的暂停进程 PID917/8334/8335，并再次验证全部旧文件与副本。结束时使用针对已核验停止进程的 SIGKILL，避免旧程序的退出/失败处理器回写封存文件；没有终止新 worker。
- 接续 manifest SHA-256：`9de7d944aa27529bf7695f9c589ef4ed898a2685cb8a437cfe40031cdfa70be8`；原始 manifest SHA-256：`615dc569dd2d255c80ab80ff03b70337694392edc445a4809beb0af83a50294f`。

完整计时样本、预检产物指纹、进程退出证据和一次正式进度快照见 [执行记录](benchmarks/distance-batched-successor-20261009.json)。这些是迁移成功的证据，**不是 66/66 已完成的证明**；实时状态以服务器新目录为准。

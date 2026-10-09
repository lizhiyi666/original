# two-city-distance-v2 执行与续跑

原始任务：聊天 `01a11abe-8ccd-7543-8bfd-2ee4f6b97994` 中明确批准的两城市完整重跑。
本次接续：`01a11c15-9253-7f83-8f38-476fcb1018d0`。

## 固定协议

- NewYork_PO1_OOD 2108 条、Istanbul_PO1_OOD 4914 条；种子 135398、135399、135400。
- 每城市每种子 11 份结果，总计 66 份、231726 条轨迹记录。
- 全新 12 份时间缓存；同城市同种子的方法共享时间输入；所有空间结果重新生成。
- 复用原始基础模型及 PO-CFG，不训练、不调参、不运行下游任务。
- batch=64，双卡 FP32，禁用 TF32，每次投影 10×50、每轨迹批次 10 次投影。
- 距离 KL 权重 1，8 路径、top-32、32 分箱、距离温度 1，约束 Gumbel 温度 3。
- 单卡显存上限 80%；非有限值、预算/随机流/指纹异常应停机，不能自动改参数。
- 完成条件：66/66、服务器审计通过、全部本地交付文件哈希通过。Full 排名不是验收条件。

## 部署快照（2026-10-08）

- SSH：`pcdg`；Python：`/root/anaconda3/envs/pcdg-exp/bin/python`。
- 新代码目录：`/root/experiments/pcdg/two-city-distance-v2`。
- 结果目录：`/root/experiments/pcdg/two-city-distance-v2/experiment_runs/two-city-distance-v2`。
- 控制器初始 PID：917。此 PID 仅为启动记录；每次先核验真实进程及命令，不能盲目使用旧 PID。
- 日志：`/root/experiments/pcdg/two-city-distance-v2-controller-20261008-01.log`。
- 启动脚本：新代码目录中的 `tools/launch-two-city-distance-v2.sh`。
- 部署包 SHA256：`614c1dfddeba7003cf9593b717cde9fcf5c1e23c5da699b9b7c73a2f216414e1`，传输前后相同。
- 服务器 113 项测试全部通过；本机同期 112 项通过，1 项 Linux 锁测试跳过。
- 四个检查点严格加载通过；NewYork 的时间缓存及双卡 No Projection 预检已通过。
- NewYork 的正常/合成空轨迹 Full 预检均通过；正常批次 64 条、268.47 秒、10 次投影、5000 次优化更新，峰值 PyTorch reserved 597688320 字节。此时尚未完成跨 GPU 同输入 Full 重复性终检。
- 本地下载工具 4 项安全测试通过：不完整实验拒绝交付、路径逃逸拒绝、归档符号链接拒绝、归档及落地文件哈希复核。
- 已创建当前聊天每 30 分钟检查一次的 heartbeat，自动化 ID 为 `distance-v2`。仅通知阶段里程碑、失败、需要用户处理或最终完成；最终完成后结束跟进。
- 正式实验还未完成；本文件不是完成凭证。实时读取 `status.json` 和各分片 `progress.json`。

原数据、配置和四个权重保留原路径只读引用。不能覆盖 `two-city-v1` 或其他旧实验目录。
部署后新增加的本地下载工具及其测试未写入封存的服务器代码，避免改变当前运行指纹。

## 检查与恢复

只读检查 `status.json`、控制器日志、运行进程、GPU 内存、分片 `status.json` / `progress.json`。
预检结束后查看 `runtime-estimate.json`；预检期间不能把 0/66 视为停滞。

若进程异常退出，先确认没有存活 worker、输入/代码指纹未变，再使用原启动脚本加 `--resume`。
指纹不符、OOM、非有限值或科学协议失败时保留证据，不能绕过检查或自动调整实验参数。
若仅 W&B 同步未完成，使用 `--resume --sync-only`；禁止因同步失败重新采样。
每次恢复使用新的外部控制器日志文件名，保留旧日志。

## 本地交付

服务器 `status.json` 为 `server-complete` 且有封存的 `delivery-files.json` 后执行：

```powershell
& 'D:/Anaconda/envs/Marionette/python.exe' -B tools/fetch_two_city_distance_v2.py
& 'D:/Anaconda/envs/Marionette/python.exe' -B tools/fetch_two_city_distance_v2.py --verify-only
```

该工具在服务器核验封存文件与代码指纹，打包后通过 SSH 下载，再在本地暂存目录逐文件验真。
交付到 `experiment_runs/two-city-distance-v2` 时拒绝覆盖不同内容；不下载原数据或权重。
暂存目录保留在 `tmp/distance-v2-delivery-*`，最终凭证为 `local-delivery-audit.json`。
`--verify-only` 不写入文件。下载工具不会启动采样或训练。

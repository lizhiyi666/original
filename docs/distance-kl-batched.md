# 距离 KL 批量实现与验证报告

日期：2026-10-09。默认后端仍为 `legacy`；优化版必须显式指定 `batched`。

## 结论与适用范围

本机 CPU/CUDA 正确性检查通过。在固定 batch=64 的合成输入上，完整 500 步投影（含距离候选/几何初始化）的中位耗时：GTX 1650 从 50.44 秒降至 7.76 秒，减少 **84.6%**；CPU 从 26.54 秒降至 11.94 秒，减少 **55.0%**。这两项合成基准达到至少降低 50% 的目标。

这些结果**不是完整模型采样、真实两城市数据回放或 RTX 3090 的性能结论**。测量使用 64 个映射 POI、72 维词表，明显小于真实模型词表。未加载模型权重，未启动新一轮实验，未修改运行中的服务器代码、参数、种子、结果、指纹或既有监控。

原始五次测量、环境、代码 SHA-256 和验证摘要见 [机器可读报告](benchmarks/distance-kl-20261009.json)。

## 实现与不变项

- `DistanceObjective` 保留为参考实现；`BatchedDistanceObjective` 复用同一候选选择和实际 POI 距离，在初始化时整理变长轨迹、有效位置和有效边掩码。内层损失无逐轨迹 Python 循环。
- 有效轨迹等权，空轨迹和单点不进入距离直方图；填充位置不贡献距离或梯度。使用候选间的 haversine 距离，不使用平均坐标近似。
- 每个外层迭代仍按原轨迹顺序，以相同形状、次数调用 `torch.rand`。随后打包噪声，供内层复用；确定性模式不消耗距离随机流。
- 保留 FP32、禁用 TF32 的测试配置；8 条路径、top-32、32 分箱、固定 `10×50` 更新，无早停。未引入混合精度、编译器、自定义 CUDA 或新依赖。
- 仅移除每个内层步用于日志的 `float(distance_loss.detach())`。最终更新后的距离诊断、首次距离梯度检查，以及每步损失/梯度/更新后 logits 的非有限值安全检查均保留。
- 距离参考文件指纹算法、随机种子派生和模型 checkpoint/state_dict 格式未改变。

## 接口、独立输出与恢复限制

`distance_backend=legacy|batched` 已贯通模型配置、`DiffusionTransformer`、`ConstraintProjection`、普通采样、合并工具、实验 worker 和两城市控制器。

| 后端 | 实现版本 |
|---|---|
| `legacy`（默认） | `distance-kl-legacy-v1` |
| `batched`（显式开启） | `distance-kl-batched-v1` |

采样元数据、worker 结果、实验 manifest 和启用距离项时的投影诊断包含后端与实现版本。旧调用仍使用参考算法，距离权重为零时输出与随机流不变。

普通采样启用 `--distance_backend batched` 时必须提供独立 `--output_tag`，分片与合并结果写入：

```text
data/<dataset>/distance-backends/distance-kl-batched-v1/<output_tag>/
```

在原有采样命令上添加上述两个参数，并保持原定采样协议。例如需使用本次验收预算时，同时显式设置 `--batch_size 64 --projection_outer_iters 10 --projection_inner_iters 50 --distance_paths 8 --distance_topk 32 --distance_bins 32`；不要依赖普通 CLI 历史默认迭代数。合并同样添加 `--distance_backend batched --output_tag <同一标签>`。

两城市控制器参数为 `--distance-backend batched --run-id <新的独立名称>`。默认旧 run-id 禁止用于批量后端。恢复时要求实现版本、manifest、代码、输入一致；不同后端/版本，以及没有版本标识的旧实验 manifest，均拒绝恢复。历史无版本 trace 仍可只读审计，**不能用新代码续跑当前旧实验**；旧实验应继续使用其封存代码。

## 正确性验证

- 原有回归在修改前为 117 项（116 通过、1 跳过）；最终为 **129 项（128 通过、1 跳过）**。跳过项为 Windows 不适用的 Linux `flock` 测试。
- CPU 与 GTX 1650 CUDA 的损失/梯度对照均使用 `rtol=1e-5, atol=1e-6`，未放宽容差。覆盖变长、空轨迹、单点、重复 POI、部分有效行、少于 32 个候选与 top-32 截断、Gumbel/确定性模式及多个种子。
- 验证真实候选路径前向值、填充屏蔽、无效位置零梯度、每外层一次噪声刷新、首次梯度探针和最终诊断。对损失、梯度、更新后 logits 注入非有限值，均按预期失败。
- 完整投影固定 500 次更新；空间、投影约束和距离三个随机流的最终状态与参考实现一致，全局 RNG 不变，无效行逐位不变，距离项关闭时输出逐位相同。批量后端在同一设备/同一输入下重复运行逐位可复现。
- CPU 全预算对照形状为 `[8, 8, 13]`；另在 CUDA 使用与性能基准相同的 `[64, 32, 72]` 全预算输入，**没有缩小 batch**。CUDA 扩展套件 10 项全部通过。

全预算参考/批量输出的最大绝对 logits 差异：

| 输入与设备 | Gumbel | 确定性 |
|---|---:|---:|
| CPU，小型正确性输入 | 1.073e-6 | 1.907e-6 |
| CUDA，batch=64 基准输入 | 2.682e-7 | 1.431e-6 |

上述输入的 argmax 与相同空间随机流下采样 token 变化数均为 0；这只是这些固定输入的观测，**不保证任意真实轨迹或多轮扩散的最终离散输出完全相同**。

## 性能测量

环境：Windows，AMD Ryzen 7 5800H，GTX 1650 4 GiB，PyTorch 2.0.0，单 CPU 计算线程，FP32/TF32 关闭。固定种子 135398，输入 `[64,32,72]`，48 条有效距离轨迹，8 路径、top-32、32 分箱，固定 500 步。每项预热 2 次，测量 5 次，报告中位数；GPU 计时前后同步。

| 设备与范围 | legacy | batched | 耗时减少 |
|---|---:|---:|---:|
| GPU，距离损失前向+反向 | 111.37 ms | 7.64 ms | 93.1% |
| GPU，完整投影，几何已准备 | 48.294 s | 6.959 s | 85.6% |
| GPU，完整投影，含距离几何初始化 | 50.443 s | 7.758 s | 84.6% |
| CPU，距离损失前向+反向 | 30.95 ms | 16.15 ms | 47.8% |
| CPU，完整投影，几何已准备 | 27.085 s | 11.540 s | 57.4% |
| CPU，完整投影，含距离几何初始化 | 26.544 s | 11.938 s | 55.0% |

GPU 参照是第一组提交时保存的**优化前源码**；CPU 比较的是两个后端共用去除日志同步后的投影器。两者不能用来单独估算那一行日志同步的贡献。参考类本身未改写。

“含初始化”包括候选选择与距离几何的构建，不包括合成输入、训练参考、projector 对象或完整模型的创建；“几何已准备”在计时前完成候选/几何构建。每次测量使用新投影器和相同随机种子。

GPU 完整投影的峰值 PyTorch allocated 从约 25.34 MiB 增至 32.05 MiB；reserved 从 30 MiB 增至 **56 MiB**，约占 4 GiB 的 1.37%，未超过 80% 阈值。该值不是载入真实模型后的总显存需求，也不包含其他桌面进程。CPU 显存记为不适用。未发生 batch=64 OOM；真实大词表/长轨迹的显存容量尚未验证。

共享桌面环境存在耗时波动，JSON 保留全部样本，包括参考测量中的慢样本，未挑选最好值。性能测试未与其他 GPU 基准重叠，未在正式计时中使用 profiler；独立 `--profile` 模式不产生正式性能指标。

## 复现命令

以下命令只使用合成输入，不加载 checkpoint、不采样实验数据；输出文件若已存在会拒绝覆盖，复跑请使用新文件名。在仓库根目录运行：

```powershell
& 'D:/Anaconda/envs/Marionette/python.exe' -B -m unittest discover -s tests -v

$env:DISTANCE_TEST_DEVICE = 'cuda'
$env:DISTANCE_TEST_BATCH = '64'
$env:DISTANCE_TEST_MAX_POIS = '16'
$env:DISTANCE_TEST_CANDIDATES = '64'
& 'D:/Anaconda/envs/Marionette/python.exe' -B -m unittest tests.test_distance_batched -v
Remove-Item Env:DISTANCE_TEST_DEVICE,Env:DISTANCE_TEST_BATCH,Env:DISTANCE_TEST_MAX_POIS,Env:DISTANCE_TEST_CANDIDATES

& 'D:/Anaconda/envs/Marionette/python.exe' -B tools/benchmark_distance_kl.py --device cuda --backend legacy --output tmp/recheck-legacy-cuda.json
& 'D:/Anaconda/envs/Marionette/python.exe' -B tools/benchmark_distance_kl.py --device cuda --backend batched --output tmp/recheck-batched-cuda.json
& 'D:/Anaconda/envs/Marionette/python.exe' -B tools/benchmark_distance_kl.py --device cpu --backend legacy --output tmp/recheck-legacy-cpu.json
& 'D:/Anaconda/envs/Marionette/python.exe' -B tools/benchmark_distance_kl.py --device cpu --backend batched --output tmp/recheck-batched-cpu.json

# 单独剖析，不与上面的正式计时同时运行
& 'D:/Anaconda/envs/Marionette/python.exe' -B tools/benchmark_distance_kl.py --device cuda --backend batched --profile --output tmp/recheck-profile.json
```

当前版本的 `legacy` 命令使用去除日志同步后的投影器。精确重现历史 GPU 基线需在独立 checkout 使用 `a2c8d76`，不能回退或覆盖运行中的实验目录。`--deterministic` 用于额外的确定性性能测量；本报告表格使用 Gumbel 模式。

## 服务器与交付边界

本轮只读检查时，原服务器实验为 29/66，两张 RTX 3090 正在使用。远端 `distance_kl.py` 与 `constraint_projection.py` 的 SHA-256 均与优化前一致（完整值见 JSON）。**没有部署、启动、重启或中断服务器任务**，现有监控未调整。

RTX 3090 验证和真实模型/真实输入性能仍待原实验结束且设备空闲后，在独立代码/输出目录进行。当前不宣称服务器加速倍数，不将本次基准结果混入旧实验的科学结果。

提交分组：修改前检查点 `f7e1f0c`；参考测试/基准 `a2c8d76`；批量实现/接口 `6141b8f`；诊断同步优化、最终测试及本报告由最后一组提交交付。每组提交后均核对 GitHub 分支哈希。

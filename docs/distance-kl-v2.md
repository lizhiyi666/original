# 距离 KL 与统计评估 v2

## 启用方式与兼容性

普通 `sample.py` 默认采样修订为 `distance-kl-v2`。配合
`--use_constraint_projection` 时，距离 KL 默认权重为 1，默认开启 Gumbel 松弛。
未开启投影的 native/baseline 不使用距离项。示例（需要替换实际 run ID）：

```powershell
& 'D:\Anaconda\envs\Marionette\python.exe' -B sample.py --run_id YOUR_RUN_ID --use_constraint_projection --sampling_revision distance-kl-v2 --projection_distance_kl_weight 1 --output_tag distance-v2-smoke --batch_size 4 --max_samples 4
```

`--projection_distance_kl_weight 0` 关闭距离项。底层 `ConstraintProjection`
和模型构造器的缺省值仍为 0；旧 checkpoint 不增加 state_dict 参数或 buffer。
显式历史 `--sampling_revision` 缺省关闭距离项；历史实验驱动也保留旧采样设置。
`tools/sample_perfcal.py` 未指定修订时仍使用 `perfcal-v1`。
所有输出仍拒绝覆盖不同输入的已存在生成文件。

可配置参数：`distance_paths=8`、`distance_topk=32`、`distance_bins=32`、
`distance_temperature=1.0`。`--no_gumbel_softmax` 为确定性消融：距离项使用
每条轨迹的期望路程，不应报告成多路径 Monte Carlo 分布。

## 距离目标

总目标保留原 token KL 和增广拉格朗日约束，并加入独立加权的
`KL(H_train || H_projected)`。训练参考只读取当前数据集的 `*_train.pkl`，
以 `poi_gps` 映射计算所有多点训练轨迹的累计 Haversine 路程（公里）。
这不使用测试轨迹，也不使用现有 Distance 评估的测试配对起止类别筛选。

参考在进程内按训练内容哈希及分箱配置缓存，包含 POI token 映射、坐标、
软分箱中心和目标概率；不持久化进模型。参考指纹进入采样元数据。

每次投影固定每个 POI 位置的 top-k 候选，噪声在外层刷新、内层复用。
前向以 straight-through Gumbel-Softmax 选择真实 POI，累计真实候选间距离；
反向使用候选概率的双线性近似。不是对平均坐标计算距离。
各有效轨迹等权，汇总候选路径后计算一个批次级 KL，而非逐轨迹匹配总体分布。

软直方图在 `log1p(km)` 上取 32 个等距高斯中心，范围为
`[0, 1.25*log1p(max(训练最大路程, 1公里))]`，核宽为中心间距。
每条路径的分箱权重归一化，再平均、加 `1e-8`、重新归一化。
训练与投影两侧使用同一变换和核，不裁剪超出训练范围的路程。

距离仅作用于原投影有效行中具有至少两个 POI 的样本；无约束、空轨迹和单点
不获得距离梯度。启用且存在有效距离行时执行配置的完整投影预算。
距离使用独立随机流，不消耗原空间采样或约束 Gumbel 随机流。

`last_projection_stats` 提供距离 KL、有效行数、初始距离项 POI 梯度范数、
候选概率覆盖率、估计器种类及参考指纹；普通采样分片及合并输出保存
`distance_projection_diagnostics`，消融 worker 保存逐调用诊断。
有限 top-k 和有限路径都是近似；低候选覆盖率、较小 batch 和分箱尾部饱和
可能降低估计质量。默认权重不是已调优的最佳权重，不保证所有评估指标改善。

## 评估 v2

所有统计结果包含数值型 `evaluation_version=2` 元数据；它不参与 `totalJSD`
或指标均值计算。六项为 Distance、Radius、CategoryTransition、DailyLoc、
Category、G-RANK。其余四项及 Distance 的起止类别筛选保持原实现。

- Category：在 `[h,h+1)` 汇总所有轨迹的类别事件，逐小时计算 JSD，对 24 小时
  等权平均。单点也参与。双侧空小时记 0，单侧空小时记 `ln(2)`；全天双侧
  无事件时为未定义值。可通过 `diagnostics` 参数取得各小时分数和两侧事件数。
- CategoryTransition：统计轨迹内部相邻有向类别对，保留自转移与重复事件。
  按起始类别归一化后逐行计算 JSD，再对两侧出边类别的并集等权平均。
  单侧无出边的行记 `ln(2)`；两侧无出边均排除；完全无转移时为未定义值。
- 对外仍使用 POI 映射得到的类别，保留原始生成类别 token 用于单独的不一致诊断。
- Interval 仅从评估与报表中删除，时序模型的时间间隔数据保持不变。
- `totalJSD` 只求六项中已定义项之和；JSON 中未定义值为 `null`。

命令行评估默认写新的 `*_Evaluation_v2_results.txt`，另存 JSON 诊断；已有路径
不会追加或覆盖。baseline、消融和统一表格使用同一函数，表格拒绝无 v2 标记
或仍含 Interval 的结果。统一表格另写 `evaluation-schema.json`。
历史综合报告构建器也检查此标记；旧结果需要另存重算，不能只改列名。

## 消融协议与验证

旧 `pcdg-ablation-v1` 的投影设置不变。新协议显式使用：

```text
tools/run_pcdg_ablation.py --projection-revision pcdg-distance-v2 ...原有必要参数...
```

新版 Full 开启两项 KL；`no_kl` 同时关闭两项；新增 `no_distance_kl` 只关闭距离项；
`no_gumbel` 使用确定性期望路程。新协议默认使用独立的运行目录。

回归与不写结果文件的训练集小样本检查：

```powershell
& 'D:\Anaconda\envs\Marionette\python.exe' -B -m unittest discover -s tests -v
& 'D:\Anaconda\envs\Marionette\python.exe' -B tools/validate_distance_v2.py --dataset NewYork_PO1
& 'D:\Anaconda\envs\Marionette\python.exe' -B tools/validate_distance_v2.py --dataset Istanbul_PO1
```

小样本检查使用训练 POI 构造测试 logits，验证距离项、掩码、梯度与随机流，
不是完整模型生成质量实验。本次不重算历史结果、不运行完整重采样。

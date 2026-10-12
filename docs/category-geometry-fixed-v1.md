# 方案A + 同类别几何细化（固定配置迁移，`ny-ist-category-geo-fixed-v1-20261012`）

在已冻结的方案A（类别一致解码）完整输出上，追加 `same_category_v1` 同类别几何细化，降低
Distance/Radius JSD，同时保持方案A的类别与约束收益。**不重新训练、不重新生成时间、不重跑 A 或
Off、不搜索参数。** 复用已冻结几何配置，生成两城×三种子共 6 份新结果；不执行旧几何实验的参数搜索、
速度筛选或 18 份消融流程，也不继承其 10% 验收门槛。

## 不可变协议

- 类别 token、POI 实际类别序列、时间、长度、顺序及全部条件保持不变；只更换 `checkins` 与对应
  catalog `gps`。实际类别以 `poi_category[POI]` 为准。
- 直接复用 `geometry_projection` 的参考加载、模型评分、候选构造、细化与不变量检查，以及现有评估、
  原子发布和哈希工具。几何核心与旧冻结实现一致；不一致则停止，不静默换实现。
- 从方案A的 `batch_traces` 恢复原批次、`indices` 与 `global_start`，按原 rank/批次边界处理，不重分
  批、不缩小 batch。是否执行过基础投影从源 trace 读取（`any(optimizer_steps>0)`），不伪造。
- 模型仅以冻结、eval、FP32、TF32 关闭做 `t=0` 一次未截断评分；只优化候选 logits，模型参数不变。
  本路径封禁 TPP 与扩散采样入口（`tpp_model.sample` / `sample_fast`），误调用立即报错。
- 几何随机流沿用 `seed/global_start` 派生，独立于原空间/投影/距离 RNG；细化前后全局 RNG 不变，
  模型张量状态不变。空轨迹、单点、单候选及未执行基础投影的批次沿用现有跳过规则。

冻结配置 `configuration_sha256`：

```text
eded18df16922bdc618f3328119bc9101022d61ced9373d72919188f201ed3a2
```

## 来源（只读）

| 用途 | 服务器目录 |
|---|---|
| 已完成的方案A输出 | `/root/experiments/pcdg/ny-ist-category-formal-v1c-20261011/experiment_runs/ny-ist-category-formal-v1c-20261011` |
| 冻结几何配置与训练参考划分 | `/root/experiments/pcdg/pcdg-geo-v1-tol10-20261010/experiment_runs/pcdg-geo-v1-tol10-20261010` |

检查点、模型配置、数据与时间缓存按上述来源的 manifest/job/registry 引用读取，不按目录名推断。
参考索引直接继承旧几何 `splits[city].reference`，不重新划分；重建的参考指纹必须与旧 manifest 相同。

## 命令

本地回归（CPU，无需检查点）：

```bash
/d/Anaconda/envs/Marionette/python.exe -m unittest tests.test_category_geometry -v
```

服务器部署目录 `/root/experiments/pcdg/ny-ist-category-geo-fixed-v1-20261012`，在其中运行完整回归与
几何 CUDA 测试后，于 tmux 内双卡执行（两 worker 各占一卡、各处理一座城市，城市内按
`135398/135399/135400` 顺序）：

```bash
python -u -B tools/run_category_geometry.py \
  --category-source <A来源目录> \
  --geometry-source <旧几何来源目录> \
  --run-id ny-ist-category-geo-fixed-v1-20261012 --gpus 0,1
```

```bash
# 身份完全一致时的显式恢复：复用匹配的完成批次，拒绝冲突或半成品
python -u -B tools/run_category_geometry.py --category-source <A> --geometry-source <G> \
  --run-id ny-ist-category-geo-fixed-v1-20261012 --gpus 0,1 --resume

# 只读核验已有输出目录（不生成结果、不加载模型、不改源文件）
python -B tools/run_category_geometry.py --verify-only experiment_runs/ny-ist-category-geo-fixed-v1-20261012
```

进程独占锁防止重复启动；新目录若已存在，先核验状态，不覆盖旧部署。

## 合并、评估与质量门槛

两城各按原全局索引合并，每种子覆盖完整测试集：NewYork 2108、Istanbul 4914，共 6 份结果、21066 条
轨迹记录（保留空轨迹，不与事件数混淆）。只在完成细化后评估，报告 `Off / A / A+Geo`；Off、A 复用
原始输出重算指标（非重采）。测试真值 GPS、真实 POI 序列与测试指标不进入候选构造、细化目标或最终选择。

逐事件精确保持冻结字段、实际类别与长度。以下指标在 A+Geo 下与 A 逐数值一致（容差 `1e-12`）：

```text
strict_ovr、ovr_skip、pair_coverage、category_coverage、Unsat、Category、
CategoryTransition、category_poi_mismatch_rate、empty_count、empty_rate
```

质量门槛（两城各自三种子均值，未舍入）：Distance 与 Radius 均低于当前 A；DailyLoc ≤ A×1.05；
G-RANK ≤ A×1.05。任一必需指标缺失或非有限视为未通过。保留 evaluation v2 的非标准 Radius 公式
`sqrt(mean(distance_to_arithmetic_centroid))`，不改标准 RMS 后与旧表混比。未达标仍交付全部有效结果并
标记 `quality_goals_met=false`，不放宽门槛。

## 下载与交付

```bash
python -B tools/fetch_category_geometry.py --run-id ny-ist-category-geo-fixed-v1-20261012
python -B tools/fetch_category_geometry.py --run-id ny-ist-category-geo-fixed-v1-20261012 --verify-only
```

下载器只接受完整 6 份结果，按常规文件打包、校验传输哈希、经硬化安全路径解包，再复核封存 audit
（6 份、逐文件哈希、质量标记），不保留旧 18 份验收逻辑。

技术完成 ≠ 本地交付完成 ≠ 质量成功：三者分别判定。本轮只是“旧固定几何配置迁移到方案A”，不是未接触
测试集的全新泛化验证；离线细化耗时不等于在线总耗时，旧 Geo 的速度比例不作为 A+Geo 的速度结论。

# PCDG-Geo v1：类别保持几何投影

## 协议

新增 `same_category_v1` 末端投影，默认 `geometry_refinement=off`。训练权重和旧输出不改写；只允许替换为同一实际类别的合法 POI。时间、长度、顺序、条件和原始类别 token 保持不变。实际类别以 `poi_category[POI]` 为准。

每批从生成记录构造模型输入，冻结模型单次 `t=0` 未截断评分，不接受测试参考 GPS。每位置候选由原 POI、模型 top15、距离原轨迹中心最近16个同类别 POI 合并，去重补齐至32个；同分按 POI ID 排序。参考概率是原 POI 点质量与候选内模型概率各50%。

优化训练侧 Distance/Radius 软直方图的 JSD，加平均候选概率 KL 保真项。Radius 保留评估 v2 的 `sqrt(mean(distance_to_arithmetic_centroid))` 公式，不替换指标定义。使用中心化弧度坐标避免 FP32 小距离消减误差；真实 POI 前向值、重复点零次梯度与数值安全检查都有测试。

8条 ST 路径、温度1、Adam学习率0.05，每20步刷新独立几何噪声。最终在原始整批和8个候选整批中，按训练侧几何目标和硬选择保真代价择优，不用测试真值挑结果。空轨迹、单点、单候选位置以及没有真正执行基础 PCDG 投影的整批输入跳过。原空间/投影/距离随机流不变。

## 先速度、后质量

- 两城市各从训练集用划分种子20261010选1024条：前512筛选、后512确认，其余构建几何参考。基础模型训练见过这些数据，属于工程校准，不能称为独立验证。
- 基础 Full/JointGen 每池、每种子只生成一次，并分批原子保存。恢复时核对身份、manifest 和文件哈希，拒绝不同内容覆盖。
- 在筛选池的普通和最长有效64条批次上，先测当前 batched Full，再测50/100/200步几何投影；2次预热、5次计时，GPU同步。所有场景中位总空间采样时间比值≤1.10才合格，选择最大的合格固定步数。50步失败则停止，不做质量搜索。
- 在线计时包括完整基础空间采样、解码、未缓存的新模型评分、候选构建、优化和离散筛选；不包括时序生成、模型加载和网络。静态参考缓存允许复用；冷启动单独记录。性能剖析为单独运行，不混入正式计时。
- 质量搜索恰好12组：半径权重 `{0.5,1,2,4}` × 保真权重 `{0.001,0.01,0.1}`，距离权重1。步数不因质量结果改变。
- 筛选池用135398，前3名在确认池用135398/135399/135400。两城市共享同一配置，按四项目标比值的最差值、平均值、新增处理耗时、配置字典序排序。确认要求几何不劣于 Full，再次进行完整在线10%速度验证；无合格配置则停止。
- 约束容忍最多1个百分点；但类别保持实现要求相关指标数值实际上不变，变化视为错误。DailyLoc/G-RANK相对退化最多5%。Istanbul Distance/Radius及NewYork Radius以JointGen为目标，NewYork Distance以当前Full为目标。

## 正式实验

只在配置冻结并通过所有门槛后读取测试侧结果。直接细化已封存的6份 Full，不重新运行基础采样；生成新版Full、Distance-only、Radius-only共18份新结果。消融不重新调参。

报告三种子均值/样本标准差、约束指标、其他统计、原有Distance端点筛选的两侧有效样本数、无筛选Distance、几何分位数和POI替换率。冻结类别及长度使端点筛选集合不变。保留旧Full的legacy/batched来源，不将历史混合后端成本与新在线总成本混为一谈；离线细化耗时不是完整采样耗时。

## 运行

本机回归：

```powershell
& 'D:/Anaconda/envs/Marionette/python.exe' -B -m unittest discover -s tests -v
$env:GEOMETRY_TEST_DEVICE='cuda'
& 'D:/Anaconda/envs/Marionette/python.exe' -B -m unittest tests.test_geometry_projection -v
Remove-Item Env:GEOMETRY_TEST_DEVICE
```

部署到独立目录 `/root/experiments/pcdg/pcdg-geo-v1-20261010`，不得覆盖源实验。用新日志后台启动 `bash tools/launch-pcdg-geo-v1.sh`；可先追加 `--stage speed` 仅执行速度门槛。已有本研究产物时必须追加 `--resume`，且代码/配置/父封存身份必须完全一致。

入口 `tools/run_geometry_study.py` 使用独占锁。结果在 `experiment_runs/pcdg-geo-v1-20261010`：

- `manifest.json`：输入、源码、划分、固定协议和参考指纹。
- `calibration/`：训练侧不可变基础输出及时间缓存；`speed/`：完整计时与独立阶段剖析。
- `performance-qualification.json`：固定步数选择；`screening.json`、`confirmation.json`、`recommendation.json`：质量选择证据。
- `formal/`、`registry.json`、`results-summary.json`：正式18份输出及未舍入结果。
- `report.md`、`audit.json`、`status.json`：完成或停止原因。审计通过不等于质量目标达到，须同时查看 `quality_goals_met`。

普通采样可用 `sample.py --geometry_refinement same_category_v1 --geometry_fit_indices <训练参考索引JSON> --geometry_steps <已合格步数> --geometry_radius_weight <已选值> --geometry_prior_weight <已选值>`，同时显式启用PCDG并提供新的output_tag。配置构造器支持相同字段。此入口不替代两城市的性能资格验证。合并需指定相同 `--geometry_refinement`，新目录与旧distance-v2隔离。

FP32、TF32关闭、名义batch64、原10×50投影预算和80%显存上限始终不变；非有限值、OOM或指纹错误会失败，不自动改参或覆盖旧产物。

## 完成或门槛停止后的本地交付

确认控制器已退出，且状态为 `complete` 或明确的门槛 `stopped` 后执行：

```powershell
& 'D:/Anaconda/envs/Marionette/python.exe' -B tools/fetch_geometry_study.py --run-id pcdg-geo-v1-20261010
& 'D:/Anaconda/envs/Marionette/python.exe' -B tools/fetch_geometry_study.py --run-id pcdg-geo-v1-20261010 --verify-only
```

交付工具在独立归档中生成清单，不向服务器实验目录添加文件。逐文件哈希通过后才生成本地 `local-delivery-audit.json`；门槛停止不被报告成18份正式结果完成。Distance-only/Radius-only仅移除末端对应惩罚，基础Full始终保持原样。

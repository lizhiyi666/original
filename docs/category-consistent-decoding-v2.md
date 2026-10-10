# 方案A修正：依赖实际采样类别的POI解码

旧版本先按类别 logits 的 argmax 筛选 POI，随后又独立随机采样类别；类别概率51%/49%的64条合成输入可出现32条实际不一致，而原测试比较手写类别计划而不是采样 token，仍会通过。该反例不是城市真实数据的性能结论。

## 修正语义

- 版本 `sampled-category-poi-v2`；开关仍默认关闭。`sample.py` 同时接受 `--category_consistent_decoding` 与 `--category-consistent-decoding`。
- 每一步仍只生成一份与原路径形状、顺序相同的 Gumbel 噪声。先取得实际类别采样，再按该类别约束对应POI，复用同一份噪声完成POI选择；不额外消耗全局/空间/投影/距离随机流。
- 使用真正的负无穷屏蔽非法候选，避免有限值 `-70` 与低分合法POI并列导致泄漏。类别位置和非POI位置不被改写。
- 扩散中间步骤允许POI mask状态；类别仍为mask或无可用POI时暂不约束。最终步骤只允许同类别真实POI；无合法类别/POI时明确失败，不静默退回不一致结果。
- 关闭时继续调用原采样签名，输出与随机流保持一致；训练侧无batch的随机采样路径不启用联合解码。checkpoint格式不变。
- 启用时将实现版本写入普通采样元数据和worker结果；旧argmax结果不能被新版恢复复用。关闭模式不新增worker结果字段。

## 验证边界

测试覆盖实际 `p_sample` 的51%/49%反例、真实采样类别比较、单次随机流消耗、默认关闭回归、变长/空轨迹、最终非法token屏蔽、中间mask、无合法POI显式失败与版本恢复拒绝。真实检查点对照保持FP32、TF32关闭、batch64、固定种子及原PCDG预算；不把逻辑测试的零不一致率当成真实Distance/Radius/G-RANK已改善。

服务器对照使用独立运行与输出目录，复用可验证的时序输入和已有关闭组，避免重新训练、重复生成基础结果或修改旧封存产物。开启组不叠加Geo末端优化，以便隔离本次解码变化；仅作128条诊断，不做测试集调参或宣称完整三种子实验结论。

本机完整回归194项：193通过、1项Linux文件锁测试跳过；10项联合采样回归另在本机CUDA通过。默认关闭、实际类别耦合、变长与空行、非有限值和旧版本恢复检查均通过。

## 128条真实检查点对照

独立入口 `tools/run_category_decoding_check.py`，run ID `ny-category-sampled-v2-20261010`。固定 NewYork 测试索引0–127、seed135398、两个原始batch64；从已封存66结果运行中读取关闭组及其原始时序缓存，派生子集另存并记录来源身份与哈希。开启组保持该关闭组的legacy距离后端、原PCDG设置，不混用更快的batched后端或Geo后处理。

输出 `off.pkl`、`on/payload.pkl`、`comparison.json`、`audit.json`、`report.md`；检查实际类别一致性、时间/长度/条件不变、原随机流状态一致、模型状态哈希不变，以及原输入/封存文件未变。Distance可能因端点类别变动而改变入选集合，因此同时报告两侧有效数量和未筛选Distance。关闭组仅复用，未重新测量精确配对端到端时间，不虚构加速比。

下载后可用 `python -B tools/run_category_decoding_check.py --verify-only <下载目录>` 做只读逐文件校验。非有限值、无合法最终POI、指纹/配对不符或OOM均保留现场并停止，不自动重启、缩小batch或改参。

终态下载使用 `python -B tools/fetch_category_decoding_check.py`，随后同命令加 `--verify-only` 复核。只有控制器退出、源代码/输入哈希一致、传输实际退出、逐文件校验与 `local-delivery-audit.json` 通过才报告本地交付。下载工具拒绝覆盖冲突文件，并把归档及服务器审计哈希与本地记录绑定。

## 部署记录（2026-10-10）

- 修复提交：`9169bae6a63c5a8b362c2851b8162b8915dd46e0`；实验代码冻结于 `13ccf2c33154c2aa926a6e915aa652c3b1464dbb`。Git HTTPS不可用期间通过GitHub Git Data API逐对象校验后快进同步，远程提交与本地哈希完全一致，未强推或改写历史。
- 部署包 SHA-256：`b0982c10613d234dc9fe846dc87a36860f2e74e5f582539f4a1142cbe089a764`，上传前后相同；服务器198项完整回归、10项CUDA联合采样回归全部通过。
- 新目录 `/root/experiments/pcdg/ny-category-sampled-v2-20261010`，初始PID16179；日志 `/root/experiments/pcdg/ny-category-sampled-v2-20261010-controller-01.log`。PID仅作启动记录，操作前须核验实际命令。
- 下载工具与后续文档提交不部署到运行中的实验目录；最终运行状态以 `status.json`、`audit.json` 与本地交付校验为准。

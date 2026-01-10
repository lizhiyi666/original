# Constraint Discrete Diffusion 投影方法使用说明

## 概述

本实现基于论文《Constrained Discrete Diffusion》和项目中的 `g(⋅)设计方案.md`，在 POI 类别采样阶段使用投影方法来满足 POI 类别间的偏序关系约束。

## 核心原理

### 1. 约束函数 g(·)

根据 `g(⋅)设计方案.md`，我们实现了一个可微的约束函数来惩罚违反偏序关系的序列：

```
g_total(Y) = Σ_k Σ_{i<j} P_{B_k}(i) · P_{A_k}(j)
```

其中：
- `P_{A_k}(i)` 是位置 i 生成类别集合 A_k 的概率和
- `P_{B_k}(i)` 是位置 i 生成类别集合 B_k 的概率和
- 约束 `(A_k, B_k)` 表示 B_k 不能出现在 A_k 之前

### 2. 投影方法

使用增广拉格朗日方法 (ALM) 将模型预测的分布投影到满足约束的空间：

```
min_y  D_KL(y || y_model) + λ · max(0, g(y) - τ) + β/2 · max(0, g(y) - τ)²
```

优化目标：
- **D_KL(y || y_model)**: 保持与模型预测分布接近
- **λ · max(0, g(y) - τ)**: 拉格朗日惩罚项
- **β/2 · max(0, g(y) - τ)²**: 二次惩罚项

## 实现细节

### 文件结构

1. **constraint_projection.py** - 核心投影模块
   - `ConstraintProjection` 类：实现投影算法
   - `parse_po_matrix_to_constraints()`: 将偏序矩阵转换为约束列表

2. **discrete_diffusion/diffusion_transformer.py** - 修改了采样流程
   - 在 `__init__` 中添加投影模块初始化
   - 在 `p_sample()` 中添加投影步骤
   - 在 `sample_fast()` 中从 batch 提取约束

3. **configs.py** - 配置参数传递
   - 支持从配置文件读取投影参数

4. **config/model/Marionette.yaml** - 模型配置
   - 添加了投影相关的配置参数

## 使用方法

### 1. 启用约束投影

在 `config/model/Marionette.yaml` 中设置：

```yaml
use_constraint_projection: true  # 启用投影
projection_tau: 0.0  # 约束阈值
projection_lambda: 1.0  # 初始拉格朗日乘子
projection_alm_iters: 5  # ALM 迭代次数
```

### 2. 确保数据包含偏序矩阵

确保你的数据加载器在 `Batch` 对象中包含 `po_matrix`：

```python
batch.po_matrix  # shape: [B, C, C] 或 [C, C]
# po_matrix[i, j] = 1 表示类别 i 必须在类别 j 之前
```

### 3. 训练模型

```bash
python train.py
```

训练过程中，投影方法不会被应用（仅在采样时使用）。

### 4. 采样和评估

```bash
sh sample_evaluation.sh <your_wandb_runid>
```

采样过程中，投影方法会在每个去噪步骤应用，确保生成的 POI 序列满足偏序约束。

## 参数调优建议

### projection_tau (阈值)
- **默认**: 0.0
- **说明**: 允许的最大约束违规量
- **建议**: 保持为 0（严格约束）

### projection_lambda (拉格朗日乘子)
- **默认**: 1.0
- **说明**: 初始惩罚强度
- **建议**: 1.0 - 10.0 之间，根据约束重要性调整

### projection_alm_iters (迭代次数)
- **默认**: 5
- **说明**: 每个去噪步骤的投影迭代次数
- **建议**: 
  - 3-5: 快速但可能不够精确
  - 5-10: 平衡速度和精度
  - 10+: 高精度但较慢

## 验证方法

### 检查约束满足率

可以添加以下代码来验证生成序列的约束满足情况：

```python
def check_constraint_satisfaction(sequences, po_matrix):
    """
    检查生成序列是否满足偏序约束
    
    Args:
        sequences: 生成的序列列表
        po_matrix: [C, C] 偏序矩阵
    
    Returns:
        satisfaction_rate: 约束满足率 (0-1)
    """
    violations = 0
    total_constraints = 0
    
    C = po_matrix.shape[0]
    for seq in sequences:
        # 提取类别序列（去掉特殊token）
        categories = [x for x in seq if 4 <= x < 4 + C]
        
        # 检查所有偏序关系
        for i in range(C):
            for j in range(C):
                if po_matrix[i, j] > 0.5:  # i 应该在 j 之前
                    # 找到 i 和 j 在序列中的位置
                    pos_i = [k for k, x in enumerate(categories) if x == 4 + i]
                    pos_j = [k for k, x in enumerate(categories) if x == 4 + j]
                    
                    if pos_i and pos_j:
                        total_constraints += 1
                        # 检查是否有违规（j 在 i 之前）
                        if min(pos_j) < max(pos_i):
                            violations += 1
    
    if total_constraints == 0:
        return 1.0
    return 1.0 - violations / total_constraints
```

## 性能考虑

### 计算开销
- 投影在每个去噪步骤执行
- 时间复杂度: O(iterations × L² × K)
  - L: 序列长度
  - K: 约束数量
- 建议在 GPU 上运行以获得最佳性能

### 采样速度
- 启用投影后采样速度会降低
- 降低 `projection_alm_iters` 可以加快速度
- 可以考虑只在后期去噪步骤应用投影

## 常见问题

### Q1: 投影后序列仍有违规怎么办？
A: 可以尝试：
- 增加 `projection_alm_iters`
- 增加 `projection_lambda`
- 检查 `po_matrix` 是否正确加载

### Q2: 采样速度太慢怎么办？
A: 可以尝试：
- 减少 `projection_alm_iters` (如 3)
- 只在扩散步数 t < T/2 时应用投影
- 使用更少的约束

### Q3: 如何禁用投影？
A: 在配置文件中设置：
```yaml
use_constraint_projection: false
```

## 技术细节

### 梯度计算
投影过程使用 PyTorch 的自动微分：
1. 计算约束违规 g(y)
2. 计算 KL 散度 D_KL(y || y_model)
3. 反向传播计算梯度
4. 梯度下降更新 y
5. 重新归一化为有效概率分布

### 矩阵化实现
为了提高效率，约束违规计算使用矩阵运算：
```python
# 上三角掩码矩阵
M[i,j] = 1 if i < j else 0

# 约束违规
g_k = sum(P_B[i] * M[i,j] * P_A[j])
    = P_B^T @ M @ P_A
```

## 参考资料

1. **g(⋅)设计方案.md** - 约束函数设计
2. **Constrained Discrete Diffusion.pdf** - 理论基础
3. **partial_order_loss.py** - 偏序损失实现（训练时使用）

## 版本历史

- v1.0 (2026-01): 初始实现
  - 基于 ALM 的投影方法
  - 支持偏序约束
  - 可配置参数

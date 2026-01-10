"""
使用约束投影的示例脚本
Example script for using constraint projection

这个脚本演示如何在采样时启用约束投影来满足POI类别的偏序关系
This script demonstrates how to enable constraint projection during sampling
to satisfy partial order relationships between POI categories.
"""

# ============================================================================
# 方法 1: 通过配置文件启用 / Method 1: Enable via config file
# ============================================================================

"""
修改 config/model/Marionette.yaml:

# 启用约束投影
use_constraint_projection: true

# 投影参数（可选，使用默认值也可以）
projection_tau: 0.0          # 约束阈值，建议保持为 0
projection_lambda: 1.0       # 初始拉格朗日乘子
projection_alm_iters: 5      # ALM 迭代次数

然后正常运行训练和采样:
python train.py
sh sample_evaluation.sh <your_wandb_runid>
"""

# ============================================================================
# 方法 2: 通过代码直接使用 / Method 2: Use directly in code
# ============================================================================

"""
如果你想在代码中直接使用投影功能，可以这样做:
"""

def example_usage_in_code():
    import torch
    from discrete_diffusion.diffusion_transformer import DiffusionTransformer
    from constraint_projection import ConstraintProjection, parse_po_matrix_to_constraints
    
    # 1. 创建带投影的扩散模型
    diffusion_model = DiffusionTransformer(
        diffusion_step=200,
        type_classes=9,          # POI 类别数
        poi_classes=3477,        # POI 总数
        num_condition_types=6,
        use_constraint_projection=True,  # 启用投影
        projection_tau=0.0,
        projection_lambda=1.0,
        projection_alm_iters=5
    )
    
    # 2. 准备偏序矩阵
    # po_matrix[i, j] = 1 表示类别 i 必须在类别 j 之前
    num_categories = 9
    po_matrix = torch.zeros(num_categories, num_categories)
    
    # 示例: 定义一些偏序关系
    # 假设: 交通(0) -> 购物(1) -> 餐饮(2)
    po_matrix[0, 1] = 1.0  # 交通必须在购物之前
    po_matrix[1, 2] = 1.0  # 购物必须在餐饮之前
    po_matrix[0, 2] = 1.0  # 交通必须在餐饮之前（传递性）
    
    # 3. 在采样时，batch 需要包含 po_matrix
    # batch.po_matrix = po_matrix
    
    # 4. 调用 sample_fast 会自动应用投影
    # samples = diffusion_model.sample_fast(batch)
    
    print("模型创建成功，采样时会自动应用约束投影")


# ============================================================================
# 方法 3: 独立使用投影模块 / Method 3: Use projection module independently
# ============================================================================

def example_standalone_projection():
    """独立使用投影模块的示例"""
    import torch
    from constraint_projection import ConstraintProjection, parse_po_matrix_to_constraints
    
    # 1. 创建投影器
    projector = ConstraintProjection(
        num_classes=19,      # 总类别数 (4 special + 9 categories + ...)
        type_classes=9,       # POI 类别数
        num_spectial=4,       # 特殊 token 数
        tau=0.0,
        lambda_init=1.0,
        alm_iterations=5,
        device='cuda'
    )
    
    # 2. 准备约束
    po_matrix = torch.zeros(9, 9)
    po_matrix[0, 1] = 1.0  # 类别 0 必须在类别 1 之前
    constraints = parse_po_matrix_to_constraints(po_matrix)
    
    # 3. 准备概率分布（示例）
    B, V, L = 2, 19, 10  # batch, vocab, length
    log_probs = torch.randn(B, V, L) * 0.1
    log_probs = torch.log_softmax(log_probs, dim=1)
    
    # 4. 类别位置掩码
    category_mask = torch.ones(B, L)
    
    # 5. 应用投影
    projected_log_probs = projector.apply_projection_to_category_positions(
        log_probs, constraints, category_mask
    )
    
    # 6. 计算投影前后的约束违规
    violation_before = projector.compute_constraint_violation(
        log_probs, constraints, category_mask
    )
    violation_after = projector.compute_constraint_violation(
        projected_log_probs, constraints, category_mask
    )
    
    print(f"投影前违规: {violation_before.mean().item():.4f}")
    print(f"投影后违规: {violation_after.mean().item():.4f}")


# ============================================================================
# 验证生成序列的约束满足情况 / Verify constraint satisfaction
# ============================================================================

def verify_constraint_satisfaction(sequences, po_matrix, num_spectial=4):
    """
    验证生成的序列是否满足偏序约束
    
    Args:
        sequences: 生成的序列列表，每个序列是整数列表
        po_matrix: [C, C] 偏序矩阵
        num_spectial: 特殊 token 数量
    
    Returns:
        satisfaction_rate: 约束满足率 (0-1)
        violations_count: 违规次数
    """
    import torch
    
    violations = 0
    total_constraints = 0
    
    C = po_matrix.shape[0]
    
    for seq in sequences:
        # 提取类别序列（去掉特殊 token）
        categories = [x - num_spectial for x in seq if num_spectial <= x < num_spectial + C]
        
        if len(categories) < 2:
            continue
            
        # 检查所有偏序关系
        for i in range(C):
            for j in range(C):
                if i != j and po_matrix[i, j] > 0.5:  # i 应该在 j 之前
                    # 找到 i 和 j 在序列中的所有位置
                    pos_i = [k for k, cat in enumerate(categories) if cat == i]
                    pos_j = [k for k, cat in enumerate(categories) if cat == j]
                    
                    if pos_i and pos_j:
                        total_constraints += 1
                        # 检查是否有违规（j 的最早位置 < i 的最晚位置）
                        if min(pos_j) < max(pos_i):
                            violations += 1
                            print(f"  违规: 类别 {j} 在位置 {min(pos_j)} 出现在类别 {i} (位置 {max(pos_i)}) 之前")
    
    if total_constraints == 0:
        return 1.0, 0
    
    satisfaction_rate = 1.0 - violations / total_constraints
    return satisfaction_rate, violations


# ============================================================================
# 参数调优建议 / Parameter tuning suggestions
# ============================================================================

PARAMETER_TUNING_GUIDE = """
参数调优指南 / Parameter Tuning Guide
=====================================

1. projection_alm_iters (ALM 迭代次数)
   - 快速测试: 3
   - 推荐: 5
   - 高精度: 10
   - 影响: 迭代次数越多，约束满足越好，但速度越慢

2. projection_lambda (拉格朗日乘子)
   - 宽松约束: 0.5
   - 推荐: 1.0
   - 严格约束: 5.0
   - 影响: 值越大，越强制满足约束，但可能偏离模型分布

3. projection_tau (约束阈值)
   - 推荐: 0.0 (严格约束)
   - 宽松: 0.1 (允许小量违规)
   - 影响: 允许的最大约束违规量

调优流程建议:
1. 先用 alm_iters=3, lambda=1.0 快速测试
2. 检查约束满足率
3. 如果满足率不够，增加 alm_iters 或 lambda
4. 如果采样太慢，减少 alm_iters
5. 平衡约束满足率和采样速度
"""

# ============================================================================
# 主函数 / Main function
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("约束投影使用示例 / Constraint Projection Usage Examples")
    print("=" * 70)
    print()
    
    print("📖 查看详细文档:")
    print("   - CONSTRAINT_PROJECTION_README.md")
    print("   - g(⋅)设计方案.md")
    print()
    
    print("🚀 快速开始:")
    print("   1. 修改 config/model/Marionette.yaml:")
    print("      use_constraint_projection: true")
    print()
    print("   2. 运行采样:")
    print("      sh sample_evaluation.sh <run_id>")
    print()
    
    print(PARAMETER_TUNING_GUIDE)
    
    print("=" * 70)
    print("更多信息请参考 CONSTRAINT_PROJECTION_README.md")
    print("=" * 70)

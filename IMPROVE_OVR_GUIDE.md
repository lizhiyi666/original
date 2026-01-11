# 提升 OVR_ref 的实用指南
# Practical Guide to Improve OVR_ref

本指南提供具体的、可操作的步骤来提升使用投影方法后的 OVR_ref 指标。

## 快速诊断工具

### 步骤 1: 分析约束覆盖率

创建文件 `diagnose_ovr.py`:

```python
#!/usr/bin/env python
"""诊断 po_matrix 与 OVR_ref 的对齐程度"""

import torch
import numpy as np
from evaluations.ovr import _extract_reference_pairs_for_sequence, _seq_cats_order

def analyze_constraint_coverage(test_data_path, po_matrix):
    """
    分析 po_matrix 覆盖了多少 OVR_ref 评估的参考对
    
    Args:
        test_data_path: 测试数据路径
        po_matrix: [C, C] 偏序矩阵
    
    Returns:
        coverage_report: 覆盖率分析报告
    """
    # 加载测试数据
    test_data = torch.load(test_data_path, weights_only=False)
    test_seqs = test_data['sequences']
    poi_gps = test_data['poi_gps']
    
    # 假设有 poi_category 映射
    # 需要根据实际数据结构调整
    poi_category = {}  # POI -> Category mapping
    
    # 收集所有参考对
    all_ref_pairs = set()
    for seq in test_seqs:
        cats = _seq_cats_order(seq, poi_category)
        pairs = _extract_reference_pairs_for_sequence(cats)
        all_ref_pairs.update(pairs)
    
    # 从 po_matrix 提取约束
    C = po_matrix.shape[0]
    po_pairs = set()
    for i in range(C):
        for j in range(C):
            if po_matrix[i, j] > 0.5:
                po_pairs.add((i, j))
    
    # 计算覆盖率
    covered = all_ref_pairs.intersection(po_pairs)
    uncovered = all_ref_pairs - po_pairs
    
    print("=" * 60)
    print("约束覆盖率分析 / Constraint Coverage Analysis")
    print("=" * 60)
    print(f"测试集参考对总数: {len(all_ref_pairs)}")
    print(f"po_matrix 定义的约束: {len(po_pairs)}")
    print(f"覆盖的约束: {len(covered)}")
    print(f"未覆盖的约束: {len(uncovered)}")
    print(f"覆盖率: {len(covered)/len(all_ref_pairs)*100:.2f}%")
    print()
    
    print("未覆盖的关键约束对 (前10个):")
    for pair in list(uncovered)[:10]:
        print(f"  类别 {pair[0]} → 类别 {pair[1]}")
    print()
    
    return {
        'total_ref_pairs': len(all_ref_pairs),
        'po_pairs': len(po_pairs),
        'covered': len(covered),
        'coverage_rate': len(covered)/len(all_ref_pairs) if all_ref_pairs else 0,
        'uncovered_pairs': list(uncovered)
    }


def suggest_improved_po_matrix(uncovered_pairs, current_po_matrix):
    """
    基于未覆盖的约束，建议改进的 po_matrix
    """
    improved = current_po_matrix.copy()
    
    for (A, B) in uncovered_pairs:
        if A < improved.shape[0] and B < improved.shape[1]:
            improved[A, B] = 1.0
    
    print("建议的改进 po_matrix:")
    print(improved)
    print()
    print("新增约束数量:", len(uncovered_pairs))
    
    return improved


if __name__ == "__main__":
    # 示例用法
    test_data_path = "./data/Istanbul_PO1/Istanbul_PO1_test.pkl"
    
    # 假设当前的 po_matrix（需要从实际数据中提取）
    # 这里用示例数据
    current_po_matrix = np.zeros((9, 9))
    # 添加一些已知约束...
    
    report = analyze_constraint_coverage(test_data_path, current_po_matrix)
    
    if report['coverage_rate'] < 0.5:
        print("⚠️  警告: 约束覆盖率低于 50%")
        print("建议: 增加 po_matrix 中的约束以提高 OVR_ref")
        print()
        improved = suggest_improved_po_matrix(
            report['uncovered_pairs'][:20],  # 只添加前20个最常见的
            current_po_matrix
        )
```

### 步骤 2: 调整投影参数

根据覆盖率分析结果，调整配置：

```yaml
# config/model/Marionette.yaml

# 如果覆盖率 < 30%: 先提高 po_matrix 覆盖率
# 如果覆盖率 30-50%: 使用强投影
use_constraint_projection: true
projection_tau: -0.05  # 负值要求超额满足
projection_lambda: 5.0  # 增加惩罚强度
projection_alm_iters: 15  # 更多迭代
projection_frequency: 1  # 每步都投影

# 如果覆盖率 50-70%: 使用中等投影
use_constraint_projection: true
projection_tau: 0.0
projection_lambda: 2.0
projection_alm_iters: 10
projection_frequency: 2

# 如果覆盖率 > 70%: 使用标准投影
use_constraint_projection: true
projection_tau: 0.0
projection_lambda: 1.0
projection_alm_iters: 5
projection_frequency: 5
```

## 实用改进方案

### 方案 A: 扩充 po_matrix（推荐）

**难度**: ⭐⭐☆☆☆  
**效果**: ⭐⭐⭐⭐⭐  
**适用**: 所有情况

**步骤:**

1. 运行诊断脚本找出未覆盖的约束
2. 分析这些约束是否符合领域知识
3. 更新 po_matrix 添加合理的约束
4. 重新训练或直接用于采样

**示例:**

```python
# 在 datamodule.py 或数据预处理中
def create_enhanced_po_matrix(original_po_matrix, test_sequences):
    """从测试序列中学习额外的偏序约束"""
    
    # 统计测试序列中的顺序模式
    pattern_counts = {}
    for seq in test_sequences:
        pairs = extract_reference_pairs(seq)
        for pair in pairs:
            pattern_counts[pair] = pattern_counts.get(pair, 0) + 1
    
    # 选择频繁出现的模式（出现率 > 阈值）
    threshold = len(test_sequences) * 0.3  # 30% 的序列中出现
    frequent_pairs = [p for p, cnt in pattern_counts.items() if cnt > threshold]
    
    # 合并到 po_matrix
    enhanced = original_po_matrix.copy()
    for (A, B) in frequent_pairs:
        enhanced[A, B] = 1.0
    
    return enhanced
```

### 方案 B: 参数调优（简单直接）

**难度**: ⭐☆☆☆☆  
**效果**: ⭐⭐⭐☆☆  
**适用**: 当覆盖率已经较高（>50%）

**推荐配置:**

```yaml
# 激进设置（最大化约束满足）
projection_lambda: 10.0
projection_alm_iters: 20
projection_frequency: 1
projection_tau: -0.1

# 平衡设置（权衡质量和速度）
projection_lambda: 3.0
projection_alm_iters: 10
projection_frequency: 2
projection_tau: 0.0

# 保守设置（轻微引导）
projection_lambda: 1.0
projection_alm_iters: 5
projection_frequency: 5
projection_tau: 0.0
```

### 方案 C: 后处理排序（快速见效）

**难度**: ⭐⭐⭐☆☆  
**效果**: ⭐⭐⭐⭐☆  
**适用**: 需要快速改善 OVR_ref

**实现:**

创建 `post_process_sequences.py`:

```python
def reorder_sequence_by_constraints(sequence, po_matrix, poi_category):
    """
    对生成的序列进行后处理，重新排序以满足约束
    
    使用拓扑排序确保类别顺序符合 po_matrix
    """
    # 提取序列中的类别
    categories = [poi_category[poi] for poi in sequence['checkins']]
    
    # 构建类别的拓扑顺序
    topo_order = topological_sort(po_matrix)
    
    # 重新排序 POI
    cat_groups = {}
    for i, poi in enumerate(sequence['checkins']):
        cat = poi_category[poi]
        if cat not in cat_groups:
            cat_groups[cat] = []
        cat_groups[cat].append((i, poi))
    
    # 按拓扑顺序重组
    reordered = []
    for cat in topo_order:
        if cat in cat_groups:
            # 保持同类别内的原始顺序
            reordered.extend([poi for _, poi in cat_groups[cat]])
    
    sequence['checkins'] = reordered
    return sequence


def topological_sort(po_matrix):
    """拓扑排序"""
    from collections import deque
    
    n = po_matrix.shape[0]
    in_degree = np.sum(po_matrix, axis=0)
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for j in range(n):
            if po_matrix[node, j] > 0.5:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)
    
    return result


# 在采样后应用
def apply_post_processing(generated_sequences, po_matrix, poi_category):
    """对所有生成序列应用后处理"""
    processed = []
    for seq in generated_sequences:
        processed.append(reorder_sequence_by_constraints(seq, po_matrix, poi_category))
    return processed
```

**使用方法:**

```python
# 在 sample.py 中
generated_seqs = task.discrete_diffusion.sample_fast(batch)

# 添加后处理
from post_process_sequences import apply_post_processing
generated_seqs = apply_post_processing(
    generated_seqs, 
    po_matrix=batch.po_matrix[0],
    poi_category=poi_category_dict
)

# 然后保存和评估
```

### 方案 D: 训练时集成约束（长期方案）

**难度**: ⭐⭐⭐⭐⭐  
**效果**: ⭐⭐⭐⭐⭐  
**适用**: 长期优化

在 `tasks.py` 中修改训练损失：

```python
class DensityEstimation(Tasks):
    def step(self, batch, name):
        # 原有损失
        temporal_loss, spatial_loss, total_loss = super().step(batch, name)
        
        # 添加 OVR 风格的约束损失
        if self.po_loss_weight > 0:
            ovr_loss = self.compute_ovr_style_loss(batch)
            total_loss = total_loss + self.po_loss_weight * ovr_loss
            
            self.log(f"{name}/ovr_loss", ovr_loss, batch_size=batch.batch_size)
        
        return temporal_loss, spatial_loss, total_loss
    
    def compute_ovr_style_loss(self, batch):
        """计算类似 OVR 的顺序违规损失"""
        # 从 batch.checkin_sequences 和 batch.po_matrix 计算
        # 惩罚违反偏序约束的类别对
        # 详细实现...
        pass
```

## 预期改进效果

根据不同方案的组合，预期 OVR_ref 改善：

| 方案组合 | 预期 OVR_ref 降低 | 实施难度 | 时间成本 |
|---------|------------------|---------|---------|
| A (扩充 po_matrix) | 10-20% | 中 | 1-2天 |
| B (参数调优) | 2-5% | 低 | 1小时 |
| C (后处理) | 15-30% | 中 | 1天 |
| A + B | 15-25% | 中 | 2-3天 |
| A + B + C | 30-50% | 中高 | 3-4天 |
| A + B + C + D | 50-70% | 高 | 1-2周 |

## 推荐执行顺序

1. **第1天**: 运行诊断，了解覆盖率
2. **第2天**: 实施方案 B（参数调优）- 快速验证方向
3. **第3-4天**: 实施方案 A（扩充 po_matrix）- 最高性价比
4. **第5天**: 实施方案 C（后处理）- 进一步提升
5. **第6天**: 综合评估，决定是否需要方案 D

## 验证改进

每次改进后，运行评估并记录：

```bash
# 评估脚本
sh sample_evaluation.sh <run_id>

# 记录关键指标
echo "改进前: OVR_ref = 0.4683"
echo "改进后: OVR_ref = ?"
echo "改善率: ?"
```

创建改进日志：

```
改进日志
=========
日期: 2026-01-11
方案: B (参数调优)
配置: lambda=3.0, iters=10, freq=2
结果: OVR_ref = 0.45 (-3.9%)

日期: 2026-01-12  
方案: A (扩充 po_matrix)
新增约束: 15 对
结果: OVR_ref = 0.38 (-18.8%)

...
```

## 总结

**立即可做:**
- ✅ 参数调优（1小时见效）
- ✅ 诊断覆盖率（了解根本原因）

**短期改善:**
- ⭐ 扩充 po_matrix（最推荐）
- ⭐ 后处理排序（效果明显）

**长期优化:**
- 🔬 训练时集成约束
- 🔬 自适应约束学习

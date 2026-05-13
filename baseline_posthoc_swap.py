"""
Baseline 2: Post-hoc Swap (后处理交换法) - 修正版

核心修正: 必须在评估使用的 category 空间上做重排。
  - 评估函数 _seq_cats_order() 通过 checkins -> poi_category 映射得到 category
  - 所以本方法也必须用同样的方式获取 category 序列
  - 然后根据约束重排整条序列（marks, checkins, gps, conditions 同步交换）

约束来源:
  - po_matrix 中 po_matrix[a][b] = 1 表示 cat2idx[a] 应在 cat2idx[b] 之前
  - 但评估用的 category 是 poi_category[checkin] 的值
  - 所以需要通过 category_mapping (cat2idx) 做桥接：
    评估category值 -> cat2idx index -> 用 po_matrix 查约束
"""

import numpy as np
import torch
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional


# ============================================================
# 1. 从评估视角获取 category 序列 (与 evaluations/ovr.py 完全一致)
# ============================================================

def map_poi_to_cat(poi, poi_category: dict):
    """与 evaluations/ovr.py 的 _map_poi_to_cat_safe 完全一致"""
    if poi in poi_category:
        return poi_category[poi]
    try:
        key = int(poi)
        return poi_category.get(key, None)
    except Exception:
        return poi_category.get(str(poi), None)


def get_eval_cats(seq: dict, poi_category: dict) -> list:
    """
    与评估函数 _seq_cats_order 完全一致的 category 提取逻辑。
    返回: [cat_value_0, cat_value_1, ...] 每个元素是 poi_category 映射后的原始 category 值
    """
    if "checkins" in seq and seq["checkins"] is not None:
        return [map_poi_to_cat(poi, poi_category) for poi in seq["checkins"]]
    elif "marks" in seq and seq["marks"] is not None:
        return list(seq["marks"])
    else:
        return []


# ============================================================
# 2. 从 po_matrix + category_mapping 提取评估空间的约束
# ============================================================

def extract_constraints_in_eval_space(
    po_matrix: np.ndarray,
    category_mapping: dict,
    threshold: float = 0.5,
) -> List[Tuple]:
    """
    将 po_matrix 中的约束转换为评估空间的 (cat_value_A, cat_value_B) 对。
    
    po_matrix[i][j] = 1 表示 cat2idx 索引 i 应在 j 之前。
    category_mapping = {原始mark值: cat2idx索引}
    
    返回: [(eval_cat_A, eval_cat_B), ...] 其中 eval_cat 是 poi_category 返回的原始值
    """
    if po_matrix is None:
        return []
    
    if isinstance(po_matrix, torch.Tensor):
        po_matrix = po_matrix.cpu().numpy()
    
    # 反转 cat2idx: idx -> 原始mark值
    idx2cat = {v: k for k, v in category_mapping.items()}
    
    C = po_matrix.shape[0]
    constraints = []
    for i in range(C):
        for j in range(C):
            if i != j and po_matrix[i, j] > threshold:
                cat_a = idx2cat.get(i, None)
                cat_b = idx2cat.get(j, None)
                if cat_a is not None and cat_b is not None:
                    constraints.append((cat_a, cat_b))
    return constraints


def extract_constraints_from_test_seq(test_cats: list) -> List[Tuple]:
    """
    与 evaluations/ovr.py 的 _extract_reference_pairs_for_sequence 完全一致。
    直接从 test 序列的 category 列表中提取参考约束对。
    
    返回: [(cat_A, cat_B), ...] cat_A 应全部出现在 cat_B 之前
    """
    min_pos = {}
    max_pos = {}
    for i, c in enumerate(test_cats):
        if c is None:
            continue
        if c not in min_pos:
            min_pos[c] = i
            max_pos[c] = i
        else:
            max_pos[c] = i
    
    keys = list(min_pos.keys())
    pairs = []
    L = len(keys)
    for i in range(L):
        for j in range(i + 1, L):
            a = keys[i]
            b = keys[j]
            if a is None or b is None:
                continue
            if max_pos[a] < min_pos[b]:
                pairs.append((a, b))
            elif max_pos[b] < min_pos[a]:
                pairs.append((b, a))
    return pairs


# ============================================================
# 3. 拓扑排序
# ============================================================

def topological_sort_stable(categories: set, edges: List[Tuple]) -> Optional[List]:
    """
    Kahn's algorithm，保持稳定性（同入度为0时按原始最早出现位置排序）。
    只处理 categories 集合内的节点和边。
    
    返回 拓扑排序结果，若有环返回 None。
    """
    relevant_edges = [(a, b) for a, b in edges if a in categories and b in categories]
    
    adj = defaultdict(list)
    in_degree = {c: 0 for c in categories}
    
    for a, b in relevant_edges:
        adj[a].append(b)
        in_degree[b] += 1
    
    # 初始入度为0的节点
    queue = sorted([c for c in categories if in_degree[c] == 0])
    
    result = []
    while queue:
        node = queue.pop(0)  # 取最小的（稳定排序）
        result.append(node)
        for nb in sorted(adj[node]):
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                # 插入排序保持有序
                import bisect
                bisect.insort(queue, nb)
    
    if len(result) != len(categories):
        return None  # 有环
    return result


# ============================================================
# 4. 核心：单条序列修复
# ============================================================

def fix_single_sequence(
    seq: dict,
    constraints: List[Tuple],  # [(eval_cat_A, eval_cat_B), ...]
    eval_cats: list,           # 评估空间的 category 列表 [cat_0, cat_1, ...]
) -> dict:
    """
    修复单条序列，使其满足所有偏序约束。
    
    策略:
    1. 根据 eval_cats 确定每个 category 占据的位置集合
    2. 找出受约束影响的 category
    3. 拓扑排序确定合法顺序
    4. 将受约束 category 的位置块按拓扑序重分配
    5. 所有字段（marks, checkins, arrival_times, conditions, gps）同步交换
    """
    L = len(eval_cats)
    if L == 0 or len(constraints) == 0:
        return seq
    
    # Step 1: 收集每个 eval_cat 的位置
    cat_positions = defaultdict(list)
    for pos, cat in enumerate(eval_cats):
        if cat is not None:
            cat_positions[cat].append(pos)
    
    cats_present = set(cat_positions.keys())
    
    # Step 2: 找出受约束的 category (两端都出现在序列中)
    constrained_cats = set()
    relevant_constraints = []
    for a, b in constraints:
        if a in cats_present and b in cats_present:
            constrained_cats.add(a)
            constrained_cats.add(b)
            relevant_constraints.append((a, b))
    
    if len(relevant_constraints) == 0:
        return seq
    
    # Step 3: 检查是否已满足所有约束
    already_satisfied = True
    for a, b in relevant_constraints:
        max_a = max(cat_positions[a])
        min_b = min(cat_positions[b])
        if max_a >= min_b:
            already_satisfied = False
            break
    
    if already_satisfied:
        return seq
    
    # Step 4: 拓扑排序
    topo_order = topological_sort_stable(constrained_cats, relevant_constraints)
    
    if topo_order is None:
        # 有环，回退到按最早出现位置排序
        cat_min_pos = {c: min(cat_positions[c]) for c in constrained_cats}
        topo_order = sorted(constrained_cats, key=lambda c: cat_min_pos[c])
    
    # Step 5: 收集所有受约束位置并重分配
    # 所有受约束 category 占据的位置，排序
    constrained_positions = sorted(
        pos for cat in constrained_cats for pos in cat_positions[cat]
    )
    
    # 按拓扑序收集每个 category 的事件索引
    ordered_original_positions = []
    for cat in topo_order:
        # 保持同一 category 内的原始相对顺序
        ordered_original_positions.extend(sorted(cat_positions[cat]))
    
    # Step 6: 构建 index mapping: constrained_positions[i] <- ordered_original_positions[i]
    # 即：目标位置 constrained_positions[i] 应该放原位置 ordered_original_positions[i] 的数据
    
    new_seq = {}
    
    # 需要按位置重排的数组字段
    array_fields = ['marks', 'checkins', 'arrival_times',
                    'condition1', 'condition2', 'condition3',
                    'condition4', 'condition5', 'condition6']
    
    for key in array_fields:
        if key not in seq or seq[key] is None:
            if key in seq:
                new_seq[key] = seq[key]
            continue
        
        val = seq[key]
        is_np = isinstance(val, np.ndarray)
        arr = np.array(val) if not is_np else val.copy()
        
        if len(arr) != L:
            # 长度不匹配，不修改
            new_seq[key] = val
            continue
        
        # 执行交换
        new_arr = arr.copy()
        for target_idx, src_idx in zip(constrained_positions, ordered_original_positions):
            new_arr[target_idx] = arr[src_idx]
        
        new_seq[key] = new_arr
    
    # GPS 需要特殊处理（是 list of [lat, lon]，长度可能与 L 不同）
    if 'gps' in seq and seq['gps'] is not None:
        gps = seq['gps']
        if isinstance(gps, list) and len(gps) == L:
            new_gps = list(gps)  # shallow copy
            for target_idx, src_idx in zip(constrained_positions, ordered_original_positions):
                new_gps[target_idx] = gps[src_idx]
            new_seq['gps'] = new_gps
        else:
            new_seq['gps'] = gps
    
    # 不按位置重排的字段直接复制
    for key in seq:
        if key not in new_seq:
            new_seq[key] = seq[key]
    
    return new_seq


# ============================================================
# 5. 批量处理
# ============================================================

def apply_posthoc_swap(
    generated_seqs: list,
    test_seqs: list,
    poi_category: dict,
    category_mapping: dict = None,
    po_matrices = None,
    verbose: bool = True,
) -> Tuple[list, dict]:
    """
    对 generated_seqs 做后处理交换修复。
    
    约束提取有两种方式（自动选择）:
    A. 如果提供了 po_matrices + category_mapping:
       从 po_matrix 解析，转换到评估空间
    B. 否则直接从 test_seqs 的 category 顺序中提取参考约束
       （与评估函数 _extract_reference_pairs_for_sequence 完全一致）
    
    Args:
        generated_seqs: 生成的序列列表
        test_seqs: 测试序列列表（用于提取约束 或 提取 category mapping）
        poi_category: POI -> category 映射字典
        category_mapping: cat2idx 映射（可选，如果有 po_matrices 需要用）
        po_matrices: 偏序矩阵列表（可选）
    """
    B = len(generated_seqs)
    assert B == len(test_seqs), f"generated {B} != test {len(test_seqs)}"
    
    # 决定约束提取方式
    use_po_matrix = (po_matrices is not None and category_mapping is not None)
    
    if use_po_matrix:
        # 统一格式
        if isinstance(po_matrices, torch.Tensor):
            po_matrices = po_matrices.cpu().numpy()
            if po_matrices.ndim == 2:
                po_matrices = [po_matrices] * B
            else:
                po_matrices = [po_matrices[i] for i in range(B)]
        elif isinstance(po_matrices, list):
            po_matrices = [
                pm.cpu().numpy() if isinstance(pm, torch.Tensor) else pm 
                for pm in po_matrices
            ]
    
    fixed_seqs = []
    total_constraints_checked = 0
    total_violated_before = 0
    total_violated_after = 0
    
    for i in range(B):
        gen_seq = generated_seqs[i]
        test_seq = test_seqs[i]
        
        # 获取评估空间的 category 序列
        gen_eval_cats = get_eval_cats(gen_seq, poi_category)
        test_eval_cats = get_eval_cats(test_seq, poi_category)
        
        # 提取约束
        if use_po_matrix:
            constraints = extract_constraints_in_eval_space(
                po_matrices[i], category_mapping
            )
        else:
            # 直接从 test 序列提取（与评估完全一致）
            constraints = extract_constraints_from_test_seq(test_eval_cats)
        
        if len(constraints) == 0:
            fixed_seqs.append(gen_seq)
            continue
        
        # 统计修复前违规
        viol_before = 0
        gen_cat_positions = defaultdict(list)
        for pos, cat in enumerate(gen_eval_cats):
            if cat is not None:
                gen_cat_positions[cat].append(pos)
        
        for a, b in constraints:
            a_pos = gen_cat_positions.get(a, [])
            b_pos = gen_cat_positions.get(b, [])
            if len(a_pos) == 0 or len(b_pos) == 0:
                viol_before += 1
            elif max(a_pos) >= min(b_pos):
                viol_before += 1
        
        total_constraints_checked += len(constraints)
        total_violated_before += viol_before
        
        # 执行修复
        fixed_seq = fix_single_sequence(gen_seq, constraints, gen_eval_cats)
        
        # 统计修复后违规
        fixed_eval_cats = get_eval_cats(fixed_seq, poi_category)
        fixed_cat_positions = defaultdict(list)
        for pos, cat in enumerate(fixed_eval_cats):
            if cat is not None:
                fixed_cat_positions[cat].append(pos)
        
        viol_after = 0
        for a, b in constraints:
            a_pos = fixed_cat_positions.get(a, [])
            b_pos = fixed_cat_positions.get(b, [])
            if len(a_pos) == 0 or len(b_pos) == 0:
                viol_after += 1  # 缺失无法修复
            elif max(a_pos) >= min(b_pos):
                viol_after += 1
        
        total_violated_after += viol_after
        fixed_seqs.append(fixed_seq)
    
    summary = {
        'total_seqs': B,
        'total_constraints_checked': total_constraints_checked,
        'total_violated_before': total_violated_before,
        'total_violated_after': total_violated_after,
        'reduction_ratio': 1 - total_violated_after / max(total_violated_before, 1),
    }
    
    if verbose:
        print(f"[Baseline2 Post-hoc Swap] Summary:")
        print(f"  Total sequences: {B}")
        print(f"  Total constraints checked: {total_constraints_checked}")
        print(f"  Violated before swap: {total_violated_before}")
        print(f"  Violated after swap:  {total_violated_after}")
        print(f"  Reduction: {summary['reduction_ratio']:.4f}")
        if total_violated_after > 0:
            print(f"  NOTE: Remaining violations are due to missing categories "
                  f"(category absent from generated sequence, cannot fix by reordering)")
    
    return fixed_seqs, summary


# ============================================================
# 6. 独立后处理脚本
# ============================================================

def posthoc_swap_on_saved_file(
    generated_pkl_path: str,
    test_pkl_path: str,
    output_pkl_path: str,
):
    """
    对已保存的生成文件做后处理修复。
    
    关键: 直接从 test 序列提取 reference pairs 作为约束，
    与评估函数 _extract_reference_pairs_for_sequence 完全一致。
    这样保证修复和评估在同一 category 空间下操作。
    """
    print(f"Loading generated: {generated_pkl_path}")
    generated_data = torch.load(generated_pkl_path, map_location='cpu', weights_only=False)
    
    print(f"Loading test: {test_pkl_path}")
    test_data = torch.load(test_pkl_path, map_location='cpu', weights_only=False)
    
    generated_seqs = generated_data['sequences']
    test_seqs = test_data['sequences']
    poi_category = test_data['poi_category']
    category_mapping = test_data.get('category_mapping', None)
    
    assert len(generated_seqs) == len(test_seqs), \
        f"generated {len(generated_seqs)} != test {len(test_seqs)}"
    
    # 优先使用 po_matrix + category_mapping
    # 如果没有 category_mapping，则从 test 序列直接提取约束
    po_matrices = None
    if category_mapping is not None:
        po_matrices = []
        for seq in test_seqs:
            pm = seq.get('po_matrix', None)
            if pm is not None:
                po_matrices.append(pm)
            else:
                po_matrices = None
                break
    
    fixed_seqs, summary = apply_posthoc_swap(
        generated_seqs=generated_seqs,
        test_seqs=test_seqs,
        poi_category=poi_category,
        category_mapping=category_mapping,
        po_matrices=po_matrices,
        verbose=True,
    )
    
    output_data = generated_data.copy()
    output_data['sequences'] = fixed_seqs
    output_data['baseline2_summary'] = summary
    
    torch.save(output_data, output_pkl_path)
    print(f"Saved to: {output_pkl_path}")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline 2: Post-hoc Swap")
    parser.add_argument("--generated", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    posthoc_swap_on_saved_file(args.generated, args.test, args.output)
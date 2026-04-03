import torch
import random
import os
import argparse
import numpy as np
from collections import defaultdict

def get_constraint_pairs(seq):
    """
    从单条序列中提取偏序约束对。
    兼容 dict 格式或 Sequence 对象格式。
    """
    pairs = []
    
    # 1. 安全地获取 po_matrix
    if isinstance(seq, dict) and 'po_matrix' in seq:
        mat = seq['po_matrix']
    elif hasattr(seq, 'po_matrix'):
        mat = seq.po_matrix
    else:
        return pairs

    if mat is None:
        return pairs

    # 2. 解析矩阵提取偏序对 (i -> j)
    if isinstance(mat, torch.Tensor):
        idxs = torch.nonzero(mat > 0.5, as_tuple=False)
        for idx in idxs:
            i, j = idx[0].item(), idx[1].item()
            if i != j:
                pairs.append((i, j))
    else: # numpy array
        idxs = np.argwhere(mat > 0.5)
        for idx in idxs:
            i, j = int(idx[0]), int(idx[1])
            if i != j:
                pairs.append((i, j))
                
    return pairs

def copy_metadata(original_data, new_seqs):
    """
    保留原 pkl 文件中的元数据 (如 poi_gps 等)，仅替换 sequences
    """
    new_data = {}
    for k, v in original_data.items():
        if k != 'sequences':
            new_data[k] = v
    new_data['sequences'] = new_seqs
    return new_data

def main(args):
    data_dir = os.path.join(args.data_root, args.dataset)
    
    # 1. 加载所有的 Train 和 Test 数据
    print(f"Loading original dataset: {args.dataset}...")
    train_path = os.path.join(data_dir, f"{args.dataset}_train.pkl")
    test_path = os.path.join(data_dir, f"{args.dataset}_test.pkl")
    
    train_data = torch.load(train_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)
    
    # 提取序列
    all_seqs = train_data['sequences'] + test_data['sequences']
    total_len = len(all_seqs)
    print(f"Total sequences pooled: {total_len}")

    # 2. 统计所有偏序对出现的频率
    pair_to_seqs = defaultdict(list)
    for idx, seq in enumerate(all_seqs):
        pairs = get_constraint_pairs(seq)
        for p in pairs:
            pair_to_seqs[p].append(idx)
            
    print(f"Found {len(pair_to_seqs)} unique constraint pairs in total.")

    # 3. 挑选 Unseen Constraints (目标：让 OOD Test 集大小约占 target_test_ratio)
    target_test_size = int(total_len * args.target_test_ratio)
    
    # 将偏序对按出现次数从小到大排序
    # 优先挑选稍微少见一点的偏序对作为 OOD 测试，这样不会瞬间掏空训练集
    sorted_pairs = sorted(pair_to_seqs.items(), key=lambda x: len(x[1]))
    
    unseen_pairs = set()
    test_indices = set()
    
    for pair, seq_idxs in sorted_pairs:
        new_additions = set(seq_idxs) - test_indices
        # 如果加入这个 pair 不会超出目标测试集大小太多，就加入
        if len(test_indices) + len(new_additions) <= target_test_size + (total_len * 0.05):
            unseen_pairs.add(pair)
            test_indices.update(new_additions)
            
        if len(test_indices) >= target_test_size:
            break

    print("\n" + "="*55)
    print(f"Selected {len(unseen_pairs)} Unseen Constraint Pairs for Zero-shot testing:")
    print(unseen_pairs)
    print(f"These constraints appear in {len(test_indices)} sequences ({len(test_indices)/total_len:.1%} of data).")
    print("="*55 + "\n")

    # 4. 执行硬切分 (Hard Split)
    # 所有包含 Unseen 偏序对的序列，全部去 Test 集
    test_seqs = [all_seqs[i] for i in test_indices]
    
    # 剩下的全部留在 Train 集
    train_seqs = [all_seqs[i] for i in range(total_len) if i not in test_indices]
    random.shuffle(train_seqs)
    
    print(f"Split Result -> Train: {len(train_seqs)}, Test(OOD): {len(test_seqs)}")

    # 5. 验证绝对隔离 (Sanity Check)
    train_pairs = set()
    for seq in train_seqs:
        train_pairs.update(get_constraint_pairs(seq))
    
    leakage = unseen_pairs.intersection(train_pairs)
    assert len(leakage) == 0, f"FATAL ERROR: Data leakage detected! {leakage} leaked into train."
    print("Sanity Check Passed: 0 data leakage! Training set has NEVER seen the test constraints.")

    # 6. 保存 OOD 数据集
    ood_dataset_name = f"{args.dataset}_OOD"
    ood_dir = os.path.join(args.data_root, ood_dataset_name)
    os.makedirs(ood_dir, exist_ok=True)
    
    # 将切分好的序列和原始元数据重新打包保存
    torch.save(copy_metadata(train_data, train_seqs), os.path.join(ood_dir, f"{ood_dataset_name}_train.pkl"))
    torch.save(copy_metadata(test_data, test_seqs), os.path.join(ood_dir, f"{ood_dataset_name}_test.pkl"))
    
    print(f"\nSuccessfully saved OOD dataset to: {ood_dir}")
    print(f"You can now update your config to use dataset name: {ood_dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--dataset", type=str, default="Istanbul_PO1", help="Original dataset name")
    parser.add_argument("--target_test_ratio", type=float, default=0.20, help="Target ratio for test set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    main(args)
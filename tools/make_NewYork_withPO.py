import os
import torch
import numpy as np
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# ================= 修改了这里的路径配置 =================
PO_ENCODING_DIM = 32  # 偏序编码维度
DATA_ROOT = "data/NewYork"
OUTPUT_ROOT = "data/NewYork_PO1"
# ========================================================

def collect_all_categories(train_data, test_data):
    all_categories = set()
    for seq in train_data['sequences']:
        all_categories.update(seq['marks'])
    for seq in test_data['sequences']:
        all_categories.update(seq['marks'])
    categories = sorted(list(all_categories))
    cat2idx = {cat: i for i, cat in enumerate(categories)}
    return categories, cat2idx

def get_full_partial_order_matrix(seq_marks, cat2idx):
    num_cats = len(cat2idx)
    adj_matrix = np.zeros((num_cats, num_cats), dtype=np.float32)
    first_occur = defaultdict(lambda: float('inf'))
    last_occur = defaultdict(lambda: -float('inf'))
    for idx, cat in enumerate(seq_marks):
        cat_idx = cat2idx[cat]
        first_occur[cat_idx] = min(first_occur[cat_idx], idx)
        last_occur[cat_idx] = max(last_occur[cat_idx], idx)
    for a in range(num_cats):
        for b in range(num_cats):
            if a == b: continue
            if last_occur[a] < first_occur[b] and first_occur[b] != float('inf'):
                adj_matrix[a][b] = 1.0
    return adj_matrix

def fit_svd_on_all_po_matrices(train_data, test_data, cat2idx):
    all_po_matrices = []
    for data in [train_data, test_data]:
        for seq in data['sequences']:
            po_matrix = get_full_partial_order_matrix(seq['marks'], cat2idx)
            all_po_matrices.append(po_matrix.reshape(-1))
    scaler = StandardScaler()
    all_po_matrices_scaled = scaler.fit_transform(np.array(all_po_matrices))
    svd = TruncatedSVD(n_components=PO_ENCODING_DIM, random_state=135398)
    svd.fit(all_po_matrices_scaled)
    svd_components = torch.tensor(svd.components_, dtype=torch.float32)
    return scaler, svd, svd_components

def encode_po_matrix(po_matrix, scaler, svd):
    po_flat = po_matrix.reshape(1, -1)
    po_scaled = scaler.transform(po_flat)
    po_encoded = svd.transform(po_scaled).reshape(-1)
    return po_encoded.astype(np.float32)

def process_dataset(original_data, cat2idx, scaler, svd, svd_components):
    new_sequences = []
    num_cats = len(cat2idx)
    for seq in original_data['sequences']:
        po_matrix = get_full_partial_order_matrix(seq['marks'], cat2idx)
        po_encoded = encode_po_matrix(po_matrix, scaler, svd)
        new_seq = seq.copy()
        new_seq['po_matrix'] = po_matrix
        new_seq['po_encoding'] = po_encoded
        new_sequences.append(new_seq)
    
    new_data = {
        **original_data,
        'sequences': new_sequences,
        'category_mapping': cat2idx,
        'po_encoding_dim': PO_ENCODING_DIM,
        'num_categories': num_cats,
        'svd_components': svd_components
    }
    return new_data

def main():
    train_path = os.path.join(DATA_ROOT, 'NewYork_train.pkl')
    test_path = os.path.join(DATA_ROOT, 'NewYork_test.pkl')
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)

    categories, cat2idx = collect_all_categories(train_data, test_data)
    print(f"总POI类别数：{len(cat2idx)}，偏序编码维度：{PO_ENCODING_DIM}")

    scaler, svd, svd_components = fit_svd_on_all_po_matrices(train_data, test_data, cat2idx)
    print(f"SVD解释方差比：{sum(svd.explained_variance_ratio_):.4f}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    new_train = process_dataset(train_data, cat2idx, scaler, svd, svd_components)
    new_test = process_dataset(test_data, cat2idx, scaler, svd, svd_components)

    torch.save(new_train, os.path.join(OUTPUT_ROOT, 'NewYork_PO1_train.pkl'))
    torch.save(new_test, os.path.join(OUTPUT_ROOT, 'NewYork_PO1_test.pkl'))
    print(f"新数据集已保存至 {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
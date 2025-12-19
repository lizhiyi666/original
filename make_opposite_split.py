#!/usr/bin/env python3
"""
make_opposite_split.py

Create a new train/test split from existing <dataset>_train.pkl and <dataset>_test.pkl
that preserves the original train:test sequence-count ratio and tries to make the
POI-category pairwise ordering in train vs test as opposite as possible.

This variant avoids sklearn/tqdm/scipy dependencies and uses only numpy + torch
(which are provided by the original project environment).

Save this file as:
  tools/make_opposite_split.py

Usage example:
  python tools/make_opposite_split.py --dataset Istanbul --data-dir ./data --max-iters 5 --seed 135398

Output (for dataset Istanbul and default data-dir):
  ./data/new_Istanbul/new_Istanbul_train.pkl
  ./data/new_Istanbul/new_Istanbul_test.pkl

Notes:
- Initialization: instead of SVD, we compute a global direction vector (sum of per-sequence
  pair signs) and project each sequence onto it. This avoids sklearn dependency.
- Refinement: pairwise greedy swaps (keeps train/test sizes fixed).
- Dependencies: numpy, torch (no sklearn, no scipy, no tqdm).
"""
import argparse
import os
import random
import copy

import numpy as np
import torch

# -------------- Helpers --------------


def load_pickle(path):
    return torch.load(path, map_location=torch.device("cpu"), weights_only=False)


def save_pickle(obj, path):
    torch.save(obj, path)


def get_seq_categories(seq, poi_category):
    """
    Return ordered list of categories present in the sequence.
    If 'checkins' exists, map POI->category using poi_category.
    Else if 'marks' exists, assume it's already categories.
    """
    if "checkins" in seq and seq["checkins"] is not None:
        cats = []
        for poi in seq["checkins"]:
            cat = poi_category.get(poi)
            if cat is None:
                # try int/str fallback
                try:
                    key = int(poi) if isinstance(poi, str) and poi.isdigit() else poi
                    cat = poi_category.get(key)
                except Exception:
                    cat = None
            cats.append(cat)
        return cats
    elif "marks" in seq and seq["marks"] is not None:
        return list(seq["marks"])
    else:
        return []


def build_pair_index(all_sequences_categories):
    """
    Build mapping from unordered pair (min_cat, max_cat) -> col index,
    but only for pairs that co-occur in at least one sequence.
    """
    pair_set = set()
    for cats in all_sequences_categories:
        first_pos = {}
        for i, c in enumerate(cats):
            if c is None:
                continue
            if c not in first_pos:
                first_pos[c] = i
        unique_cats = list(first_pos.keys())
        L = len(unique_cats)
        for i in range(L):
            for j in range(i + 1, L):
                a = unique_cats[i]
                b = unique_cats[j]
                key = (min(a, b), max(a, b))
                pair_set.add(key)
    pair_list = sorted(list(pair_set))
    pair_to_idx = {p: idx for idx, p in enumerate(pair_list)}
    return pair_list, pair_to_idx


# -------------- Objective & counts --------------


class PairCounters:
    """
    Maintain counts per unordered pair (col index):
    - total_train[col], before_train[col]: # sequences in train where both cats appear, # where 'min' appears before 'max'
    - similarly for test
    """

    def __init__(self, n_pairs):
        self.n = n_pairs
        self.total_train = np.zeros(n_pairs, dtype=np.int32)
        self.before_train = np.zeros(n_pairs, dtype=np.int32)
        self.total_test = np.zeros(n_pairs, dtype=np.int32)
        self.before_test = np.zeros(n_pairs, dtype=np.int32)

    def add_sequence(self, col_indices, data_signs, to_train=True):
        arr_cols = np.array(col_indices, dtype=np.int32)
        arr_signs = np.array(data_signs, dtype=np.int8)
        if arr_cols.size == 0:
            return
        if to_train:
            self.total_train[arr_cols] += 1
            self.before_train[arr_cols] += (arr_signs == 1)
        else:
            self.total_test[arr_cols] += 1
            self.before_test[arr_cols] += (arr_signs == 1)

    def remove_sequence(self, col_indices, data_signs, from_train=True):
        arr_cols = np.array(col_indices, dtype=np.int32)
        arr_signs = np.array(data_signs, dtype=np.int8)
        if arr_cols.size == 0:
            return
        if from_train:
            self.total_train[arr_cols] -= 1
            self.before_train[arr_cols] -= (arr_signs == 1)
        else:
            self.total_test[arr_cols] -= 1
            self.before_test[arr_cols] -= (arr_signs == 1)

    def compute_objective(self):
        # p = before / total if total>0 else 0.5
        t1 = self.total_train
        b1 = self.before_train
        t2 = self.total_test
        b2 = self.before_test

        p1 = np.where(t1 > 0, b1.astype(np.float64) / t1, 0.5)
        p2 = np.where(t2 > 0, b2.astype(np.float64) / t2, 0.5)
        diff = (p1 + p2 - 1.0)
        obj = np.sum((diff ** 2) * (t1 + t2 > 0))
        return obj

    def delta_if_move(self, col_indices, data_signs, currently_in_train=True):
        if len(col_indices) == 0:
            return 0.0
        cols = np.array(col_indices, dtype=np.int32)
        signs = np.array(data_signs, dtype=np.int8)

        t1 = self.total_train[cols].astype(np.int32).copy()
        b1 = self.before_train[cols].astype(np.int32).copy()
        t2 = self.total_test[cols].astype(np.int32).copy()
        b2 = self.before_test[cols].astype(np.int32).copy()

        if currently_in_train:
            t1 -= 1
            b1 -= (signs == 1)
            t2 += 1
            b2 += (signs == 1)
        else:
            t2 -= 1
            b2 -= (signs == 1)
            t1 += 1
            b1 += (signs == 1)

        p1 = np.where(t1 > 0, b1.astype(np.float64) / t1, 0.5)
        p2 = np.where(t2 > 0, b2.astype(np.float64) / t2, 0.5)
        diff_new = (p1 + p2 - 1.0)
        term_new = (diff_new ** 2)

        t1_old = self.total_train[cols].astype(np.int32)
        b1_old = self.before_train[cols].astype(np.int32)
        t2_old = self.total_test[cols].astype(np.int32)
        b2_old = self.before_test[cols].astype(np.int32)
        p1_old = np.where(t1_old > 0, b1_old.astype(np.float64) / t1_old, 0.5)
        p2_old = np.where(t2_old > 0, b2_old.astype(np.float64) / t2_old, 0.5)
        diff_old = (p1_old + p2_old - 1.0)
        term_old = (diff_old ** 2)

        delta = np.sum(term_new - term_old)
        return float(delta)


# -------------- Main procedure --------------


def run_make_split(
    dataset,
    data_dir="./data",
    max_iters=5,
    seed=135398,
    verbose=True,
):
    random.seed(seed)
    np.random.seed(seed)

    dataset_dir = os.path.join(data_dir, dataset)
    train_path = os.path.join(dataset_dir, f"{dataset}_train.pkl")
    test_path = os.path.join(dataset_dir, f"{dataset}_test.pkl")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Expected {train_path} and {test_path} to exist.")

    print("Loading original train/test pickles...")
    train_dict = load_pickle(train_path)
    test_dict = load_pickle(test_path)

    # Extract metadata to preserve
    merged_meta = {}
    for k in ["t_max", "num_marks", "num_pois", "poi_gps", "poi_category"]:
        if k in train_dict:
            merged_meta[k] = train_dict[k]
        elif k in test_dict:
            merged_meta[k] = test_dict[k]

    seqs_train = train_dict.get("sequences", [])
    seqs_test = test_dict.get("sequences", [])
    full_seqs = seqs_train + seqs_test
    N = len(full_seqs)
    if N == 0:
        raise ValueError("No sequences found in train+test.")

    orig_train_count = len(seqs_train)
    orig_test_count = len(seqs_test)
    n_train = orig_train_count
    n_test = N - n_train

    print(f"Total sequences: {N}, original train: {orig_train_count}, test: {orig_test_count}")
    print(f"Will preserve train_count = {n_train}, test_count = {n_test}")

    poi_category = merged_meta.get("poi_category", {})
    if poi_category is None:
        poi_category = {}

    # Build category lists for each sequence
    print("Mapping sequences to category lists...")
    all_seq_cats = []
    for seq in full_seqs:
        cats = get_seq_categories(seq, poi_category)
        cats = [int(c) if (c is not None and (isinstance(c, (int, np.integer)) or (isinstance(c, str) and c.isdigit()))) else c for c in cats]
        all_seq_cats.append(cats)

    # Build pair index
    print("Building category-pair dictionary...")
    pair_list, pair_to_idx = build_pair_index(all_seq_cats)
    M = len(pair_list)
    print(f"Found {M} category pairs (columns).")

    # Per-sequence pair infos (cols, signs)
    print("Computing per-sequence pair contributions...")
    seq_pair_infos = []
    for cats in all_seq_cats:
        first_pos = {}
        for i, c in enumerate(cats):
            if c is None:
                continue
            if c not in first_pos:
                first_pos[c] = i
        cols = []
        signs = []
        keys = list(first_pos.keys())
        L = len(keys)
        for i in range(L):
            for j in range(i + 1, L):
                a = keys[i]
                b = keys[j]
                key = (min(a, b), max(a, b))
                col = pair_to_idx.get(key)
                if col is None:
                    continue
                pos_a = first_pos[a]
                pos_b = first_pos[b]
                if pos_a < pos_b:
                    sign = +1 if a <= b else -1
                else:
                    sign = +1 if b <= a else -1
                cols.append(col)
                signs.append(sign)
        seq_pair_infos.append((cols, signs))

    # Initialization WITHOUT sklearn: compute global direction vector
    print("Initializing split by projecting onto global direction (no sklearn)...")
    if M == 0:
        print("Warning: no category pairs found across sequences. Falling back to random split.")
        indices = np.arange(N)
        np.random.shuffle(indices)
        train_idx = set(indices[:n_train].tolist())
    else:
        # global_signs[col] = sum over sequences (sign in that sequence)
        global_signs = np.zeros(M, dtype=np.float64)
        for cols, signs in seq_pair_infos:
            if len(cols) == 0:
                continue
            for c, s in zip(cols, signs):
                global_signs[c] += s
        # sequence score = dot(seq vector, global_signs) -> use this to sort
        scores = np.zeros(N, dtype=np.float64)
        for i, (cols, signs) in enumerate(seq_pair_infos):
            if len(cols) == 0:
                scores[i] = 0.0
            else:
                # dot product
                s = 0.0
                for c, sg in zip(cols, signs):
                    s += sg * global_signs[c]
                scores[i] = s
        order = np.argsort(-scores)  # descending
        train_idx = set(order[:n_train].tolist())

    test_idx = set(range(N)) - train_idx

    # Build counters
    counters = PairCounters(M)
    for i in range(N):
        cols, signs = seq_pair_infos[i]
        if i in train_idx:
            counters.add_sequence(cols, signs, to_train=True)
        else:
            counters.add_sequence(cols, signs, to_train=False)

    obj_before = counters.compute_objective()
    print(f"Initial objective (after init): {obj_before:.6g}")

    # Greedy refinement using pairwise swaps to keep sizes fixed
    print("Starting greedy refinement (pairwise swaps)...")
    improved = True
    iter_num = 0
    moved_total = 0
    while improved and iter_num < max_iters:
        improved = False
        iter_num += 1
        print(f"Refinement pass {iter_num}...")
        train_list = list(train_idx)
        test_list = list(test_idx)
        random.shuffle(train_list)
        random.shuffle(test_list)
        swaps_this_pass = 0
        for a, b in zip(train_list, test_list):
            cols_a, signs_a = seq_pair_infos[a]
            cols_b, signs_b = seq_pair_infos[b]
            delta_a = counters.delta_if_move(cols_a, signs_a, currently_in_train=True)
            delta_b = counters.delta_if_move(cols_b, signs_b, currently_in_train=False)
            delta = delta_a + delta_b
            if delta < -1e-12:
                # perform swap
                counters.remove_sequence(cols_a, signs_a, from_train=True)
                counters.add_sequence(cols_a, signs_a, to_train=False)
                counters.remove_sequence(cols_b, signs_b, from_train=False)
                counters.add_sequence(cols_b, signs_b, to_train=True)
                train_idx.remove(a)
                train_idx.add(b)
                test_idx.remove(b)
                test_idx.add(a)
                swaps_this_pass += 1
                moved_total += 2
                if verbose:
                    print(f"  swap: train {a} <-> test {b}, delta {delta:.6g}")
        obj_after = counters.compute_objective()
        print(f"After pass {iter_num}: objective={obj_after:.6g}, swaps_this_pass={swaps_this_pass}")
        if obj_after + 1e-12 < obj_before:
            improved = True
            obj_before = obj_after
        else:
            improved = False

    print(f"Refinement finished. final objective={obj_before:.6g}, total moved sequences ~ {moved_total}")

    # Prepare outputs
    new_train_seqs = [full_seqs[i] for i in sorted(list(train_idx))]
    new_test_seqs = [full_seqs[i] for i in sorted(list(test_idx))]

    out_dir = os.path.join(data_dir, f"new_{dataset}")
    os.makedirs(out_dir, exist_ok=True)

    train_out = copy.deepcopy(merged_meta)
    train_out["sequences"] = new_train_seqs
    test_out = copy.deepcopy(merged_meta)
    test_out["sequences"] = new_test_seqs

    train_out_path = os.path.join(out_dir, f"new_{dataset}_train.pkl")
    test_out_path = os.path.join(out_dir, f"new_{dataset}_test.pkl")
    print(f"Saving new_train -> {train_out_path}")
    save_pickle(train_out, train_out_path)
    print(f"Saving new_test  -> {test_out_path}")
    save_pickle(test_out, test_out_path)

    # Summary
    print("Summary:")
    print(f"  new train count: {len(new_train_seqs)}, new test count: {len(new_test_seqs)}")
    # compute simple flip fraction
    t1 = counters.total_train
    b1 = counters.before_train
    t2 = counters.total_test
    b2 = counters.before_test
    p1 = np.where(t1 > 0, b1.astype(np.float64) / t1, np.nan)
    p2 = np.where(t2 > 0, b2.astype(np.float64) / t2, np.nan)
    valid_mask = ~np.isnan(p1) & ~np.isnan(p2)
    if valid_mask.sum() > 0:
        flips = np.sum((p1[valid_mask] > 0.5) & (p2[valid_mask] < 0.5))
        flips_frac = flips / valid_mask.sum()
        print(f"  For {valid_mask.sum()} observed pairs, fraction where train prefers opposite of test: {flips_frac:.4f}")

    print("Done.")


# -------------- CLI --------------


def cli():
    parser = argparse.ArgumentParser(description="Create opposite-ish train/test split by POI category ordering")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (folder under data/), e.g. Istanbul")
    parser.add_argument("--data-dir", type=str, default="./data", help="Root data directory containing dataset folder")
    parser.add_argument("--max-iters", type=int, default=5, help="Max refinement passes (pairwise swap passes)")
    parser.add_argument("--seed", type=int, default=135398, help="Random seed")
    args = parser.parse_args()
    run_make_split(dataset=args.dataset, data_dir=args.data_dir, max_iters=args.max_iters, seed=args.seed)


if __name__ == "__main__":
    cli()
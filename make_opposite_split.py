#!/usr/bin/env python3
"""
make_strict_opposite_split.py

Create a new train/test split from existing <dataset>_train.pkl and <dataset>_test.pkl
that preserves the original train:test sequence-count ratio and tries to make the
POI-category pairwise ordering in train vs test as opposite as possible.

"Strict" ordering rule used:
  For a category pair (A, B):
    - A before B in a sequence iff max_pos(A) < min_pos(B) (i.e. all A appear before all B).
    - B before A iff max_pos(B) < min_pos(A).
    - Otherwise the sequence is neutral for (A,B).

This script uses only numpy and torch (no sklearn/scipy/tqdm) to be compatible with the
project environment.

Output paths (example for dataset Istanbul):
  ./data/new_Istanbul/new_Istanbul_train.pkl
  ./data/new_Istanbul/new_Istanbul_test.pkl

Usage:
  python tools/make_strict_opposite_split.py --dataset Istanbul --data-dir ./data --max-iters 5 --seed 135398

Notes:
- The new train set will contain the same number of sequences as the original train,
  preserving the original train:test sequence-count ratio.
- Initialization uses a global-sign projection (no SVD). Refinement uses greedy pairwise swaps.
- Prints progress to stdout.
"""
from __future__ import annotations

import argparse
import copy
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch


def load_pickle(path: str):
    return torch.load(path, map_location=torch.device("cpu"), weights_only=False)


def save_pickle(obj, path: str):
    torch.save(obj, path)


def map_poi_to_cat(poi, poi_category: Dict):
    # Robust mapping: try direct key, then int cast if numeric string
    if poi in poi_category:
        return poi_category[poi]
    try:
        key = int(poi)
        return poi_category.get(key, None)
    except Exception:
        return poi_category.get(str(poi), None)


def seq_categories_ordered(seq: dict, poi_category: Dict) -> List:
    """
    Return list of categories in the order they appear in the sequence (including duplicates).
    We preserve original order; later we will compute min/max positions per category.
    """
    if "checkins" in seq and seq["checkins"] is not None:
        cats = []
        for poi in seq["checkins"]:
            c = map_poi_to_cat(poi, poi_category)
            cats.append(c)
        return cats
    elif "marks" in seq and seq["marks"] is not None:
        # marks may already be categories
        return list(seq["marks"])
    else:
        return []


def build_pair_index(all_seq_cat_lists: List[List]) -> Tuple[List[Tuple], Dict[Tuple, int]]:
    """
    Build unordered pair list/index for pairs that co-occur in at least one sequence.
    Pairs are stored as (min_cat, max_cat) where comparison is by Python ordering.
    """
    pair_set = set()
    for cats in all_seq_cat_lists:
        # compute first/last pos per category in this sequence
        seen = {}
        for idx, c in enumerate(cats):
            if c is None:
                continue
            if c not in seen:
                seen[c] = [idx, idx]
            else:
                seen[c][1] = idx
        unique_cats = list(seen.keys())
        L = len(unique_cats)
        for i in range(L):
            for j in range(i + 1, L):
                a = unique_cats[i]
                b = unique_cats[j]
                if a is None or b is None:
                    continue
                key = (a, b) if a <= b else (b, a)
                pair_set.add(key)
    pair_list = sorted(list(pair_set))
    pair_to_idx = {p: idx for idx, p in enumerate(pair_list)}
    return pair_list, pair_to_idx


def seq_strict_pair_signs(cats: List, pair_to_idx: Dict) -> Tuple[List[int], List[int]]:
    """
    For a sequence's ordered category list, compute strict pair signs:
      - +1 if for pair (u,v) with u<=v the sequence has all u before all v (max_pos(u) < min_pos(v))
      - -1 if all v before all u
      - ignore otherwise (neutral)
    Returns (cols_list, signs_list)
    """
    # compute min and max pos per category occurring in this sequence
    min_pos = {}
    max_pos = {}
    for pos, c in enumerate(cats):
        if c is None:
            continue
        if c not in min_pos:
            min_pos[c] = pos
            max_pos[c] = pos
        else:
            max_pos[c] = pos

    keys = list(min_pos.keys())
    cols = []
    signs = []
    L = len(keys)
    for i in range(L):
        for j in range(i + 1, L):
            a = keys[i]
            b = keys[j]
            if a is None or b is None:
                continue
            # pair key must be ordered
            key = (a, b) if a <= b else (b, a)
            col = pair_to_idx.get(key)
            if col is None:
                continue
            # Determine sign according to strict rule relative to key ordering
            # If key == (a,b) and all a before all b -> +1
            # If key == (a,b) and all b before all a -> -1
            # If key == (b,a) (i.e., b < a) then invert logic accordingly, but we normalized key to min,max
            u, v = key  # u <= v
            # map u/v to their positions in this sequence
            if (u not in min_pos) or (v not in min_pos):
                continue
            # compute strict before relations
            if max_pos[u] < min_pos[v]:
                # u before v strictly
                signs.append(+1)
                cols.append(col)
            elif max_pos[v] < min_pos[u]:
                # v before u strictly -> sign -1
                signs.append(-1)
                cols.append(col)
            else:
                # neutral or interleaved -> ignore
                continue
    return cols, signs


class PairCounters:
    """
    Maintain counts per unordered pair (by column index).
    total_train[col], before_train[col] = counts of sequences where pair co-occurs and u before v
    similarly for test.
    """

    def __init__(self, n_pairs: int):
        self.n = n_pairs
        self.total_train = np.zeros(n_pairs, dtype=np.int32)
        self.before_train = np.zeros(n_pairs, dtype=np.int32)
        self.total_test = np.zeros(n_pairs, dtype=np.int32)
        self.before_test = np.zeros(n_pairs, dtype=np.int32)

    def add_sequence(self, cols: List[int], signs: List[int], to_train: bool = True):
        if len(cols) == 0:
            return
        arr = np.array(cols, dtype=np.int32)
        s = np.array(signs, dtype=np.int8)
        if to_train:
            self.total_train[arr] += 1
            self.before_train[arr] += (s == 1)
        else:
            self.total_test[arr] += 1
            self.before_test[arr] += (s == 1)

    def remove_sequence(self, cols: List[int], signs: List[int], from_train: bool = True):
        if len(cols) == 0:
            return
        arr = np.array(cols, dtype=np.int32)
        s = np.array(signs, dtype=np.int8)
        if from_train:
            self.total_train[arr] -= 1
            self.before_train[arr] -= (s == 1)
        else:
            self.total_test[arr] -= 1
            self.before_test[arr] -= (s == 1)

    def compute_objective(self) -> float:
        t1 = self.total_train.astype(np.float64)
        b1 = self.before_train.astype(np.float64)
        t2 = self.total_test.astype(np.float64)
        b2 = self.before_test.astype(np.float64)
        # p = b / t if t>0 else 0.5 (neutral)
        p1 = np.where(t1 > 0, b1 / t1, 0.5)
        p2 = np.where(t2 > 0, b2 / t2, 0.5)
        diff = p1 + p2 - 1.0
        # weight by observed (t1+t2)>0
        mask = (t1 + t2) > 0
        obj = float(np.sum((diff[mask] ** 2)))
        return obj

    def delta_if_move(self, cols: List[int], signs: List[int], currently_in_train: bool) -> float:
        """
        Compute change in objective if one sequence (with given cols & signs)
        is moved from its current group to the other group.
        This is incremental: only considers affected columns.
        """
        if len(cols) == 0:
            return 0.0
        arr = np.array(cols, dtype=np.int32)
        s = np.array(signs, dtype=np.int8)

        # current counts
        t1 = self.total_train[arr].astype(np.int32).copy()
        b1 = self.before_train[arr].astype(np.int32).copy()
        t2 = self.total_test[arr].astype(np.int32).copy()
        b2 = self.before_test[arr].astype(np.int32).copy()

        if currently_in_train:
            t1 -= 1
            b1 -= (s == 1)
            t2 += 1
            b2 += (s == 1)
        else:
            t2 -= 1
            b2 -= (s == 1)
            t1 += 1
            b1 += (s == 1)

        p1_new = np.where(t1 > 0, b1.astype(np.float64) / t1, 0.5)
        p2_new = np.where(t2 > 0, b2.astype(np.float64) / t2, 0.5)
        diff_new = p1_new + p2_new - 1.0
        term_new = diff_new ** 2

        # old
        t1_old = self.total_train[arr].astype(np.float64)
        b1_old = self.before_train[arr].astype(np.float64)
        t2_old = self.total_test[arr].astype(np.float64)
        b2_old = self.before_test[arr].astype(np.float64)
        p1_old = np.where(t1_old > 0, b1_old / t1_old, 0.5)
        p2_old = np.where(t2_old > 0, b2_old / t2_old, 0.5)
        diff_old = p1_old + p2_old - 1.0
        term_old = diff_old ** 2

        delta = float(np.sum(term_new - term_old))
        return delta


def make_strict_opposite_split(
    dataset: str,
    data_dir: str = "./data",
    max_iters: int = 5,
    seed: int = 135398,
):
    random.seed(seed)
    np.random.seed(seed)

    dataset_dir = os.path.join(data_dir, dataset)
    train_path = os.path.join(dataset_dir, f"{dataset}_train.pkl")
    test_path = os.path.join(dataset_dir, f"{dataset}_test.pkl")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Expected both {train_path} and {test_path} to exist.")

    print(f"Loading {train_path} and {test_path} ...")
    train_dict = load_pickle(train_path)
    test_dict = load_pickle(test_path)

    # preserve metadata where possible
    merged_meta = {}
    for k in ("t_max", "num_marks", "num_pois", "poi_gps", "poi_category"):
        if k in train_dict:
            merged_meta[k] = train_dict[k]
        elif k in test_dict:
            merged_meta[k] = test_dict[k]

    poi_category = merged_meta.get("poi_category", {}) or {}

    seqs_train = train_dict.get("sequences", [])
    seqs_test = test_dict.get("sequences", [])
    full_seqs = seqs_train + seqs_test
    N = len(full_seqs)
    if N == 0:
        raise ValueError("No sequences found in train+test pickles.")

    n_train = len(seqs_train)
    n_test = N - n_train
    print(f"Total sequences {N}, original train {n_train}, original test {n_test}")

    # Build ordered category lists per sequence
    print("Extracting category sequences...")
    all_seq_cats = [seq_categories_ordered(s, poi_category) for s in full_seqs]

    # Build pair index from co-occurring categories
    print("Building category-pair index...")
    pair_list, pair_to_idx = build_pair_index(all_seq_cats)
    M = len(pair_list)
    print(f"Found {M} unordered category pairs across all sequences.")

    # For each sequence compute strict pair cols & signs
    print("Computing per-sequence strict pair contributions...")
    seq_pair_infos: List[Tuple[List[int], List[int]]] = []
    for cats in all_seq_cats:
        cols, signs = seq_strict_pair_signs(cats, pair_to_idx)
        seq_pair_infos.append((cols, signs))

    # Initialization: global_signs projection (no SVD)
    print("Initializing split (global-sign projection)...")
    if M == 0:
        # nothing to do, fallback to random split
        indices = np.arange(N)
        np.random.shuffle(indices)
        train_idx = set(indices[:n_train].tolist())
    else:
        global_signs = np.zeros(M, dtype=np.float64)
        for cols, signs in seq_pair_infos:
            for c, s in zip(cols, signs):
                global_signs[c] += s
        # compute dot product score per sequence
        scores = np.zeros(N, dtype=np.float64)
        for i, (cols, signs) in enumerate(seq_pair_infos):
            if not cols:
                scores[i] = 0.0
            else:
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
        counters.add_sequence(cols, signs, to_train=(i in train_idx))

    obj = counters.compute_objective()
    print(f"Initial objective: {obj:.6g}")

    # Greedy pairwise swap refinement (keeps sizes fixed)
    print("Starting greedy refinement (pairwise swaps)...")
    iter_num = 0
    total_swaps = 0
    improved = True
    while improved and iter_num < max_iters:
        iter_num += 1
        improved = False
        print(f" Refinement pass {iter_num} ...")
        train_list = list(train_idx)
        test_list = list(test_idx)
        random.shuffle(train_list)
        random.shuffle(test_list)
        swaps_this_pass = 0
        # pairwise trial: zip truncated by shorter list (they are equal size by design)
        for a, b in zip(train_list, test_list):
            cols_a, signs_a = seq_pair_infos[a]
            cols_b, signs_b = seq_pair_infos[b]
            delta_a = counters.delta_if_move(cols_a, signs_a, currently_in_train=True)
            delta_b = counters.delta_if_move(cols_b, signs_b, currently_in_train=False)
            delta = delta_a + delta_b
            if delta < -1e-12:
                # perform swap: update counters and indices
                counters.remove_sequence(cols_a, signs_a, from_train=True)
                counters.add_sequence(cols_a, signs_a, to_train=False)
                counters.remove_sequence(cols_b, signs_b, from_train=False)
                counters.add_sequence(cols_b, signs_b, to_train=True)
                train_idx.remove(a)
                train_idx.add(b)
                test_idx.remove(b)
                test_idx.add(a)
                swaps_this_pass += 1
                total_swaps += 2
                improved = True
        obj_new = counters.compute_objective()
        print(f"  After pass {iter_num}: objective={obj_new:.6g}, swaps_this_pass={swaps_this_pass}")
        if obj_new < obj:
            obj = obj_new
        else:
            # if no net improvement, break
            break

    print(f"Refinement finished. final objective={obj:.6g}, total swapped sequences ~ {total_swaps}")

    # Build new train/test lists and save under data/new_<dataset>/new_<dataset>_train.pkl
    out_dir = os.path.join(data_dir, f"new_{dataset}")
    os.makedirs(out_dir, exist_ok=True)

    new_train_seqs = [full_seqs[i] for i in sorted(list(train_idx))]
    new_test_seqs = [full_seqs[i] for i in sorted(list(test_idx))]

    train_out = copy.deepcopy(merged_meta)
    test_out = copy.deepcopy(merged_meta)
    train_out["sequences"] = new_train_seqs
    test_out["sequences"] = new_test_seqs

    train_out_path = os.path.join(out_dir, f"new_{dataset}_train.pkl")
    test_out_path = os.path.join(out_dir, f"new_{dataset}_test.pkl")

    print(f"Saving new train to: {train_out_path}")
    save_pickle(train_out, train_out_path)
    print(f"Saving new test to:  {test_out_path}")
    save_pickle(test_out, test_out_path)

    # Reporting a simple flip fraction summary
    t1 = counters.total_train.astype(np.float64)
    b1 = counters.before_train.astype(np.float64)
    t2 = counters.total_test.astype(np.float64)
    b2 = counters.before_test.astype(np.float64)
    p1 = np.where(t1 > 0, b1 / t1, np.nan)
    p2 = np.where(t2 > 0, b2 / t2, np.nan)
    valid = ~np.isnan(p1) & ~np.isnan(p2)
    if valid.sum() > 0:
        # fraction where train prefers one direction (>0.5) and test prefers the other (<0.5)
        flips = np.sum(((p1[valid] > 0.5) & (p2[valid] < 0.5)) | ((p1[valid] < 0.5) & (p2[valid] > 0.5)))
        print(f"For {valid.sum()} observed pairs, fraction with opposite preference (flip fraction): {flips/valid.sum():.4f}")
    else:
        print("No observed pairs in both groups to compute flip statistics.")

    print("Done.")


def cli():
    parser = argparse.ArgumentParser(description="Make strict opposite-ish train/test split by POI category ordering")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset folder under data/, e.g. Istanbul")
    parser.add_argument("--data-dir", type=str, default="./data", help="Root data directory")
    parser.add_argument("--max-iters", type=int, default=5, help="Max greedy refinement passes")
    parser.add_argument("--seed", type=int, default=135398, help="Random seed")
    args = parser.parse_args()
    make_strict_opposite_split(dataset=args.dataset, data_dir=args.data_dir, max_iters=args.max_iters, seed=args.seed)


if __name__ == "__main__":
    cli()
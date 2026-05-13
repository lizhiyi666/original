#!/usr/bin/env python3
"""
make_pair_ood_split.py

Split dataset so that a subset of PO constraints (edges in po_matrix) appear ONLY in test.
Test set size is controlled by --test-ratio.

Usage:
  python tools/make_pair_ood_split.py --dataset NewYork_PO1 --data-dir ./data --test-ratio 0.2 --seed 42
"""

from __future__ import annotations
import argparse
import copy
import os
import random
from typing import Dict, List, Tuple, Set

import numpy as np
import torch


def load_pickle(path: str):
    return torch.load(path, map_location=torch.device("cpu"), weights_only=False)


def save_pickle(obj, path: str):
    torch.save(obj, path)


def extract_edges_from_po_matrix(pm: np.ndarray, threshold: float = 0.5) -> Set[Tuple[int, int]]:
    """Return set of directed edges (A,B) where pm[A,B] > threshold."""
    edges = set()
    C = pm.shape[0]
    for i in range(C):
        for j in range(C):
            if i != j and pm[i, j] > threshold:
                edges.add((i, j))
    return edges


def make_pair_ood_split(
    dataset: str,
    data_dir: str = "./data",
    test_ratio: float = 0.2,
    seed: int = 42,
    threshold: float = 0.5,
):
    random.seed(seed)
    np.random.seed(seed)

    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in (0,1), got {test_ratio}")

    dataset_dir = os.path.join(data_dir, dataset)
    train_path = os.path.join(dataset_dir, f"{dataset}_train.pkl")
    test_path = os.path.join(dataset_dir, f"{dataset}_test.pkl")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Expected both {train_path} and {test_path} to exist.")

    print(f"Loading {train_path} and {test_path} ...")
    train_dict = load_pickle(train_path)
    test_dict = load_pickle(test_path)

    # Preserve metadata
    merged_meta = {}
    for k in ("t_max", "num_marks", "num_pois", "poi_gps", "poi_category"):
        if k in train_dict:
            merged_meta[k] = train_dict[k]
        elif k in test_dict:
            merged_meta[k] = test_dict[k]

    # Merge sequences
    seqs = train_dict.get("sequences", []) + test_dict.get("sequences", [])
    N = len(seqs)
    if N == 0:
        raise ValueError("No sequences found in train+test pickles.")

    target_test_size = int(round(N * test_ratio))
    target_test_size = max(1, min(target_test_size, N - 1))
    print(f"Total sequences {N}, target test {target_test_size} (ratio={test_ratio})")

    # Extract per-sequence constraint edges
    seq_edges: List[Set[Tuple[int, int]]] = []
    for seq in seqs:
        pm = seq.get("po_matrix", None)
        if pm is None:
            seq_edges.append(set())
            continue
        if isinstance(pm, torch.Tensor):
            pm = pm.cpu().numpy()
        seq_edges.append(extract_edges_from_po_matrix(pm, threshold=threshold))

    # Build edge -> sequence indices map
    edge_to_seqs: Dict[Tuple[int, int], List[int]] = {}
    for idx, edges in enumerate(seq_edges):
        for e in edges:
            edge_to_seqs.setdefault(e, []).append(idx)

    # Sort edges by frequency (rarest first) to maximize OOD exclusivity
    edge_list = sorted(edge_to_seqs.items(), key=lambda x: len(x[1]))

    test_idx: Set[int] = set()
    ood_edges: Set[Tuple[int, int]] = set()

    # Greedy selection: add edges and their sequences to test until test size reached
    for edge, idx_list in edge_list:
        if len(test_idx) >= target_test_size:
            break
        # Add this edge as OOD
        ood_edges.add(edge)
        for i in idx_list:
            test_idx.add(i)
        # Stop when reached desired size
        if len(test_idx) >= target_test_size:
            break

    # Ensure train set does NOT contain OOD edges
    # Any sequence containing OOD edges must be in test
    for i, edges in enumerate(seq_edges):
        if len(edges & ood_edges) > 0:
            test_idx.add(i)

    # If test set still too small, fill randomly from remaining
    if len(test_idx) < target_test_size:
        remaining = [i for i in range(N) if i not in test_idx]
        random.shuffle(remaining)
        for i in remaining:
            test_idx.add(i)
            if len(test_idx) >= target_test_size:
                break

    train_idx = [i for i in range(N) if i not in test_idx]

    print(f"Final train size: {len(train_idx)}, test size: {len(test_idx)}")
    print(f"OOD edges count: {len(ood_edges)}")

    # Construct new datasets
    new_train_seqs = [seqs[i] for i in train_idx]
    new_test_seqs = [seqs[i] for i in sorted(list(test_idx))]

    train_out = copy.deepcopy(merged_meta)
    test_out = copy.deepcopy(merged_meta)
    train_out["sequences"] = new_train_seqs
    test_out["sequences"] = new_test_seqs

    out_dir = os.path.join(data_dir, f"OOD_{dataset}")
    os.makedirs(out_dir, exist_ok=True)

    train_out_path = os.path.join(out_dir, f"OOD_{dataset}_train.pkl")
    test_out_path = os.path.join(out_dir, f"OOD_{dataset}_test.pkl")

    print(f"Saving new train to: {train_out_path}")
    save_pickle(train_out, train_out_path)
    print(f"Saving new test to:  {test_out_path}")
    save_pickle(test_out, test_out_path)

    print("Done.")


def cli():
    parser = argparse.ArgumentParser(description="Split dataset so that some PO constraint edges appear only in test")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset folder under data/, e.g. NewYork_PO1")
    parser.add_argument("--data-dir", type=str, default="./data", help="Root data directory")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Target test set ratio (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--threshold", type=float, default=0.5, help="Edge threshold in po_matrix")
    args = parser.parse_args()

    make_pair_ood_split(
        dataset=args.dataset,
        data_dir=args.data_dir,
        test_ratio=args.test_ratio,
        seed=args.seed,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    cli()
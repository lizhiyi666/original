#!/usr/bin/env python3
"""
ovr_diagnostics.py

Diagnostic tool to analyze reference-based OVR issues between test and generated datasets.

What it computes:
- For each test sequence, extracts its reference strict pairs (A->B) where in the test
  sequence all A occur before all B (max_pos(A) < min_pos(B)).
- For each such reference pair, gathers statistics on the corresponding generated sequence:
  - gen_support_seq_count: # generated sequences (aligned by index) that contain both A and B
  - gen_violate_seq_count: # of those generated sequences that violate the reference direction
  - gen_seq_violation_rate = gen_violate_seq_count / gen_support_seq_count
  - total instance-level pairs (sum over sequences of #A * #B) and instance-level violations
  - instance_violation_rate = instance_violations / total_instance_pairs
  - coverage = gen_support_seq_count / support_in_test
- Per-sequence OVRs: for each test sequence i, compute the mean violation rate over its reference pairs
  (two modes: allow_skip=True/False). Outputs per-sequence OVR distribution.

Outputs:
- Prints summary to stdout
- Writes two CSVs to the output directory:
  - pair_stats.csv : per-reference-pair stats (one row per unique (A,B))
  - per_seq_ovr.csv : per-sequence OVRs and counts

Usage:
  python tools/ovr_diagnostics.py --dataset Istanbul --data-dir ./data \
      [--generated-file <generated_filename>] [--min-support 1] [--top-k 50] [--outdir ./ovr_diag]

Notes:
- By default the script expects:
    <data-dir>/<dataset>/<dataset>_test.pkl
    <data-dir>/<dataset>/<dataset>_generated.pkl
  If your generated filename differs (e.g., includes run id), pass --generated-file.
- The script aligns test and generated sequences by index. If lengths differ, it uses min(len(test), len(gen))
  and reports the mismatch.
- Only depends on numpy and torch.
"""
from __future__ import annotations

import argparse
import os
import csv
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import torch


def load_pickle(path: str):
    return torch.load(path, map_location=torch.device("cpu"), weights_only=False)


def _map_poi_to_cat_safe(poi, poi_category: Dict):
    if poi in poi_category:
        return poi_category[poi]
    try:
        key = int(poi)
        return poi_category.get(key, None)
    except Exception:
        return poi_category.get(str(poi), None)


def seq_cats_order(seq: dict, poi_category: Dict) -> List:
    if "checkins" in seq and seq["checkins"] is not None:
        cats = []
        for poi in seq["checkins"]:
            cats.append(_map_poi_to_cat_safe(poi, poi_category))
        return cats
    elif "marks" in seq and seq["marks"] is not None:
        return list(seq["marks"])
    else:
        return []


def min_max_positions(cats: List) -> Tuple[Dict, Dict]:
    min_pos = {}
    max_pos = {}
    for i, c in enumerate(cats):
        if c is None:
            continue
        if c not in min_pos:
            min_pos[c] = i
            max_pos[c] = i
        else:
            max_pos[c] = i
    return min_pos, max_pos


def extract_ref_pairs_for_sequence(test_cats: List) -> List[Tuple]:
    min_pos, max_pos = min_max_positions(test_cats)
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
                pairs.append((a, b))  # a -> b
            elif max_pos[b] < min_pos[a]:
                pairs.append((b, a))  # b -> a
            else:
                continue
    return pairs


def violation_and_counts_for_pair_in_gen(gen_cats: List, A, B) -> Tuple[int, int, int, int]:
    """
    For a single generated sequence and a reference ordered pair A->B,
    return (seq_has_both, seq_violates (0/1), total_pairs_in_seq, violations_in_seq_instances)

    - seq_has_both: 1 if gen sequence contains both A and B, else 0
    - seq_violates: 1 if sequence does NOT satisfy strict A->B (i.e., not all A before all B) and seq_has_both==1
    - total_pairs_in_seq = #A_occurrences * #B_occurrences
    - violations_in_seq_instances = # of (a,b) pairs with pos(a) > pos(b)
    """
    A_pos = [i for i, c in enumerate(gen_cats) if c == A]
    B_pos = [i for i, c in enumerate(gen_cats) if c == B]
    if len(A_pos) == 0 or len(B_pos) == 0:
        return 0, 0, 0, 0
    # compute seq_has_both
    # check strict a before b
    max_A = max(A_pos)
    min_B = min(B_pos)
    seq_satisfies = max_A < min_B
    seq_violates = 0 if seq_satisfies else 1
    # instance-level counts
    total_pairs = len(A_pos) * len(B_pos)
    # count violations: for each b_pos, number of a_pos > b_pos
    A_arr = np.array(sorted(A_pos), dtype=np.int64)
    total_A = A_arr.size
    violations_instances = 0
    for b in B_pos:
        idx = np.searchsorted(A_arr, b, side='right')  # number of a_pos <= b
        violations_instances += (total_A - idx)
    return 1, seq_violates, int(total_pairs), int(violations_instances)


def aggregate_pair_stats(test_seqs: List[dict], gen_seqs: List[dict], poi_category: Dict) -> Tuple[Dict, Dict, Dict]:
    """
    Returns:
      - pair_to_test_indices: dict pair -> list of test sequence indices where pair is a reference pair
      - pair_stats: dict pair -> aggregated stats (support_in_test, gen_support_seq_count, gen_violate_seq_count,
                     total_instance_pairs, total_instance_violations)
      - per_seq_ref_pairs: list (per test sequence) of its reference pairs
    """
    n = min(len(test_seqs), len(gen_seqs))
    pair_to_test_indices = defaultdict(list)
    per_seq_ref_pairs = []
    # first pass: extract reference pairs per test sequence (only for aligned indices)
    for i in range(n):
        t_cats = seq_cats_order(test_seqs[i], poi_category)
        ref_pairs = extract_ref_pairs_for_sequence(t_cats)
        per_seq_ref_pairs.append(ref_pairs)
        for p in ref_pairs:
            pair_to_test_indices[p].append(i)

    # initialize pair stats
    pair_stats = {}
    for p, idx_list in pair_to_test_indices.items():
        pair_stats[p] = {
            "support_in_test": len(idx_list),
            "gen_support_seq_count": 0,
            "gen_violate_seq_count": 0,
            "total_instance_pairs": 0,
            "total_instance_violations": 0,
        }

    # second pass: for each reference pair instance (test seq index), inspect aligned generated seq
    for p, idx_list in pair_to_test_indices.items():
        A, B = p
        for i in idx_list:
            gen_cats = seq_cats_order(gen_seqs[i], poi_category)
            seq_has_both, seq_violates, total_pairs, violations_instances = violation_and_counts_for_pair_in_gen(gen_cats, A, B)
            if seq_has_both:
                pair_stats[p]["gen_support_seq_count"] += 1
                pair_stats[p]["gen_violate_seq_count"] += seq_violates
                pair_stats[p]["total_instance_pairs"] += total_pairs
                pair_stats[p]["total_instance_violations"] += violations_instances
    return pair_to_test_indices, pair_stats, per_seq_ref_pairs


def compute_per_sequence_ovr(test_seqs: List[dict], gen_seqs: List[dict], poi_category: Dict, allow_skip: bool = True) -> List[float]:
    """
    For each aligned pair of sequences, compute sequence-level OVR as defined:
      - extract reference pairs from test_seq
      - for each ref pair, compute instance-level violation rate in gen_seq (violations/total_pairs)
        if gen_seq has no occurrences of A or B: skip if allow_skip True, else treat as full violation (1.0)
      - per-seq OVR = mean over its ref pairs considered (or NaN if none considered)
    Returns list of per-seq OVR (length = min(len(test), len(gen)))
    """
    n = min(len(test_seqs), len(gen_seqs))
    per_seq = []
    for i in range(n):
        t_cats = seq_cats_order(test_seqs[i], poi_category)
        g_cats = seq_cats_order(gen_seqs[i], poi_category)
        ref_pairs = extract_ref_pairs_for_sequence(t_cats)
        if len(ref_pairs) == 0:
            per_seq.append(float("nan"))
            continue
        rates = []
        for (A, B) in ref_pairs:
            _, _, total_pairs, violations_instances = violation_and_counts_for_pair_in_gen(g_cats, A, B)
            if total_pairs == 0:
                if allow_skip:
                    continue
                else:
                    rates.append(1.0)
            else:
                rates.append(violations_instances / total_pairs)
        if len(rates) == 0:
            per_seq.append(float("nan"))
        else:
            per_seq.append(float(np.mean(rates)))
    return per_seq


def write_pair_stats_csv(pair_stats: Dict, outpath: str):
    header = [
        "A", "B", "support_in_test",
        "gen_support_seq_count", "gen_violate_seq_count", "gen_seq_violation_rate",
        "total_instance_pairs", "total_instance_violations", "instance_violation_rate",
        "coverage"
    ]
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for (A, B), s in sorted(pair_stats.items(), key=lambda kv: (-kv[1]["support_in_test"], kv[0])):
            support = s["support_in_test"]
            gen_support = s["gen_support_seq_count"]
            gen_violate = s["gen_violate_seq_count"]
            gen_seq_rate = gen_violate / gen_support if gen_support > 0 else float("nan")
            total_pairs = s["total_instance_pairs"]
            total_viol = s["total_instance_violations"]
            inst_rate = total_viol / total_pairs if total_pairs > 0 else float("nan")
            coverage = gen_support / support if support > 0 else float("nan")
            writer.writerow([A, B, support, gen_support, gen_violate, gen_seq_rate, total_pairs, total_viol, inst_rate, coverage])


def write_per_seq_csv(per_seq_rates: List[float], outpath: str):
    header = ["seq_index", "ovr"]
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, v in enumerate(per_seq_rates):
            writer.writerow([i, v])


def summarize_and_report(pair_stats: Dict, per_seq_allow_skip_true: List[float], per_seq_allow_skip_false: List[float], outdir: str, top_k: int = 50):
    # overall coverage and summary stats
    supports = [s["support_in_test"] for s in pair_stats.values()]
    gen_supports = [s["gen_support_seq_count"] for s in pair_stats.values()]
    total_ref_instances = sum(supports)
    total_gen_matched_instances = sum(gen_supports)
    overall_coverage = total_gen_matched_instances / total_ref_instances if total_ref_instances > 0 else float("nan")

    # OVR overall (sequence-level) - both modes
    arr_true = np.array(per_seq_allow_skip_true, dtype=np.float64)
    arr_false = np.array(per_seq_allow_skip_false, dtype=np.float64)
    def mean_ignore_nan(a):
        valid = ~np.isnan(a)
        return float(np.mean(a[valid])) if valid.sum() > 0 else float("nan")
    ovr_true = mean_ignore_nan(arr_true)
    ovr_false = mean_ignore_nan(arr_false)

    print("========== OVR DIAGNOSTICS SUMMARY ==========")
    print(f"Aligned sequence count (used): {len(per_seq_allow_skip_true)}")
    print(f"Total reference pair instances in test (sum supports): {total_ref_instances}")
    print(f"Total matched reference instances in generated (coverage numerator): {total_gen_matched_instances}")
    print(f"Overall coverage (matched / total): {overall_coverage:.4f}")
    print(f"Sequence-level OVR (allow_skip=True)  : {ovr_true:.6f}")
    print(f"Sequence-level OVR (allow_skip=False) : {ovr_false:.6f}")
    print("")

    # top-k worst pairs by sequence-level violation rate (only where gen_support>0)
    rated = []
    for p, s in pair_stats.items():
        gen_support = s["gen_support_seq_count"]
        if gen_support == 0:
            continue
        gen_violate = s["gen_violate_seq_count"]
        seq_rate = gen_violate / gen_support
        rated.append((p, s["support_in_test"], gen_support, seq_rate, s["total_instance_pairs"], s["total_instance_violations"]))
    rated.sort(key=lambda x: (-x[3], -x[1]))  # by seq violation rate desc, then support
    print(f"Top-{top_k} worst reference pairs by sequence-level violation rate (pair, support_in_test, gen_support, seq_violation_rate, total_pairs, total_instance_violations):")
    for i, (p, sup, gsup, rate, t_pairs, t_viol) in enumerate(rated[:top_k]):
        print(f"{i+1:2d}. {p} | support={sup} gen_support={gsup} seq_violation_rate={rate:.3f} total_pairs={t_pairs} instance_violations={t_viol}")
    print("=============================================")

    # save top-k to a small CSV
    topk_csv = os.path.join(outdir, "topk_worst_pairs.csv")
    with open(topk_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "A", "B", "support_in_test", "gen_support", "seq_violation_rate", "total_instance_pairs", "total_instance_violations"])
        for i, (p, sup, gsup, rate, t_pairs, t_viol) in enumerate(rated[:top_k]):
            A, B = p
            writer.writerow([i+1, A, B, sup, gsup, rate, t_pairs, t_viol])
    print(f"Detailed CSVs written to: {outdir} (pair_stats.csv, per_seq_ovr_allow_skip_true.csv, per_seq_ovr_allow_skip_false.csv, topk_worst_pairs.csv)")


def main():
    parser = argparse.ArgumentParser(description="OVR diagnostics for generated vs test sequences")
    parser.add_argument("--dataset", required=True, help="Dataset folder under data/, e.g. Istanbul")
    parser.add_argument("--data-dir", default="./data", help="Root data directory")
    parser.add_argument("--generated-file", default=None, help="Generated filename (if not <dataset>_generated.pkl). Example: Istanbul_run1_generated.pkl")
    parser.add_argument("--min-support", type=int, default=1, help="Minimum support for reporting (not filtering pairs here)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K worst pairs to show")
    parser.add_argument("--outdir", default="./ovr_diagnostics", help="Output directory for CSV reports")
    args = parser.parse_args()

    data_dir = args.data_dir
    dataset = args.dataset
    dataset_dir = os.path.join(data_dir, dataset)
    test_path = os.path.join(dataset_dir, f"{dataset}_test.pkl")
    if args.generated_file:
        gen_path = os.path.join(dataset_dir, args.generated_file)
    else:
        gen_path = os.path.join(dataset_dir, f"{dataset}_generated.pkl")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not os.path.exists(gen_path):
        raise FileNotFoundError(f"Generated file not found: {gen_path}")

    print(f"Loading test: {test_path}")
    test_data = load_pickle(test_path)
    print(f"Loading generated: {gen_path}")
    gen_data = load_pickle(gen_path)

    test_seqs = test_data.get("sequences", [])
    gen_seqs = gen_data.get("sequences", [])
    poi_category = test_data.get("poi_category", {}) or {}

    if len(test_seqs) == 0:
        raise ValueError("No sequences in test data")
    if len(gen_seqs) == 0:
        raise ValueError("No sequences in generated data")

    if len(test_seqs) != len(gen_seqs):
        print(f"Warning: test seqs length = {len(test_seqs)}, gen seqs length = {len(gen_seqs)}. Aligning to min length and reporting mismatch.")
    n = min(len(test_seqs), len(gen_seqs))

    os.makedirs(args.outdir, exist_ok=True)

    print("Aggregating pair-level stats...")
    pair_to_test_indices, pair_stats, per_seq_ref_pairs = aggregate_pair_stats(test_seqs[:n], gen_seqs[:n], poi_category)

    # write pair_stats.csv
    pair_stats_with_support = {}
    for p, s in pair_stats.items():
        # include support_in_test for convenience
        pair_stats_with_support[p] = s

    write_pair_stats_csv(pair_stats_with_support, os.path.join(args.outdir, "pair_stats.csv"))

    print("Computing per-sequence OVR (allow_skip=True)...")
    per_seq_true = compute_per_sequence_ovr(test_seqs[:n], gen_seqs[:n], poi_category, allow_skip=True)
    write_per_seq_csv(per_seq_true, os.path.join(args.outdir, "per_seq_ovr_allow_skip_true.csv"))

    print("Computing per-sequence OVR (allow_skip=False)...")
    per_seq_false = compute_per_sequence_ovr(test_seqs[:n], gen_seqs[:n], poi_category, allow_skip=False)
    write_per_seq_csv(per_seq_false, os.path.join(args.outdir, "per_seq_ovr_allow_skip_false.csv"))

    summarize_and_report(pair_stats_with_support, per_seq_true, per_seq_false, args.outdir, top_k=args.top_k)


if __name__ == "__main__":
    main()
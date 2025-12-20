#!/usr/bin/env python3
"""
ovr_diagnostics_v2.py

Enhanced diagnostic tool for analyzing OVR / coverage / ordering issues between
test and generated datasets.

(This is the corrected version: fixed a KeyError in write_category_stats_csv where key names
didn't match compute_category_stats output.)
"""
from __future__ import annotations

import argparse
import os
import csv
from collections import defaultdict, Counter
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
    A_pos = [i for i, c in enumerate(gen_cats) if c == A]
    B_pos = [i for i, c in enumerate(gen_cats) if c == B]
    if len(A_pos) == 0 or len(B_pos) == 0:
        return 0, 0, 0, 0
    max_A = max(A_pos)
    min_B = min(B_pos)
    seq_satisfies = max_A < min_B
    seq_violates = 0 if seq_satisfies else 1
    total_pairs = len(A_pos) * len(B_pos)
    A_arr = np.array(sorted(A_pos), dtype=np.int64)
    total_A = A_arr.size
    violations_instances = 0
    for b in B_pos:
        idx = np.searchsorted(A_arr, b, side='right')  # number of a_pos <= b
        violations_instances += (total_A - idx)
    return 1, seq_violates, int(total_pairs), int(violations_instances)


def aggregate_pair_stats(test_seqs: List[dict], gen_seqs: List[dict], poi_category: Dict) -> Tuple[Dict, Dict, List[List[Tuple]]]:
    n = min(len(test_seqs), len(gen_seqs))
    pair_to_test_indices = defaultdict(list)
    per_seq_ref_pairs: List[List[Tuple]] = []
    for i in range(n):
        t_cats = seq_cats_order(test_seqs[i], poi_category)
        ref_pairs = extract_ref_pairs_for_sequence(t_cats)
        per_seq_ref_pairs.append(ref_pairs)
        for p in ref_pairs:
            pair_to_test_indices[p].append(i)

    pair_stats = {}
    for p, idx_list in pair_to_test_indices.items():
        pair_stats[p] = {
            "support_in_test": len(idx_list),
            "gen_support_seq_count": 0,
            "gen_violate_seq_count": 0,
            "total_instance_pairs": 0,
            "total_instance_violations": 0,
        }

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


def compute_category_stats(test_seqs: List[dict], gen_seqs: List[dict], poi_category: Dict) -> Dict:
    """
    Compute per-category:
      - test_total_occurrences, gen_total_occurrences
      - test_seq_presence (# sequences containing category), gen_seq_presence
    """
    test_total = Counter()
    gen_total = Counter()
    test_seq_presence = Counter()
    gen_seq_presence = Counter()

    # count across entire test and generated individually
    for seq in test_seqs:
        cats = seq_cats_order(seq, poi_category)
        test_total.update([c for c in cats if c is not None])
        unique = set([c for c in cats if c is not None])
        test_seq_presence.update(unique)
    for seq in gen_seqs:
        cats = seq_cats_order(seq, poi_category)
        gen_total.update([c for c in cats if c is not None])
        unique = set([c for c in cats if c is not None])
        gen_seq_presence.update(unique)

    # combine keys
    keys = sorted(set(list(test_total.keys()) + list(gen_total.keys())), key=lambda x: (x is None, x))
    stats = {}
    for k in keys:
        stats[k] = {
            "test_total_occurrences": int(test_total.get(k, 0)),
            "gen_total_occurrences": int(gen_total.get(k, 0)),
            "test_seq_presence": int(test_seq_presence.get(k, 0)),
            "gen_seq_presence": int(gen_seq_presence.get(k, 0)),
        }
    return stats


def compute_length_and_distinct_stats(seqs: List[dict], poi_category: Dict) -> Tuple[List[int], List[int]]:
    lengths = []
    distincts = []
    for seq in seqs:
        cats = seq_cats_order(seq, poi_category)
        lengths.append(len(cats))
        distincts.append(len(set([c for c in cats if c is not None])))
    return lengths, distincts


# ---------- CSV writers ----------

def write_category_stats_csv(cat_stats: Dict, outpath: str):
    # Use keys produced by compute_category_stats (test_total_occurrences, gen_total_occurrences, etc.)
    header = ["category", "test_total_occurrences", "gen_total_occurrences", "test_seq_presence", "gen_seq_presence", "occurrence_ratio_gen_to_test", "presence_ratio_gen_to_test"]
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # sort by test_total_occurrences descending (use safe get)
        for k, v in sorted(cat_stats.items(), key=lambda kv: -kv[1].get("test_total_occurrences", 0)):
            test_occ = v.get("test_total_occurrences", 0)
            gen_occ = v.get("gen_total_occurrences", 0)
            test_pres = v.get("test_seq_presence", 0)
            gen_pres = v.get("gen_seq_presence", 0)
            occ_ratio = (gen_occ / test_occ) if test_occ > 0 else float("nan")
            pres_ratio = (gen_pres / test_pres) if test_pres > 0 else float("nan")
            writer.writerow([k, test_occ, gen_occ, test_pres, gen_pres, occ_ratio, pres_ratio])


def write_length_stats_csv(lengths: List[int], distincts: List[int], outpath: str):
    header = ["metric", "value"]
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        def stats(arr):
            arr = np.array(arr, dtype=np.float64)
            return {
                "count": int(arr.size),
                "mean": float(np.mean(arr)) if arr.size>0 else float("nan"),
                "median": float(np.median(arr)) if arr.size>0 else float("nan"),
                "p25": float(np.percentile(arr,25)) if arr.size>0 else float("nan"),
                "p75": float(np.percentile(arr,75)) if arr.size>0 else float("nan"),
                "min": float(np.min(arr)) if arr.size>0 else float("nan"),
                "max": float(np.max(arr)) if arr.size>0 else float("nan"),
            }
        lstats = stats(lengths)
        dstats = stats(distincts)
        writer.writerow(["seq_length_count", lstats["count"]])
        writer.writerow(["seq_length_mean", lstats["mean"]])
        writer.writerow(["seq_length_median", lstats["median"]])
        writer.writerow(["seq_length_p25", lstats["p25"]])
        writer.writerow(["seq_length_p75", lstats["p75"]])
        writer.writerow(["seq_length_min", lstats["min"]])
        writer.writerow(["seq_length_max", lstats["max"]])
        writer.writerow(["distinct_cat_count", dstats["count"]])
        writer.writerow(["distinct_cat_mean", dstats["mean"]])
        writer.writerow(["distinct_cat_median", dstats["median"]])
        writer.writerow(["distinct_cat_p25", dstats["p25"]])
        writer.writerow(["distinct_cat_p75", dstats["p75"]])
        writer.writerow(["distinct_cat_min", dstats["min"]])
        writer.writerow(["distinct_cat_max", dstats["max"]])


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
        for (A, B), s in sorted(pair_stats.items(), key=lambda kv: -kv[1]["support_in_test"]):
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


def write_topk_csv(rows: List[Tuple], outpath: str, header: List[str]):
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="OVR diagnostics v2")
    parser.add_argument("--dataset", required=True, help="Dataset folder under data/, e.g. Istanbul")
    parser.add_argument("--data-dir", default="./data", help="Root data directory")
    parser.add_argument("--generated-file", default=None, help="Generated filename (if not <dataset>_generated.pkl).")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K lists size")
    parser.add_argument("--outdir", default="./ovr_diag_v2", help="Output directory for CSV reports")
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
        print(f"Warning: test seqs length = {len(test_seqs)}, gen seqs length = {len(gen_seqs)}. Aligning to min length.")
    n = min(len(test_seqs), len(gen_seqs))

    os.makedirs(args.outdir, exist_ok=True)

    # Category stats
    print("Computing category statistics...")
    cat_stats = compute_category_stats(test_seqs, gen_seqs, poi_category)
    write_category_stats_csv(cat_stats, os.path.join(args.outdir, "category_stats.csv"))

    # Sequence length and distinct stats
    print("Computing sequence length and distinct-category statistics...")
    test_lengths, test_distincts = compute_length_and_distinct_stats(test_seqs, poi_category)
    gen_lengths, gen_distincts = compute_length_and_distinct_stats(gen_seqs, poi_category)
    write_length_stats_csv(test_lengths, test_distincts, os.path.join(args.outdir, "seq_length_stats_test.csv"))
    write_length_stats_csv(gen_lengths, gen_distincts, os.path.join(args.outdir, "seq_length_stats_gen.csv"))

    # Pair stats
    print("Aggregating pair-level stats...")
    pair_to_test_indices, pair_stats, per_seq_ref_pairs = aggregate_pair_stats(test_seqs[:n], gen_seqs[:n], poi_category)
    write_pair_stats_csv(pair_stats, os.path.join(args.outdir, "pair_stats.csv"))

    # Per-sequence OVRs
    print("Computing per-sequence OVR (allow_skip=True)...")
    per_seq_true = compute_per_sequence_ovr(test_seqs[:n], gen_seqs[:n], poi_category, allow_skip=True)
    write_per_seq_csv(per_seq_true, os.path.join(args.outdir, "per_seq_ovr_allow_skip_true.csv"))

    print("Computing per-sequence OVR (allow_skip=False)...")
    per_seq_false = compute_per_sequence_ovr(test_seqs[:n], gen_seqs[:n], poi_category, allow_skip=False)
    write_per_seq_csv(per_seq_false, os.path.join(args.outdir, "per_seq_ovr_allow_skip_false.csv"))

    # Summaries & top-k lists
    supports = [s["support_in_test"] for s in pair_stats.values()]
    gen_supports = [s["gen_support_seq_count"] for s in pair_stats.values()]
    total_ref_instances = sum(supports)
    total_gen_matched_instances = sum(gen_supports)
    overall_coverage = total_gen_matched_instances / total_ref_instances if total_ref_instances > 0 else float("nan")

    def mean_ignore_nan(arr):
        a = np.array(arr, dtype=np.float64)
        valid = ~np.isnan(a)
        return float(np.mean(a[valid])) if valid.sum() > 0 else float("nan")

    ovr_true = mean_ignore_nan(per_seq_true)
    ovr_false = mean_ignore_nan(per_seq_false)

    print("========== OVR DIAGNOSTICS SUMMARY (v2) ==========")
    print(f"Aligned sequence count (used): {n}")
    print(f"Total reference pair instances in test (sum supports): {total_ref_instances}")
    print(f"Total matched reference instances in generated (coverage numerator): {total_gen_matched_instances}")
    print(f"Overall coverage (matched / total): {overall_coverage:.4f}")
    print(f"Sequence-level OVR (allow_skip=True)  : {ovr_true:.6f}")
    print(f"Sequence-level OVR (allow_skip=False) : {ovr_false:.6f}")
    print("")

    # top-k missing (support high but gen_support == 0)
    missing = [(p, s["support_in_test"]) for p, s in pair_stats.items() if s["gen_support_seq_count"] == 0]
    missing_sorted = sorted(missing, key=lambda x: -x[1])[: args.top_k]
    rows = [(i+1, p[0], p[1], sup) for i, (p, sup) in enumerate(missing_sorted)]
    write_topk_csv(rows, os.path.join(args.outdir, "topk_missing_pairs.csv"),
                  ["rank", "A", "B", "support_in_test"])

    # top-k low coverage (support high, coverage low but gen_support>0)
    low_cov = []
    for p, s in pair_stats.items():
        sup = s["support_in_test"]
        gsup = s["gen_support_seq_count"]
        if sup > 0 and gsup > 0:
            cov = gsup / sup
            low_cov.append((p, sup, gsup, cov))
    low_cov_sorted = sorted(low_cov, key=lambda x: (x[3], -x[1]))[: args.top_k]
    rows = [(i+1, p[0], p[1], sup, gsup, cov) for i, (p, sup, gsup, cov) in enumerate(low_cov_sorted)]
    write_topk_csv(rows, os.path.join(args.outdir, "topk_low_coverage_pairs.csv"),
                  ["rank", "A", "B", "support_in_test", "gen_support", "coverage"])

    # top-k by instance-level violation rate (requires total_instance_pairs>0)
    inst_list = []
    for p, s in pair_stats.items():
        tot = s["total_instance_pairs"]
        viol = s["total_instance_violations"]
        if tot > 0:
            inst_rate = viol / tot
            inst_list.append((p, s["support_in_test"], s["gen_support_seq_count"], tot, viol, inst_rate))
    inst_sorted = sorted(inst_list, key=lambda x: (-x[5], -x[1]))[: args.top_k]
    rows = [(i+1, p[0], p[1], sup, gsup, tot, viol, rate) for i, (p, sup, gsup, tot, viol, rate) in enumerate(inst_sorted)]
    write_topk_csv(rows, os.path.join(args.outdir, "topk_high_instance_violation_pairs.csv"),
                  ["rank", "A", "B", "support_in_test", "gen_support", "total_instance_pairs", "total_instance_violations", "instance_violation_rate"])

    print(f"Detailed CSVs written to: {args.outdir}")
    print("=============================================")


if __name__ == "__main__":
    main()
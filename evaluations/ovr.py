#!/usr/bin/env python3
"""
OVR (per-test-sequence reference pairs → per-generated-sequence violation rate)

Definition implemented:
- For each test sequence t:
    - extract reference strict pairs R_t = {(A,B), ...} where in t all A are before all B
      (i.e., max_pos(A) < min_pos(B)).
- For the corresponding generated sequence g (same index):
    - for each (A,B) in R_t, consider all POI-instance pairs (a,b) where a in g has category A and b in g has category B.
    - compute violation rate for that pair = (# of (a,b) with pos(a) > pos(b)) / (total # of (a,b))
      (if either A or B absent in g, behavior controlled by allow_skip; default = skip this pair)
    - aggregate per-test-sequence violation as the mean of its reference-pair violation rates (equal weight per pair).
- Dataset-level OVR = mean over sequences (ignoring sequences that had no considered pairs).

Functions:
- dataset_ovr_by_test_pairs(test_seqs, gen_seqs, poi_category, allow_skip=True, skip_nan=True)
    returns float OVR (or np.nan if no considered instances)

Only uses numpy + torch.
"""
from typing import Dict, List, Tuple
import numpy as np
import torch
import math


def _map_poi_to_cat_safe(poi, poi_category: Dict):
    if poi in poi_category:
        return poi_category[poi]
    try:
        key = int(poi)
        return poi_category.get(key, None)
    except Exception:
        return poi_category.get(str(poi), None)


def _seq_cats_order(seq: dict, poi_category: Dict) -> List:
    """Return ordered list of categories for the sequence (one per checkin)."""
    if "checkins" in seq and seq["checkins"] is not None:
        cats = []
        for poi in seq["checkins"]:
            cats.append(_map_poi_to_cat_safe(poi, poi_category))
        return cats
    elif "marks" in seq and seq["marks"] is not None:
        return list(seq["marks"])
    else:
        return []


def _min_max_positions(cats: List) -> Tuple[Dict, Dict]:
    """Return min_pos and max_pos dicts for categories present in cats."""
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


def _extract_reference_pairs_for_sequence(test_cats: List) -> List[Tuple]:
    """
    For a single test sequence (category list), return list of reference strict ordered pairs (u,v)
    where all u appear before all v in this sequence.
    Pairs returned as (u, v) with Python ordering preserved (u <= v is not required here; the
    tuple (u,v) encodes direction: u->v).
    """
    min_pos, max_pos = _min_max_positions(test_cats)
    keys = list(min_pos.keys())
    pairs = []
    L = len(keys)
    for i in range(L):
        for j in range(i + 1, L):
            a = keys[i]
            b = keys[j]
            if a is None or b is None:
                continue
            # check strict orders
            if max_pos[a] < min_pos[b]:
                pairs.append((a, b))  # a -> b
            elif max_pos[b] < min_pos[a]:
                pairs.append((b, a))  # b -> a
            else:
                # mixed / interleaved -> not a reference strict pair
                continue
    return pairs


def _violation_rate_for_pair_in_generated(gen_cats: List, A, B) -> Tuple[int, int]:
    """
    For generated sequence category list gen_cats and a reference ordered pair (A->B),
    compute (violations, total_pairs) where:
      - total_pairs = #A_occurrences * #B_occurrences
      - violations = # of pairs (pos_a, pos_b) with pos_a > pos_b
    If either list empty, returns (0, 0).
    Implementation is O(|A| log |A| + |B| log |A|) using numpy searchsorted.
    """
    # collect positions
    A_pos = [i for i, c in enumerate(gen_cats) if c == A]
    B_pos = [i for i, c in enumerate(gen_cats) if c == B]
    if len(A_pos) == 0 or len(B_pos) == 0:
        return 0, 0
    A_arr = np.array(sorted(A_pos), dtype=np.int64)
    total_A = A_arr.size
    # count violations: for each b_pos, number of a_pos > b_pos
    # idx = np.searchsorted(A_arr, b_pos, side='right') -> number of a_pos <= b_pos
    # then greater = total_A - idx
    violations = 0
    for b in B_pos:
        idx = np.searchsorted(A_arr, b, side='right')
        violations += (total_A - idx)
    total_pairs = total_A * len(B_pos)
    return int(violations), int(total_pairs)


def sequence_ovr_by_test_reference(test_seq: dict, gen_seq: dict, poi_category: Dict, allow_skip: bool = True) -> float:
    """
    Compute per-sequence OVR:
      - extract reference pairs from test_seq;
      - for each reference pair (A->B), compute violation rate in gen_seq as violations/total_pairs;
      - if allow_skip True, skip pairs where gen_seq has 0 occurrences of A or B;
        otherwise treat missing as full violation (violations = total_pairs where total_pairs computed as 0 -> treated as 1)
      - return mean violation rate over considered pairs (or np.nan if none considered)
    """
    test_cats = _seq_cats_order(test_seq, poi_category)
    gen_cats = _seq_cats_order(gen_seq, poi_category)
    ref_pairs = _extract_reference_pairs_for_sequence(test_cats)
    if len(ref_pairs) == 0:
        return float("nan")
    rates = []
    for (A, B) in ref_pairs:
        violations, total_pairs = _violation_rate_for_pair_in_generated(gen_cats, A, B)
        if total_pairs == 0:
            if allow_skip:
                continue
            else:
                # treat as full violation (1.0) for this pair
                rates.append(1.0)
        else:
            rates.append(violations / total_pairs)
    if len(rates) == 0:
        return float("nan")
    return float(np.mean(rates))


def dataset_ovr_by_test_pairs(test_seqs: List[dict], gen_seqs: List[dict], poi_category: Dict, allow_skip: bool = True, skip_nan: bool = True) -> float:
    """
    Compute dataset-level OVR per your definition:
      - For each test sequence i, compute per-sequence OVR between test_seqs[i] and gen_seqs[i]
        using sequence_ovr_by_test_reference(...).
      - Then average per-sequence OVRs across sequences. If skip_nan True, ignore sequences with nan
        (no reference pairs or no considered pairs); otherwise include nan in mean resulting in nan.
    Returns float (OVR) or np.nan.
    """
    if len(test_seqs) != len(gen_seqs):
        raise ValueError("test_seqs and gen_seqs must have same length (corresponding sequences).")

    per_seq = []
    for t_seq, g_seq in zip(test_seqs, gen_seqs):
        r = sequence_ovr_by_test_reference(t_seq, g_seq, poi_category, allow_skip=allow_skip)
        per_seq.append(r)

    arr = np.array(per_seq, dtype=np.float64)
    if skip_nan:
        valid = ~np.isnan(arr)
        if valid.sum() == 0:
            return float("nan")
        return float(np.mean(arr[valid]))
    else:
        return float(np.nanmean(arr))  # will be nan if all nan
"""Small, shared invariants for reproducible sampling and fail-closed merging."""
import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile

import numpy as np
import torch

from evaluations.ovr import _extract_reference_pairs_for_sequence, _seq_cats_order


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_tag(value):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in (".", ".."):
        raise ValueError(f"Invalid output tag: {value!r}")
    return value


def strict_test_matrix(sequence, poi_category, category_mapping):
    """Use exactly the reference-pair definition used by OVR evaluation.

    Missing categories never produce edges. Original PKLs/encodings are untouched.
    """
    mapping = {int(k): int(v) for k, v in category_mapping.items()}
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("Category mapping must cover 0..C-1")
    cats = _seq_cats_order(sequence, poi_category)
    if any(cat is None or int(cat) not in mapping for cat in cats):
        raise ValueError("Unknown POI/category in reference sequence")
    matrix = torch.zeros((len(mapping), len(mapping)), dtype=torch.float32)
    for a, b in _extract_reference_pairs_for_sequence(cats):
        matrix[mapping[int(a)], mapping[int(b)]] = 1
    return matrix


def seed_sampling(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def empty_generated_record(batch, index):
    """Serialize an empty trajectory without inventing events or losing conditions."""
    record = dict(arrival_times=np.empty(0, dtype=np.float32),
                  marks=np.empty(0, dtype=np.int64), checkins=np.empty(0, dtype=np.int64),
                  gps=[], generation_status="empty_temporal")
    for condition in range(1, 7):
        record[f"condition{condition}"] = np.empty(0, dtype=np.int64)
        record[f"condition{condition}_indicator"] = getattr(
            batch, f"condition{condition}_indicator")[index].detach().cpu().numpy().copy()
    return record


def decode_preserving_empty(task, time_samples, gps_dict, **sample_kwargs):
    """Retain the original mixed batch shape/RNG layout, bypass only all-empty batches."""
    empty = time_samples.unpadded_length == 0
    if bool(empty.all()):
        return [empty_generated_record(time_samples, i) for i in range(time_samples.batch_size)]
    dd = task.discrete_diffusion
    geometry = getattr(dd, 'geometry_config', None)
    geometry_enabled = getattr(geometry, 'geometry_refinement', 'off') == 'same_category_v1'
    before = getattr(dd, 'projection_call_count', 0) if geometry_enabled else 0
    samples = dd.sample_fast(time_samples.to(task.device), **sample_kwargs).to_seq_list(gps_dict)
    for index in torch.where(empty)[0].tolist():
        samples[index] = empty_generated_record(time_samples, index)
    if geometry_enabled:
        from geometry_projection import refine_records
        samples, dd.last_geometry_stats = refine_records(samples, dd, dd.geometry_reference, geometry,
            seed=dd.geometry_seed, global_start=dd.geometry_global_start,
            projection_executed=dd.projection_call_count > before)
    return samples


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False, suffix=".tmp") as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_torch(path, value):
    """Atomically publish a NEW file, never silently replace an existing result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp") as stream:
        temporary = Path(stream.name)
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)  # fails if another process has already published it
    finally:
        temporary.unlink(missing_ok=True)


def validate_sequences(sequences, poi_category=None):
    for index, sequence in enumerate(sequences):
        times = np.asarray(sequence["arrival_times"], dtype=np.float64)
        pois = sequence["checkins"]
        if len(times) != len(pois) or not np.isfinite(times).all():
            raise ValueError(f"Invalid generated sequence {index}")
        if np.any(times < 0) or np.any(times >= 24) or np.any(np.diff(times) <= 0):
            raise ValueError(f"Invalid generated times at sequence {index}")
        if poi_category is not None and any(int(poi) not in poi_category for poi in pois):
            raise ValueError(f"Unknown generated POI at sequence {index}")


def validate_part(part, metadata, rank, indices):
    if part.get("metadata") != metadata or part.get("rank") != rank:
        raise ValueError("Shard fingerprint/parameters/rank mismatch")
    if part.get("test_indices") != indices or len(part["sequences"]) != len(indices):
        raise ValueError("Shard indices/count mismatch")
    if float(np.asarray(part["t_max"]).item()) != 24.0:
        raise ValueError("Shard t_max mismatch")
    validate_sequences(part["sequences"])
    if "empty_test_indices" in part:
        actual = [i for i, seq in zip(indices, part["sequences"]) if len(seq["checkins"]) == 0]
        if part["empty_test_indices"] != actual:
            raise ValueError("Empty trajectory indices do not match the output")
        temporal = part.get("temporal_empty_test_indices", [])
        if temporal != sorted(set(temporal)) or not set(temporal).issubset(actual):
            raise ValueError("Invalid temporal-empty trajectory indices")

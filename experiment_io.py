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

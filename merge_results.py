"""Fail-closed merge: retain shards and never publish incomplete generations."""
import argparse
import math
from pathlib import Path
import torch
from experiment_io import publish_torch, safe_tag, sha256_file, validate_part


def merge_parts(data_name, run_id, world_size=4, output_tag=None, data_dir="data", expected_count=None):
    if world_size < 1:
        raise ValueError("world_size must be positive")
    tag = safe_tag(output_tag or run_id)
    base = Path(data_dir) / safe_tag(data_name)
    paths = [base / f"{data_name}_{tag}_generated_part{rank}.pkl" for rank in range(world_size)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing shards; nothing merged: {missing}")
    parts = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    metadata = parts[0].get("metadata")
    if not metadata:
        raise ValueError("Legacy shards lack validation metadata; regenerate with the current sampler")
    total = metadata["total_samples"]
    if expected_count is not None and total != expected_count:
        raise ValueError("Unexpected total sample count")
    if (metadata["world_size"], metadata["run_id"], metadata["output_tag"], metadata["data_name"]) != (world_size, run_id, tag, data_name):
        raise ValueError("Requested experiment does not match shards")
    if metadata["dataset_sha256"] != sha256_file(base / f"{data_name}_test.pkl"):
        raise ValueError("Test dataset changed since sampling")
    sequences, indices = [], []
    chunk = math.ceil(total / world_size)
    start_index = metadata.get("start_index", 0)
    for rank, part in enumerate(parts):
        expected_indices = list(range(start_index + min(rank * chunk, total), start_index + min((rank + 1) * chunk, total)))
        validate_part(part, metadata, rank, expected_indices)
        sequences.extend(part["sequences"])
        indices.extend(part["test_indices"])
    if indices != list(range(start_index, start_index + total)) or len(sequences) != total:
        raise ValueError("Missing, duplicate, or out-of-order test indices")
    destination = base / f"{data_name}_{tag}_generated.pkl"
    merged = dict(sequences=sequences, t_max=24.0, test_indices=indices, metadata=metadata,
                  empty_test_indices=[i for i, seq in zip(indices, sequences) if len(seq['checkins']) == 0],
                  temporal_empty_test_indices=[i for part in parts for i in part.get('temporal_empty_test_indices', [])],
                  eligible_projection_samples=sum(part.get('eligible_projection_samples', len(part['sequences'])) for part in parts),
                  projection_calls=sum(p.get("projection_calls", 0) for p in parts),
                  elapsed_seconds=max(p.get("elapsed_seconds", 0) for p in parts),
                  shard_sha256=[sha256_file(path) for path in paths])
    if any('distance_projection_diagnostics' in part for part in parts):
        merged['distance_projection_diagnostics'] = [dict(entry, rank=rank)
            for rank, part in enumerate(parts) for entry in part.get('distance_projection_diagnostics', [])]
    if destination.exists():
        previous = torch.load(destination, map_location="cpu", weights_only=False)
        if any(previous.get(k) != merged[k] for k in ("metadata", "test_indices", "shard_sha256")):
            raise FileExistsError("Existing merged result belongs to different inputs")
        if len(previous["sequences"]) != total:
            raise ValueError("Existing merged result is incomplete")
    else:
        publish_torch(destination, merged)
    print(f"Verified {total} samples: {destination}; all source shards retained")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--data_name", required=True)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--output_tag")
    parser.add_argument("--expected_count", type=int)
    args = parser.parse_args()
    merge_parts(args.data_name, args.run_id, args.world_size, args.output_tag, expected_count=args.expected_count)

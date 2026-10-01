"""Isolated singleton regression check using an existing, completed preflight model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiment_io import atomic_json, safe_tag, sha256_file, validate_sequences
from tools.run_newyork_ood import PROJECTION, DATASET, DATA_HASHES, code_fingerprint
from tools.continue_newyork_sampling import verify_compatible_sources, PATCHED_FILE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--preflight-run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    source = Path(args.source_experiment).resolve()
    source_manifest = json.loads((source / "manifest.json").read_text())
    verify_compatible_sources(source_manifest["code_sha256"], code_fingerprint())
    preflight_id = safe_tag(args.preflight_run_id)
    preflight = source.parent / preflight_id
    if json.loads((preflight / "status.json").read_text())["state"] != "complete":
        raise RuntimeError("Validation requires a completed preflight")
    checkpoint = preflight / "final.ckpt"
    patch_hash = sha256_file(ROOT / PATCHED_FILE)
    folder = ROOT / "validation"
    folder.mkdir(exist_ok=True)
    receipt_path = folder / "singleton-validation.json"
    receipt = dict(state="running", patched_file_sha256=patch_hash,
                   validation_script_sha256=sha256_file(__file__),
                   source_manifest_sha256=sha256_file(source / "manifest.json"),
                   validation_checkpoint_sha256=sha256_file(checkpoint),
                   started_at=time.time(), projection=PROJECTION)
    atomic_json(receipt_path, receipt)
    try:
        with (folder / "unit-tests.log").open("w") as log:
            subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        for method in ("native", "projection"):
            tag = f"{preflight_id}_sg{patch_hash[:8]}_{method}"
            command = [sys.executable, "-B", "sample.py", "--run_id", preflight_id,
                       "--checkpoint", str(checkpoint), "--output_tag", tag,
                       "--seed", "135398", "--batch_size", "1", "--max_samples", "1",
                       "--constraint_source", "strict_test"]
            if args.resume:
                command += ["--resume"]
            if method == "projection":
                command += ["--use_constraint_projection", "--use_gumbel_softmax"]
                for key, value in PROJECTION.items():
                    command += [f"--{key}", str(value)]
            print(f"Validating singleton {method}", flush=True)
            start = time.monotonic()
            with (folder / f"singleton-{method}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            path = ROOT / f"data/{DATASET}/{DATASET}_{tag}_generated_part0.pkl"
            part = torch.load(path, map_location="cpu", weights_only=False)
            if part["test_indices"] != [0] or len(part["sequences"]) != 1:
                raise RuntimeError("Singleton output count/index mismatch")
            if method == "native" and part["projection_calls"] != 0:
                raise RuntimeError("Native path unexpectedly used projection")
            if method == "projection" and part["projection_calls"] <= 0:
                raise RuntimeError("Projection was not executed")
            validate_sequences(part["sequences"])
            receipt[f"singleton_{method}_samples"] = 1
            receipt[f"singleton_{method}_calls"] = part["projection_calls"]
            receipt[f"singleton_{method}_seconds"] = time.monotonic() - start
            receipt[f"singleton_{method}_output_sha256"] = sha256_file(path)
            atomic_json(receipt_path, receipt)
        for split, expected in DATA_HASHES.items():
            relative = f"data/{DATASET}/{DATASET}_{split}.pkl"
            for root in (ROOT, source.parent.parent):
                if sha256_file(root / relative) != expected:
                    raise RuntimeError("Dataset fingerprint changed")
        receipt.update(state="passed", completed_at=time.time())
    except BaseException as exc:
        receipt.update(state="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        atomic_json(receipt_path, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()

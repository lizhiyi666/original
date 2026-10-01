"""Versioned inference continuation; never edit the active training checkout.

Run in a separate deployment snapshot with a read-only-by-convention copy of the two
input PKLs and a `wandb` symlink to the training checkout. Wait for the original
controller to stop at the specifically identified singleton benchmark failure, then
use its completed 1000-epoch checkpoint. Original manifests/checkpoints stay intact.
"""
import argparse
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_newyork_ood import Experiment, ROOT as RUNNER_ROOT, code_fingerprint, disk_guard
from experiment_io import atomic_json, safe_tag, sha256_file
import torch
import wandb

PATCHED_FILE = "add_thin/distributions/intensities.py"


def verify_compatible_sources(training_hashes, sampling_hashes):
    changed = {name for name in training_hashes.keys() | sampling_hashes.keys()
               if training_hashes.get(name) != sampling_hashes.get(name)}
    if changed != {PATCHED_FILE}:
        raise RuntimeError(f"Expected only the approved singleton inference fix; changed={sorted(changed)}")


class SamplingContinuation(Experiment):
    def __init__(self, args):
        self.source_directory = Path(args.source_experiment).resolve()
        self.source_root = self.source_directory.parent.parent
        if ROOT == self.source_root or ROOT != RUNNER_ROOT:
            raise ValueError("Continuation must use a separate deployment directory")
        self.source_manifest = json.loads((self.source_directory / "manifest.json").read_text())
        self.source_manifest_hash = sha256_file(self.source_directory / "manifest.json")
        args.run_id = self.source_manifest["run_id"]
        args.stage = "sample"
        args.preflight_id = None
        super().__init__(args)
        self.phase = "waiting-for-training"

    def wait_for_source(self):
        import fcntl
        last = None
        while True:
            if sha256_file(self.source_directory / "manifest.json") != self.source_manifest_hash:
                raise RuntimeError("Original training manifest changed while waiting")
            state = json.loads((self.source_directory / "status.json").read_text())
            marker = (state.get("state"), state.get("phase"))
            if marker != last:
                self.status("waiting", source_state=state)
                last = marker
            if state["state"] == "complete":
                raise RuntimeError("Original pipeline already completed; duplicate sampling refused")
            if state["state"] == "failed":
                if state["phase"] != "projection-benchmark":
                    raise RuntimeError(f"Original pipeline failed outside the approved recovery point: {state}")
                command = json.loads((self.source_directory / "command-projection-benchmark.json").read_text())
                text = Path(command["log"]).read_text(errors="replace")
                if not all(value in text for value in (
                    "IndexError: too many indices for tensor of dimension 0",
                    "sequence_len[:, None]", "distributions/intensities.py")):
                    raise RuntimeError("Benchmark failure does not match the verified singleton bug")
                with (self.source_directory / "pipeline.lock").open("r+") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        time.sleep(5)
                        continue
                    return
            if not self.args.wait:
                raise RuntimeError("Training is still running; use --wait to defer safely")
            try:
                os.kill(int(state["pid"]), 0)
            except ProcessLookupError:
                refreshed = json.loads((self.source_directory / "status.json").read_text())
                if refreshed.get("state") == "running":
                    raise RuntimeError("Original controller exited without recording a terminal state")
            time.sleep(30)

    def precheck(self):
        self.wait_for_source()
        self.phase = "continuation-precheck"
        disk_guard()
        source = self.source_manifest
        if (source["profile"] != "formal" or source["epochs"] != 1000
                or source["train_batch_size"] != 64 or source["sample_count"] != 2108
                or source["po_loss_weight"] != 0 or source["seed"] != 135398):
            raise RuntimeError("Original training profile does not match the approved experiment")
        if torch.cuda.device_count() != 2:
            raise RuntimeError("Two visible GPUs required for full sampling")
        self.run([sys.executable, "-m", "pip", "check"], "continuation-pip-check")
        current_hashes = code_fingerprint()
        verify_compatible_sources(source["code_sha256"], current_hashes)
        for name, expected in source["code_sha256"].items():
            if sha256_file(self.source_root / name) != expected:
                raise RuntimeError(f"Original source changed: {name}")
        for split, expected in source["data_sha256"].items():
            relative = f"data/{source['dataset']}/{source['dataset']}_{split}.pkl"
            if any(sha256_file(root / relative) != expected for root in (ROOT, self.source_root)):
                raise RuntimeError("Training/sampling input fingerprints differ")
        versions = {name: importlib.metadata.version(name) for name in source["packages"]}
        if versions != source["packages"]:
            raise RuntimeError("Python package versions changed")
        receipt_path = Path(self.args.validation_receipt).resolve()
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("state") != "passed" or receipt.get("singleton_native_samples") != 1
                or receipt.get("singleton_projection_calls", 0) <= 0
                or receipt.get("source_manifest_sha256") != self.source_manifest_hash
                or receipt.get("patched_file_sha256") != current_hashes[PATCHED_FILE]):
            raise RuntimeError("Singleton fix has not passed isolated validation")
        self.entity = wandb.Api(timeout=30).default_entity
        if self.entity != source["entity"]:
            raise RuntimeError("W&B account differs from original experiment")
        original_run = wandb.Api(timeout=30).run(f"{self.entity}/Marionette/{self.run_id}")
        if original_run.state != "finished":
            raise RuntimeError("W&B does not confirm completed training")
        self.manifest = deepcopy(source)
        self.manifest.update(
            code_sha256=current_hashes, training_code_sha256=source["code_sha256"],
            source_experiment=str(self.source_directory),
            source_manifest_sha256=self.source_manifest_hash,
            continuation_script_sha256=sha256_file(__file__),
            validation_receipt_sha256=sha256_file(receipt_path),
            continuation_reason="User-approved singleton count-axis fix; training untouched")
        path = self.directory / "manifest.json"
        if path.exists():
            if not self.args.resume or json.loads(path.read_text()) != self.manifest:
                raise RuntimeError("Existing continuation requires an exactly matching --resume")
        elif self.args.resume:
            raise RuntimeError("Cannot resume a continuation that has not started")
        else:
            atomic_json(path, self.manifest)
        # Check epoch, optimizer/scheduler state, dataset, seed, and batch size before sampling.
        self.checkpoint(require_complete=True)
        self.status("running", source_experiment=str(self.source_directory))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--validation-receipt", required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    experiment = SamplingContinuation(args)
    import fcntl
    with (experiment.directory / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            experiment.execute()
        except BaseException as exc:
            experiment.status("failed", error_type=type(exc).__name__, error=str(exc))
            raise


if __name__ == "__main__":
    main()

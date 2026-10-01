"""Persistent, fail-closed NewYork OOD experiment. Run inside tmux.

Examples:
  python tools/run_newyork_ood.py --run-id nyood-preflight --stage preflight
  python tools/run_newyork_ood.py --run-id nyood-1000 --preflight-id nyood-preflight
  python tools/run_newyork_ood.py --run-id nyood-1000 --resume
  python tools/run_newyork_ood.py --run-id nyood-1000 --stage sample --resume
"""
import argparse
from contextlib import ExitStack
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from experiment_io import atomic_json, safe_tag, sha256_file, validate_sequences
from evaluate_utils import get_run_data
from merge_results import merge_parts

DATASET = "NewYork_PO1_OOD"
SEED = 135398
DATA_HASHES = {
    "train": "afdadf2d60329afc5dfb86aa792f5a359455f8de720eab8d56e23c376593afec",
    "test": "6f127b1bb0562500e1e679ac564fdaab778ccb76fecc7087be63f6590df077ab",
}
PROJECTION = dict(projection_last_k_steps=40, projection_frequency=4,
                  projection_outer_iters=200, projection_inner_iters=100,
                  projection_tau=0, projection_lambda=1.0, projection_eta=1.0,
                  projection_mu=1.0, projection_mu_max=1000.0, projection_mu_alpha=2.0,
                  projection_delta_tol=1e-6, gumbel_temperature=0.1,
                  projection_existence_weight=5.0)


def training_profile(preflight):
    # 3160 sequences / batch 64 => 50 batches per epoch (last batch is partial).
    # User-approved warm-up: 20 full epochs, exactly 1000 batches, always a new run.
    return dict(epochs=20 if preflight else 1000, train_batch_size=64,
                limit_train_batches=50 if preflight else None,
                expected_train_batches=1000 if preflight else 50000)


def disk_guard():
    if shutil.disk_usage(ROOT).free < 5 * 1024**3:
        raise RuntimeError("Less than 5 GiB free; stopping before the next stage")


def code_fingerprint():
    paths = list(ROOT.glob("*.py"))
    for folder in ("add_thin", "discrete_diffusion", "evaluations"):
        paths.extend((ROOT / folder).rglob("*.py"))
    paths.extend(ROOT / name for name in (
        "config/train.yaml", "config/data/NewYork_PO1_OOD.yaml",
        "config/model/Marionette.yaml", "config/task/density.yaml", "config/hydra/default.yaml"))
    paths.append(Path(__file__))
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(set(paths))}


class Experiment:
    def __init__(self, args):
        self.args = args
        self.run_id = safe_tag(args.run_id)
        self.preflight = args.stage == "preflight"
        self.directory = ROOT / "experiment_runs" / self.run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.phase = "precheck"
        self.entity = None
        self.command_number = 0
        self.manifest = None

    def status(self, state, **extra):
        atomic_json(self.directory / "status.json", dict(
            state=state, phase=self.phase, run_id=self.run_id, updated_at=time.time(),
            pid=os.getpid(), **extra))
        print(json.dumps(dict(state=state, phase=self.phase, **extra)), flush=True)

    def run(self, command, label, env=None):
        self.command_number += 1
        log = self.directory / f"{label}-{time.time_ns()}.log"
        print(f"Running {label}; log={log}", flush=True)
        atomic_json(self.directory / f"command-{label}.json", dict(command=command, log=str(log)))
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(command, cwd=ROOT, env=env or os.environ.copy(),
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    def precheck(self):
        disk_guard()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError("Exactly two visible CUDA GPUs are required")
        self.run([sys.executable, "-m", "pip", "check"], "pip-check")
        for split, expected in DATA_HASHES.items():
            path = ROOT / f"data/{DATASET}/{DATASET}_{split}.pkl"
            if sha256_file(path) != expected:
                raise RuntimeError(f"{split} dataset fingerprint changed")
        self.entity = wandb.Api(timeout=30).default_entity
        if not self.entity:
            raise RuntimeError("W&B account has no default entity")
        self.manifest = dict(
            run_id=self.run_id, profile="preflight" if self.preflight else "formal",
            dataset=DATASET, data_sha256=DATA_HASHES, code_sha256=code_fingerprint(),
            seed=SEED, **training_profile(self.preflight),
            sample_batch_size=4 if self.preflight else 64,
            po_loss_weight=0, temporal_steps=100, spatial_steps_effective=256,
            sample_count=4 if self.preflight else 2108, world_size=2,
            constraint_source="strict_test", projection=PROJECTION,
            entity=self.entity, project="Marionette",
            packages={name: importlib.metadata.version(name) for name in
                      ("torch", "pytorch-lightning", "numpy", "wandb", "hydra-core")})
        path = self.directory / "manifest.json"
        if path.exists():
            if not self.args.resume:
                raise FileExistsError("Experiment exists; use --resume explicitly")
            if json.loads(path.read_text()) != self.manifest:
                raise RuntimeError("Resume refused: code, data, or experiment settings changed")
        elif self.args.resume:
            raise FileNotFoundError("Cannot resume an experiment without its original manifest")
        else:
            if not self.preflight:
                if not self.args.preflight_id:
                    raise ValueError("A completed --preflight-id is required for a new formal run")
                directory = ROOT / "experiment_runs" / safe_tag(self.args.preflight_id)
                proof = json.loads((directory / "manifest.json").read_text())
                state = json.loads((directory / "status.json").read_text())
                if state["state"] != "complete" or proof["profile"] != "preflight":
                    raise RuntimeError("Preflight did not finish successfully")
                if any(proof[k] != self.manifest[k] for k in ("code_sha256", "data_sha256", "packages")):
                    raise RuntimeError("Preflight is stale for the current code/data/environment")
            atomic_json(path, self.manifest)
        self.status("running")

    def checkpoint(self, require_complete=False):
        _, _, run_path = get_run_data(self.run_id, ROOT / "wandb")
        path = Path(run_path) / "checkpoints/last.ckpt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if len(checkpoint.get("optimizer_states", [])) != 2:
            raise RuntimeError("Checkpoint does not contain both optimizer states")
        if len(checkpoint.get("lr_schedulers", [])) != 2:
            raise RuntimeError("Checkpoint does not contain both scheduler states")
        if require_complete and int(checkpoint["epoch"]) != self.manifest["epochs"] - 1:
            raise RuntimeError("Checkpoint has not completed the requested epochs")
        config = OmegaConf.load(Path(run_path) / "config_hydra.yaml")
        if (config.data.name != DATASET or float(config.task.po_loss_weight) != 0
                or int(config.seed) != SEED or int(config.data.batch_size) != self.manifest["train_batch_size"]):
            raise RuntimeError("Checkpoint config does not match experiment")
        return path, Path(run_path)

    def train(self):
        self.phase = "train"
        disk_guard()
        self.status("running")
        arguments = ["train.py", f"data={DATASET}", "task.po_loss_weight=0",
                     "model.use_constraint_projection=false", "trainer.devices=[0]",
                     f"trainer.max_epochs={self.manifest['epochs']}",
                     f"data.batch_size={self.manifest['train_batch_size']}", f"seed={SEED}",
                     "mode=online", f"entity={self.entity}", f"id={self.run_id}",
                     f"name={self.run_id}", f"group={self.run_id}", "run_dir=."]
        if self.preflight:
            arguments += [f"+trainer.limit_train_batches={self.manifest['limit_train_batches']}",
                          "+trainer.enable_progress_bar=false"]
        else:
            arguments += ["+trainer.enable_progress_bar=false"]
        if self.args.resume:
            checkpoint, _ = self.checkpoint()
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if int(saved["epoch"]) >= self.manifest["epochs"] - 1:
                self.checkpoint(require_complete=True)
                print("Requested epochs already complete; skipping training")
                return
            arguments += [f"trainer.resume_from_checkpoint={checkpoint}"]
        self.run([sys.executable, *arguments], "train")
        self.checkpoint(require_complete=True)
        remote = wandb.Api(timeout=30).run(f"{self.entity}/Marionette/{self.run_id}")
        if remote.state != "finished":
            raise RuntimeError(f"W&B training run is not finished: {remote.state}")

    def freeze_checkpoint(self):
        checkpoint, run_path = self.checkpoint(require_complete=True)
        destination = self.directory / "final.ckpt"
        if destination.exists():
            if sha256_file(destination) != sha256_file(checkpoint):
                raise RuntimeError("Final checkpoint differs from completed training checkpoint")
        else:
            temporary = self.directory / "final.ckpt.tmp"
            shutil.copy2(checkpoint, temporary)
            os.replace(temporary, destination)
        shutil.copy2(run_path / "config_hydra.yaml", self.directory / "config_hydra.yaml")
        atomic_json(self.directory / "checkpoint.json", dict(path=str(destination),
                    sha256=sha256_file(destination), epoch=self.manifest["epochs"]-1))
        return destination

    def sample_command(self, method, checkpoint, tag, rank, *, benchmark=False):
        cmd = [sys.executable, "sample.py", "--run_id", self.run_id, "--checkpoint", str(checkpoint),
               "--output_tag", tag, "--rank", str(rank), "--world_size", "1" if benchmark else "2",
               "--seed", str(SEED), "--batch_size", "1" if benchmark else str(self.manifest["sample_batch_size"]),
               "--constraint_source", "strict_test"]
        if benchmark:
            cmd += ["--max_samples", "1"]
        elif self.preflight:
            cmd += ["--max_samples", "4"]
        if self.args.resume:
            cmd += ["--resume"]
        if method == "projection":
            options = dict(PROJECTION)
            if self.preflight and not benchmark:
                # Isolated functional smoke test, never used for formal results.
                options.update(projection_outer_iters=2, projection_inner_iters=2,
                               projection_last_k_steps=4)
            cmd += ["--use_constraint_projection", "--use_gumbel_softmax"]
            for key, value in options.items():
                cmd += [f"--{key}", str(value)]
        return cmd

    def sample(self, method, checkpoint):
        self.phase = f"sample-{method}"
        disk_guard()
        self.status("running")
        tag = f"{self.run_id}_{method}"
        started = time.monotonic()
        with ExitStack() as stack:
            processes = []
            try:
                for rank in range(2):
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank))
                    log = stack.enter_context((self.directory / f"{method}-rank{rank}-{time.time_ns()}.log").open("w"))
                    processes.append(subprocess.Popen(self.sample_command(method, checkpoint, tag, rank),
                                     cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
                while any(p.poll() is None for p in processes):
                    if any(p.poll() not in (None, 0) for p in processes):
                        raise RuntimeError("Sampling worker failed; merge/evaluation refused")
                    time.sleep(2)
                if any(p.returncode != 0 for p in processes):
                    raise RuntimeError("Sampling worker failed")
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        path = merge_parts(DATASET, self.run_id, 2, tag,
                           expected_count=self.manifest["sample_count"])
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data["metadata"]["checkpoint_sha256"] != sha256_file(checkpoint):
            raise RuntimeError("Merged result has a different checkpoint")
        if method == "native" and data["projection_calls"] != 0:
            raise RuntimeError("Native result used projection")
        if method == "projection" and data["projection_calls"] <= 0:
            raise RuntimeError("Projection result never executed projection")
        return path, dict(wall_seconds=time.monotonic()-started,
                          worker_seconds=data["elapsed_seconds"], projection_calls=data["projection_calls"],
                          samples=len(data["sequences"]))

    def benchmark(self, checkpoint):
        self.phase = "projection-benchmark"
        disk_guard()
        self.status("running")
        started = time.monotonic()
        self.run(self.sample_command("projection", checkpoint, f"{self.run_id}_benchmark", 0, benchmark=True),
                 "projection-benchmark", dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        atomic_json(self.directory / "benchmark.json", dict(samples=1,
                    elapsed_seconds=time.monotonic()-started, projection=PROJECTION,
                    note="Single-sample timing, not a reliable full-batch ETA; formal parameters unchanged"))

    def evaluate(self, method, generated_path, timing):
        self.phase = f"evaluate-{method}"
        disk_guard()
        self.status("running")
        from evaluation import run_Statistical
        stats, skip, strict, coverage, unsat = run_Statistical(DATASET, f"{self.run_id}_{method}")
        metrics = {str(k): float(v) for k, v in stats.items()}
        metrics.update(OVR_ref_skip=float(skip), OVR_ref_strict=float(strict),
                       coverage=float(coverage), Unsat_ref=float(unsat), **timing)
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Non-finite evaluation metric; result is not accepted")
        run = wandb.init(project="Marionette", entity=self.entity,
                         id=f"{self.run_id}-{method}", resume="allow", group=self.run_id,
                         name=f"{self.run_id}-{method}", job_type="sampling-evaluation", mode="online",
                         dir=str(self.directory), save_code=False,
                         config={**self.manifest, "method": method,
                                 "checkpoint_sha256": sha256_file(self.directory / "final.ckpt")})
        try:
            run.log(metrics)
            run.summary.update(metrics)
            run.summary["result_sha256"] = sha256_file(generated_path)
            run.summary["complete"] = True
            run.finish(exit_code=0)
        except BaseException:
            run.finish(exit_code=1)
            raise
        verified = wandb.Api(timeout=30).run(f"{self.entity}/Marionette/{run.id}")
        if verified.summary.get("complete") is not True:
            raise RuntimeError("W&B summary readback failed")
        atomic_json(self.directory / f"metrics-{method}.json", dict(
            metrics=metrics, output=str(generated_path), output_sha256=sha256_file(generated_path),
            wandb_url=run.url))
        return metrics

    def execute(self):
        self.precheck()
        if self.args.stage != "sample":
            self.train()
        checkpoint = self.freeze_checkpoint()
        if not self.preflight:
            self.benchmark(checkpoint)
        metrics = {}
        for method in ("native", "projection"):
            generated, timing = self.sample(method, checkpoint)
            if not self.preflight:
                metrics[method] = self.evaluate(method, generated, timing)
        for split, expected in DATA_HASHES.items():
            if sha256_file(ROOT / f"data/{DATASET}/{DATASET}_{split}.pkl") != expected:
                raise RuntimeError("Input dataset was modified")
        if metrics:
            atomic_json(self.directory / "comparison.json", metrics)
        self.phase = "complete"
        self.status("complete", checkpoint=str(checkpoint), preflight=self.preflight)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=["all", "sample", "preflight"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-id")
    args = parser.parse_args()
    os.chdir(ROOT)
    experiment = Experiment(args)
    # flock releases automatically on process exit; do not delete a possibly live lock.
    import fcntl
    with (experiment.directory / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            experiment.execute()
        except BaseException as exc:
            experiment.status("failed", error_type=type(exc).__name__, error=str(exc))
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()

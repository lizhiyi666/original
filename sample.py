import torch
import argparse
import math
import json
import time
from pathlib import Path
from evaluate_utils import get_task, get_run_data
from experiment_io import (publish_torch, safe_tag, seed_sampling, sha256_file,
                           strict_test_matrix, validate_part, validate_sequences, decode_preserving_empty)

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, default="marionette")
parser.add_argument("--output_tag", default=None)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--cfg_checkpoint", default=None, help="Completed baseline-suite CFG spatial checkpoint")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--start_index", type=int, default=0, help="Diagnostic slice; retains global test indices/seeds")
parser.add_argument("--sampling_revision", default="distance-kl-v2")
parser.add_argument("--constraint_source", choices=["stored", "strict_test"], default="stored")
parser.add_argument("--resume", action="store_true")

parser.add_argument("--use_constraint_projection", action="store_true")

# 投影参数
parser.add_argument("--projection_frequency", type=int, default=10)
parser.add_argument("--projection_tau", type=float, default=0.0)
parser.add_argument("--projection_lambda", type=float, default=0.0)
parser.add_argument("--projection_eta", type=float, default=1.0)
parser.add_argument("--projection_mu", type=float, default=1.0)
parser.add_argument("--projection_mu_max", type=float, default=1000.0)
parser.add_argument("--projection_outer_iters", type=int, default=10)
parser.add_argument("--projection_inner_iters", type=int, default=10)
parser.add_argument("--projection_mu_alpha", type=float, default=2.0)
parser.add_argument("--projection_delta_tol", type=float, default=1e-6)
parser.add_argument("--projection_existence_weight", type=float, default=0.02)
parser.add_argument("--projection_distance_kl_weight", type=float, default=None,
                    help="Defaults to 1 for distance-kl-v2, 0 for explicitly historical sampling revisions")
parser.add_argument("--distance_paths", type=int, default=8)
parser.add_argument("--distance_topk", type=int, default=32)
parser.add_argument("--distance_bins", type=int, default=32)
parser.add_argument("--distance_temperature", type=float, default=1.0)
parser.add_argument("--distance_backend", choices=['legacy', 'batched'], default='legacy')
parser.add_argument('--geometry_refinement', choices=['off', 'same_category_v1'], default='off')
parser.add_argument('--geometry_steps', type=int, choices=[50,100,200], default=50)
parser.add_argument('--geometry_distance_weight', type=float, default=1.)
parser.add_argument('--geometry_radius_weight', type=float, default=1.)
parser.add_argument('--geometry_prior_weight', type=float, default=.01)
parser.add_argument('--geometry_fit_indices', help='JSON list of train-only prior indices, excluding calibration pools')
parser.add_argument("--use_gumbel_softmax", action="store_true", default=None, help="Enable Gumbel-Softmax for gradient estimation")
parser.add_argument("--no_gumbel_softmax", action="store_false", dest="use_gumbel_softmax",
                    help="Deterministic relaxation, including expected-route distance KL")
parser.add_argument("--gumbel_temperature", type=float, default=1.0)
parser.add_argument("--projection_last_k_steps", type=int, default=60)
parser.add_argument("--cond_dropout_rate", type=float, default=0.1, help="CFG dropout rate")

# 并行采样参数
parser.add_argument("--rank", type=int, default=0, help="当前进程的索引 (0 ~ world_size-1)")
parser.add_argument("--world_size", type=int, default=1, help="总进程数 (GPU数量)")

# debug 开关
parser.add_argument("--debug_constraint_projection", action="store_true")

# ========== Baseline 选择 ==========
parser.add_argument("--baseline", type=str, default=None,
    choices=["posthoc_swap", "energy_guidance", "cfg"], # [新增 cfg]
    help="Baseline方法: posthoc_swap=Baseline2, energy_guidance=Baseline3, cfg=Baseline4")

# ========== Baseline 3: Energy-Based Guidance 参数 ==========
parser.add_argument("--guidance_scale", type=float, default=10.0,
    help="Guidance scale (越大约束越强，但可能影响生成质量)")
parser.add_argument("--guidance_last_k_steps", type=int, default=40,
    help="Only apply guidance in the last k diffusion steps")
parser.add_argument("--guidance_frequency", type=int, default=4,
    help="Apply guidance every N steps")
parser.add_argument("--guidance_temperature", type=float, default=1.0,
                    help="Softmax temperature for guidance violation (higher = softer = better gradients)")
args = None


def simulation(RUN_ID="marionette", WANDB_DIR="wandb", PROJECT_ROOT="./"):
    from distance_kl import distance_metadata, distance_output_directory
    implementation = distance_metadata(args.distance_backend)
    geometry_on = args.geometry_refinement != 'off'
    if geometry_on and (not args.output_tag or not args.use_constraint_projection or not args.geometry_fit_indices):
        raise ValueError('Geometry requires PCDG, an independent output tag and training reference indices')
    if args.distance_backend == 'batched' and not args.output_tag:
        raise ValueError('Batched distance sampling requires an explicit independent --output_tag')
    if args.use_gumbel_softmax is None:
        args.use_gumbel_softmax = args.sampling_revision == 'distance-kl-v2'
    if args.projection_distance_kl_weight is None:
        args.projection_distance_kl_weight = 1.0 if args.sampling_revision == 'distance-kl-v2' else 0.0
    if not math.isfinite(args.projection_distance_kl_weight) or args.projection_distance_kl_weight < 0:
        raise ValueError('Distance KL weight must be finite and nonnegative')
    if args.baseline is not None:
        from tools.baseline_common import precision
        precision()
        if args.use_constraint_projection:
            raise ValueError('Baseline methods cannot be combined with ALM projection')
    if args.use_constraint_projection and args.projection_distance_kl_weight > 0:
        from tools.baseline_common import precision
        precision()
    data_name, seed, run_path = get_run_data(RUN_ID, WANDB_DIR)
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("Invalid rank/world_size")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max_samples must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    task, datamodule = get_task(run_path, data_root=PROJECT_ROOT, checkpoint=args.checkpoint)
    if args.baseline == 'cfg':
        if not args.cfg_checkpoint:
            raise ValueError('--baseline cfg requires --cfg_checkpoint')
        from tools.baseline_common import install_cfg
        install_cfg(task, datamodule, run_path, args.cfg_checkpoint)
    if not 0 <= args.start_index < len(datamodule.test_data.sequences):
        raise ValueError("start_index must identify an existing test sequence")
    if args.batch_size is not None:
        datamodule.batch_size = args.batch_size
    checkpoint = Path(args.checkpoint or Path(run_path) / "checkpoints/last.ckpt")
    test_path = Path(datamodule.root) / data_name / f"{data_name}_test.pkl"
    test_data = torch.load(test_path, map_location="cpu", weights_only=False)
    output_tag = safe_tag(args.output_tag or RUN_ID)
    seed_base = int(seed if args.seed is None else args.seed)
    distance_reference = None
    if args.use_constraint_projection and args.projection_distance_kl_weight > 0:
        from distance_kl import load_distance_reference
        distance_reference = load_distance_reference(
            Path(datamodule.root) / datamodule.name / f'{datamodule.name}_train.pkl', args.distance_bins)
    remaining = len(datamodule.test_data.sequences) - args.start_index
    total_len = min(remaining, args.max_samples or remaining)
    chunk_size = int(math.ceil(total_len / args.world_size))
    start_idx = args.start_index + min(args.rank * chunk_size, total_len)
    end_idx = args.start_index + min((args.rank + 1) * chunk_size, total_len)
    indices = list(range(start_idx, end_idx))
    sampling_config = {key: value for key, value in vars(args).items()
                       if key not in {"rank", "resume", "checkpoint", "output_tag", "run_id"}}
    sampling_config.update(batch_size=datamodule.batch_size, seed=seed_base)
    metadata = dict(schema_version=1, data_name=data_name, run_id=RUN_ID,
                    output_tag=output_tag, total_samples=total_len, world_size=args.world_size,
                    start_index=args.start_index, empty_policy="keep", sampling_revision=args.sampling_revision,
                    checkpoint_sha256=sha256_file(checkpoint),
                    dataset_sha256=sha256_file(test_path), sampling_config=sampling_config, **implementation)
    if distance_reference is not None:
        metadata['distance_reference_sha256'] = distance_reference.fingerprint
    if not geometry_on:
        sampling_config_keys = [k for k in sampling_config if k.startswith('geometry_')]
        for k in sampling_config_keys:
            sampling_config.pop(k)
    output_dir = distance_output_directory(test_path.parent, args.distance_backend, args.output_tag)
    geometry_config = geometry_reference = None
    if geometry_on:
        from geometry_projection import GeometryConfig, load_geometry_reference, geometry_output_directory
        geometry_config = GeometryConfig(**{k:getattr(args,k) for k in GeometryConfig.__dataclass_fields__ if hasattr(args,k)})
        fit_indices = json.loads(Path(args.geometry_fit_indices).read_text(encoding='utf-8'))
        geometry_reference = load_geometry_reference(Path(datamodule.root)/datamodule.name/f'{datamodule.name}_train.pkl', fit_indices)
        metadata.update(geometry_config.metadata(), geometry_reference_sha256=geometry_reference.fingerprint,
                        geometry_fit_indices_sha256=sha256_file(args.geometry_fit_indices))
        output_dir = geometry_output_directory(test_path.parent, args.geometry_refinement, args.output_tag)
    save_name = output_dir / f"{data_name}_{output_tag}_generated_part{args.rank}.pkl"
    if save_name.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite {save_name}")
        previous = torch.load(save_name, map_location="cpu", weights_only=False)
        validate_part(previous, metadata, args.rank, indices)
        print(f"Verified existing shard: {save_name}")
        return
    if args.constraint_source == "strict_test":
        for index in indices:
            datamodule.test_data.sequences[index].po_matrix = strict_test_matrix(
                test_data["sequences"][index], test_data["poi_category"], test_data["category_mapping"])

    dd = task.discrete_diffusion
    dd.projection_call_count = 0
    dd.geometry_config = geometry_config
    dd.geometry_reference = geometry_reference
    geometry_diagnostics = []

    # ========== Baseline1: 强制关闭投影 ==========
    # 规则A：baseline=None 且未显式 --use_constraint_projection 时，视为 baseline1
    if args.baseline in (None, 'posthoc_swap') and (not args.use_constraint_projection):
        dd.use_constraint_projection = False
        dd.debug_constraint_projection = False
        dd.projection_last_k_steps = 0
        dd.projection_frequency = 10**9  # 防止内部误触发
        dd.constraint_projector = None

    # ========== 采样时强制开启投影，并补建 projector ==========
    if args.use_constraint_projection:
        from constraint_projection import ConstraintProjection

        dd.use_constraint_projection = True
        dd.projection_frequency = args.projection_frequency
        dd.debug_constraint_projection = args.debug_constraint_projection
        dd._debug_projection_printed = False
        dd._debug_viol_printed = False
        dd._debug_po_printed = False

        dd.projection_last_k_steps = args.projection_last_k_steps
        dd.use_gumbel_softmax = args.use_gumbel_softmax
        dd.gumbel_temperature = args.gumbel_temperature

        dd.projection_tau = args.projection_tau
        dd.projection_lambda = args.projection_lambda
        dd.projection_eta = args.projection_eta
        dd.projection_mu = args.projection_mu
        dd.projection_mu_max = args.projection_mu_max
        dd.projection_outer_iters = args.projection_outer_iters
        dd.projection_inner_iters = args.projection_inner_iters
        dd.projection_mu_alpha = args.projection_mu_alpha
        dd.projection_delta_tol = args.projection_delta_tol
        dd.projection_existence_weight = args.projection_existence_weight

        if not hasattr(dd, "constraint_projector") or dd.constraint_projector is None or args.projection_distance_kl_weight > 0:
            device = next(dd.parameters()).device
            dd.constraint_projector = ConstraintProjection(
                num_classes=dd.num_classes,
                type_classes=dd.type_classes,
                num_spectial=dd.num_spectial,
                tau=args.projection_tau,
                lambda_init=args.projection_lambda,
                mu_init=args.projection_mu,
                mu_alpha=args.projection_mu_alpha,
                mu_max=args.projection_mu_max,
                outer_iterations=args.projection_outer_iters,
                inner_iterations=args.projection_inner_iters,
                eta=args.projection_eta,
                delta_tol=args.projection_delta_tol,
                projection_existence_weight=args.projection_existence_weight,
                use_gumbel_softmax=args.use_gumbel_softmax,
                gumbel_temperature=args.gumbel_temperature,
                device=str(device),
                projection_distance_kl_weight=args.projection_distance_kl_weight,
                distance_reference=distance_reference,
                distance_paths=args.distance_paths, distance_topk=args.distance_topk,
                distance_bins=args.distance_bins, distance_temperature=args.distance_temperature,
                distance_backend=args.distance_backend,
            )
            dd.constraint_projector.distance_seed = seed_base + args.rank

        if hasattr(dd, "constraint_projector") and dd.constraint_projector is not None:
            dd.constraint_projector.distance_backend = args.distance_backend
            dd.constraint_projector.distance_implementation_version = implementation['distance_implementation_version']
            dd.constraint_projector.projection_distance_kl_weight = args.projection_distance_kl_weight
            dd.constraint_projector.projection_existence_weight = args.projection_existence_weight
            dd.constraint_projector.lambda_init = args.projection_lambda
            dd.constraint_projector.eta = args.projection_eta
            dd.constraint_projector.inner_iterations = args.projection_inner_iters
            dd.constraint_projector.outer_iterations = args.projection_outer_iters

            print(f"[DEBUG] Force updated projector.projection_existence_weight to {dd.constraint_projector.projection_existence_weight}")

        print("[DEBUG] sample.py forced use_constraint_projection=True")
        print("[DEBUG] projection params:",
              dict(freq=dd.projection_frequency,
                   tau=args.projection_tau,
                    lambda_init=args.projection_lambda,
                    mu_init=args.projection_mu,
                    mu_alpha=args.projection_mu_alpha,
                    mu_max=args.projection_mu_max,
                    outer_iterations=args.projection_outer_iters,
                    inner_iterations=args.projection_inner_iters,
                    eta=args.projection_eta,
                    delta_tol=args.projection_delta_tol,
                    projection_existence_weight=args.projection_existence_weight,
                    use_gumbel_softmax=args.use_gumbel_softmax,
                    gumbel_temperature=args.gumbel_temperature,
                   mu=dd.projection_mu))

    if args.baseline == "energy_guidance":
        if args.use_constraint_projection:
            raise ValueError('Energy guidance and ALM are mutually exclusive')
        from baseline_models import configure_energy
        configure_energy(dd, args.guidance_temperature, args.guidance_scale,
                         args.guidance_last_k_steps, args.guidance_frequency)
    if args.baseline == "cfg":
        if args.use_constraint_projection:
            raise ValueError('CFG and ALM are mutually exclusive')
        dd.cfg_scale = args.guidance_scale

    distance_projection_stats = []
    if args.use_constraint_projection and args.projection_distance_kl_weight > 0:
        project_distance = dd.constraint_projector.project_with_matrices
        def trace_distance(*values, **options):
            output = project_distance(*values, **options)
            distance_projection_stats.append(dict(dd.constraint_projector.last_projection_stats))
            return output
        dd.constraint_projector.project_with_matrices = trace_distance

    # ======================================================
    all_sequences = datamodule.test_data.sequences
    my_sequences = all_sequences[start_idx:end_idx]
    datamodule.test_data.sequences = my_sequences

    print(f"[GPU {args.rank}] Processing {len(my_sequences)} sequences (Range: {start_idx} -> {end_idx})")

    if len(my_sequences) == 0:
        publish_torch(save_name, dict(sequences=[], t_max=24.0, test_indices=[],
                                     metadata=metadata, rank=args.rank, projection_calls=0, elapsed_seconds=0,
                                     empty_test_indices=[], temporal_empty_test_indices=[], eligible_projection_samples=0))
        print(f"[GPU {args.rank}] Published verified empty shard.")
        return

    gps_dict = test_data['poi_gps']

    collected_po_matrices = []

    generated_seqs = []
    temporal_empty_indices = []
    eligible_projection_samples = 0
    started = time.monotonic()
    for batch in datamodule.test_dataloader():
        # Paired methods start each global batch from the same RNG state.
        seed_sampling(seed_base + start_idx + len(generated_seqs))
        if args.use_constraint_projection and args.projection_distance_kl_weight > 0:
            from distance_kl import distance_generator
            dd.constraint_projector.distance_generator = distance_generator(
                seed_base + start_idx + len(generated_seqs), task.device)
        with torch.no_grad():
            time_samples = task.tpp_model.sample(
                batch.batch_size,
                tmax=batch.tmax.to(task.device),
                x_n=batch.to(task.device)
            )
        generated_length = int(time_samples.unpadded_length.max().item())
        max_supported = min(dd.condition_encoder.max_position_embeddings,
                            (dd.transformer.positional_encoding.num_embeddings - 3) // 2)
        if generated_length > max_supported:
            raise RuntimeError(
                f"Temporal sampler generated {generated_length} events, but the unchanged "
                f"spatial model supports at most {max_supported}. Stopping without truncating "
                "trajectories or changing batch size; inspect temporal model convergence.")
        time_samples = time_samples.mask_check()

        if hasattr(batch, "po_matrix"):
            if batch.po_matrix is not None:
                print("po_matrix shape:", batch.po_matrix.shape, "sum0:", batch.po_matrix[0].sum().item())
            else:
                print("po_matrix missing")
            if batch.po_matrix is not None:
                time_samples.po_matrix = batch.po_matrix.to(task.device)

        batch_global_start = start_idx + len(generated_seqs)
        temporal_empty_indices.extend(batch_global_start + i for i in
                                      torch.where(time_samples.unpadded_length == 0)[0].tolist())
        if batch.po_matrix is not None:
            eligible_projection_samples += int(((time_samples.unpadded_length > 0) &
                                                 batch.po_matrix.bool().flatten(1).any(1)).sum().item())

        assert len(time_samples) == batch.batch_size, "not enough samples"

        dd.geometry_seed = seed_base
        dd.geometry_global_start = batch_global_start
        dd.last_geometry_stats = None
        samples = decode_preserving_empty(
            task, time_samples, gps_dict,
            baseline_method=args.baseline,      # [新增]
            guidance_scale=args.guidance_scale  # [新增]
        )
        assert len(samples) == batch.batch_size, "not enough samples"
        if geometry_on:
            geometry_diagnostics.append(dict(global_start=batch_global_start, diagnostics=dd.last_geometry_stats))
        generated_seqs += samples

    # ========== Baseline 2: Post-hoc Swap ==========
    if args.baseline == "posthoc_swap":
        from baseline_posthoc_swap import apply_posthoc_swap, get_eval_cats, extract_constraints_from_test_seq

        poi_category = test_data['poi_category']
        category_mapping = test_data.get('category_mapping', None)

        all_test_seqs = test_data['sequences']
        my_test_seqs = all_test_seqs[start_idx:end_idx]

        po_matrices = None
        if category_mapping is not None:
            po_matrices = []
            for seq in my_test_seqs:
                pm = seq.get('po_matrix', None)
                if pm is not None:
                    po_matrices.append(pm)
                else:
                    po_matrices = None
                    break

        print(f"\n[Baseline2] Applying post-hoc swap to {len(generated_seqs)} sequences...")

        generated_seqs, swap_summary = apply_posthoc_swap(
            generated_seqs=generated_seqs,
            test_seqs=my_test_seqs,
            poi_category=poi_category,
            category_mapping=category_mapping,
            po_matrices=None,  # Always derive the same actual reference pairs as evaluation.
            verbose=True,
        )
        print(f"[Baseline2] Done. Summary: {swap_summary}")

    # ================= 保存 =================
    validate_sequences(generated_seqs, test_data["poi_category"])
    if len(generated_seqs) != len(indices):
        raise ValueError("Generated sample count mismatch")
    if args.use_constraint_projection and eligible_projection_samples > 0:
        if dd.projection_call_count == 0:
            raise RuntimeError("Projection requested but never executed")
    if not args.use_constraint_projection and args.baseline is None and dd.projection_call_count:
        raise RuntimeError("Native sampling unexpectedly executed projection")
    data_new = dict(sequences=generated_seqs, t_max=24.0, test_indices=indices,
                    metadata=metadata, rank=args.rank, projection_calls=dd.projection_call_count,
                    empty_test_indices=[i for i, seq in zip(indices, generated_seqs) if len(seq['checkins']) == 0],
                    temporal_empty_test_indices=temporal_empty_indices,
                    eligible_projection_samples=eligible_projection_samples,
                    elapsed_seconds=time.monotonic() - started)
    if args.use_constraint_projection and args.projection_distance_kl_weight > 0:
        data_new['distance_projection_diagnostics'] = distance_projection_stats
    if geometry_on:
        data_new['geometry_projection_diagnostics'] = geometry_diagnostics
    publish_torch(save_name, data_new)
    print(f"[GPU {args.rank}] Saved part file to {save_name}")

if __name__ == "__main__":
    args = parser.parse_args()
    simulation(RUN_ID=args.run_id)

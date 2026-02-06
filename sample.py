import torch
import argparse
import math
from evaluate_utils import get_task, get_run_data

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, default="marionette")

parser.add_argument("--use_constraint_projection", action="store_true")

# [新增] 采样时强制开启投影（不依赖训练时配置）
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
parser.add_argument("--use_gumbel_softmax", action="store_true", help="Enable Gumbel-Softmax for gradient estimation")
parser.add_argument("--gumbel_temperature", type=float, default=1.0)
parser.add_argument("--projection_last_k_steps", type=int, default=60)
# [新增] 并行采样参数
parser.add_argument("--rank", type=int, default=0, help="当前进程的索引 (0 ~ world_size-1)")
parser.add_argument("--world_size", type=int, default=1, help="总进程数 (GPU数量)")
# [新增] debug 开关
parser.add_argument("--debug_constraint_projection", action="store_true")

# ========== [新增] Baseline 选择 ==========
parser.add_argument("--baseline", type=str, default=None,
    choices=[None, "posthoc_swap"],
    help="选择 baseline 方法。posthoc_swap: 先正常采样再后处理交换。")

args = parser.parse_args()


def simulation(RUN_ID="marionette", WANDB_DIR="wandb", PROJECT_ROOT="./"):
    data_name, seed, run_path = get_run_data(RUN_ID, WANDB_DIR)
    task, datamodule = get_task(run_path, data_root=PROJECT_ROOT)

    # ========== [新增] 采样时强制开启投影，并补建 projector ==========
    if args.use_constraint_projection:
        from constraint_projection import ConstraintProjection

        dd = task.discrete_diffusion
        dd.use_constraint_projection = True
        dd.projection_frequency = args.projection_frequency
        dd.debug_constraint_projection = args.debug_constraint_projection
        dd._debug_projection_printed = False
        dd._debug_viol_printed = False
        dd._debug_po_printed = False
        
        dd.projection_last_k_steps = args.projection_last_k_steps 
        dd.use_gumbel_softmax = args.use_gumbel_softmax
        dd.gumbel_temperature = args.gumbel_temperature

        # 保存参数（供内部使用/打印）
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
        
        if not hasattr(dd, "constraint_projector") or dd.constraint_projector is None:
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
            )

        if hasattr(dd, "constraint_projector") and dd.constraint_projector is not None:
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
    # ======================================================
    all_sequences = datamodule.test_data.sequences
    total_len = len(all_sequences)
    
    # 计算当前 GPU 应该负责的范围
    chunk_size = int(math.ceil(total_len / args.world_size))
    start_idx = args.rank * chunk_size
    end_idx = min((args.rank + 1) * chunk_size, total_len)
    
    # 切分数据
    my_sequences = all_sequences[start_idx:end_idx]
    
    # 覆盖 datamodule 中的数据，这样 dataloader 就会只加载这一部分
    datamodule.test_data.sequences = my_sequences
    
    print(f"[GPU {args.rank}] Processing {len(my_sequences)} sequences (Range: {start_idx} -> {end_idx})")
    
    if len(my_sequences) == 0:
        print(f"[GPU {args.rank}] No data to process, exiting.")
        return

    test_data = torch.load(PROJECT_ROOT + 'data/' + data_name + f'/{data_name}_test.pkl', weights_only=False)
    gps_dict = test_data['poi_gps']

    # ========== [新增] 收集 po_matrix 用于 Baseline2 后处理 ==========
    collected_po_matrices = []

    generated_seqs = []
    for batch in datamodule.test_dataloader():
        # tpp 采样出时间序列
        time_samples = task.tpp_model.sample(
            batch.batch_size,
            tmax=batch.tmax.to(task.device),
            x_n=batch.to(task.device)
        ).mask_check()
        
        if hasattr(batch, "po_matrix"):
            if batch.po_matrix is not None:
                print("po_matrix shape:", batch.po_matrix.shape, "sum0:", batch.po_matrix[0].sum().item())
            else:
                print("po_matrix missing")
            time_samples.po_matrix = batch.po_matrix.to(task.device)
        
        assert len(time_samples) == batch.batch_size, "not enough samples"

        samples = task.discrete_diffusion.sample_fast(time_samples.to(task.device)).to_seq_list(gps_dict)
        assert len(samples) == batch.batch_size, "not enough samples"
        generated_seqs += samples


    # ========== [新增] Baseline 2: Post-hoc Swap ==========
    if args.baseline == "posthoc_swap":
        from baseline_posthoc_swap import apply_posthoc_swap, get_eval_cats, extract_constraints_from_test_seq
        
        # 需要 test 序列和 poi_category 来做修复
        # test_data 已经在前面加载过了
        poi_category = test_data['poi_category']
        category_mapping = test_data.get('category_mapping', None)
        
        # 获取当前 rank 对应的 test 序列切片
        all_test_seqs = test_data['sequences']
        my_test_seqs = all_test_seqs[start_idx:end_idx]
        
        # 收集 po_matrices（如果有的话）
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
            po_matrices=po_matrices,
            verbose=True,
        )
        print(f"[Baseline2] Done. Summary: {swap_summary}")

    # ================= 保存文件带上 rank 后缀 =================
    save_name = f'./data/{data_name}/{data_name}_{RUN_ID}_generated_part{args.rank}.pkl'
    data_new = {'sequences': generated_seqs, 't_max': batch.tmax.detach().cpu().numpy()}
    torch.save(data_new, save_name)
    print(f"[GPU {args.rank}] Saved part file to {save_name}")

simulation(RUN_ID=args.run_id)
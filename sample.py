import torch
import argparse
from evaluate_utils import get_task, get_run_data

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, default="marionette")

# [新增] 采样时强制开启投影（不依赖训练时配置）
parser.add_argument("--use_constraint_projection", action="store_true")
parser.add_argument("--projection_frequency", type=int, default=10)
parser.add_argument("--projection_alm_iters", type=int, default=10)
parser.add_argument("--projection_tau", type=float, default=0.0)
parser.add_argument("--projection_lambda", type=float, default=1.0)
parser.add_argument("--projection_eta", type=float, default=0.2)
parser.add_argument("--projection_mu", type=float, default=1.0)

# [新增] debug 开关
parser.add_argument("--debug_constraint_projection", action="store_true")

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

        # 保存参数（供内部使用/打印）
        dd.projection_tau = args.projection_tau
        dd.projection_lambda = args.projection_lambda
        dd.projection_alm_iters = args.projection_alm_iters
        dd.projection_eta = args.projection_eta
        dd.projection_mu = args.projection_mu

        # 关键：训练时若 use_constraint_projection=False，则 __init__ 不会创建 constraint_projector，这里补建
        if not hasattr(dd, "constraint_projector") or dd.constraint_projector is None:
            device = next(dd.parameters()).device
            dd.constraint_projector = ConstraintProjection(
                num_classes=dd.num_classes,
                type_classes=dd.type_classes,
                num_spectial=dd.num_spectial,
                tau=args.projection_tau,
                lambda_init=args.projection_lambda,
                alm_iterations=args.projection_alm_iters,
                eta=args.projection_eta,
                mu=args.projection_mu,
                device=str(device),
            )

        print("[DEBUG] sample.py forced use_constraint_projection=True")
        print("[DEBUG] projection params:",
              dict(freq=dd.projection_frequency,
                   iters=dd.projection_alm_iters,
                   tau=dd.projection_tau,
                   lambda_=dd.projection_lambda,
                   eta=dd.projection_eta,
                   mu=dd.projection_mu))
    # ======================================================

    test_data = torch.load(PROJECT_ROOT + 'data/' + data_name + f'/{data_name}_test.pkl', weights_only=False)
    gps_dict = test_data['poi_gps']

    generated_seqs = []
    for batch in datamodule.test_dataloader():
        # tpp 采样出时间序列（这个 time_samples 通常不包含 po_matrix）
        time_samples = task.tpp_model.sample(
            batch.batch_size,
            tmax=batch.tmax.to(task.device),
            x_n=batch.to(task.device)
        ).mask_check()
        #print("time_samples.time", time_samples.time.shape)
        #print("time_samples.tau ", time_samples.tau.shape)
        # ========== [新增] 把每条轨迹自己的 po_matrix 从原始 batch 传给 time_samples ==========
        # 这是投影能否生效的关键：diffusion 的 sample_fast 只看到 time_samples
        if hasattr(batch, "po_matrix"):
            if batch.po_matrix is not None:
                print("po_matrix shape:", batch.po_matrix.shape, "sum0:", batch.po_matrix[0].sum().item())
            else:
                print("po_matrix missing")
            time_samples.po_matrix = batch.po_matrix.to(task.device)
        # 同时把 category_mask 等也带上（通常 time_samples 已有，但这里做兜底）
        '''if hasattr(batch, "category_mask"):
            time_samples.category_mask = batch.category_mask.to(task.device)
        if hasattr(batch, "poi_mask"):
            time_samples.poi_mask = batch.poi_mask.to(task.device)
        if hasattr(batch, "tau"):
            time_samples.tau = batch.tau.to(task.device) if hasattr(batch.tau, "to") else batch.tau
        if hasattr(batch, "unpadded_length"):
            time_samples.unpadded_length = batch.unpadded_length.to(task.device) if hasattr(batch.unpadded_length, "to") else batch.unpadded_length'''
        # ============================================================================
        #print("time_samples.time", time_samples.time.shape)
        #print("time_samples.tau ", time_samples.tau.shape)
        assert len(time_samples) == batch.batch_size, "not enough samples"

        samples = task.discrete_diffusion.sample_fast(time_samples.to(task.device)).to_seq_list(gps_dict)
        assert len(samples) == batch.batch_size, "not enough samples"
        generated_seqs += samples

    data_new = {'sequences': generated_seqs, 't_max': batch.tmax.detach().cpu().numpy()}
    torch.save(data_new, f'./data/{data_name}/{data_name}_{RUN_ID}_generated.pkl')


simulation(RUN_ID=args.run_id)
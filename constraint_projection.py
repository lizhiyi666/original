
"""
Constraint Discrete Diffusion - Projection Method Implementation

"""

import torch
import torch.nn.functional as F


class ConstraintProjection:
    def __init__(
            self,
            num_classes: int,
            type_classes: int,
            num_spectial: int,
            tau: float = 0.0,
            lambda_init: float = 0.0,          # 论文参数：λinit
            mu_init: float = 1.0,              # 论文参数：μinit
            mu_alpha: float = 2.0,             # 每轮外层放大系数 α（可调）
            mu_max: float = 1000.0,            # 论文参数：μmax
            outer_iterations: int = 10,      # 论文参数：outer_itermax
            inner_iterations: int = 10,       # 论文参数：inner_itermax
            eta: float = 1.0,                  # 论文参数：η
            delta_tol: float = 0.25,           # 外层硬判定容忍 δ
            use_gumbel_softmax: bool = True,
            gumbel_temperature: float = 1.0,
            device: str = "cuda",
            projection_existence_weight: float = 0.02,
        ):
            self.num_classes = num_classes
            self.type_classes = type_classes
            self.num_spectial = num_spectial
            self.tau = tau
            self.lambda_init = lambda_init
            self.mu_init = mu_init
            self.mu_alpha = mu_alpha
            self.mu_max = mu_max
            self.outer_iterations = outer_iterations
            self.inner_iterations = inner_iterations
            self.eta = eta
            self.delta_tol = delta_tol
            self.device = device

            self.category_start = num_spectial
            self.category_end = num_spectial + type_classes

            self.use_gumbel_softmax = use_gumbel_softmax
            self.gumbel_temperature = gumbel_temperature
            self.projection_existence_weight=projection_existence_weight

    def _sample_gumbel(self, shape, device, dtype):
        U = torch.rand(shape, device=device, dtype=dtype)
        return -torch.log(-torch.log(U + 1e-20) + 1e-20)

    def _gumbel_softmax_relax(self, logits_lv, tau, gumbel_noise=None):
        if gumbel_noise is None:
            gumbel_noise = self._sample_gumbel(logits_lv.shape, logits_lv.device, logits_lv.dtype)
        y = (logits_lv + gumbel_noise) / max(tau, 1e-6)
        return torch.softmax(y, dim=-1), gumbel_noise

    # def compute_hard_constraint_violation(
    #     self,
    #     log_probs: torch.Tensor,
    #     po_constraints: list,
    #     category_mask: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     基于 argmax 的硬违规度 g(y*): 使用 one-hot 序列计算 P_B @ M · P_A。
    #     """
    #     B, V, L = log_probs.shape
    #     dev = log_probs.device

    #     # y* = argmax(log_probs)
    #     idx = log_probs.argmax(dim=1)  # [B, L]
    #     probs = F.one_hot(idx, num_classes=self.num_classes).float()  # [B, L, V]
    #     if category_mask is not None:
    #         probs = probs * category_mask.unsqueeze(-1).float()

    #     category_probs = probs[:, :, self.category_start:self.category_end]
    #     M = torch.triu(torch.ones(L, L, device=dev), diagonal=1)

    #     total_violation = torch.zeros(B, device=dev)
    #     for A_indices, B_indices in po_constraints:
    #         P_A = category_probs[:, :, A_indices].sum(dim=-1)  # [B, L]
    #         P_B = category_probs[:, :, B_indices].sum(dim=-1)  # [B, L]
    #         P_B_weighted = torch.matmul(P_B.unsqueeze(1), M).squeeze(1)  # [B, L]
    #         violation_k = (P_B_weighted * P_A).sum(dim=1)  # [B]
    #         total_violation += violation_k
    #     return total_violation

    def compute_hard_constraint_violation_optimized(
        self,
        log_probs: torch.Tensor, # [B, V, L]
        W_A: torch.Tensor,       # [V_type, K]
        W_B: torch.Tensor,       # [V_type, K]
        category_mask: torch.Tensor,
        constraint_mask: torch.Tensor = None,
        ):
        B, V, L = log_probs.shape
        
        idx = log_probs.argmax(dim=1)
        probs_hard = F.one_hot(idx, num_classes=self.num_classes).float()
        
        if category_mask is not None:
            probs_hard = probs_hard * category_mask.unsqueeze(-1).float()

        probs_type = probs_hard[:, :, self.category_start : self.category_end]

        P_A_all = torch.matmul(probs_type, W_A) 
        P_B_all = torch.matmul(probs_type, W_B)

        if constraint_mask is not None:
            mask_expanded = constraint_mask.unsqueeze(1) # [B, 1, K]
            P_A_all = P_A_all * mask_expanded
            P_B_all = P_B_all * mask_expanded

        # 顺序违规
        P_B_cumsum = torch.cumsum(P_B_all, dim=1)
        P_B_prefix = torch.zeros_like(P_B_cumsum)
        P_B_prefix[:, 1:, :] = P_B_cumsum[:, :-1, :]
        order_per_k = (P_B_prefix * P_A_all).sum(dim=1) # [B, K]

        # 存在性违规
        count_A = P_A_all.sum(dim=1) # [B, K]
        count_B = P_B_all.sum(dim=1) # [B, K]
        
        target_count = 1.0
        viol_exist_A = F.relu(target_count - count_A)
        viol_exist_B = F.relu(target_count - count_B)
        exist_per_k = viol_exist_A + viol_exist_B # [B, K]

        # [修改] 独立返回
        return order_per_k, exist_per_k

    def compute_constraint_violation_optimized(
        self,
        log_probs: torch.Tensor,
        W_A: torch.Tensor,
        W_B: torch.Tensor,
        category_mask: torch.Tensor,
        constraint_mask: torch.Tensor = None,
        gumbel_noise=None,
        ):
        B, V, L = log_probs.shape
        
        logits_lv = log_probs.transpose(1, 2)
        T = float(self.gumbel_temperature)
        if T <= 0:
            T = 1.0
        if self.use_gumbel_softmax:
            probs, gumbel_noise = self._gumbel_softmax_relax(
                logits_lv, tau=T, gumbel_noise=gumbel_noise
            )
        else:
            probs, gumbel_noise = self._gumbel_softmax_relax(
                logits_lv, tau=T, gumbel_noise=gumbel_noise
            )

        if category_mask is not None:
            probs = probs * category_mask.unsqueeze(-1).float()

        probs_type = probs[:, :, self.category_start : self.category_end]

        P_A_all = torch.matmul(probs_type, W_A) 
        P_B_all = torch.matmul(probs_type, W_B)

        if constraint_mask is not None:
            mask_expanded = constraint_mask.unsqueeze(1)
            P_A_all = P_A_all * mask_expanded
            P_B_all = P_B_all * mask_expanded
            
        # 顺序违规
        P_B_cumsum = torch.cumsum(P_B_all, dim=1)
        P_B_prefix = torch.zeros_like(P_B_cumsum)
        P_B_prefix[:, 1:, :] = P_B_cumsum[:, :-1, :]
        violation_matrix = P_B_prefix * P_A_all # [B, L, K]
        order_per_k = violation_matrix.sum(dim=1) # [B, K]

        # 存在性违规
        count_A = P_A_all.sum(dim=1)
        count_B = P_B_all.sum(dim=1)
        target_count = 1.0
        viol_exist_A = F.relu(target_count - count_A)
        viol_exist_B = F.relu(target_count - count_B)
        exist_per_k = (viol_exist_A + viol_exist_B) # [B, K]

        if constraint_mask is not None:
            order_per_k = order_per_k * constraint_mask
            exist_per_k = exist_per_k * constraint_mask

        # [修改] 独立返回，不再求和
        return order_per_k, exist_per_k, gumbel_noise

    def project_with_matrices(
            self,
            log_probs: torch.Tensor,
            W_A: torch.Tensor,
            W_B: torch.Tensor,
            category_mask: torch.Tensor,
            constraint_mask = None,
        ) -> torch.Tensor:
            
        y_model = log_probs.transpose(1, 2).detach() 
        y = y_model.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.SGD([y], lr=self.eta)
        B = log_probs.shape[0] 
        K = W_A.shape[-1]
            
        # 独立初始化乘子
        lambda_order = torch.full((B, K), self.lambda_init, device=log_probs.device)
        mu_order = torch.full((B, K), self.mu_init, device=log_probs.device)
        lambda_exist = torch.full((B, K), self.lambda_init, device=log_probs.device)
        mu_exist = torch.full((B, K), self.mu_init, device=log_probs.device)

        # 增加一个标记打印起始点
        print(f"\n[Projection Start] Batch: {B}, Constraints: {K}")

        with torch.enable_grad(): 
            for outer_idx in range(self.outer_iterations):

                gumbel_noise = None
                last_kl_loss = 0.0
                last_const_loss = 0.0
                
                for _ in range(self.inner_iterations):
                    optimizer.zero_grad()
                    
                    g_soft_order, g_soft_exist, gumbel_noise = self.compute_constraint_violation_optimized(
                        y.transpose(1, 2), W_A, W_B, 
                        category_mask, constraint_mask, gumbel_noise=gumbel_noise
                    )
                    
                    delta_soft_order = F.relu(g_soft_order - self.tau)
                    delta_soft_exist = F.relu(g_soft_exist - self.tau)
                    
                    log_p = F.log_softmax(y, dim=-1)
                    log_q = F.log_softmax(y_model, dim=-1) 
                    kl_loss = F.kl_div(log_p, log_q, reduction='batchmean', log_target=True) 

                    # 独立计算惩罚项
                    penalty_order = lambda_order * delta_soft_order + 0.5 * mu_order * (delta_soft_order ** 2)
                    penalty_exist = lambda_exist * delta_soft_exist + 0.5 * mu_exist * (delta_soft_exist ** 2)
                    
                    # 惩罚项在 K 维度累加成为每个 Batch 的标量
                    constraint_loss = (penalty_order.sum(dim=1) + self.projection_existence_weight * penalty_exist.sum(dim=1)).sum()
                    
                    loss = kl_loss + constraint_loss
                    loss.backward()
                    optimizer.step()

                    # 记录最后一次内层循环的 Loss 用于打印
                    last_kl_loss = kl_loss.item()
                    #[关键修复] 强制限制梯度最大范数，防止 mu 和 lambda 变大时梯度爆炸
                    torch.nn.utils.clip_grad_norm_([y], max_norm=10.0) 
                    last_const_loss = constraint_loss.item()
                
                                # 3. 外层参数更新 (不需要梯度)
                with torch.no_grad():
                    g_hard_order, g_hard_exist = self.compute_hard_constraint_violation_optimized(
                        y.transpose(1, 2), W_A, W_B, category_mask, constraint_mask
                    )
                    
                    delta_hard_order = F.relu(g_hard_order - self.tau)
                    delta_hard_exist = F.relu(g_hard_exist - self.tau)
                    

                     # ===============================================================
                    # [新增日志输出] 每隔 10 次外层循环，或者在第一次和最后一次打印状态
                    # ===============================================================
                    if outer_idx == 0 or (outer_idx + 1) % 10 == 0 or outer_idx == self.outer_iterations - 1:
                        print(f"  [Outer {outer_idx+1:02d}/{self.outer_iterations}] "
                              f"Loss (KL={last_kl_loss:.4f}, Const={last_const_loss:.4f}) | "
                              f"Viol_Order(max={delta_hard_order.max():.2f}, mean={delta_hard_order.mean():.4f}) | "
                              f"Viol_Exist(max={delta_hard_exist.max():.2f}, mean={delta_hard_exist.mean():.4f}) | "
                              f"Mu_O(max={mu_order.max():.1f}) Mu_E(max={mu_exist.max():.1f}) | "
                              f"Lam_O(max={lambda_order.max():.2f}) Lam_E(max={lambda_exist.max():.2f})")

                    # 独立更新 lambda
                    lambda_order += mu_order * delta_hard_order
                    lambda_exist += mu_exist * delta_hard_exist
                    
                    # [关键修复] 只有当该约束的硬违规依然存在时，才放大对应的 mu
                    mu_order = torch.where(delta_hard_order > self.delta_tol, mu_order * self.mu_alpha, mu_order)
                    mu_order = torch.clamp(mu_order, max=self.mu_max)
                    
                    mu_exist = torch.where(delta_hard_exist > self.delta_tol, mu_exist * self.mu_alpha, mu_exist)
                    mu_exist = torch.clamp(mu_exist, max=self.mu_max)
                    
                    # 早停检查
                    max_delta = torch.max(delta_hard_order.max(), delta_hard_exist.max())
                    if max_delta < self.delta_tol:
                        break

        return y.transpose(1, 2).detach()

    # def project_to_constraint_space(
    #         self,
    #         log_probs: torch.Tensor,
    #         po_constraints: list,
    #         category_mask: torch.Tensor,
    #         existence_weight: float = 0.02,  # [新增] 存在性约束的权重
    #     ) -> torch.Tensor:
    #         """
    #         外层：硬判定 Δg_hard < δ 则提前返回；否则进入 ALM 内层优化。
    #         内层：KL + λ·Δg_soft + 0.5·μ·Δg_soft^2，迭代 inner_iterations 次。
    #         外层更新：λ ← λ + μ·Δg_hard；μ ← min(α·μ, μ_max)。
    #         """
    #         if not po_constraints or category_mask is None:
    #             return log_probs

    #         W_A, W_B = self._compile_constraints(po_constraints, log_probs.device)
        
    #         if W_A is None or W_B is None: 
    #             return log_probs

    #         y_model = log_probs.detach()
    #         y = y_model.clone().detach().requires_grad_(True)

    #         lambda_multiplier = torch.tensor(self.lambda_init, device=log_probs.device)
    #         mu = torch.tensor(self.mu_init, device=log_probs.device)

    #         for _ in range(self.outer_iterations):
    #             # 硬判定
    #             with torch.no_grad():
    #                 g_hard = self.compute_hard_constraint_violation(y, po_constraints, category_mask)
    #                 delta_hard = F.relu(g_hard - self.tau)
    #                 if delta_hard.max().item() <= self.delta_tol:
    #                     return y.detach()

    #             # 内层 ALM 优化
    #             gumbel_noise = None
    #             for _ in range(self.inner_iterations):
    #                 if y.grad is not None:
    #                     y.grad.zero_()

    #                 g_soft, gumbel_noise = self.compute_constraint_violation(
    #                     y, W_A,W_B, category_mask, gumbel_noise=gumbel_noise,projection_existence_weight=existence_weight
    #                 )
    #                 delta_soft = F.relu(g_soft - self.tau)

    #                 kl_div = (torch.exp(y) * (y - y_model)).sum(dim=(1, 2))
    #                 loss = (kl_div + lambda_multiplier * delta_soft + 0.5 * mu * (delta_soft ** 2)).mean()
    #                 loss.backward()

    #                 with torch.no_grad():
    #                     y = y - self.eta * y.grad
    #                     probs = torch.exp(y)
    #                     probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)
    #                     y = torch.log(probs + 1e-30).clamp(-70, 0).detach().requires_grad_(True)

    #             # 外层参数更新
    #             with torch.no_grad():
    #                 g_hard = self.compute_hard_constraint_violation(y, po_constraints, category_mask)
    #                 delta_hard = F.relu(g_hard - self.tau)
    #                 lambda_multiplier = lambda_multiplier + mu * delta_hard
    #                 mu = torch.clamp(mu * self.mu_alpha, max=self.mu_max)

    #         return y.detach()

    def apply_projection_to_category_positions(
        self,
        log_probs: torch.Tensor,
        po_constraints: list,
        category_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        原始版本：直接对整个分布做投影（内部按 category_mask 只关注类别位）
        """
        if not po_constraints or category_mask is None:
            return log_probs

        with torch.enable_grad():
            projected = self.project_to_constraint_space(
                log_probs,
                po_constraints,
                category_mask
            )
        return projected

    def compile_batched_constraints(self, po_constraints_list: list, device):
        """
        处理 Per-Sample Constraints (List[List[Tuple]])
        返回: W_A, W_B 形状为 [Batch, Type_Classes, Max_K]
        """
        B = len(po_constraints_list)
        # 找到最大的约束数量 Max_K
        max_k = max([len(c) for c in po_constraints_list])
        if max_k == 0:
            return None, None

        # 初始化 3D 矩阵 [B, V, K]
        W_A = torch.zeros((B, self.type_classes, max_k), device=device, dtype=torch.float32)
        W_B = torch.zeros((B, self.type_classes, max_k), device=device, dtype=torch.float32)

        # [新增] 约束掩码
        c_mask = torch.zeros((B, max_k), device=device) 
        # 填充矩阵
        for b, constraints in enumerate(po_constraints_list):
            for k, (indices_A, indices_B) in enumerate(constraints):
                # indices_A/B 是列表，例如 [0]
                if len(indices_A) > 0:
                    W_A[b, indices_A, k] = 1.0
                if len(indices_B) > 0:
                    W_B[b, indices_B, k] = 1.0
                c_mask[b,k] = 1.0
        
        return W_A, W_B, c_mask

    def _compile_constraints(self, po_constraints, device):
        """
        将列表形式的约束 [(A_idxs, B_idxs), ...] 编译为稀疏/稠密矩阵
        W_A, W_B: [Type_Classes, Num_Constraints]
        """
        if not po_constraints:
            return None, None
            
        num_constraints = len(po_constraints)
        # 初始化映射矩阵，形状 [V, K]
        # 注意：这里只映射 type_classes 部分，不用管 special tokens
        W_A = torch.zeros((self.type_classes, num_constraints), device=device, dtype=torch.float32)
        W_B = torch.zeros((self.type_classes, num_constraints), device=device, dtype=torch.float32)
        
        for k, (indices_A, indices_B) in enumerate(po_constraints):
            # 假设 indices 是相对于 category_start 的偏移量
            # 如果 indices 是绝对 id，请自行调整
            if len(indices_A) > 0:
                W_A[indices_A, k] = 1.0
            if len(indices_B) > 0:
                W_B[indices_B, k] = 1.0
                
        return W_A, W_B

def parse_po_matrix_to_constraints(po_matrix: torch.Tensor, threshold: float = 0.5) -> list:
    C = po_matrix.shape[0]
    constraints = []
    for i in range(C):
        for j in range(C):
            if i != j and po_matrix[i, j] > threshold:
                constraints.append(([i], [j]))
    return constraints
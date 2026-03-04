
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
    ) -> torch.Tensor:
        """
        [高效版] 基于 Argmax 的硬违规计算
        复用 W_A/W_B 矩阵，逻辑与 Soft 版本完全一致，但输入是 One-Hot。
        """
        B, V, L = log_probs.shape
        
        # 1. 硬解码 (Argmax -> One-Hot)
        # log_probs: [B, V, L] -> argmax dim=1 -> [B, L]
        idx = log_probs.argmax(dim=1)
        
        # 转为 One-Hot Float [B, L, V]
        probs_hard = F.one_hot(idx, num_classes=self.num_classes).float()
        
        # 2. 应用 Mask (序列长度 Mask)
        if category_mask is not None:
            probs_hard = probs_hard * category_mask.unsqueeze(-1).float()

        # 3. 截取类别段 [B, L, V_type]
        probs_type = probs_hard[:, :, self.category_start : self.category_end]

        # 4. 投影到约束空间 [B, L, K]
        # P_A_all 现在的含义是：在位置 L, 约束 K 的 A 类是否出现 (0 或 1)
        P_A_all = torch.matmul(probs_type, W_A) 
        P_B_all = torch.matmul(probs_type, W_B)

        # 5. Mask 掉无效约束 (针对 Batched 且约束数量不一的情况)
        if constraint_mask is not None:
            mask_expanded = constraint_mask.unsqueeze(1) # [B, 1, K]
            P_A_all = P_A_all * mask_expanded
            P_B_all = P_B_all * mask_expanded

        # ==========================================
        # 6. 计算顺序违规 (Order Violation) - 硬逻辑
        # ==========================================
        # 计算前缀和 (Prefix Sum)
        P_B_cumsum = torch.cumsum(P_B_all, dim=1)
        P_B_prefix = torch.zeros_like(P_B_cumsum)
        P_B_prefix[:, 1:, :] = P_B_cumsum[:, :-1, :]
        
        # 如果 A 在这里出现 (1.0), 且前面出现过 N 次 B, 则违规 N 次
        order_per_k = (P_B_prefix * P_A_all).sum(dim=1) # [B, K]

        # ==========================================
        # 7. 计算存在性违规 (Existence Violation) - 硬逻辑
        # ==========================================
        count_A = P_A_all.sum(dim=1) # [B, K]
        count_B = P_B_all.sum(dim=1) # [B, K]

        # 硬阈值检查：如果没有出现 (count=0), 则违规为 1.0
        target_count = 1.0
        viol_exist_A = F.relu(target_count - count_A)
        viol_exist_B = F.relu(target_count - count_B)
        
        exist_per_k = viol_exist_A + viol_exist_B

        # ==========================================
        # 8. 总和
        # ==========================================
        # 注意：这里也必须加上 existence_weight，保持与内层优化目标一致
        total_violation = order_per_k.sum(dim=1) + self.projection_existence_weight * exist_per_k.sum(dim=1)

        return total_violation

    def compute_constraint_violation_optimized(
        self,
        log_probs: torch.Tensor,
        W_A: torch.Tensor, # [V_type, K] 预计算好的矩阵
        W_B: torch.Tensor, # [V_type, K]
        category_mask: torch.Tensor,
        constraint_mask: torch.Tensor = None,
        gumbel_noise=None,
        #projection_existence_weight: float = 0.02,  # [新增] 存在性约束的权重
    ) -> torch.Tensor:
        """
        优化版 g(Y) 计算：
        复杂度: O(L) (相对于 O(L^2) 的矩阵乘法)
        并行度: 同时计算所有 K 个约束
        """
        # log_probs: [B, V, L]
        B, V, L = log_probs.shape
        
        # 1. 获取概率分布 [B, L, V]
        logits_lv = log_probs.transpose(1, 2)
        if self.use_gumbel_softmax:
            probs, gumbel_noise = self._gumbel_softmax_relax(
                logits_lv, tau=self.gumbel_temperature, gumbel_noise=gumbel_noise
            )
        else:
            probs = torch.softmax(logits_lv, dim=-1)

        # 2. 应用 Category Mask (只保留类别位置的概率)
        # category_mask: [B, L] -> [B, L, 1]
        if category_mask is not None:
            probs = probs * category_mask.unsqueeze(-1).float()

        # 3. 截取类别部分的概率 [B, L, V_type]
        # 假设 category 在中间段
        probs_type = probs[:, :, self.category_start : self.category_end]

        # 4. 投影到约束空间 [B, L, K]
        # P_A[b, l, k] 表示在 batch b, 时刻 l, 约束 k 的 A 类发生概率
        # 矩阵乘法: [B, L, V_type] @ [V_type, K] -> [B, L, K]
        P_A_all = torch.matmul(probs_type, W_A) 
        P_B_all = torch.matmul(probs_type, W_B)

        # [新增] 在计算前缀和之前，先 Mask 掉无效约束的概率值
        # 这样无效约束列的全是 0，cumsum 也是 0，彻底杜绝干扰
        if constraint_mask is not None:
            # constraint_mask: [B, K] -> 广播到 [B, L, K]
            mask_expanded = constraint_mask.unsqueeze(1)
            P_A_all = P_A_all * mask_expanded
            P_B_all = P_B_all * mask_expanded
        # 5. 计算前缀和 (Prefix Sum) 代替矩阵乘法 M
        # 我们需要 sum_{i < j} P_B(i) * P_A(j)
        # 令 Cum_P_B(j) = sum_{i=0}^{j} P_B(i)
        # 则 sum_{i < j} P_B(i) = Cum_P_B(j-1)
        # 也就是 P_B_all 的 exclusive cumsum
        
        # 计算包含当前位置的累加和 [B, L, K]
        P_B_cumsum = torch.cumsum(P_B_all, dim=1)
        
        # 偏移一位得到 exclusive cumsum (第0位补0)
        # [B, L, K] -> shift -> [B, L, K]
        P_B_prefix = torch.zeros_like(P_B_cumsum)
        P_B_prefix[:, 1:, :] = P_B_cumsum[:, :-1, :]
        
        # 6. 计算违规积并求和
        # term[b, l, k] = (在此之前出现过B的概率总和) * (当前是A的概率)
        violation_matrix = P_B_prefix * P_A_all # [B, L, K]
        
        # 对 Time(L) 和 Constraints(K) 维度求和，得到每个 Batch 的总违规
        order_per_k = violation_matrix.sum(dim=1) # [b,k]

        # -----------------------------------------------------------
        # 2. 新增：存在性违规 (Existence Violation)
        # -----------------------------------------------------------
        # 计算整条序列中，A 和 B 的期望出现次数 (Expected Count)
        # sum over Length -> [Batch, K]
        count_A = P_A_all.sum(dim=1)
        count_B = P_B_all.sum(dim=1)

        # 目标：期望次数至少为 1.0 (或者 0.9 以留有余地)
        # 如果 count < 1.0, 产生惩罚 (1.0 - count)
        # 使用 ReLU 确保如果次数 > 1.0 则无惩罚
        target_count = 1.0
        viol_exist_A = F.relu(target_count - count_A)
        viol_exist_B = F.relu(target_count - count_B)

        # 对所有约束 K 求和 -> [B]
        exist_per_k = (viol_exist_A + viol_exist_B)

        if constraint_mask is not None:
            order_per_k = order_per_k * constraint_mask
            exist_per_k = exist_per_k * constraint_mask

        total_violation = order_per_k.sum(dim=1) + self.projection_existence_weight * exist_per_k.sum(dim=1)

        return total_violation, gumbel_noise

    def project_with_matrices(
            self,
            log_probs: torch.Tensor,
            W_A: torch.Tensor,
            W_B: torch.Tensor,
            category_mask: torch.Tensor,
            constraint_mask = None,
        ) -> torch.Tensor:
            
        # 1. 准备优化变量 (Detach from original graph, create new leaf)
        # log_probs: [B, V, L] -> [B, L, V]
        y_model = log_probs.transpose(1, 2).detach() 
        y = y_model.clone().detach().requires_grad_(True)
        
        # 使用优化器
        optimizer = torch.optim.SGD([y], lr=self.eta)
        B = log_probs.shape[0] 
            
        # [修正] 初始化为形状 [B] 的向量，而不是标量
        lambda_multiplier = torch.full((B,), self.lambda_init, device=log_probs.device)
        #lambda_multiplier = torch.tensor(self.lambda_init, device=log_probs.device)
        mu = torch.tensor(self.mu_init, device=log_probs.device)

        # ========================================================
        # [CRITICAL FIX] 显式开启梯度计算，覆盖外部的 @torch.no_grad()
        # ========================================================
        with torch.enable_grad(): 
            for _ in range(self.outer_iterations):
                # 1. 硬判定 (Optional: 只有在外层需要检查时才做，为了速度可以跳过)
                # ...

                # 2. 内层循环 (ALM Optimization)
                gumbel_noise = None
                for _ in range(self.inner_iterations):
                    optimizer.zero_grad()
                    
                    # [重要] 这里的计算现在会构建计算图了
                    g_soft, gumbel_noise = self.compute_constraint_violation_optimized(
                        y.transpose(1, 2), # compute 需要 [B, V, L] 格式? 
                        # 修正：compute_constraint_violation_optimized 第一行是 transpose(1,2)
                        # 这意味着它期望输入是 [B, V, L]。
                        # 而这里的 y 是 [B, L, V]。
                        # 所以这里应该传入 y.transpose(1, 2) 是对的！
                        W_A, W_B, 
                        category_mask, 
                        constraint_mask,
                        gumbel_noise=gumbel_noise
                    )
                    
                    delta_soft = F.relu(g_soft - self.tau)
                    
                    # KL 计算
                    log_p = F.log_softmax(y, dim=-1)
                    # y_model 是 detach 过的，且不需要梯度，视为 Target
                    log_q = F.log_softmax(y_model, dim=-1) 
                    
                    # Forward KL: sum P_new * (log P_new - log P_old)
                    kl_loss = F.kl_div(log_p, log_q, reduction='batchmean',log_target=True) 

                    constraint_loss = (lambda_multiplier * delta_soft + 0.5 * mu * (delta_soft ** 2)).mean()
                    
                    loss = kl_loss + constraint_loss
                    
                    # 现在这里可以正常反向传播了
                    loss.backward()
                    
                    # grad_norm = y.grad.norm().item()
                    # if _ == 0: # 只在第一次迭代打印
                    #     print(f"[DEBUG] Inner Iter 0: Loss={loss.item():.4f}, Grad Norm={grad_norm:.4f}")
                    #     print(f"[DEBUG] Params: exist_w={self.projection_existence_weight}, eta={self.eta}")

                    # 梯度裁剪 (建议加上，防止 NaN)
                    #torch.nn.utils.clip_grad_norm_([y], 1.0)
                    #print(f"GRAD NORM: {y.grad.norm().item()}") # 如果是 0，说明梯度没传回来
                    optimizer.step()
                
                # 3. 外层参数更新 (不需要梯度)
                with torch.no_grad():
                    g_hard = self.compute_hard_constraint_violation_optimized(
                    y.transpose(1, 2), W_A, W_B,  category_mask,constraint_mask
                    )
                    delta_hard = F.relu(g_hard - self.tau)
                    
                    # 更新 lambda 和 mu
                    lambda_multiplier += mu * delta_hard
                    mu = torch.clamp(mu * self.mu_alpha, max=self.mu_max)
                    
                    # 早停检查 (可选)
                    if delta_hard.max() < self.delta_tol:
                        break

        # 返回优化后的结果 [B, V, L]
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
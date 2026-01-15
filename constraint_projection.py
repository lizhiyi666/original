"""
Constraint Discrete Diffusion - Projection Method Implementation
实现基于 g(⋅)设计方案.md 的可微约束投影
"""

import torch
import torch.nn.functional as F


class ConstraintProjection:
    """
    实现 Constrained Discrete Diffusion 中的投影方法
    用于在采样阶段强制满足 POI 类别间的偏序关系
    """
    
    def __init__(
        self,
        num_classes: int,
        type_classes: int,
        num_spectial: int,
        tau: float = 0.0,
        lambda_init: float = 1.0,
        alm_iterations: int = 10,
        eta: float = 0.2,
        mu: float = 1.0,
        device: str = 'cuda',
        use_gumbel_softmax: bool = True,
        gumbel_temperature: float = 1.0,
    ):
        """
        Args:
            num_classes: 总类别数
            type_classes: POI 类别数量
            num_spectial: 特殊 token 数量
            tau: 约束阈值（论文中建议设为 0）
            lambda_init: ALM 算法初始拉格朗日乘子
            alm_iterations: ALM 内循环迭代次数（论文建议 10-20）
            eta: ALM 学习率 η，用于更新 y 的步长（论文建议 0.2）
            mu: ALM 惩罚权重 μ，二次惩罚项系数（论文建议 1.0）
            device: 设备
        """
        self.num_classes = num_classes
        self.type_classes = type_classes
        self.num_spectial = num_spectial
        self.tau = tau
        self.lambda_init = lambda_init
        self.alm_iterations = alm_iterations
        self.eta = eta  # 学习率 η
        self.mu = mu  # 惩罚权重 μ
        self.device = device
        
        # 类别范围：[num_spectial : num_spectial + type_classes]
        self.category_start = num_spectial
        self.category_end = num_spectial + type_classes

        self.use_gumbel_softmax = use_gumbel_softmax
        self.gumbel_temperature = gumbel_temperature

    def _sample_gumbel(self, shape, device, dtype):
        U = torch.rand(shape, device=device, dtype=dtype)
        return -torch.log(-torch.log(U + 1e-20) + 1e-20)

    def _gumbel_softmax_relax(self, logits_lv, tau, gumbel_noise=None):
        """
        logits_lv: [B,L,V]
        gumbel_noise: [B,L,V] or None
        return:
        x_tilde: [B,L,V]
        gumbel_noise: [B,L,V]
        """
        if gumbel_noise is None:
            gumbel_noise = self._sample_gumbel(logits_lv.shape, logits_lv.device, logits_lv.dtype)
        y = (logits_lv + gumbel_noise) / max(tau, 1e-6)
        return torch.softmax(y, dim=-1), gumbel_noise
    
    # def compute_constraint_violation(
    #     self,
    #     log_probs: torch.Tensor,
    #     po_constraints: list,
    #     category_mask: torch.Tensor,
    #     gumbel_noise
    # ) -> torch.Tensor:
    #     """
    #     计算约束违规程度 g(Y)
        
    #     根据 g(⋅)设计方案.md:
    #     g_total(Y) = Σ_k Σ_{i<j} P_{B_k}(i) * P_{A_k}(j)
        
    #     Args:
    #         log_probs: [B, V, L] log概率分布
    #         po_constraints: [(A_indices, B_indices), ...] 偏序约束列表
    #                        要求 A 必须出现在 B 之前（B 不能在 A 之前）
    #         category_mask: [B, L] 类别位置的掩码
            
    #     Returns:
    #         constraint_violation: [B] 每个样本的约束违规值
    #     """
    #     B, V, L = log_probs.shape
    #     dev = log_probs.device

    #     '''        # 转换为概率分布 [B, L, V]
    #     probs = torch.exp(log_probs).transpose(1, 2)
        
    #     # 只考虑类别位置
    #     if category_mask is not None:
    #         # [B, L, 1]
    #         mask_expanded = category_mask.unsqueeze(-1).float()
    #         probs = probs * mask_expanded
        
    #     # 提取类别概率 [B, L, type_classes]
    #     category_probs = probs[:, :, self.category_start:self.category_end]'''
    #     '''        logits_lv = log_probs.transpose(1, 2)
        
    #     if self.use_gumbel_softmax:
    #         x_tilde = F.gumbel_softmax(
    #             logits_lv,
    #             tau=self.gumbel_temperature,
    #             hard=False,
    #             dim=-1,
    #         )  # [B, L, V]
    #     else:
    #         # fallback：普通 softmax（仍可微，但不近似 argmax）
    #         x_tilde = torch.softmax(logits_lv, dim=-1)'''

        
    #     logits_lv = log_probs.transpose(1, 2)  # [B,L,V]

    #     if self.use_gumbel_softmax:
    #         x_tilde, gumbel_noise = self._gumbel_softmax_relax(
    #             logits_lv, tau=self.gumbel_temperature, gumbel_noise=gumbel_noise
    #         )
    #     else:
    #         x_tilde = torch.softmax(logits_lv, dim=-1)

    #     # 只考虑类别位置
    #     if category_mask is not None:
    #         x_tilde = x_tilde * category_mask.unsqueeze(-1).float()  # [B, L, V]

    #     # 提取类别 token 的概率段：[B, L, type_classes]
    #     category_probs = x_tilde[:, :, self.category_start:self.category_end]

    #     # 构建上三角掩码矩阵 M[i,j] = 1 if i < j else 0
    #     # [L, L]
    #     position_indices = torch.arange(L, device=dev)
    #     #M = (position_indices.unsqueeze(1) < position_indices.unsqueeze(0)).float()
    #     M = torch.triu(torch.ones(L, L, device=dev), diagonal=1)
        
    #     total_violation = torch.zeros(B, device=dev)
        
    #     # 遍历每个约束 (A_k, B_k)
    #     for A_indices, B_indices in po_constraints:
    #         # P_A[b, i] = Σ_{v ∈ A} probs[b, i, v]
    #         # [B, L]
    #         P_A = category_probs[:, :, A_indices].sum(dim=-1)
            
    #         # P_B[b, i] = Σ_{v ∈ B} probs[b, i, v]  
    #         # [B, L]
    #         P_B = category_probs[:, :, B_indices].sum(dim=-1)
            
    #         # 计算逆序对惩罚：Σ_{i<j} P_B(i) * P_A(j)
    #         # 使用矩阵乘法: P_B^T @ M @ P_A
    #         # [B, L] @ [L, L] -> [B, L] 
    #         #P_B_weighted = torch.matmul(P_B.unsqueeze(1), M).squeeze(1)
    #         # [B, L] * [B, L] -> [B]
    #         #violation_k = (P_B_weighted * P_A).sum(dim=1)
    #         prefix_B = torch.cumsum(P_B, dim=1) - P_B
    #         violation_k = (prefix_B * P_A).sum(dim=1)
    #         total_violation += violation_k
            
    #     return total_violation,gumbel_noise
    def _constraints_to_padded_tensors(self, po_constraints_per_sample: list, device):
        """
        po_constraints_per_sample: List[List[([a],[b]), ...]] length=B
        returns:
        A_idx: [B,K] long
        B_idx: [B,K] long
        valid: [B,K] bool
        """
        B = len(po_constraints_per_sample)
        K = max((len(x) for x in po_constraints_per_sample), default=0)
        if K == 0:
            return None, None, None

        A_idx = torch.zeros((B, K), dtype=torch.long, device=device)
        B_idx = torch.zeros((B, K), dtype=torch.long, device=device)
        valid = torch.zeros((B, K), dtype=torch.bool, device=device)

        for b, cs in enumerate(po_constraints_per_sample):
            for k, (A_list, B_list) in enumerate(cs):
                A_idx[b, k] = int(A_list[0])
                B_idx[b, k] = int(B_list[0])
                valid[b, k] = True
        return A_idx, B_idx, valid


    def compute_constraint_violation_batched(
        self,
        log_probs: torch.Tensor,       # [B,V,L]
        A_idx: torch.Tensor,           # [B,K]
        B_idx: torch.Tensor,           # [B,K]
        valid: torch.Tensor,           # [B,K] bool
        category_mask: torch.Tensor,   # [B,L]
        gumbel_noise: torch.Tensor | None = None,
    ):
        """
        return:
        g_y: [B]
        gumbel_noise: [B,L,V] (复用)
        """
        B, V, L = log_probs.shape
        logits_lv = log_probs.transpose(1, 2)  # [B,L,V]

        if self.use_gumbel_softmax:
            x_tilde, gumbel_noise = self._gumbel_softmax_relax(
                logits_lv, tau=self.gumbel_temperature, gumbel_noise=gumbel_noise
            )  # [B,L,V]
        else:
            x_tilde = torch.softmax(logits_lv, dim=-1)

        if category_mask is not None:
            x_tilde = x_tilde * category_mask.unsqueeze(-1).float()

        category_probs = x_tilde[:, :, self.category_start:self.category_end]  # [B,L,C], C=9

        K = A_idx.shape[1]
        A_g = A_idx[:, None, :].expand(B, L, K)
        B_g = B_idx[:, None, :].expand(B, L, K)

        P_A = torch.gather(category_probs, dim=2, index=A_g)  # [B,L,K]
        P_B = torch.gather(category_probs, dim=2, index=B_g)  # [B,L,K]

        # O(L) 前缀和：sum_{i<j} P_B(i)
        prefix_B = torch.cumsum(P_B, dim=1) - P_B            # [B,L,K]
        viol_k = (prefix_B * P_A).sum(dim=1)                 # [B,K]
        viol_k = viol_k * valid.float()

        g_y = viol_k.sum(dim=1)                               # [B]
        return g_y, gumbel_noise    
    

    def project_to_constraint_space(
        self,
        log_probs: torch.Tensor,
        po_constraints: list,
        category_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        使用 ALM (Augmented Lagrangian Method) 投影到约束空间
        
        优化目标:
        min_y  D_KL(y || y_model) + λ * max(0, g(y) - τ) + μ/2 * max(0, g(y) - τ)^2
        
        根据 Constrained Discrete Diffusion 论文（Appendix D）：
        - η (eta): 学习率，控制 y 的更新步长，推荐 0.2
        - μ (mu): 惩罚权重，二次惩罚项系数，推荐 1.0
        - 内循环次数: 推荐 10-20 次
        
        这是一个独立的优化过程，需要临时启用梯度计算
        
        Args:
            log_probs: [B, V, L] 模型输出的 log 概率
            po_constraints: 偏序约束列表
            category_mask: [B, L] 类别位置掩码
            
        Returns:
            projected_log_probs: [B, V, L] 投影后的 log 概率
        """
        # if not po_constraints or len(po_constraints) == 0:
        #     # 没有约束，直接返回
        #     return log_probs
        A_idx, B_idx, valid = self._constraints_to_padded_tensors(po_constraints, B, log_probs.device)
        if A_idx is None:
            return log_probs

        log_probs = torch.log_softmax(log_probs, dim=1)
        B, V, L = log_probs.shape
        
        # 初始化 - detach 并创建新的需要梯度的张量
        # 注意：即使在 torch.no_grad() 上下文中，这里也需要梯度
        y = log_probs.detach().clone()
        y.requires_grad_(True)
        
        lambda_multiplier = self.lambda_init
        mu = self.mu  # 使用配置的惩罚权重
        eta = self.eta  # 使用配置的学习率
        
        gumbel_noise=None

        # ALM 迭代 - 这是一个独立的优化循环
        for alm_iter in range(self.alm_iterations):
            # 清零梯度
            if y.grad is not None:
                y.grad.zero_()
                
            # 计算约束违规
            #g_y = self.compute_constraint_violation(y, po_constraints, category_mask)
            #g_y, gumbel_noise = self.compute_constraint_violation(y, po_constraints, category_mask, gumbel_noise=gumbel_noise)
            g_y, gumbel_noise = self.compute_constraint_violation_batched(
                y, A_idx, B_idx, valid, category_mask, gumbel_noise=gumbel_noise
            )
            # 计算违规量 Δg = max(0, g(y) - τ)
            delta_g = F.relu(g_y - self.tau)
            
            # KL 散度: D_KL(y || y_model)
            # KL(p||q) = Σ p * (log p - log q)
            #kl_div = (torch.exp(y) * (y - log_probs)).sum(dim=(1, 2))
            log_probs_const = log_probs.detach()
            kl_div = (torch.exp(y) * (y - log_probs_const)).sum(dim=(1, 2))
            
            # 增广拉格朗日函数
            # L = KL + λ * Δg + μ/2 * Δg^2
            aug_lagrangian = kl_div + lambda_multiplier * delta_g + (mu / 2) * (delta_g ** 2)
            
            # 总损失（对批次求平均）
            loss = aug_lagrangian.mean()
            
            # 计算梯度 - 这在 torch.no_grad() 外也能工作，因为 y.requires_grad=True
            loss.backward()
            
            # 梯度下降更新 y (不使用 torch.no_grad，因为我们需要跟踪这个操作)
            with torch.no_grad():
                # 使用配置的学习率 η 进行更新
                y_new = y - eta * y.grad
                
                # 重新归一化为有效的 log 概率
                # 先转换为概率空间
                probs = torch.exp(y_new)
                probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)
                # 转回 log 空间
                y_new = torch.log(probs + 1e-30).clamp(-70, 0)
                
                # 更新 y，并重新启用梯度追踪
                y = y_new.detach().requires_grad_(True)
                
            # 更新拉格朗日乘子
            with torch.no_grad():
                #g_y_current = self.compute_constraint_violation(y, po_constraints, category_mask)
                #g_y_current, _ = self.compute_constraint_violation(y, po_constraints, category_mask, gumbel_noise=gumbel_noise)
                g_y_current, _ = self.compute_constraint_violation_batched(
                    y, A_idx, B_idx, valid, category_mask, gumbel_noise=gumbel_noise
                )
                delta_g_current = F.relu(g_y_current - self.tau)
                # 乘子更新：λ ← λ + μ * max(0, g(y) - τ)
                lambda_multiplier = lambda_multiplier + mu * delta_g_current.mean().item()
        
        # 返回优化后的结果（detach 以避免梯度问题）
        return y.detach()
    
    def apply_projection_to_category_positions(
        self,
        log_probs: torch.Tensor,
        po_constraints: list,
        category_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        仅对类别位置应用投影，保持其他位置不变
        
        Args:
            log_probs: [B, V, L] log 概率分布
            po_constraints: 偏序约束
            category_mask: [B, L] 类别位置掩码
            
        Returns:
            log_probs_projected: [B, V, L] 投影后的分布
        """
        if not po_constraints or category_mask is None:
            return log_probs
            
        # 重要：即使在 torch.no_grad() 上下文中调用，
        # 投影优化也需要临时启用梯度
        # 使用 torch.enable_grad() 来覆盖外层的 no_grad
        with torch.enable_grad():
            # [关键] 让优化变量成为 leaf tensor
            log_probs_opt = log_probs.detach().clone()
            log_probs_opt.requires_grad_(True)
            # 对整个序列应用投影
            # 投影算法会利用 category_mask 来只关注类别位置
            projected = self.project_to_constraint_space(
                log_probs, 
                po_constraints, 
                category_mask
            )
        
        return projected


def parse_po_matrix_to_constraints(po_matrix: torch.Tensor, threshold: float = 0.5) -> list:
    """
    将偏序矩阵转换为约束列表格式
    
    po_matrix[i, j] = 1 表示类别 i 必须在类别 j 之前
    转换为约束格式: [(A_indices, B_indices), ...]
    其中 A 必须在 B 之前（B 不能在 A 之前）
    
    Args:
        po_matrix: [C, C] 偏序关系矩阵
        threshold: 判定阈值
        
    Returns:
        constraints: [(A_indices, B_indices), ...] 
                    A 是前置类别的索引列表，B 是后置类别的索引列表
    """
    C = po_matrix.shape[0]
    constraints = []
    
    # 找出所有的偏序关系
    for i in range(C):
        for j in range(C):
            if i != j and po_matrix[i, j] > threshold:
                # i 必须在 j 之前
                # 约束：j 不能在 i 之前
                # 所以 A=i, B=j
                constraints.append(([i], [j]))
    
    return constraints

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
        alm_iterations: int = 5,
        device: str = 'cuda'
    ):
        """
        Args:
            num_classes: 总类别数
            type_classes: POI 类别数量
            num_spectial: 特殊 token 数量
            tau: 约束阈值（论文中建议设为 0）
            lambda_init: ALM 算法初始拉格朗日乘子
            alm_iterations: ALM 内循环迭代次数
            device: 设备
        """
        self.num_classes = num_classes
        self.type_classes = type_classes
        self.num_spectial = num_spectial
        self.tau = tau
        self.lambda_init = lambda_init
        self.alm_iterations = alm_iterations
        self.device = device
        
        # 类别范围：[num_spectial : num_spectial + type_classes]
        self.category_start = num_spectial
        self.category_end = num_spectial + type_classes
        
    def compute_constraint_violation(
        self,
        log_probs: torch.Tensor,
        po_constraints: list,
        category_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算约束违规程度 g(Y)
        
        根据 g(⋅)设计方案.md:
        g_total(Y) = Σ_k Σ_{i<j} P_{B_k}(i) * P_{A_k}(j)
        
        Args:
            log_probs: [B, V, L] log概率分布
            po_constraints: [(A_indices, B_indices), ...] 偏序约束列表
                           要求 A 必须出现在 B 之前（B 不能在 A 之前）
            category_mask: [B, L] 类别位置的掩码
            
        Returns:
            constraint_violation: [B] 每个样本的约束违规值
        """
        B, V, L = log_probs.shape
        
        # 转换为概率分布 [B, L, V]
        probs = torch.exp(log_probs).transpose(1, 2)
        
        # 只考虑类别位置
        if category_mask is not None:
            # [B, L, 1]
            mask_expanded = category_mask.unsqueeze(-1).float()
            probs = probs * mask_expanded
        
        # 提取类别概率 [B, L, type_classes]
        category_probs = probs[:, :, self.category_start:self.category_end]
        
        # 构建上三角掩码矩阵 M[i,j] = 1 if i < j else 0
        # [L, L]
        position_indices = torch.arange(L, device=self.device)
        M = (position_indices.unsqueeze(1) < position_indices.unsqueeze(0)).float()
        
        total_violation = torch.zeros(B, device=self.device)
        
        # 遍历每个约束 (A_k, B_k)
        for A_indices, B_indices in po_constraints:
            # P_A[b, i] = Σ_{v ∈ A} probs[b, i, v]
            # [B, L]
            P_A = category_probs[:, :, A_indices].sum(dim=-1)
            
            # P_B[b, i] = Σ_{v ∈ B} probs[b, i, v]  
            # [B, L]
            P_B = category_probs[:, :, B_indices].sum(dim=-1)
            
            # 计算逆序对惩罚：Σ_{i<j} P_B(i) * P_A(j)
            # 使用矩阵乘法: P_B^T @ M @ P_A
            # [B, L] @ [L, L] -> [B, L] 
            P_B_weighted = torch.matmul(P_B.unsqueeze(1), M).squeeze(1)
            # [B, L] * [B, L] -> [B]
            violation_k = (P_B_weighted * P_A).sum(dim=1)
            
            total_violation += violation_k
            
        return total_violation
    
    def project_to_constraint_space(
        self,
        log_probs: torch.Tensor,
        po_constraints: list,
        category_mask: torch.Tensor,
        beta: float = 1.0
    ) -> torch.Tensor:
        """
        使用 ALM (Augmented Lagrangian Method) 投影到约束空间
        
        优化目标:
        min_y  D_KL(y || y_model) + λ * max(0, g(y) - τ) + β/2 * max(0, g(y) - τ)^2
        
        Args:
            log_probs: [B, V, L] 模型输出的 log 概率
            po_constraints: 偏序约束列表
            category_mask: [B, L] 类别位置掩码
            beta: 惩罚参数
            
        Returns:
            projected_log_probs: [B, V, L] 投影后的 log 概率
        """
        if not po_constraints or len(po_constraints) == 0:
            # 没有约束，直接返回
            return log_probs
            
        B, V, L = log_probs.shape
        
        # 初始化
        y = log_probs.detach().clone().requires_grad_(True)
        lambda_multiplier = self.lambda_init
        
        # ALM 迭代
        for alm_iter in range(self.alm_iterations):
            # 确保 y 需要梯度
            if not y.requires_grad:
                y.requires_grad_(True)
                
            # 计算约束违规
            g_y = self.compute_constraint_violation(y, po_constraints, category_mask)
            
            # 计算违规量 Δg = max(0, g(y) - τ)
            delta_g = F.relu(g_y - self.tau)
            
            # KL 散度: D_KL(y || y_model)
            # KL(p||q) = Σ p * (log p - log q)
            kl_div = (torch.exp(y) * (y - log_probs)).sum(dim=(1, 2))
            
            # 增广拉格朗日函数
            # L = KL + λ * Δg + β/2 * Δg^2
            aug_lagrangian = kl_div + lambda_multiplier * delta_g + (beta / 2) * (delta_g ** 2)
            
            # 总损失（对批次求平均）
            loss = aug_lagrangian.mean()
            
            # 梯度下降更新 y
            if y.grad is not None:
                y.grad.zero_()
            loss.backward()
            
            with torch.no_grad():
                # 梯度下降步长
                step_size = 0.1 / (alm_iter + 1)
                y = y - step_size * y.grad
                
                # 重新归一化为有效的 log 概率
                # 先转换为概率空间
                probs = torch.exp(y)
                probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)
                # 转回 log 空间
                y = torch.log(probs + 1e-30).clamp(-70, 0)
                
                # 需要再次设置 requires_grad
                y = y.detach().requires_grad_(True)
                
            # 更新拉格朗日乘子
            with torch.no_grad():
                g_y_current = self.compute_constraint_violation(y, po_constraints, category_mask)
                delta_g_current = F.relu(g_y_current - self.tau)
                lambda_multiplier = lambda_multiplier + beta * delta_g_current.mean().item()
        
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


"""
Constraint Discrete Diffusion - Projection Method Implementation
实现基于 g(⋅)设计方案.md 的可微约束投影（已还原到最原始版本）
- softmax/exp(log_probs)
- 上三角矩阵 M 做 matmul
- 不使用 Gumbel-Softmax relaxation
- 不复用噪声
- 不使用 batched 约束张量化
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
        lambda_init: float = 1.0,
        alm_iterations: int = 10,
        eta: float = 0.2,
        mu: float = 1.0,
        device: str = 'cuda',
        use_gumbel_softmax: bool = True,   # 保留参数但原始版本不使用
        gumbel_temperature: float = 1.0,   # 保留参数但原始版本不使用
    ):
        self.num_classes = num_classes
        self.type_classes = type_classes
        self.num_spectial = num_spectial
        self.tau = tau
        self.lambda_init = lambda_init
        self.alm_iterations = alm_iterations
        self.eta = eta
        self.mu = mu
        self.device = device

        self.category_start = num_spectial
        self.category_end = num_spectial + type_classes

        # 原始版本：不使用 gumbel
        self.use_gumbel_softmax = use_gumbel_softmax
        self.gumbel_temperature = gumbel_temperature

    def compute_constraint_violation(
        self,
        log_probs: torch.Tensor,
        po_constraints: list,
        category_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        原始版本 g(Y):
          g_total(Y) = Σ_k Σ_{i<j} P_{B_k}(i) * P_{A_k}(j)

        使用上三角矩阵 M 实现：
          violation_k = (P_B @ M) * P_A  再按位置求和
        """
        B, V, L = log_probs.shape
        dev = log_probs.device

        # 转为概率 [B,L,V]
        probs = torch.exp(log_probs).transpose(1, 2)

        # 只考虑类别位置
        if category_mask is not None:
            probs = probs * category_mask.unsqueeze(-1).float()

        # 取类别段 [B,L,type_classes]
        category_probs = probs[:, :, self.category_start:self.category_end]

        # 上三角矩阵 M: M[i,j]=1 if i<j else 0
        # 注意：你原来的实现用的是 torch.triu(diagonal=1)
        M = torch.triu(torch.ones(L, L, device=dev), diagonal=1)

        total_violation = torch.zeros(B, device=dev)

        for A_indices, B_indices in po_constraints:
            P_A = category_probs[:, :, A_indices].sum(dim=-1)  # [B,L]
            P_B = category_probs[:, :, B_indices].sum(dim=-1)  # [B,L]

            # [B,1,L] @ [L,L] -> [B,1,L] -> [B,L]
            P_B_weighted = torch.matmul(P_B.unsqueeze(1), M).squeeze(1)
            violation_k = (P_B_weighted * P_A).sum(dim=1)  # [B]
            total_violation += violation_k

        return total_violation

    def project_to_constraint_space(
        self,
        log_probs: torch.Tensor,
        po_constraints: list,
        category_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        原始 ALM 投影版本（不复用噪声、每次 g 都是确定性的 softmax/exp）。
        """
        if not po_constraints or len(po_constraints) == 0:
            return log_probs

        log_probs = torch.log_softmax(log_probs, dim=1)
        y_model = log_probs.detach()

        y = y_model.clone().requires_grad_(True)

        lambda_multiplier = self.lambda_init
        mu = self.mu
        eta = self.eta

        for _ in range(self.alm_iterations):
            if y.grad is not None:
                y.grad.zero_()

            g_y = self.compute_constraint_violation(y, po_constraints, category_mask)
            delta_g = F.relu(g_y - self.tau)

            # KL(p_y || p_model)，这里沿用你当前实现方式（exp(y) * (y - y_model)）
            kl_div = (torch.exp(y) * (y - y_model)).sum(dim=(1, 2))

            loss = (kl_div + lambda_multiplier * delta_g + (mu / 2) * (delta_g ** 2)).mean()
            loss.backward()

            with torch.no_grad():
                y_new = y - eta * y.grad

                probs = torch.exp(y_new)
                probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)
                y = torch.log(probs + 1e-30).clamp(-70, 0).detach().requires_grad_(True)

            with torch.no_grad():
                g_y_current = self.compute_constraint_violation(y, po_constraints, category_mask)
                delta_g_current = F.relu(g_y_current - self.tau)
                lambda_multiplier = lambda_multiplier + mu * delta_g_current.mean().item()

        return y.detach()

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


def parse_po_matrix_to_constraints(po_matrix: torch.Tensor, threshold: float = 0.5) -> list:
    C = po_matrix.shape[0]
    constraints = []
    for i in range(C):
        for j in range(C):
            if i != j and po_matrix[i, j] > threshold:
                constraints.append(([i], [j]))
    return constraints
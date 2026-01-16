
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
            outer_iterations: int = 1000,      # 论文参数：outer_itermax
            inner_iterations: int = 100,       # 论文参数：inner_itermax
            eta: float = 1.0,                  # 论文参数：η
            delta_tol: float = 0.25,           # 外层硬判定容忍 δ
            use_gumbel_softmax: bool = True,
            gumbel_temperature: float = 1.0,
            device: str = "cuda",
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

    def _sample_gumbel(self, shape, device, dtype):
        U = torch.rand(shape, device=device, dtype=dtype)
        return -torch.log(-torch.log(U + 1e-20) + 1e-20)

    def _gumbel_softmax_relax(self, logits_lv, tau, gumbel_noise=None):
        if gumbel_noise is None:
            gumbel_noise = self._sample_gumbel(logits_lv.shape, logits_lv.device, logits_lv.dtype)
        y = (logits_lv + gumbel_noise) / max(tau, 1e-6)
        return torch.softmax(y, dim=-1), gumbel_noise

    def compute_hard_constraint_violation(
        self,
        log_probs: torch.Tensor,
        po_constraints: list,
        category_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        基于 argmax 的硬违规度 g(y*): 使用 one-hot 序列计算 P_B @ M · P_A。
        """
        B, V, L = log_probs.shape
        dev = log_probs.device

        # y* = argmax(log_probs)
        idx = log_probs.argmax(dim=1)  # [B, L]
        probs = F.one_hot(idx, num_classes=self.num_classes).float()  # [B, L, V]
        if category_mask is not None:
            probs = probs * category_mask.unsqueeze(-1).float()

        category_probs = probs[:, :, self.category_start:self.category_end]
        M = torch.triu(torch.ones(L, L, device=dev), diagonal=1)

        total_violation = torch.zeros(B, device=dev)
        for A_indices, B_indices in po_constraints:
            P_A = category_probs[:, :, A_indices].sum(dim=-1)  # [B, L]
            P_B = category_probs[:, :, B_indices].sum(dim=-1)  # [B, L]
            P_B_weighted = torch.matmul(P_B.unsqueeze(1), M).squeeze(1)  # [B, L]
            violation_k = (P_B_weighted * P_A).sum(dim=1)  # [B]
            total_violation += violation_k
        return total_violation

    def compute_constraint_violation(
        self,
        log_probs: torch.Tensor,
        po_constraints: list,
        category_mask: torch.Tensor,
        gumbel_noise=None,
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
        #probs = torch.exp(log_probs).transpose(1, 2)
        
        logits_lv = log_probs.transpose(1, 2)  # [B,L,V]
        if self.use_gumbel_softmax:
            probs, gumbel_noise = self._gumbel_softmax_relax(
                logits_lv, tau=self.gumbel_temperature, gumbel_noise=gumbel_noise
            )  # probs = x_tilde
        else:
            probs = torch.softmax(logits_lv, dim=-1)
        
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

        return total_violation,gumbel_noise

    def project_to_constraint_space(
            self,
            log_probs: torch.Tensor,
            po_constraints: list,
            category_mask: torch.Tensor
        ) -> torch.Tensor:
            """
            外层：硬判定 Δg_hard < δ 则提前返回；否则进入 ALM 内层优化。
            内层：KL + λ·Δg_soft + 0.5·μ·Δg_soft^2，迭代 inner_iterations 次。
            外层更新：λ ← λ + μ·Δg_hard；μ ← min(α·μ, μ_max)。
            """
            if not po_constraints or category_mask is None:
                return log_probs

            y_model = log_probs.detach()
            y = y_model.clone().detach().requires_grad_(True)

            lambda_multiplier = torch.tensor(self.lambda_init, device=log_probs.device)
            mu = torch.tensor(self.mu_init, device=log_probs.device)

            for _ in range(self.outer_iterations):
                # 硬判定
                with torch.no_grad():
                    g_hard = self.compute_hard_constraint_violation(y, po_constraints, category_mask)
                    delta_hard = F.relu(g_hard - self.tau)
                    if delta_hard.max().item() <= self.delta_tol:
                        return y.detach()

                # 内层 ALM 优化
                gumbel_noise = None
                for _ in range(self.inner_iterations):
                    if y.grad is not None:
                        y.grad.zero_()

                    g_soft, gumbel_noise = self.compute_constraint_violation(
                        y, po_constraints, category_mask, gumbel_noise=gumbel_noise
                    )
                    delta_soft = F.relu(g_soft - self.tau)

                    kl_div = (torch.exp(y) * (y - y_model)).sum(dim=(1, 2))
                    loss = (kl_div + lambda_multiplier * delta_soft + 0.5 * mu * (delta_soft ** 2)).mean()
                    loss.backward()

                    with torch.no_grad():
                        y = y - self.eta * y.grad
                        probs = torch.exp(y)
                        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)
                        y = torch.log(probs + 1e-30).clamp(-70, 0).detach().requires_grad_(True)

                # 外层参数更新
                with torch.no_grad():
                    g_hard = self.compute_hard_constraint_violation(y, po_constraints, category_mask)
                    delta_hard = F.relu(g_hard - self.tau)
                    lambda_multiplier = lambda_multiplier + mu * delta_hard
                    mu = torch.clamp(mu * self.mu_alpha, max=self.mu_max)

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
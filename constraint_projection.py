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
        lambda_init: float = 0.0,
        mu_init: float = 1.0,
        mu_alpha: float = 2.0,
        mu_max: float = 1000.0,
        outer_iterations: int = 10,
        inner_iterations: int = 10,
        eta: float = 1.0,
        delta_tol: float = 0.25,
        use_gumbel_softmax: bool = True,
        gumbel_temperature: float = 1.0,
        device: str = "cuda",
        projection_existence_weight: float = 0.02,
    ):
        self.num_classes = int(num_classes)
        self.type_classes = int(type_classes)
        self.num_spectial = int(num_spectial)

        self.tau = float(tau)
        self.lambda_init = float(lambda_init)
        self.mu_init = float(mu_init)
        self.mu_alpha = float(mu_alpha)
        self.mu_max = float(mu_max)

        self.outer_iterations = int(outer_iterations)
        self.inner_iterations = int(inner_iterations)
        self.eta = float(eta)
        self.delta_tol = float(delta_tol)
        self.device = str(device)

        self.category_start = self.num_spectial
        self.category_end = self.num_spectial + self.type_classes

        self.use_gumbel_softmax = bool(use_gumbel_softmax)
        self.gumbel_temperature = float(gumbel_temperature)
        self.projection_existence_weight = float(projection_existence_weight)

    def _sample_gumbel(self, shape, device, dtype):
        U = torch.rand(shape, device=device, dtype=dtype)
        return -torch.log(-torch.log(U + 1e-20) + 1e-20)

    def _gumbel_softmax_relax(self, logits_lv, tau, gumbel_noise=None):
        if gumbel_noise is None:
            gumbel_noise = self._sample_gumbel(logits_lv.shape, logits_lv.device, logits_lv.dtype)
        y = (logits_lv + gumbel_noise) / max(float(tau), 1e-6)
        return torch.softmax(y, dim=-1), gumbel_noise

    def compute_hard_constraint_violation_optimized(
        self,
        log_probs: torch.Tensor,  # [B, V, L]
        W_A: torch.Tensor,        # [B, V_type, K] or [V_type, K] after batching; your code uses [B, V_type, K]
        W_B: torch.Tensor,
        category_mask: torch.Tensor,
        constraint_mask: torch.Tensor = None,
    ):
        B, V, L = log_probs.shape

        idx = log_probs.argmax(dim=1)  # [B, L]
        probs_hard = F.one_hot(idx, num_classes=self.num_classes).float()  # [B, L, V]

        if category_mask is not None:
            probs_hard = probs_hard * category_mask.unsqueeze(-1).float()

        probs_type = probs_hard[:, :, self.category_start:self.category_end]  # [B, L, V_type]

        P_A_all = torch.matmul(probs_type, W_A)  # [B, L, K]
        P_B_all = torch.matmul(probs_type, W_B)  # [B, L, K]

        if constraint_mask is not None:
            mask_expanded = constraint_mask.unsqueeze(1)  # [B, 1, K]
            P_A_all = P_A_all * mask_expanded
            P_B_all = P_B_all * mask_expanded

        # 顺序违规
        P_B_cumsum = torch.cumsum(P_B_all, dim=1)
        P_B_prefix = torch.zeros_like(P_B_cumsum)
        P_B_prefix[:, 1:, :] = P_B_cumsum[:, :-1, :]
        order_per_k = (P_B_prefix * P_A_all).sum(dim=1)  # [B, K]

        # 存在性违规
        count_A = P_A_all.sum(dim=1)  # [B, K]
        count_B = P_B_all.sum(dim=1)  # [B, K]
        target_count = 1.0
        viol_exist_A = F.relu(target_count - count_A)
        viol_exist_B = F.relu(target_count - count_B)
        exist_per_k = viol_exist_A + viol_exist_B  # [B, K]

        return order_per_k, exist_per_k

    def compute_constraint_violation_optimized(
        self,
        log_probs: torch.Tensor,  # [B, V, L]
        W_A: torch.Tensor,
        W_B: torch.Tensor,
        category_mask: torch.Tensor,
        constraint_mask: torch.Tensor = None,
        gumbel_noise=None,
    ):
        B, V, L = log_probs.shape

        logits_lv = log_probs.transpose(1, 2)  # [B, L, V]
        T = float(self.gumbel_temperature)
        if T <= 0:
            T = 1.0

        if self.use_gumbel_softmax:
            probs, gumbel_noise = self._gumbel_softmax_relax(logits_lv, tau=T, gumbel_noise=gumbel_noise)
        else:
            probs = torch.softmax(logits_lv / T, dim=-1)
            gumbel_noise = None

        if category_mask is not None:
            probs = probs * category_mask.unsqueeze(-1).float()

        probs_type = probs[:, :, self.category_start:self.category_end]  # [B, L, V_type]

        P_A_all = torch.matmul(probs_type, W_A)  # [B, L, K]
        P_B_all = torch.matmul(probs_type, W_B)  # [B, L, K]

        if constraint_mask is not None:
            mask_expanded = constraint_mask.unsqueeze(1)
            P_A_all = P_A_all * mask_expanded
            P_B_all = P_B_all * mask_expanded

        # 顺序违规
        P_B_cumsum = torch.cumsum(P_B_all, dim=1)
        P_B_prefix = torch.zeros_like(P_B_cumsum)
        P_B_prefix[:, 1:, :] = P_B_cumsum[:, :-1, :]
        violation_matrix = P_B_prefix * P_A_all
        order_per_k = violation_matrix.sum(dim=1)  # [B, K]

        # 存在性违规
        count_A = P_A_all.sum(dim=1)
        count_B = P_B_all.sum(dim=1)
        target_count = 1.0
        viol_exist_A = F.relu(target_count - count_A)
        viol_exist_B = F.relu(target_count - count_B)
        exist_per_k = viol_exist_A + viol_exist_B  # [B, K]

        if constraint_mask is not None:
            order_per_k = order_per_k * constraint_mask
            exist_per_k = exist_per_k * constraint_mask

        return order_per_k, exist_per_k, gumbel_noise

    def project_with_matrices(
        self,
        log_probs: torch.Tensor,
        W_A: torch.Tensor,
        W_B: torch.Tensor,
        category_mask: torch.Tensor,
        constraint_mask=None,
    ) -> torch.Tensor:

        y_model = log_probs.transpose(1, 2).detach()  # [B, L, V]
        y = y_model.clone().detach().requires_grad_(True)

        optimizer = torch.optim.SGD([y], lr=self.eta)
        B = log_probs.shape[0]
        K = W_A.shape[-1]

        dev = log_probs.device

        # ============================
        # 关键修复：在初始化处强制 float dtype
        # ============================
        lambda_order = torch.full((B, K), self.lambda_init, device=dev, dtype=torch.float32)
        mu_order     = torch.full((B, K), self.mu_init,     device=dev, dtype=torch.float32)
        lambda_exist = torch.full((B, K), self.lambda_init, device=dev, dtype=torch.float32)
        mu_exist     = torch.full((B, K), self.mu_init,     device=dev, dtype=torch.float32)

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

                    penalty_order = lambda_order * delta_soft_order + 0.5 * mu_order * (delta_soft_order ** 2)
                    penalty_exist = lambda_exist * delta_soft_exist + 0.5 * mu_exist * (delta_soft_exist ** 2)

                    constraint_loss = (
                        penalty_order.sum(dim=1) + self.projection_existence_weight * penalty_exist.sum(dim=1)
                    ).sum()

                    loss = 1.0 * kl_loss + constraint_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([y], max_norm=10.0)
                    optimizer.step()

                    last_kl_loss = float(kl_loss.item())
                    last_const_loss = float(constraint_loss.item())

                with torch.no_grad():
                    g_hard_order, g_hard_exist = self.compute_hard_constraint_violation_optimized(
                        y.transpose(1, 2), W_A, W_B, category_mask, constraint_mask
                    )

                    delta_hard_order = F.relu(g_hard_order - self.tau)
                    delta_hard_exist = F.relu(g_hard_exist - self.tau)

                    if outer_idx == 0 or (outer_idx + 1) % 10 == 0 or outer_idx == self.outer_iterations - 1:
                        print(
                            f"  [Outer {outer_idx+1:02d}/{self.outer_iterations}] "
                            f"Loss (KL={last_kl_loss:.4f}, Const={last_const_loss:.4f}) | "
                            f"Viol_Order(max={delta_hard_order.max():.2f}, mean={delta_hard_order.mean():.4f}) | "
                            f"Viol_Exist(max={delta_hard_exist.max():.2f}, mean={delta_hard_exist.mean():.4f}) | "
                            f"Mu_O(max={mu_order.max():.1f}) Mu_E(max={mu_exist.max():.1f}) | "
                            f"Lam_O(max={lambda_order.max():.2f}) Lam_E(max={lambda_exist.max():.2f})"
                        )

                    # 现在 lambda_* 一定是 float，不会再触发 Long cast 错误
                    lambda_order += mu_order * delta_hard_order
                    lambda_exist += mu_exist * delta_hard_exist

                    mu_order = torch.where(delta_hard_order > self.delta_tol, mu_order * self.mu_alpha, mu_order)
                    mu_order = torch.clamp(mu_order, max=self.mu_max)

                    mu_exist = torch.where(delta_hard_exist > self.delta_tol, mu_exist * self.mu_alpha, mu_exist)
                    mu_exist = torch.clamp(mu_exist, max=self.mu_max)

                    max_delta = torch.max(delta_hard_order.max(), delta_hard_exist.max())
                    if max_delta < self.delta_tol:
                        break

        return y.transpose(1, 2).detach()

    def compile_batched_constraints(self, po_constraints_list: list, device):
        B = len(po_constraints_list)
        max_k = max([len(c) for c in po_constraints_list]) if B > 0 else 0
        if max_k == 0:
            return None, None, None

        W_A = torch.zeros((B, self.type_classes, max_k), device=device, dtype=torch.float32)
        W_B = torch.zeros((B, self.type_classes, max_k), device=device, dtype=torch.float32)
        c_mask = torch.zeros((B, max_k), device=device, dtype=torch.float32)

        for b, constraints in enumerate(po_constraints_list):
            for k, (indices_A, indices_B) in enumerate(constraints):
                if len(indices_A) > 0:
                    W_A[b, indices_A, k] = 1.0
                if len(indices_B) > 0:
                    W_B[b, indices_B, k] = 1.0
                c_mask[b, k] = 1.0

        return W_A, W_B, c_mask

    def _compile_constraints(self, po_constraints, device):
        if not po_constraints:
            return None, None

        num_constraints = len(po_constraints)
        W_A = torch.zeros((self.type_classes, num_constraints), device=device, dtype=torch.float32)
        W_B = torch.zeros((self.type_classes, num_constraints), device=device, dtype=torch.float32)

        for k, (indices_A, indices_B) in enumerate(po_constraints):
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
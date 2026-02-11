"""
Baseline 3: Constraint Classifier for Classifier-Based Guidance

训练一个噪声感知的约束分类器:
  输入: x_t (噪声后的 log-onehot), t (时间步), category_mask
  输出: p(所有偏序约束满足 | x_t, t) ∈ [0, 1]

关键: forward 中使用 soft embedding (softmax 加权) 保持可微性,
      使得分类器梯度可以传回 log_x_t 输入。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class ConstraintClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        type_classes: int,
        num_spectial: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        num_timesteps: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.type_classes = type_classes
        self.num_spectial = num_spectial
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps

        # Token embedding 矩阵: 训练时和推理时都通过矩阵乘法使用
        self.token_embedding = nn.Embedding(num_classes, hidden_dim)

        self.time_embedding = nn.Sequential(
            nn.Embedding(num_timesteps + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.max_positions = 3000
        self.position_embedding = nn.Embedding(self.max_positions, hidden_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.max_positions).expand((1, -1))
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        log_x_t: torch.Tensor,
        t: torch.Tensor,
        category_mask: torch.Tensor,
        use_soft: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            log_x_t: [B, V, L] log-onehot 分布
            t: [B] 时间步
            category_mask: [B, L] category 位置 mask
            use_soft: 是否使用可微的 soft embedding（推理求梯度时为 True）

        Returns:
            logits: [B] 约束满足的 logit
        """
        B, V, L = log_x_t.shape

        if use_soft:
            # ====== 可微路径: softmax 加权 embedding ======
            # log_x_t: [B, V, L] -> [B, L, V]
            logits_lv = log_x_t.transpose(1, 2)
            probs = torch.softmax(logits_lv, dim=-1)  # [B, L, V] 可微

            # 加权求和: [B, L, V] @ [V, D] -> [B, L, D]
            emb_weight = self.token_embedding.weight  # [V, D]
            token_emb = torch.matmul(probs, emb_weight)  # [B, L, D] 可微

            # padding mask: 使用 argmax（不可微但只用于 mask，不参与梯度）
            x_indices = log_x_t.argmax(dim=1)  # [B, L]
            padding_mask = (x_indices == 3)
        else:
            # ====== 硬路径: argmax + lookup（训练时用，更快更稳定）======
            x_indices = log_x_t.argmax(dim=1)  # [B, L]
            token_emb = self.token_embedding(x_indices)  # [B, L, D]
            padding_mask = (x_indices == 3)

        # 位置 embedding
        pos_ids = self.position_ids[:, :L]
        pos_emb = self.position_embedding(pos_ids)

        # 时间步 embedding
        time_emb = self.time_embedding(t).unsqueeze(1)  # [B, 1, D]

        # 合并
        hidden = token_emb + pos_emb + time_emb  # [B, L, D]

        # Transformer 编码
        hidden = self.transformer(hidden, src_key_padding_mask=padding_mask)

        # 池化（只在 category 位置）
        if category_mask is not None:
            mask_float = category_mask.float().unsqueeze(-1)  # [B, L, 1]
            pooled = (hidden * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-8)
        else:
            valid_mask = (~padding_mask).float().unsqueeze(-1)
            pooled = (hidden * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)

        logits = self.classifier_head(pooled).squeeze(-1)  # [B]
        return logits

    def get_log_prob_gradient(
        self,
        log_x_t: torch.Tensor,
        t: torch.Tensor,
        category_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 ∇_{log_x_t} log p(y=satisfied | x_t, t)

        使用 use_soft=True 确保 softmax 加权 embedding 路径可微，
        梯度可以从 classifier_head 一直传回 log_x_t。
        """
        # detach + clone 创建新叶节点
        log_x_t_grad = log_x_t.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            # 关键: use_soft=True 使用可微路径
            logits = self.forward(log_x_t_grad, t, category_mask, use_soft=True)

            # log p(y=1) = log sigmoid(logits) = -softplus(-logits)
            log_prob = -F.softplus(-logits)
            log_prob_sum = log_prob.sum()
            log_prob_sum.backward()

        gradient = log_x_t_grad.grad.detach()
        return gradient


def compute_constraint_label(
    x_0_indices: torch.Tensor,
    po_matrix: torch.Tensor,
    category_mask: torch.Tensor,
    num_spectial: int,
    type_classes: int,
) -> torch.Tensor:
    """
    根据 ground truth x_0 和 po_matrix 计算二分类标签。
    1.0 = 满足所有约束，0.0 = 至少一个违规。
    """
    B, L = x_0_indices.shape

    if po_matrix.dim() == 2:
        po_matrix = po_matrix.unsqueeze(0).expand(B, -1, -1)

    C = type_classes
    device = x_0_indices.device
    labels = torch.ones(B, device=device)

    for b in range(B):
        cat_positions = category_mask[b].bool()
        cat_tokens = x_0_indices[b, cat_positions]
        cat_ids = cat_tokens - num_spectial

        valid = (cat_ids >= 0) & (cat_ids < C)
        if valid.sum() == 0:
            continue

        cat_ids_valid = cat_ids[valid]
        pm = po_matrix[b]

        violated = False
        for k in range(pm.shape[0]):
            if violated:
                break
            for j in range(pm.shape[1]):
                if k != j and pm[k, j] > 0.5:
                    k_positions = (cat_ids_valid == k).nonzero(as_tuple=True)[0]
                    j_positions = (cat_ids_valid == j).nonzero(as_tuple=True)[0]

                    if len(k_positions) == 0 or len(j_positions) == 0:
                        labels[b] = 0.0
                        violated = True
                        break

                    if k_positions.max().item() >= j_positions.min().item():
                        labels[b] = 0.0
                        violated = True
                        break

    return labels
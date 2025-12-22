import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialOrderLoss(nn.Module):
    def __init__(self, svd_components=None, device='cuda'):
        """
        Args:
            svd_components: 占位符，为了兼容现有接口，不再使用
        """
        super().__init__()
        # 不需要注册任何 buffer，极度轻量

    def forward(self, logits, target_matrix, mask=None):
        """
        Args:
            logits: [Batch, Num_Cats, Seq_Len] 模型预测的 Logits
            target_matrix: [Batch, Num_Cats, Num_Cats] 真实的偏序矩阵 (来自 batch.po_matrix)
            mask: [Batch, Seq_Len] Padding Mask
        """
        B, C, L = logits.shape
        
        # 1. 计算概率分布 [B, L, C]
        probs = F.softmax(logits.transpose(1, 2), dim=-1)
        
        if mask is not None:
            # Mask out padding
            probs = probs * mask.unsqueeze(-1).float()

        # 2. 构建预测的 "软邻接矩阵"
        # future_probs[b, t, c] = sum(probs[b, k, c] for k > t)
        future_probs = torch.flip(torch.cumsum(torch.flip(probs, [1]), dim=1), [1]) - probs
        
        # pred_adj[b, i, j] = \sum_t ( P(x_t=i) * \sum_{k>t} P(x_k=j) )
        # 结果形状: [B, C, C]
        pred_adj = torch.bmm(probs.transpose(1, 2), future_probs)
        
        # 3. 归一化 (关键)
        # 真实的 target_matrix 是 0/1 矩阵。
        # 预测的 pred_adj 是累积概率，其值可能很大。需要归一化到 [0, 1] 区间或类似量级。
        # 简单归一化：除以序列有效长度的平方的一半 (L*(L-1)/2) 近似值
        valid_len = mask.sum(dim=1).view(B, 1, 1).float() if mask is not None else float(L)
        scale_factor = (valid_len * (valid_len - 1)) / 2 + 1e-6
        pred_adj_norm = pred_adj / scale_factor

        # 4. 计算 MSE Loss
        # 直接对比预测的矩阵和真实的偏序矩阵
        loss = F.mse_loss(pred_adj_norm, target_matrix)
        
        return loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialOrderLoss(nn.Module):
    def __init__(self, svd_components=None, device='cuda'):
        """
        Args:
            svd_components: 占位符，不再使用
        """
        super().__init__()
        self.device = device

    def forward(self, logits, target_matrix, mask=None):
        """
        Args:
            logits: [Batch, Num_Cats, Seq_Len] 模型预测的 Logits
            target_matrix: [Batch, Num_Cats, Num_Cats] 真实的偏序矩阵 (来自 batch.po_matrix)
            mask: [Batch, Seq_Len] Padding Mask
        """
        B, C, L = logits.shape
        
        # 1. 计算概率分布 [B, L, C]
        # Transpose: [B, C, L] -> [B, L, C]
        probs = F.softmax(logits.transpose(1, 2), dim=-1)
        
        if mask is not None:
            # Mask out padding (set prob to 0)
            probs = probs * mask.unsqueeze(-1).float()

        # 2. 构建 "软邻接矩阵" (Soft Adjacency Matrix)
        # M[i, j] 表示 "类别 i 出现在 类别 j 之前" 的期望强度
        # 公式: M = Probs^T @ (Future_Probs)
        
        # 计算 "未来累积概率" (Future Probabilities)
        # future_probs[b, t, c] = sum(probs[b, k, c] for k > t)
        # 技巧：通过 翻转 -> cumsum -> 翻转 - probs 实现后缀和
        # [B, L, C]
        future_probs = torch.flip(torch.cumsum(torch.flip(probs, [1]), dim=1), [1]) - probs
        
        # pred_adj: [B, C, C]
        # pred_adj[b, i, j] = \sum_t ( P(x_t=i) * P(x_{>t}=j) )
        pred_adj = torch.bmm(probs.transpose(1, 2), future_probs)
        
        # 3. 归一化 (关键步骤)
        # 预测矩阵中的值是概率乘积的累加，其最大值与序列长度 L 有关 (约等于 L^2/4)。
        # 而 target_matrix 是 0/1 二值矩阵。
        # 为了让 MSE Loss 有意义，必须将 pred_adj 归一化到 [0, 1] 附近的量级。
        
        if mask is not None:
            # 获取每个序列的有效长度 [B, 1, 1]
            valid_len = mask.sum(dim=1).view(B, 1, 1).float()
        else:
            valid_len = torch.tensor(float(L), device=probs.device).view(1, 1, 1)
            
        # 归一化因子：总的有效 Pair 对数约为 L*(L-1)/2
        scale_factor = (valid_len * (valid_len - 1)) / 2.0 + 1e-6
        
        pred_adj_norm = pred_adj / scale_factor

        # 我们只关心 A->B (A!=B) 的顺序，不惩罚 A->A
        num_cats = logits.shape[1]
        eye_mask = torch.eye(num_cats, device=self.device).bool().unsqueeze(0) # [1, C, C]
        
        # 将预测矩阵和目标矩阵的对角线都置为 0，或者只选非对角线元素计算 Loss
        pred_adj_norm = pred_adj_norm.masked_fill(eye_mask, 0.0)
        target_matrix = target_matrix.masked_fill(eye_mask, 0.0)
        
        # 4. 计算 MSE Loss
        # 确保 target 是 float 类型
        loss = F.mse_loss(pred_adj_norm, target_matrix.float())
        
        return loss
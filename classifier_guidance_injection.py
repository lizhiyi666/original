"""
Classifier-Based Guidance 的核心注入逻辑。
在 p_sample 中调用，将分类器梯度加到 model_log_prob 上。
"""

import torch
import torch.nn.functional as F


def apply_classifier_guidance(
    model_log_prob: torch.Tensor,
    log_x_t: torch.Tensor,
    t: torch.Tensor,
    classifier,
    category_mask: torch.Tensor,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """
    log p_guided(x_{t-1}) = log p(x_{t-1}|x_t) + s * ∇_{log_x_t} log p(y=1|x_t, t)
    """
    if classifier is None or guidance_scale == 0:
        return model_log_prob

    # 分类器梯度计算（内部自动开启 enable_grad）
    gradient = classifier.get_log_prob_gradient(log_x_t, t, category_mask)

    # 应用引导
    guided_log_prob = model_log_prob + guidance_scale * gradient

    # Clamp 防止数值问题
    guided_log_prob = torch.clamp(guided_log_prob, -70, 0)

    return guided_log_prob
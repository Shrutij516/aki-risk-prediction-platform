"""Binary focal loss, same formula as binary_focal_loss() in the notebooks.

Keras version: clip probabilities to [eps, 1-eps] (eps = 1e-7), pt = p if y=1
else 1-p, alpha_t = alpha if y=1 else 1-alpha, loss = mean(-alpha_t (1-pt)^gamma log pt).
"""
from __future__ import annotations

import torch

KERAS_EPSILON = 1e-7


def binary_focal_loss(logits: torch.Tensor, y: torch.Tensor, alpha: float, gamma: float = 2.0) -> torch.Tensor:
    p = torch.sigmoid(logits).clamp(KERAS_EPSILON, 1.0 - KERAS_EPSILON)
    y = y.float()
    pt = torch.where(y == 1, p, 1 - p)
    alpha_t = torch.where(y == 1, torch.full_like(p, alpha), torch.full_like(p, 1 - alpha))
    return (-alpha_t * (1 - pt).pow(gamma) * torch.log(pt)).mean()

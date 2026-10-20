"""A functional AdamW update on flat tensors.

ZeRO shards the optimizer by cutting one flat parameter vector into equal slices, so
the update has to run on an arbitrary slice that does not line up with parameter
boundaries. AdamW is elementwise, which is exactly what makes that legal: the update
of element ``i`` depends only on element ``i`` of the parameter, gradient and moments.
The operation order mirrors ``torch.optim.AdamW`` (single-tensor path) so that the
sharded optimizer can be tested against it at float64 precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AdamWConfig:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-2


@torch.no_grad()
def adamw_step_(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    cfg: AdamWConfig,
) -> None:
    """Apply one in-place AdamW update; ``step`` counts from 1."""
    param.mul_(1 - cfg.lr * cfg.weight_decay)  # decoupled weight decay
    exp_avg.lerp_(grad, 1 - cfg.beta1)
    exp_avg_sq.mul_(cfg.beta2).addcmul_(grad, grad, value=1 - cfg.beta2)
    bias_correction1 = 1 - cfg.beta1**step
    bias_correction2 = 1 - cfg.beta2**step
    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(cfg.eps)
    param.addcdiv_(exp_avg, denom, value=-cfg.lr / bias_correction1)

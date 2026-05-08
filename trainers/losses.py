from __future__ import annotations

import torch
import torch.nn.functional as F


def rectified_flow_loss(
    pred_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    return F.mse_loss(
        pred_velocity.float(),
        target_velocity.float(),
        reduction=reduction,
    )

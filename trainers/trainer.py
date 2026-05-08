from __future__ import annotations

from dataclasses import dataclass

import torch

from .losses import rectified_flow_loss


@dataclass
class TrainStepOutput:
    loss: torch.Tensor
    pred_velocity: torch.Tensor
    target_velocity: torch.Tensor


class EmotionIdentityTrainer:


    def __init__(self, pipeline, scheduler) -> None:
        self.pipeline = pipeline
        self.scheduler = scheduler

    def loss_from_prediction(
        self,
        pred_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
    ) -> torch.Tensor:
        return rectified_flow_loss(pred_velocity, target_velocity)

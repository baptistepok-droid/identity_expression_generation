from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .expression_adapter import ExpressionAdapter


@dataclass
class ConditionBranchOutput:
    tokens: torch.Tensor
    identity_token_count: int
    expression_token_count: int
    identity_grid: tuple[int, int, int] | None = None
    expression_grid: tuple[int, int, int] | None = None
    expression_token_indices: torch.Tensor | None = None


class DualConditionBuilder(nn.Module):
    """Build [identity tokens ; expression tokens] for a DiT-like backbone.

    The object expects a backbone module exposing:
    - `patchify(latents) -> (tokens, grid)`
    - `dim`
    - `num_heads`
    """

    def __init__(
        self,
        dit,
        expression_adapter: Optional[ExpressionAdapter] = None,
        identity_scale: float = 1.0,
        expression_scale: float = 1.0,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "dit", dit)
        self.expression_adapter = expression_adapter or ExpressionAdapter(
            dim=dit.dim,
        )
        self.identity_scale = identity_scale
        self.expression_scale = expression_scale

    def forward(
        self,
        identity_latents: Optional[torch.Tensor] = None,
        expression_latents: Optional[torch.Tensor] = None,
        expression_face_boxes: Optional[torch.Tensor] = None,
    ) -> Optional[ConditionBranchOutput]:
        pieces = []
        identity_count = 0
        expression_count = 0
        identity_grid = None
        expression_grid = None
        expression_token_indices = None

        if identity_latents is not None:
            identity_tokens, identity_grid = self.dit.patchify(identity_latents)
            identity_tokens = identity_tokens * self.identity_scale
            pieces.append(identity_tokens)
            identity_count = identity_tokens.shape[1]

        if expression_latents is not None:
            expression_vae_tokens, expression_grid = self.dit.patchify(expression_latents)
            expression_output = self.expression_adapter.select_tokens_with_indices(
                expression_vae_tokens,
                grid=expression_grid,
                face_boxes=expression_face_boxes,
            )
            expression_tokens = expression_output.tokens
            expression_token_indices = expression_output.token_indices
            expression_tokens = expression_tokens * self.expression_scale
            pieces.append(expression_tokens)
            expression_count = expression_tokens.shape[1]

        if not pieces:
            return None

        return ConditionBranchOutput(
            tokens=torch.cat(pieces, dim=1),
            identity_token_count=identity_count,
            expression_token_count=expression_count,
            identity_grid=identity_grid,
            expression_grid=expression_grid,
            expression_token_indices=expression_token_indices,
        )

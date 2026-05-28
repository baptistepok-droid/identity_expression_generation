from __future__ import annotations

from typing import Optional

import torch

from .condition_builder import DualConditionBuilder
from .rope_utils import condition_freqs_from_geometry, video_freqs
from .time_embedding import sinusoidal_embedding_1d


def model_fn_emotion_identity(
    dit,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    condition_builder: Optional[DualConditionBuilder] = None,
    identity_latents: Optional[torch.Tensor] = None,
    expression_latents: Optional[torch.Tensor] = None,
    expression_face_boxes: Optional[torch.Tensor] = None,
    use_gradient_checkpointing: bool = False,
) -> torch.Tensor:
    """Architecture-level emotion/identity forward for a DiT-like backbone.

    `condition_tokens` are passed as a separate condition sequence to DiT
    blocks that implement restricted self-attention.
    """

    time_emb = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    time_mod = dit.time_projection(time_emb).unflatten(1, (6, dit.dim))
    text_context = dit.text_embedding(context)

    video_tokens, grid = dit.patchify(latents)
    freqs = video_freqs(dit, grid, video_tokens.device)

    condition = None
    if condition_builder is not None:
        condition = condition_builder(
            identity_latents=identity_latents,
            expression_latents=expression_latents,
            expression_face_boxes=expression_face_boxes,
        )

    condition_tokens = None
    condition_time_mod = None
    condition_token_counts = None
    if condition is not None:
        condition_tokens = condition.tokens
        condition_token_counts = (
            condition.identity_token_count,
            condition.expression_token_count,
        )
        condition_time = torch.zeros_like(timestep)
        condition_time_emb = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, condition_time)
        )
        condition_time_mod = dit.time_projection(condition_time_emb).unflatten(1, (6, dit.dim))
        cond_freqs = condition_freqs_from_geometry(
            dit=dit,
            main_grid=grid,
            device=video_tokens.device,
            identity_grid=condition.identity_grid,
            expression_grid=condition.expression_grid,
            expression_token_indices=condition.expression_token_indices,
        )
        freqs = torch.cat([freqs, cond_freqs], dim=0)

    for block in dit.blocks:
        if use_gradient_checkpointing:
            video_tokens, condition_tokens = torch.utils.checkpoint.checkpoint(
                block,
                video_tokens,
                text_context,
                time_mod,
                freqs,
                condition_tokens,
                condition_time_mod,
                condition_token_counts,
                use_reentrant=False,
            )
        else:
            video_tokens, condition_tokens = block(
                video_tokens,
                text_context,
                time_mod,
                freqs,
                x_ip=condition_tokens,
                t_mod_ip=condition_time_mod,
                condition_token_counts=condition_token_counts,
            )

    pred = dit.head(video_tokens, time_emb)
    return dit.unpatchify(pred, grid)

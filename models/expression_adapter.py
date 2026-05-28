from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ExpressionAdapterOutput:
    tokens: torch.Tensor
    token_indices: torch.Tensor


class ExpressionAdapter(nn.Module):
    """Select expression-video tokens for the condition branch.

    Input must already be tokenized from the expression video:

        expression video -> VAE latents -> DiT patchify -> expression_vae_tokens

    The adapter receives `expression_vae_tokens` with shape `[B, N, D]` and the
    patch grid `(F, H, W)`. If face boxes are provided, it keeps the tokens that
    spatially correspond to the face area. If no boxes are provided, it keeps all
    expression-video tokens.

    `face_boxes` are normalized boxes in image/latent coordinates:
    `[x1, y1, x2, y2]` in `[0, 1]`, with shape `[B, F, 4]` or `[B, 4]`.
    If no box is provided, all expression-video tokens are used.
    """

    def __init__(
        self,
        dim: int,
        max_expression_tokens: int = 8192,
        fallback_face_box: tuple[float, float, float, float] = (0.25, 0.12, 0.75, 0.82),
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_expression_tokens = max_expression_tokens
        self.fallback_face_box = fallback_face_box

    def forward(
        self,
        expression_vae_tokens: torch.Tensor,
        grid: tuple[int, int, int],
        face_boxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return selected expression tokens with shape `[B, K, D]`."""

        return self.select_tokens_with_indices(
            expression_vae_tokens,
            grid=grid,
            face_boxes=face_boxes,
        ).tokens

    def select_tokens_with_indices(
        self,
        expression_vae_tokens: torch.Tensor,
        grid: tuple[int, int, int],
        face_boxes: torch.Tensor | None = None,
    ) -> ExpressionAdapterOutput:
        """Return selected expression tokens and their flattened grid indices."""

        if face_boxes is None:
            token_count = expression_vae_tokens.shape[1]
            selected_ids = torch.arange(
                token_count,
                device=expression_vae_tokens.device,
            ).unsqueeze(0).expand(expression_vae_tokens.shape[0], token_count)
            return ExpressionAdapterOutput(expression_vae_tokens, selected_ids)

        selected_ids = self.face_token_indices(
            batch_size=expression_vae_tokens.shape[0],
            grid=grid,
            device=expression_vae_tokens.device,
            face_boxes=face_boxes,
        )
        gather_ids = selected_ids.unsqueeze(-1).expand(-1, -1, expression_vae_tokens.shape[-1])
        selected_tokens = expression_vae_tokens.gather(dim=1, index=gather_ids)
        return ExpressionAdapterOutput(selected_tokens, selected_ids)

    def face_token_indices(
        self,
        batch_size: int,
        grid: tuple[int, int, int],
        device: torch.device,
        face_boxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map face boxes to flattened patch-token indices."""

        frames, height, width = grid
        boxes = self._prepare_boxes(batch_size, frames, device, face_boxes)
        per_batch_indices = []

        yy, xx = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        x_centers = (xx.float() + 0.5) / width
        y_centers = (yy.float() + 0.5) / height

        for batch_id in range(batch_size):
            frame_indices = []
            for frame_id in range(frames):
                x1, y1, x2, y2 = boxes[batch_id, frame_id]
                mask = (
                    (x_centers >= x1)
                    & (x_centers <= x2)
                    & (y_centers >= y1)
                    & (y_centers <= y2)
                )
                spatial_ids = mask.flatten().nonzero(as_tuple=False).flatten()
                token_ids = frame_id * height * width + spatial_ids
                frame_indices.append(token_ids)

            indices = torch.cat(frame_indices, dim=0)
            if indices.numel() == 0:
                indices = torch.arange(frames * height * width, device=device)

            indices = self._fit_token_count(indices)
            per_batch_indices.append(indices)

        return torch.stack(per_batch_indices, dim=0)

    def _prepare_boxes(
        self,
        batch_size: int,
        frames: int,
        device: torch.device,
        face_boxes: torch.Tensor | None,
    ) -> torch.Tensor:
        if face_boxes is None:
            box = torch.tensor(self.fallback_face_box, device=device, dtype=torch.float32)
            return box.view(1, 1, 4).expand(batch_size, frames, 4)

        face_boxes = face_boxes.to(device=device, dtype=torch.float32)
        if face_boxes.dim() == 1 and face_boxes.shape[0] == 4:
            face_boxes = face_boxes.view(1, 1, 4)
        if face_boxes.dim() == 2:
            if face_boxes.shape[0] == batch_size:
                face_boxes = face_boxes[:, None, :]
            else:
                face_boxes = face_boxes[None, :, :]
        if face_boxes.dim() != 3 or face_boxes.shape[-1] != 4:
            raise ValueError(f"face_boxes must have shape [B, F, 4], [F, 4], or [B, 4], got {tuple(face_boxes.shape)}")

        if face_boxes.shape[0] == 1 and batch_size > 1:
            face_boxes = face_boxes.expand(batch_size, -1, -1)
        elif face_boxes.shape[0] != batch_size:
            raise ValueError(f"face_boxes batch size {face_boxes.shape[0]} does not match {batch_size}")

        if face_boxes.shape[1] == 1:
            face_boxes = face_boxes.expand(batch_size, frames, 4)
        elif face_boxes.shape[1] != frames:
            frame_ids = torch.linspace(
                0,
                face_boxes.shape[1] - 1,
                steps=frames,
                device=device,
            ).round().long()
            face_boxes = face_boxes.index_select(dim=1, index=frame_ids)
        return face_boxes.clamp(0.0, 1.0)

    def _fit_token_count(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() <= self.max_expression_tokens:
            return indices

        positions = torch.linspace(
            0,
            indices.numel() - 1,
            steps=self.max_expression_tokens,
            device=indices.device,
        ).long()
        return indices[positions]

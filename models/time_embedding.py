from __future__ import annotations

import torch


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    half_dim = dim // 2
    exponent = -torch.arange(
        half_dim,
        dtype=torch.float64,
        device=position.device,
    ) / half_dim
    freqs = 10000**exponent
    sinusoid = torch.outer(position.to(torch.float64), freqs)
    embedding = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return embedding.to(position.dtype)


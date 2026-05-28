from __future__ import annotations

import torch


def video_freqs(dit, grid, device):
    f, h, w = grid
    offset = 1
    freqs = torch.cat(
        [
            dit.freqs[0][offset : f + offset].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][offset : h + offset].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][offset : w + offset].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    )
    return freqs.reshape(f * h * w, 1, -1).to(device)


def _axis_freqs(freqs: torch.Tensor, start: int, length: int) -> torch.Tensor:
    ids = torch.arange(start, start + length, device=freqs.device)
    ids = ids.clamp(0, freqs.shape[0] - 1)
    return freqs.index_select(0, ids)


def _condition_grid_freqs(
    dit,
    grid: tuple[int, int, int],
    h_start: int,
    w_start: int,
    device,
    use_grid_time: bool = False,
):
    f, h, w = grid
    # Stand-In style for static identity uses the dedicated precomputed t=-1
    # slot. Video expression keeps its temporal grid. Both condition branches
    # can share the same h/w offset so neither is positionally farther from the
    # noisy video than the other.
    if use_grid_time:
        t_freq = _axis_freqs(dit.freqs[0], 1, f).view(f, 1, 1, -1).expand(f, h, w, -1)
    else:
        t_freq = dit.freqs[0][0].view(1, 1, 1, -1).expand(f, h, w, -1)
    h_freq = _axis_freqs(dit.freqs[1], h_start, h).view(1, h, 1, -1).expand(f, h, w, -1)
    w_freq = _axis_freqs(dit.freqs[2], w_start, w).view(1, 1, w, -1).expand(f, h, w, -1)
    return torch.cat([t_freq, h_freq, w_freq], dim=-1).reshape(f * h * w, 1, -1).to(device)


def condition_freqs_from_geometry(
    dit,
    main_grid,
    device,
    identity_grid: tuple[int, int, int] | None = None,
    expression_grid: tuple[int, int, int] | None = None,
    expression_token_indices: torch.Tensor | None = None,
):
    _, h, w = main_grid
    offset = 1
    condition_h_start = h + offset
    condition_w_start = w + offset
    pieces = []

    if identity_grid is not None:
        identity_freqs = _condition_grid_freqs(
            dit,
            identity_grid,
            h_start=condition_h_start,
            w_start=condition_w_start,
            device=device,
            use_grid_time=False,
        )
        pieces.append(identity_freqs)

    if expression_grid is not None:
        expression_freqs = _condition_grid_freqs(
            dit,
            expression_grid,
            h_start=condition_h_start,
            w_start=condition_w_start,
            device=device,
            use_grid_time=True,
        )
        if expression_token_indices is not None:
            # RoPE freqs are sequence-level, so this assumes the same selected
            # token positions across the batch. Current training/inference uses
            # batch size 1 for this branch.
            expression_freqs = expression_freqs.index_select(
                0,
                expression_token_indices[0].to(device=expression_freqs.device, dtype=torch.long),
            )
        pieces.append(expression_freqs)

    if not pieces:
        freq_dim = dit.freqs[0].shape[-1] + dit.freqs[1].shape[-1] + dit.freqs[2].shape[-1]
        return torch.empty(0, 1, freq_dim, device=device)
    return torch.cat(pieces, dim=0).to(device)

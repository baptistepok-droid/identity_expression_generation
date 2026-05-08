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


def _condition_anchor_freq(dit, h_position: int, w_position: int):
    return torch.cat(
        [
            dit.freqs[0][0].view(1, 1, 1, -1),
            dit.freqs[1][h_position].view(1, 1, 1, -1),
            dit.freqs[2][w_position].view(1, 1, 1, -1),
        ],
        dim=-1,
    ).reshape(1, 1, -1)


def condition_freqs(
    dit,
    main_grid,
    condition_count: int,
    device,
    condition_group_sizes: tuple[int, int] | None = None,
):
    _, h, w = main_grid
    offset = 1
    identity_h = min(h + offset, dit.freqs[1].shape[0] - 1)
    identity_w = min(w + offset, dit.freqs[2].shape[0] - 1)
    expression_h = min(h + offset + 1, dit.freqs[1].shape[0] - 1)
    expression_w = min(w + offset + 1, dit.freqs[2].shape[0] - 1)
    if condition_group_sizes is None:
        head_freq = _condition_anchor_freq(dit, identity_h, identity_w)
        return head_freq.repeat(condition_count, 1, 1).to(device)

    identity_count, expression_count = condition_group_sizes
    pieces = []
    if identity_count > 0:
        identity_freq = _condition_anchor_freq(dit, identity_h, identity_w)
        pieces.append(identity_freq.repeat(identity_count, 1, 1))
    if expression_count > 0:
        expression_freq = _condition_anchor_freq(dit, expression_h, expression_w)
        pieces.append(expression_freq.repeat(expression_count, 1, 1))
    if not pieces:
        freq_dim = dit.freqs[0].shape[-1] + dit.freqs[1].shape[-1] + dit.freqs[2].shape[-1]
        return torch.empty(0, 1, freq_dim, device=device)
    return torch.cat(pieces, dim=0).to(device)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from typing import Optional


try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn

    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    compatibility_mode=False,
):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x, tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def rope_apply(x, freqs, num_heads):
    dtype = x.dtype
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    x_out = torch.view_as_complex(
        x.to(compute_dtype).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    freqs = freqs.to(device=x.device, dtype=x_out.dtype)
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        device="cuda",
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)
        return up_hidden_states.to(orig_dtype)


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)


class WanSelfAttention(nn.Module):
    """Wan DiT self-attention with an optional identity/expression branch.

    When condition tokens are appended to the video tokens, video tokens attend
    to `[video ; condition]`, while condition tokens attend only to themselves.
    This is the attachment point for identity/expression tokens built by
    `DualConditionBuilder`.
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)
        self.kv_cache = None
        self.cond_size = None
        self.cond_group_sizes = None

    def init_lora(self, train=False, device=None, dtype=None, rank=128):
        dim = self.dim
        device = device or self.q.weight.device
        dtype = dtype or self.q.weight.dtype
        self.q_loras = LoRALinearLayer(dim, dim, rank=rank, device=device, dtype=dtype)
        self.k_loras = LoRALinearLayer(dim, dim, rank=rank, device=device, dtype=dtype)
        self.v_loras = LoRALinearLayer(dim, dim, rank=rank, device=device, dtype=dtype)

        for lora in [self.q_loras, self.k_loras, self.v_loras]:
            for param in lora.parameters():
                param.requires_grad = train

    def forward(self, x, freqs):
        if self.cond_size is not None:
            return self._forward_with_condition(x, freqs)

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)
    

    def _store_condition_winner_attention(
        self,
        q_main,
        k_cond,
        identity_count,
        expression_count,
        chunk_size,
    ):
        if identity_count <= 0 and expression_count <= 0:
            return

        q = rearrange(q_main.float(), "b s (n d) -> b n s d", n=self.num_heads)
        k_cond = rearrange(
            k_cond[:, : identity_count + expression_count].float(),
            "b s (n d) -> b n s d",
            n=self.num_heads,
        )

        heat_chunks = []
        group_chunks = []
        token_chunks = []
        for start in range(0, q.shape[2], chunk_size):
            q_chunk = q[:, :, start : start + chunk_size]
            scores = torch.matmul(q_chunk, k_cond.transpose(-2, -1)) / math.sqrt(self.head_dim)
            probs = scores.softmax(dim=-1)
            mean_probs = probs.mean(dim=1)
            top_prob, top_token = mean_probs.max(dim=-1)
            top_group = (top_token >= identity_count).long()
            heat_chunks.append(top_prob)
            group_chunks.append(top_group)
            token_chunks.append(top_token)

        self.last_condition_attention = torch.cat(heat_chunks, dim=1).detach().cpu()
        self.last_condition_attention_group = torch.cat(group_chunks, dim=1).detach().cpu()
        self.last_condition_attention_top_token = torch.cat(token_chunks, dim=1).detach().cpu()

    def _store_condition_attention(self, q_main, k_cond):
        if self.cond_group_sizes is None:
            return

        identity_count = int(self.cond_group_sizes[0])
        expression_count = int(self.cond_group_sizes[1]) if len(self.cond_group_sizes) > 1 else 0

        if getattr(self, "capture_condition_winner_attention", False):
            self._store_condition_winner_attention(
                q_main,
                k_cond,
                identity_count,
                expression_count,
                int(getattr(self, "condition_attention_chunk_size", 512)),
            )

    def _forward_with_condition(self, x, freqs):
        if not hasattr(self, "q_loras"):
            self.init_lora(train=True, device=x.device, dtype=self.q.weight.dtype)

        if self.kv_cache is None:
            x_main, x_cond = x[:, : -self.cond_size], x[:, -self.cond_size :]
            split_point = freqs.shape[0] - self.cond_size
            freqs_main = freqs[:split_point]
            freqs_cond = freqs[split_point:]

            q_main = self.norm_q(self.q(x_main))
            k_main = self.norm_k(self.k(x_main))
            v_main = self.v(x_main)
            q_main = rope_apply(q_main, freqs_main, self.num_heads)
            k_main = rope_apply(k_main, freqs_main, self.num_heads)

            q_cond = self.norm_q(self.q(x_cond) + self.q_loras(x_cond))
            k_cond = self.norm_k(self.k(x_cond) + self.k_loras(x_cond))
            v_cond = self.v(x_cond) + self.v_loras(x_cond)
            q_cond = rope_apply(q_cond, freqs_cond, self.num_heads)
            k_cond = rope_apply(k_cond, freqs_cond, self.num_heads)
            
            self._store_condition_attention(q_main, k_cond)

            self.kv_cache = {"k_cond": k_cond.detach(), "v_cond": v_cond.detach()}
            full_k = torch.concat([k_main, k_cond], dim=1)
            full_v = torch.concat([v_main, v_cond], dim=1)
            main_out = self.attn(q_main, full_k, full_v)
            cond_out = self._condition_group_attention(q_cond, k_cond, v_cond)
            return self.o(torch.concat([main_out, cond_out], dim=1))

        k_cond = self.kv_cache["k_cond"]
        v_cond = self.kv_cache["v_cond"]
        q_main = self.norm_q(self.q(x))
        k_main = self.norm_k(self.k(x))
        v_main = self.v(x)
        q_main = rope_apply(q_main, freqs, self.num_heads)
        k_main = rope_apply(k_main, freqs, self.num_heads)

        full_k = torch.concat([k_main, k_cond], dim=1)
        full_v = torch.concat([v_main, v_cond], dim=1)
        x = self.attn(q_main, full_k, full_v)
        return self.o(x)

    def _condition_group_attention(self, q_cond, k_cond, v_cond):
        group_sizes = self._valid_group_sizes()
        if group_sizes is None:
            return self.attn(q_cond, k_cond, v_cond)

        outputs = []
        start = 0
        for size in group_sizes:
            stop = start + size
            outputs.append(
                self.attn(
                    q_cond[:, start:stop],
                    k_cond[:, start:stop],
                    v_cond[:, start:stop],
                )
            )
            start = stop
        return torch.concat(outputs, dim=1)

    def _valid_group_sizes(self):
        if self.cond_group_sizes is None:
            return None
        group_sizes = [int(size) for size in self.cond_group_sizes if int(size) > 0]
        if len(group_sizes) <= 1:
            return None
        if sum(group_sizes) != self.cond_size:
            return None
        return group_sizes

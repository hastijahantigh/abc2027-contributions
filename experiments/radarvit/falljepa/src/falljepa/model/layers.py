import torch
import torch.nn as nn
import torch.nn.functional as F


def rotary(x: torch.Tensor) -> torch.Tensor:
    """RoPE on [B, heads, N, dim_head] at positions 0..N-1 (base 10000): q . k then depends on the
    positions only through their difference."""
    N, dh = x.shape[-2:]
    inv = 1.0 / 10000 ** (torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh)
    ang = torch.arange(N, device=x.device, dtype=torch.float32)[:, None] * inv[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Attention(nn.Module):
    """
    Multi-head attention over a dense batch [B, N, dim].

    Copied from gir (src/model/layers.py). gir packs lines of different widths into one sequence
    and needs FlexAttention block masks; every window here has the same length, so plain
    scaled_dot_product_attention with `is_causal` (or a boolean key mask) is enough, and it runs
    forward and backward on CPU too.
    """
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False, dropout: float = 0.0, rope: bool = False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.rope = rope                # self-attention only: q and k at positions 0..N-1
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, q_x: torch.Tensor, kv_x: torch.Tensor = None, causal: bool = False,
                key_valid: torch.Tensor | None = None) -> torch.Tensor:
        """key_valid: optional [B, N_kv] bool, False keys are never attended to."""
        kv_x = q_x if kv_x is None else kv_x
        B, N_q, _ = q_x.shape
        N_kv = kv_x.size(1)

        q = self.q_proj(q_x).view(B, N_q, self.num_heads, self.dim_head).transpose(1, 2)
        k, v = self.kv_proj(kv_x).view(B, N_kv, 2, self.num_heads, self.dim_head).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = rotary(q), rotary(k)

        mask = None
        if key_valid is not None:
            mask = key_valid[:, None, None, :]
            if causal:
                mask = mask & torch.ones(N_q, N_kv, dtype=torch.bool, device=q_x.device).tril()
            # A query with no visible key would give NaN: let it see itself
            mask = mask | torch.eye(N_q, N_kv, dtype=torch.bool, device=q_x.device)
            causal = False
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.out_proj(out.transpose(1, 2).reshape(B, N_q, self.dim))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-LN self-attention block over [B, N, dim]: causal or bidirectional, optionally with RoPE."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1, rope: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, rope=rope)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio * dim, dropout)

    def forward(self, x: torch.Tensor, causal: bool = False, key_valid: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal=causal, key_valid=key_valid)
        return x + self.ffn(self.norm2(x))

"""
ViT trunk parts for FallSetJEPA: a STARTING POINT, not a tuned model (handover/VIT_IDEAS.md).

They are drop-in replacements with the same call signatures as the JF6 trunk parts in model/stem.py, so
FallJEPA.encode() does not change:

    RadarViT    replaces RadarStem      x [B, 13, 184, L] , frames  -> z tokens [B, Fn, d]
    LocalViT    replaces TCN            h [B, Fn, d] , frame_valid  -> [B, Fn, d]

RadarViT (idea 2 of VIT_IDEAS.md): per radar, a complex "tubelet" patch embedding with kernel
(patch_bins x frame_width) and stride (patch_bins x frame_stride), so patch column i covers exactly
steps [4 i, 4 i + 32) = frame i, as the stem's token i does. |.| -> log1p makes it invariant to each
radar's global phase (as the stem's L0). Then `depth` transformer blocks attend over the
radars x range patches of the same frame (spatial attention only: frames stay independent, time is
mixed later by the local / bidirectional blocks), and a learned query pools them into one token.

LocalViT (idea 1): pre-LN transformer blocks with RoPE whose attention is limited to +- `radius`
tokens (radius 30 = 61 tokens = 4.9 s, the TCN's receptive field); invalid frames are masked as keys
and zeroed at the output, as the TCN does.

Checked by handover/test_vit_sketch.py (shapes, frame alignment, phase invariance, the band, and a
forward + backward pass through FallSetJEPA.encode with both parts swapped in).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from falljepa.model.layers import Block, FeedForward, rotary
from falljepa.model.stem import CLUT_I, CLUT_Q, RAW_I, RAW_Q, ComplexConv2d


class RadarViT(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4, depth: int = 2, patch_bins: int = 23, n_bins: int = 184,
                 frame_width: int = 32, frame_stride: int = 4, embed_complex: int = 32, mlp_ratio: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        assert n_bins % patch_bins == 0
        self.rows = n_bins // patch_bins
        # 2 complex inputs per radar (raw z and clutter-removed c) -> embed_complex complex features
        self.embed = ComplexConv2d(2, embed_complex, (patch_bins, frame_width), (patch_bins, frame_stride), (0, 0))
        self.norm_in = nn.LayerNorm(embed_complex)
        self.proj = nn.Linear(embed_complex, d_model)
        self.range_embed = nn.Parameter(torch.randn(self.rows, d_model) * 0.02)
        self.radar_embed = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.blocks = nn.ModuleList([Block(d_model, nhead, mlp_ratio, dropout) for _ in range(depth)])
        self.pool_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.pool = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm_out = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, mlp_ratio * d_model, dropout)

    def forward(self, x: torch.Tensor, frames: int) -> torch.Tensor:
        B = x.size(0)
        re = torch.stack([x[:, [RAW_I + r, CLUT_I + r]] for r in range(3)], 1)    # [B, 3, 2, H, L]
        im = torch.stack([x[:, [RAW_Q + r, CLUT_Q + r]] for r in range(3)], 1)
        y = self.embed(torch.cat([re, im], 2).flatten(0, 1))                       # [B 3, 2 e, rows, Fn']
        e = y.size(1) // 2
        a = torch.log1p(torch.sqrt(y[:, :e] ** 2 + y[:, e:] ** 2 + 1e-6))          # phase-invariant energies
        assert a.size(-1) >= frames, (a.shape, frames)
        a = a[..., :frames]                                                         # column i = frame i
        tok = self.proj(self.norm_in(a.permute(0, 3, 2, 1)))                       # [B 3, Fn, rows, d]
        tok = tok.view(B, 3, frames, self.rows, -1) + self.range_embed[None, None, None]
        tok = tok + self.radar_embed[None, :, None, None]
        # A radar zeroed by radar dropout is not attended to
        sig = torch.stack([x[:, [RAW_I + r, RAW_Q + r, CLUT_I + r, CLUT_Q + r]] for r in range(3)], 1)
        present = sig.flatten(2).abs().amax(-1) > 0                                 # [B, 3]
        present = present | ~present.any(1, keepdim=True)
        key_valid = present[:, None, :, None].expand(B, frames, 3, self.rows).reshape(B * frames, 3 * self.rows)
        s = tok.permute(0, 2, 1, 3, 4).reshape(B * frames, 3 * self.rows, -1)      # one sequence per frame
        for blk in self.blocks:
            s = blk(s, key_valid=key_valid)
        q = self.pool_query.expand(s.size(0), 1, -1)
        z, _ = self.pool(q, s, s, key_padding_mask=~key_valid, need_weights=False)
        z = z + self.ffn(self.norm_out(z))
        return z.view(B, frames, -1)


class BandBlock(nn.Module):
    """Pre-LN self-attention block with RoPE, each token attending to tokens within +- radius."""
    def __init__(self, d_model: int, nhead: int, radius: int, mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        self.h, self.radius, self.dropout = nhead, radius, dropout
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.ffn = FeedForward(d_model, mlp_ratio * d_model, dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        B, N, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, N, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = rotary(q), rotary(k)
        i = torch.arange(N, device=x.device)
        band = (i[:, None] - i[None, :]).abs() <= self.radius
        mask = band[None, None] & valid[:, None, None, :]
        mask = mask | torch.eye(N, dtype=torch.bool, device=x.device)               # no query without a key
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.out(y.transpose(1, 2).reshape(B, N, d))
        return x + self.ffn(self.norm2(x))


class LocalViT(nn.Module):
    """[B, F, d] -> [B, F, d], the TCN's signature: forward(x, frame_valid)."""
    def __init__(self, d_model: int = 64, nhead: int = 4, depth: int = 2, radius: int = 30, mlp_ratio: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([BandBlock(d_model, nhead, radius, mlp_ratio, dropout) for _ in range(depth)])

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        m = valid[..., None].to(x.dtype)
        x = x * m
        for blk in self.blocks:
            x = blk(x, valid)
        return x * m

"""Frame-aligned spatial encoder from handover/vit_sketch.py (Idea 2)."""
import torch
import torch.nn as nn

from .layers import Block, FeedForward
from .stem import CLUT_I, CLUT_Q, RAW_I, RAW_Q, ComplexConv2d


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

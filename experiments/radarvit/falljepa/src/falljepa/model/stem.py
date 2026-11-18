"""
The JF4 trunk parts (docs/falljepa_JF4_design.md, section 3): a convolutional stem run once over a
window instead of the per-frame patch embedding and frame transformer, the range pooling, the radar
fusion and the TCN local encoder.

    ComplexConv2d   a complex convolution on [re channels, im channels], no bias: it commutes with a
                    global phase rotation of its input, so |output| is phase-invariant
    RadarStem       per radar (shared weights): L0 complex conv (raw z and clutter-removed c) -> |.| ->
                    log1p -> BN, then L1-L3 real convs (+ the contrast maps at L1) down to 23 range
                    positions and one token per 4 steps; token i covers steps [4 i, 4 i + 32], frame i's
                    footprint. `fusion` "early" stacks the 3 radars' L0 energies as channels at L1 (one
                    map per window); "attention" keeps the radars apart up to RadarFusion
    RangePool       softmax attention pooling over the range positions (with a range embedding)
    RadarFusion     a learned query attends to the 3 radar tokens of each step; absent radars are masked
    TCN             non-causal dilated residual 1-D convolutions over the tokens, no pooling

Input layout (data.amplitude false): [I x3, Q x3, I_c x3, Q_c x3, contrast (k-major, x3 each), valid].
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Block, FeedForward

RAW_I, RAW_Q, CLUT_I, CLUT_Q, CONTRAST = 0, 3, 6, 9, 12


def contrast_from_context(ctx: torch.Tensor, ctx_valid: torch.Tensor, scales: list[tuple[int, int]]) -> torch.Tensor:
    """data.uwb.contrast_maps on the GPU (data.contrast_in_model): ctx [B, 6, H, L + 2 R] (raw I x3, Q x3
    over the item's steps and R = max(g + w) more on each side), ctx_valid [B, L + 2 R] -> [B, 3 K, H, L]
    k-major, per (gap g, width w) in timesteps and radar: |mean z[t + g, t + g + w) - mean z[t - g - w, t - g)|
    over the valid steps, 0 where a side has none."""
    R = max(g + w for g, w in scales)
    L = ctx.size(-1) - 2 * R
    v = ctx_valid.to(torch.float64)
    z = ctx.to(torch.float64) * v[:, None, None]
    mean = z.sum(-1, keepdim=True) / v.sum(-1).clamp_min(1)[:, None, None, None]
    C = F.pad(((z - mean) * v[:, None, None]).cumsum(-1), (1, 0)).float()   # a constant cancels; small sums
    N = F.pad(v.cumsum(-1), (1, 0)).float()
    out = []
    for g, w in scales:
        before = C[..., R - g:R - g + L] - C[..., R - g - w:R - g - w + L]
        after = C[..., R + g + w:R + g + w + L] - C[..., R + g:R + g + L]
        na = N[:, R - g:R - g + L] - N[:, R - g - w:R - g - w + L]
        nb = N[:, R + g + w:R + g + w + L] - N[:, R + g:R + g + L]
        d = after / nb.clamp_min(1)[:, None, None] - before / na.clamp_min(1)[:, None, None]
        d = d * ((na > 0) & (nb > 0))[:, None, None]
        out.append(torch.sqrt(d[:, :3] ** 2 + d[:, 3:] ** 2))
    return torch.cat(out, 1)


class ComplexConv2d(nn.Module):
    """[N, 2 in, H, W] (re channels, then im channels) -> [N, 2 out, H', W'], no bias."""
    def __init__(self, in_channels: int, out_channels: int, kernel: tuple[int, int], stride: tuple[int, int],
                 padding: tuple[int, int]):
        super().__init__()
        bound = 1 / math.sqrt(2 * in_channels * kernel[0] * kernel[1])
        self.wr = nn.Parameter(torch.empty(out_channels, in_channels, *kernel).uniform_(-bound, bound))
        self.wi = nn.Parameter(torch.empty(out_channels, in_channels, *kernel).uniform_(-bound, bound))
        self.stride, self.padding = stride, padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.cat([torch.cat([self.wr, -self.wi], 1), torch.cat([self.wi, self.wr], 1)], 0)
        return F.conv2d(x, w, stride=self.stride, padding=self.padding)


def _conv_bn(cin: int, cout: int, kernel, stride, padding) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, stride, padding, bias=False), nn.BatchNorm2d(cout), nn.GELU())


class RangePool(nn.Module):
    """[N, d, P, T] -> [N, T, d]: `heads` softmax distributions over the P range positions."""
    def __init__(self, d_model: int, positions: int, heads: int = 4, range_attention: bool = False,
                 mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.pos = nn.Parameter(torch.randn(positions, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.score = nn.Linear(d_model, heads)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.range_block = Block(d_model, heads, mlp_ratio, dropout) if range_attention else None

    def forward(self, u: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """bias: optional [N, d] added at every position (the radar embedding)."""
        N, d, P, T = u.shape
        x = u.permute(0, 3, 2, 1) + self.pos                               # [N, T, P, d]
        if bias is not None:
            x = x + bias[:, None, None]
        if self.range_block is not None:
            # Each frame is a separate example. Only its range positions attend to one another.
            x = self.range_block(x.reshape(N * T, P, d)).reshape(N, T, P, d)
        x = self.norm(x)
        a = torch.softmax(self.score(x), dim=2)                            # [N, T, P, h]
        v = self.value(x).view(N, T, P, self.heads, d // self.heads)
        return self.out((a[..., None] * v).sum(2).reshape(N, T, d))


class RadarFusion(nn.Module):
    """Pre-LN cross-attention of one learned query to the 3 radar tokens of each step, then an FFN:
    tokens [B, 3, T, d], present [B, 3] -> [B, T, d]. `last_weights` [B, T, 3] keeps the attention
    (head mean) of the last call, for diagnostics."""
    def __init__(self, d_model: int, nhead: int, mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        self.h, self.dh = nhead, d_model // nhead
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.norm_q, self.norm_kv = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2, self.ffn = nn.LayerNorm(d_model), FeedForward(d_model, mlp_ratio * d_model, dropout)
        self.last_weights = None

    def forward(self, tokens: torch.Tensor, present: torch.Tensor, radar_embed: torch.Tensor) -> torch.Tensor:
        B, R, T, d = tokens.shape
        kv = self.norm_kv((tokens + radar_embed[None, :, None]).permute(0, 2, 1, 3).reshape(B * T, R, d))
        q = self.query.expand(B * T, 1, d)
        k, v = self.kv_proj(kv).view(B * T, R, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)   # [B T, h, R, dh]
        qq = self.q_proj(self.norm_q(q)).view(B * T, 1, self.h, self.dh).transpose(1, 2)    # [B T, h, 1, dh]
        # A window whose radars all look absent (no signal at all) attends to all of them
        present = present | ~present.any(1, keepdim=True)
        mask = present[:, None].expand(B, T, R).reshape(B * T, 1, 1, R)
        w = torch.softmax((qq @ k.transpose(-1, -2) / math.sqrt(self.dh)).masked_fill(~mask, -torch.inf), -1)
        self.last_weights = w.mean(1).view(B, T, R).detach()
        y = q + self.out_proj((w @ v).transpose(1, 2).reshape(B * T, 1, d))
        y = y + self.ffn(self.norm2(y))
        return y.view(B, T, d)


class RadarStem(nn.Module):
    """x [B, C, H, L] (the layout above, H = 184 complex bins) -> tokens [B, Fn, d], token i centred on
    frame i (steps [4 i, 4 i + w])."""
    def __init__(self, d_model: int, nhead: int, channels: tuple[int, ...] = (16, 32, 48, 64), contrast: int = 0,
                 fusion: str = "attention", range_heads: int = 4, n_bins: int = 184, frame_width: int = 32,
                 frame_stride: int = 4, mlp_ratio: int = 2, dropout: float = 0.1, range_attention: bool = False):
        super().__init__()
        assert fusion in ("early", "attention"), f"unknown radar fusion {fusion!r}"
        assert frame_stride == 4 and frame_width % 8 == 0, "the stem has a total time stride of 4"
        assert n_bins % 8 == 0 and channels[-1] == d_model
        c0, c1, c2, c3 = channels
        self.fusion, self.contrast = fusion, contrast
        self.offset = frame_width // 8                    # stem output j is centred on step 4 j
        self.l0 = ComplexConv2d(2, c0, (5, 9), (2, 2), (2, 4))
        self.bn0 = nn.BatchNorm2d(c0)
        radars = 3 if fusion == "early" else 1
        self.l1 = _conv_bn(radars * (c0 + contrast), c1, (3, 5), (1, 2), (1, 2))
        self.l2 = _conv_bn(c1, c2, (3, 3), (2, 1), (1, 1))
        self.l3 = _conv_bn(c2, c3, (3, 3), (2, 1), (1, 1))
        self.pool = RangePool(d_model, n_bins // 8, range_heads, range_attention, mlp_ratio, dropout)
        if fusion == "attention":
            self.radar_embed = nn.Parameter(torch.randn(3, d_model) * 0.02)
            self.fuse = RadarFusion(d_model, nhead, mlp_ratio, dropout)

    def energies(self, x: torch.Tensor) -> torch.Tensor:
        """L0 per radar: [B 3, c0 (+ contrast), H / 2, ceil(L / 2)]."""
        re = torch.stack([x[:, [RAW_I + r, CLUT_I + r]] for r in range(3)], 1)
        im = torch.stack([x[:, [RAW_Q + r, CLUT_Q + r]] for r in range(3)], 1)
        y = self.l0(torch.cat([re, im], 2).flatten(0, 1))
        c0 = y.size(1) // 2
        e = self.bn0(torch.log1p(torch.sqrt(y[:, :c0] ** 2 + y[:, c0:] ** 2 + 1e-6)))
        if self.contrast:
            d = torch.stack([x[:, CONTRAST + r:CONTRAST + 3 * self.contrast:3] for r in range(3)], 1).flatten(0, 1)
            d = F.avg_pool2d(torch.log1p(d.clamp_min(0)), 2, ceil_mode=True)
            e = torch.cat([e, d[..., :e.size(-2), :e.size(-1)]], 1)
        return e

    def forward(self, x: torch.Tensor, frames: int) -> torch.Tensor:
        B = x.size(0)
        e = self.energies(x)
        if self.fusion == "early":
            e = e.view(B, -1, *e.shape[-2:])
        u = self.l3(self.l2(self.l1(e)))                                   # [N, d, P, ceil(L / 4)]
        assert u.size(-1) >= self.offset + frames, (u.shape, frames)
        u = u[..., self.offset:self.offset + frames]
        if self.fusion == "early":
            return self.pool(u)
        tok = self.pool(u, self.radar_embed.repeat(B, 1)).view(B, 3, frames, -1)
        sig = torch.stack([x[:, [RAW_I + r, RAW_Q + r, CLUT_I + r, CLUT_Q + r]] for r in range(3)], 1)
        present = sig.flatten(2).abs().amax(-1) > 0                        # [B, 3]; radar dropout zeroes a radar
        return self.fuse(tok, present, self.radar_embed)


class TCNBlock(nn.Module):
    """x + Drop(GELU(LN(conv(GELU(LN(conv(x))))))), both convs k 3 at `dilation`, centred."""
    def __init__(self, d_model: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(d_model, d_model, 3, padding=dilation, dilation=dilation)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.norm1(self.conv1((x * m).transpose(1, 2)).transpose(1, 2)))
        y = F.gelu(self.norm2(self.conv2((y * m).transpose(1, 2)).transpose(1, 2)))
        return x + self.drop(y)


class TCN(nn.Module):
    """[B, F, d] -> [B, F, d]; frames that are not valid enter every convolution as zeros."""
    def __init__(self, d_model: int, dilations: tuple[int, ...] = (1, 2, 4, 8), dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([TCNBlock(d_model, d, dropout) for d in dilations])

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        m = valid[..., None].to(x.dtype)
        for blk in self.blocks:
            x = blk(x, m)
        return x * m

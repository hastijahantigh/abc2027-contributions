import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import FeedForward


class FrameEncoderBlock(nn.Module):
    """Pre-LN transformer block over a dense batch of independent frames [N, L, dim]."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio * dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, L, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(N, L, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.out_proj(attn.transpose(1, 2).reshape(N, L, D))
        return x + self.ffn(self.norm2(x))


class FrameEncoder(nn.Module):
    """
    Encodes each frame into one vector, independently of every other frame: a frame can never
    see values outside itself. Copied unchanged from gir, where a frame is a full-height slice of
    a line image; here it is a [C, range bins, w timesteps] slice of a radar window (rows = range,
    columns = time, channels = radars, their temporal differences and the valid mask).
    Positions are within-frame only, so the same motion gives the same vector wherever it
    sits in the window; the frame's place in the window is added later, by the context.

    pooling: how the tokens become one vector: "cls" (a learned token), "mean" (of the patch
    tokens) or "attn" (softmax attention of one learned query over the patch tokens).
    """
    def __init__(
        self, dim: int, depth: int, num_heads: int, in_channels: int = 1,
        patch_size: tuple[int, int] = (8, 8), frame_width: int = 32, max_rows: int = 8,
        dropout: float = 0.1, pooling: str = "cls", mlp_ratio: int = 4
    ):
        super().__init__()
        assert frame_width % patch_size[1] == 0, "frame_width must be a multiple of the patch width"
        assert pooling in ("cls", "mean", "attn"), f"unknown pooling {pooling!r}"
        self.patch_size = patch_size
        self.num_cols = frame_width // patch_size[1]
        self.pooling = pooling

        self.patch_embed = nn.Linear(in_channels * patch_size[0] * patch_size[1], dim)
        self.pos_embed_y = nn.Parameter(torch.randn(max_rows, dim) * 0.02)
        self.pos_embed_x = nn.Parameter(torch.randn(self.num_cols, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02) if pooling == "cls" else None
        self.pool_query = nn.Parameter(torch.randn(dim) * 0.02) if pooling == "attn" else None

        self.blocks = nn.ModuleList([
            FrameEncoderBlock(dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, frames: torch.Tensor, frame_embed: torch.Tensor | None = None) -> torch.Tensor:
        """
        frames: [N, C, H, frame_width] with H a multiple of the patch height -> [N, dim].
        frame_embed: optional [N, dim] added to every token of its frame (the frame's place in
        the line, for the ablation that gives it to the encoder).
        """
        N, C, H, W = frames.shape
        ph, pw = self.patch_size
        rows = H // ph

        patches = frames[:, :, :rows * ph].reshape(N, C, rows, ph, self.num_cols, pw).permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(N, rows * self.num_cols, C * ph * pw)

        pos = (self.pos_embed_y[:rows, None, :] + self.pos_embed_x[None, :, :]).reshape(rows * self.num_cols, -1)
        x = self.patch_embed(patches) + pos
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(N, -1, -1), x], dim=1)
        if frame_embed is not None:
            x = x + frame_embed[:, None, :]

        for blk in self.blocks:
            x = blk(x)

        if self.pooling == "cls":
            return self.norm(x[:, 0])
        x = self.norm(x)
        if self.pooling == "mean":
            return x.mean(dim=1)
        weights = torch.softmax(x @ self.pool_query / x.size(-1) ** 0.5, dim=1)
        return (weights.unsqueeze(-1) * x).sum(dim=1)

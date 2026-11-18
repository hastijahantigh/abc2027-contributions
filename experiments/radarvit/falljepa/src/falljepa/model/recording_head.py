import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregate import frames_mean
from .layers import Block

NEG = -1e4


class RecordingDecoder(nn.Module):
    """
    The recording head (docs/falljepa_recording_head.md): a second stage on the trunk's frame
    tokens whose region of interest is a source recording (a unit, data/units.py). Per unit it
    predicts whether the unit holds a fall and where: what Amir's post-processing did outside the
    model with a threshold, merge, NMS, the official rule and the persistence gate.

        tokens      the trunk tokens r_i of the frames whose centre lies in the unit +- `context`
                    timesteps (2 s: it sees both joins and a fall that runs past a false join), plus Linear(unit channels averaged over the
                    frame, an anchor-radar mix of them, and an inside-the-unit flag)
        soft anchor the per-radar channels mixed with softmax(beta * log profile change): the
                    official "anchor = the radar whose profile changed most", differentiable
        [REC]       a learned token + Linear(unit scalars); `layers` bidirectional blocks over
                    [REC] + the unit's tokens, attention confined to them
        fall head   logit of P(the unit holds a fall), from [REC]
        span head   start and end logits for every timestep (each frame gives w of each, merged
                    over overlapping frames with the trunk's Hann weights), softmax over the unit
                    from `before` timesteps before it to `after` timesteps after it (a fall belongs to
                    the unit it starts in and may end past it, data/units.py): exactly one span per
                    unit, by construction

    Inputs (window coordinates: timestep 0 is the first timestep of the window's centre part; the
    frames see `margin` more on each side): r [B, F, d], frame_valid [B, F], units [B, U, 2],
    unit_mask [B, U], unit_scalars [B, U, S], uphys [B, C, L + 2 margin].
    Output rows are the real units, in (batch, slot) order.
    """
    def __init__(self, d_model: int, nhead: int, frame_width: int, frame_stride: int, unit_channels: int = 11,
                 unit_scalars: int = 10, layers: int = 2, mlp_ratio: int = 2, dropout: float = 0.1,
                 max_frames: int = 340, context: int = 100, before: int = 25, after: int = 100):
        super().__init__()
        self.w, self.s = frame_width, frame_stride
        self.margin = (frame_width - frame_stride) // 2
        self.max_frames, self.context, self.before, self.after = max_frames, context, before, after
        self.radar_channels = 3
        self.anchor_beta = nn.Parameter(torch.tensor(5.0))
        self.token_in = nn.Linear(unit_channels + self.radar_channels + 1, d_model)
        self.rec_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.scalar_in = nn.Sequential(nn.Linear(unit_scalars, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([Block(d_model, nhead, mlp_ratio, dropout=dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fall_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.span_head = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(),
                                       nn.Linear(d_model, frame_width * 2))

    def forward(self, r, frame_valid, units, unit_mask, unit_scalars, uphys) -> dict[str, torch.Tensor]:
        B, Fn, d = r.shape
        w, s, m, S = self.w, self.s, self.margin, self.max_frames
        rows, slots = unit_mask.nonzero(as_tuple=True)                        # real units
        N = rows.numel()
        lo = units[rows, slots, 0] + m                                        # extended timesteps
        hi = units[rows, slots, 1] + m
        # The frames of each unit +- context: frame i is centred at i s + w / 2
        first = torch.div(lo - self.context - w // 2 + s - 1, s, rounding_mode="floor").clamp_min(0)
        idx = first[:, None] + torch.arange(S, device=r.device)[None, :]      # [N, S]
        centre = idx * s + w // 2
        keep = (idx < Fn) & (centre < hi[:, None] + self.context)
        idx = idx.clamp_max(Fn - 1)
        keep &= frame_valid[rows[:, None], idx]
        tokens = r[rows[:, None], idx]                                        # [N, S, d]

        # Unit channels per frame (mean over its w timesteps) and the soft anchor
        ph = uphys.unfold(-1, w, s).mean(-1).transpose(1, 2)                  # [B, F, C]
        ph = ph[rows[:, None], idx]                                           # [N, S, C]
        inside = ((centre >= lo[:, None]) & (centre < hi[:, None])).float()[..., None]
        ph = ph * inside
        per_radar = ph[..., :3 * self.radar_channels].reshape(N, S, 3, self.radar_channels)
        change = unit_scalars[rows, slots, :3]                                # log10 change per radar
        anchor = torch.softmax(self.anchor_beta * change, dim=-1)             # [N, 3]
        mixed = (per_radar * anchor[:, None, :, None]).sum(2)                 # [N, S, 3]
        tokens = tokens + self.token_in(torch.cat([ph, mixed, inside], -1))

        rec = self.rec_token + self.scalar_in(unit_scalars[rows, slots])      # [N, d]
        x = torch.cat([rec[:, None], tokens], 1)
        key_valid = torch.cat([torch.ones(N, 1, dtype=torch.bool, device=r.device), keep], 1)
        for blk in self.blocks:
            x = blk(x, key_valid=key_valid)
        x = self.norm(x)
        fall_logit = self.fall_head(x[:, 0]).squeeze(-1)

        # Start / end logits per timestep of the gathered frames: local timestep 0 = frame `first`
        frame_logits = self.span_head(x[:, 1:]).view(N, S, w, 2)
        frame_logits = frame_logits.masked_fill(~keep[..., None, None], 0.0)
        span_logits = frames_mean(frame_logits, s)                            # [N, (S - 1) s + w, 2]
        origin = first * s - m                                                # centre-part timestep of local 0
        t = origin[:, None] + torch.arange(span_logits.size(1), device=r.device)[None, :]
        domain = (t >= (lo - m - self.before)[:, None]) & (t < (hi - m + self.after)[:, None])
        span_logits = span_logits.masked_fill(~domain[..., None], NEG)
        return {"unit_rows": rows, "unit_slots": slots, "unit_logit": fall_logit, "span_logits": span_logits,
                "span_origin": origin, "span_domain": domain}

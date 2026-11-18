"""
The heads of the join-free model (docs/falljepa_join_free.md): no joins, units or recordings.

    Pointer         start / end logits per 20 ms step from a query: q . (key of the token that owns
                    the step + one of `stride` learned offsets inside it + a projection of the step's
                    own features), so a query points at a step without a dense decoder
    FallQueries     stage 1, per window: K learned queries cross-attend to the window's tokens (DETR);
                    each gives P(fall), a start and an end distribution and its hidden state
    GroupRefiner    the stream stage, per group of overlapping proposals: slots attend to the group's
                    proposals and to the stream tokens of its region; each slot gives P(fall) and a
                    start and an end distribution over the region
    event_physics   parameter-free features of a region relative to a candidate span [a, b): the
                    scene just before a against just after b, per radar (the labeller's quantities
                    around the event instead of around a recording); with `stable`, the before and
                    after profiles are the stillest stretches near a and b instead of fixed offsets
    transition_bounds  the labeller's fall interval (abc2027/annotate.py `_transition_bounds`) from a
                    state curve and a motion curve: the 1.5 s step of the state that coincides with motion
    match           the lowest-cost assignment of predictions to target falls, with ignore targets
"""
import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Attention, Block, FeedForward

NEG = -1e4

# Target kinds of `match`
SCORED, FLAGGED, CUT = 1, 2, 3
BEFORE = (60, 10)          # the before profile: [a - 1.2 s, a - 0.2 s)
AFTER = (10, 85)           # the after profile: [b + 0.2 s, b + 1.7 s)
STABLE_BEFORE = (200, 10, 25)   # stable anchors: the stillest 0.5 s inside [a - 4 s, a - 0.2 s)
STABLE_AFTER = (10, 200, 50)    # and the stillest 1 s inside [b + 0.2 s, b + 4 s)
RISE = 75                  # the rule's 1.5 s
NUM_EVENT_CHANNELS = 11    # state, motion, rise per radar; position to a and to b
NUM_EVENT_SCALARS = 4      # log change per radar, stillness after on the anchor radar
NUM_GROUP_STATS = 4        # log1p(members), max P, mean P, span length / 5 s
NUM_MEMBER_FEATURES = 5    # logit, start and end in the region (fractions), length / 5 s, centrality
PHYSICS_GROUPS = 16        # groups per event_physics call (bounds its temporary memory)
NUM_TRANSITION_CHANNELS = 2   # the labeller's transition score, and its span as an indicator
NUM_TRANSITION_FEATURES = 6   # found, start and end in the region (fractions), length / 5 s, rise, log1p(score)
NUM_MEMBER_CHANNELS = 2    # JF7: the members' starts and ends as P-weighted bumps over the region's steps
MEMBER_SIGMA = 10          # their std in steps (0.2 s)


class Pointer(nn.Module):
    """Start and end logits [B, Q, F s, 2] of queries [B, Q, d] over the steps owned by tokens [B, F, d]
    (token i owns steps [i s, (i + 1) s)), with optional step features [B, C, F s]."""
    def __init__(self, d_model: int, stride: int, step_channels: int = 0):
        super().__init__()
        self.d, self.s = d_model, stride
        self.query = nn.Linear(d_model, 2 * d_model)
        self.key = nn.Linear(d_model, 2 * d_model)
        self.offset = nn.Parameter(torch.zeros(stride, 2, d_model))
        self.step = nn.Linear(step_channels, 2 * d_model) if step_channels else None

    def forward(self, q: torch.Tensor, tokens: torch.Tensor, steps: torch.Tensor | None = None) -> torch.Tensor:
        B, Fn, d = tokens.shape
        k = self.key(tokens).view(B, Fn, 1, 2, d) + self.offset[None, None]           # [B, F, s, 2, d]
        k = k.reshape(B, Fn * self.s, 2, d)
        if self.step is not None and steps is not None:
            k = k + self.step(steps.transpose(1, 2)).view(B, Fn * self.s, 2, d)
        qq = self.query(q).view(B, q.size(1), 2, d)
        return torch.einsum("bqcd,btcd->bqtc", qq, k) / math.sqrt(d)


class QueryLayer(nn.Module):
    """Pre-LN DETR decoder layer: self-attention among the queries, cross-attention to the tokens, MLP."""
    def __init__(self, d_model: int, nhead: int, mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm1, self.self_attn = nn.LayerNorm(d_model), Attention(d_model, nhead)
        self.norm2, self.norm_kv, self.cross = nn.LayerNorm(d_model), nn.LayerNorm(d_model), Attention(d_model, nhead)
        self.norm3, self.ffn = nn.LayerNorm(d_model), FeedForward(d_model, mlp_ratio * d_model, dropout)

    def forward(self, q: torch.Tensor, memory: torch.Tensor, memory_valid: torch.Tensor,
                memory_pos: torch.Tensor | None = None) -> torch.Tensor:
        q = q + self.self_attn(self.norm1(q))
        kv = self.norm_kv(memory) if memory_pos is None else self.norm_kv(memory) + memory_pos
        q = q + self.cross(self.norm2(q), kv, key_valid=memory_valid)
        return q + self.ffn(self.norm3(q))


def _objectness(d_model: int, prior: float = 0.1) -> nn.Sequential:
    head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
    nn.init.constant_(head[-1].bias, math.log(prior / (1 - prior)))
    return head


class FallQueries(nn.Module):
    """Stage 1: K queries over one window's tokens [B, F, d] -> hidden [B, K, d], fall logits [B, K] and
    start / end logits [B, K, F s, 2] over the window's centre steps (masked where `step_valid` is 0).
    key_pos: optional [F, d] added to the tokens in the cross-attention only (JF4: the trunk has RoPE
    and no absolute positions; the pointer still reads the plain tokens)."""
    def __init__(self, d_model: int, nhead: int, num_queries: int = 3, layers: int = 2, stride: int = 4,
                 step_channels: int = 0, mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        self.layers = nn.ModuleList([QueryLayer(d_model, nhead, mlp_ratio, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fall = _objectness(d_model)
        self.pointer = Pointer(d_model, stride, step_channels)

    def forward(self, tokens, token_valid, step_valid, steps=None, key_pos=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.queries[None].expand(tokens.size(0), -1, -1)
        for layer in self.layers:
            q = layer(q, tokens, token_valid, key_pos)
        q = self.norm(q)
        logits = self.pointer(q, tokens, steps).masked_fill(~step_valid[:, None, :, None], NEG)
        return q, self.fall(q).squeeze(-1), logits


class GroupRefiner(nn.Module):
    """
    The stream stage on a padded batch of groups (see `pad_groups` for the fields): `slots` learned
    slots, started from the group scalars (event scalars and member statistics), attend with
    `layers` blocks to the group's member proposals and its region tokens (stream tokens plus a
    projection of the event channels and their soft-anchor mix). Per slot: hidden, fall logit and
    start / end logits over the region's steps.

    event_physics false (the ablation) zeroes the event channels and scalars; stable_anchors takes the
    before and after profiles from the stillest stretches near the span (see event_physics).
    transition (JF3) adds the labeller's transition span: two event channels (its score and its span)
    and one more member token (a learned vector plus its features), present when a transition is found.

    JF7 (docs/falljepa_JF7_design.md): `member_channels` adds two step channels, the members' starts and
    ends as Gaussian bumps weighted by the members' P, so the pointer can copy a member's boundary and
    adjust it; `quality` adds a head that predicts each slot's tIoU with its fall (a logit).
    """
    def __init__(self, d_model: int, nhead: int, slots: int = 2, layers: int = 2, stride: int = 4,
                 mlp_ratio: int = 2, dropout: float = 0.1, event_physics: bool = True, stable_anchors: bool = False,
                 transition: bool = False, member_channels: bool = False, quality: bool = False):
        super().__init__()
        self.s, self.use_physics, self.stable, self.transition = stride, event_physics, stable_anchors, transition
        self.member_channels = member_channels
        channels = (NUM_EVENT_CHANNELS + 3 + (NUM_TRANSITION_CHANNELS if transition else 0)
                    + (NUM_MEMBER_CHANNELS if member_channels else 0))
        self.channels = channels
        scalars = 3 + 3 + 1 + 1 + NUM_GROUP_STATS
        self.anchor_beta = nn.Parameter(torch.tensor(5.0))
        self.token_in = nn.Linear(channels, d_model)
        self.slots = nn.Parameter(torch.randn(slots, d_model) * 0.02)
        self.scalar_in = nn.Sequential(nn.Linear(scalars, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.member_in = nn.Linear(d_model + NUM_MEMBER_FEATURES, d_model)
        self.kind = nn.Parameter(torch.randn(3, d_model) * 0.02)      # slot, member, token
        self.blocks = nn.ModuleList([Block(d_model, nhead, mlp_ratio, dropout=dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fall = _objectness(d_model)
        self.pointer = Pointer(d_model, stride, channels)
        self.quality = _objectness(d_model, prior=0.5) if quality else None
        if transition:
            self.rule_member = nn.Parameter(torch.randn(d_model) * 0.02)
            self.rule_in = nn.Linear(NUM_TRANSITION_FEATURES, d_model)

    def forward(self, g: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens, tvalid, svalid = g["tokens"], g["token_valid"], g["step_valid"]
        G, R, _ = tokens.shape
        # A few groups at a time: the profiles and distances are [groups, 3, 2 H, R s] each
        parts = [event_physics(g["signal"][i:i + PHYSICS_GROUPS], svalid[i:i + PHYSICS_GROUPS],
                               g["span"][i:i + PHYSICS_GROUPS], int(g["groups"]), self.stable, self.transition)
                 for i in range(0, G, PHYSICS_GROUPS)]
        chans, scal = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
        rule = torch.cat([p[2] for p in parts]) if self.transition else None
        if not self.use_physics:
            chans, scal = torch.zeros_like(chans), torch.zeros_like(scal)
            rule = torch.zeros_like(rule) if rule is not None else None
        change = scal[:, :3]
        anchor = torch.softmax(self.anchor_beta * change, dim=-1)                      # [G, 3]
        mixed = (chans[:, :9].reshape(G, 3, 3, -1) * anchor[:, None, :, None]).sum(2)   # state, motion, rise
        steps = torch.cat([chans, mixed], 1)                                            # [G, 14, R s]
        if self.member_channels:
            steps = torch.cat([steps, member_bumps(g, steps.size(-1))], 1)
        per_token = steps.reshape(G, steps.size(1), R, self.s).mean(-1).transpose(1, 2)
        x_tok = tokens + self.token_in(per_token) + self.kind[2]

        lin = torch.pow(10.0, change)
        share = lin.max(1).values / lin.sum(1).clamp_min(1e-12)
        sorted_change = torch.sort(change, dim=1, descending=True).values
        scalars = torch.cat([change, sorted_change, share[:, None], scal[:, 3:4], g["stats"]], 1)
        x_slot = self.slots[None] + self.scalar_in(scalars)[:, None] + self.kind[0]
        x_mem = self.member_in(torch.cat([g["members"], g["member_features"]], -1)) + self.kind[1]
        member_valid = g["member_valid"]
        if rule is not None:
            x_rule = self.rule_member + self.rule_in(rule) + self.kind[1]
            x_mem = torch.cat([x_mem, x_rule[:, None]], 1)
            member_valid = torch.cat([member_valid, rule[:, :1] > 0], 1)

        Q, P = x_slot.size(1), x_mem.size(1)
        x = torch.cat([x_slot, x_mem, x_tok], 1)
        valid = torch.cat([torch.ones(G, Q, dtype=torch.bool, device=x.device), member_valid, tvalid], 1)
        for blk in self.blocks:
            x = blk(x, key_valid=valid)
        x = self.norm(x)
        h, tok = x[:, :Q], x[:, Q + P:]
        logits = self.pointer(h, tok, steps).masked_fill(~svalid[:, None, :, None], NEG)
        out = {"hidden": h, "logit": self.fall(h).squeeze(-1), "span_logits": logits, "tokens": tok, "steps": steps}
        if self.quality is not None:
            out["quality"] = self.quality(h).squeeze(-1)
        return out


def member_bumps(g: dict[str, torch.Tensor], L: int) -> torch.Tensor:
    """[G, 2, L]: per region step, the max over the members of P(member) x a Gaussian (std MEMBER_SIGMA)
    at the member's start (channel 0) and end (channel 1). Positions come from the member features
    (fractions of the unpadded region length)."""
    f, valid = g["member_features"], g["member_valid"]                  # [G, P, 5], [G, P]
    length = g["region_len"].float()[:, None]
    p = torch.sigmoid(f[..., 0]) * valid.float()
    t = torch.arange(L, device=f.device, dtype=torch.float32)
    out = []
    for c in (1, 2):
        pos = f[..., c] * length                                        # [G, P]
        bump = torch.exp(-0.5 * ((t[None, None] - pos[..., None]) / MEMBER_SIGMA) ** 2) * p[..., None]
        out.append(bump.amax(1) if bump.size(1) else torch.zeros(f.size(0), L, device=f.device))
    return torch.stack(out, 1) * g["step_valid"][:, None].float()


class BoundaryRefiner(nn.Module):
    """
    JF7b: one boundary (a start or an end) re-placed from a crop of 2 `half` + 1 steps (50 Hz) centred on
    its current position: per step the region's step channels (the refiner's event physics and member
    bumps), the refined stream token that owns the step, the slot's hidden state, the boundary kind and
    the offset from the centre; a residual stack of dilated convolutions gives logits over the crop
    (masked where the step is outside the region or invalid).
    """
    def __init__(self, d_model: int, channels: int, stride: int = 4, half: int = 150, width: int = 64,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16), dropout: float = 0.1):
        super().__init__()
        self.s, self.half = stride, half
        self.step_in = nn.Linear(channels, width)
        self.token_in = nn.Linear(d_model, width)
        self.slot_in = nn.Linear(d_model, width)
        self.pos_in = nn.Linear(1, width)
        self.kind = nn.Parameter(torch.randn(2, width) * 0.02)
        self.blocks = nn.ModuleList([nn.Sequential(nn.GroupNorm(1, width), nn.Conv1d(width, width, 5, padding=2 * d, dilation=d),
                                                   nn.GELU(), nn.Dropout(dropout), nn.Conv1d(width, width, 1))
                                     for d in dilations])
        self.out = nn.Conv1d(width, 1, 1)

    def forward(self, tokens, steps, step_valid, hidden, group, slot, centre, kind) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens [G, R, d], steps [G, C, L], step_valid [G, L], hidden [G, Q, d]; per boundary (n): its
        group, slot, centre (region step) and kind (0 start, 1 end). Returns logits [n, 2 half + 1] and the
        crop's valid steps [n, 2 half + 1] (crop step j is region step centre - half + j)."""
        L = steps.size(-1)
        rel = torch.arange(-self.half, self.half + 1, device=steps.device)
        pos = centre[:, None] + rel[None]                                          # [n, C]
        inside = (pos >= 0) & (pos < L)
        pc = pos.clamp(0, L - 1)
        mask = inside & step_valid[group[:, None], pc]
        x = (self.step_in(steps.transpose(1, 2)[group[:, None], pc])               # [n, C, w]
             + self.token_in(tokens[group[:, None], pc // self.s])
             + self.slot_in(hidden[group, slot])[:, None]
             + self.kind[kind][:, None]
             + self.pos_in(rel[None, :, None].float() / self.half))
        x = (x * mask[..., None].float()).transpose(1, 2)                          # [n, w, C]
        m = mask[:, None].float()
        for blk in self.blocks:
            x = (x + blk(x)) * m
        return self.out(x).squeeze(1).masked_fill(~mask, NEG), mask


def _smooth(x: torch.Tensor, valid: torch.Tensor, width: int) -> torch.Tensor:
    """Moving mean of x [G, C, L] over `width` steps, counting only valid steps [G, L]."""
    m = valid.float()[:, None]
    num = F.avg_pool1d(x * m, width, 1, width // 2, count_include_pad=True)
    den = F.avg_pool1d(m, width, 1, width // 2, count_include_pad=True)
    return num / den.clamp_min(1e-6)


def _stillest(motion: torch.Tensor, valid: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, width: int) -> torch.Tensor:
    """[G, L] mask of the `width`-step stretch of [lo, hi) (per group) with the lowest mean motion
    [G, L] over valid steps; all False where no whole valid stretch fits."""
    G, L = motion.shape
    t = torch.arange(L, device=motion.device)[None]
    vf = valid.float()
    run = F.avg_pool1d((motion * vf)[:, None], width, 1).squeeze(1)              # [G, L - width + 1], stretch at its first step
    full = F.avg_pool1d(vf[:, None], width, 1).squeeze(1) > 0.999
    first = t[:, :run.size(1)]
    ok = full & (first >= lo) & (first + width <= hi)
    best = torch.where(ok, run, torch.full_like(run, torch.inf)).argmin(1, keepdim=True)
    return (t >= best) & (t < best + width) & ok.any(1, keepdim=True)


@torch.no_grad()
def event_physics(signal: torch.Tensor, valid: torch.Tensor, span: torch.Tensor, groups: int,
                  stable: bool = False, transition: bool = False) -> tuple[torch.Tensor, ...]:
    """
    signal [G, 3 groups, H, L]: the normalized signal channels of each region (groups of 3 radars: I
    and Q, or z), valid [G, L] bool, span [G, 2] the candidate fall [a, b) in region steps ->
    channels [G, 11, L] (per radar: the state between the before and the after profile, motion
    relative to the region's mean, the rise of the state over 1.5 s; then the position relative to a
    and to b in s / 5, clipped) and scalars [G, 4] (log10 of the before-after profile change per
    radar, the stillness after the span on the radar that changed most). The profiles fall back to
    everything before a (after b), then the whole region, near the region's ends.

    stable: the before profile is the stillest 0.5 s (lowest mean motion over the radars) inside
    [a - 4 s, a - 0.2 s), the after profile the stillest 1 s inside [b + 0.2 s, b + 4 s), each
    falling back to the fixed windows when none fits. A span that starts late inside a slow descent,
    or a fall shortly after a crossfade, then still compares the standing scene with the floor scene.

    transition: the labeller's interval (`transition_bounds`) on the radar whose profile changes most,
    from this state and the linear motion smoothed 0.4 s, as the labeller does on a recording. Adds
    2 channels (its score rise x log1p(motion) at the centre of each 1.5 s window, divided by the
    region's largest; 1 inside its span) and returns a third tensor [G, NUM_TRANSITION_FEATURES]:
    found, start and end as fractions of the valid region, length / 5 s, the state's rise across the
    span, log1p of the best score. A region shorter than 1.5 s + 2 steps has none (all zero).
    """
    G, _, H, L = signal.shape
    v = torch.stack([torch.cat([signal[:, k * 3 + r] for k in range(groups)], 1) for r in range(3)], 1).float()
    dev = v.device
    t = torch.arange(L, device=dev)[None]
    a, b = span[:, :1].float(), span[:, 1:].float()

    def pick(primary, fallback):
        m = torch.where(primary.sum(1, keepdim=True) >= 5, primary, fallback)
        return torch.where(m.sum(1, keepdim=True) >= 1, m, valid)

    diff = (v[..., 1:] - v[..., :-1]).square().mean(2)
    lin = torch.cat([diff[..., :1], diff], -1)
    motion = _smooth(torch.log(lin + 1e-3), valid, 11)
    before = pick(valid & (t >= a - BEFORE[0]) & (t < a - BEFORE[1]), valid & (t < a))
    after = pick(valid & (t >= b + AFTER[0]) & (t < b + AFTER[1]), valid & (t >= b))
    if stable:
        m_all = motion.mean(1)
        sb = _stillest(m_all, valid, a - STABLE_BEFORE[0], a - STABLE_BEFORE[1], STABLE_BEFORE[2])
        sa = _stillest(m_all, valid, b + STABLE_AFTER[0], b + STABLE_AFTER[1], STABLE_AFTER[2])
        before = torch.where(sb.any(1, keepdim=True), sb, before)
        after = torch.where(sa.any(1, keepdim=True), sa, after)

    def profile(m):
        w = m.float()
        return (v * w[:, None, None, :]).sum(-1) / w.sum(1).clamp_min(1)[:, None, None]

    pb, pa = profile(before), profile(after)                                   # [G, 3, gH]
    db = (v - pb[..., None]).square().mean(2)
    da = (v - pa[..., None]).square().mean(2)
    state = _smooth(db / (db + da + 1e-6), valid, 21)
    rise = torch.cat([state[..., RISE:], state[..., -1:].expand(-1, -1, min(RISE, L))], -1)[..., :L] - state
    vf = valid.float()
    motion = motion - ((motion * vf[:, None]).sum(-1) / vf.sum(-1).clamp_min(1)[:, None])[..., None]
    pos_a = ((t - a) / 50.0).clamp(-5, 5) / 5
    pos_b = ((t - b) / 50.0).clamp(-5, 5) / 5
    chans = torch.cat([state, motion, rise, pos_a[:, None], pos_b[:, None]], 1) * vf[:, None]

    change = (pa - pb).square().mean(-1)                                       # [G, 3]
    anchor = change.argmax(1)
    m = motion[torch.arange(G, device=dev), anchor]                            # [G, L]
    inside = valid & (t >= a) & (t < b)
    peak = torch.where(inside, m, torch.full_like(m, -1e9)).max(1).values
    peak = torch.where(inside.any(1), peak, torch.zeros_like(peak))
    still = (m * after.float()).sum(1) / after.float().sum(1).clamp_min(1) - peak
    scal = torch.cat([torch.log10(change + 1e-4), still[:, None]], 1)
    if not transition:
        return chans, scal
    st = state[torch.arange(G, device=dev), anchor].float().cpu().numpy()
    mo = _smooth(lin, valid, 21)[torch.arange(G, device=dev), anchor].float().cpu().numpy()
    vn = valid.cpu().numpy()
    extra = np.zeros((G, NUM_TRANSITION_CHANNELS, L), np.float32)
    rule = np.zeros((G, NUM_TRANSITION_FEATURES), np.float32)
    for i in range(G):
        idx = np.flatnonzero(vn[i])
        n = int(idx[-1]) + 1 if len(idx) else 0
        if n < TRANSITION_WIDTH + 2:
            continue
        left, right, score, rise = transition_bounds(st[i, :n], mo[i, :n])
        top = float(np.max(score))
        if top > 0:
            extra[i, 0, TRANSITION_WIDTH // 2:TRANSITION_WIDTH // 2 + len(score)] = np.clip(score / top, -1, 1)
        extra[i, 1, left:right + 1] = 1.0
        rule[i] = (1.0, left / n, (right + 1) / n, (right + 1 - left) / 250.0, rise, np.log1p(max(top, 0.0)))
    chans = torch.cat([chans, torch.from_numpy(extra).to(dev) * vf[:, None]], 1)
    return chans, scal, torch.from_numpy(rule).to(dev)


TRANSITION_WIDTH = 75      # the labeller's 1.5 s
TRANSITION_RISE = 0.12     # a window must raise the state this much to be eligible
TRANSITION_EDGE = 0.15     # the span widens over motion above this fraction of its peak
TRANSITION_CAPS = (150, 100)   # at most 3 s to the left and 2 s to the right of the step's start


def transition_bounds(state: np.ndarray, motion: np.ndarray) -> tuple[int, int, np.ndarray, float]:
    """
    The labeller's fall interval (abc2027/annotate.py `_transition_bounds`, step 0.02 s) from a smoothed
    state curve [L] (0 = before profile, 1 = after) and a smoothed linear motion curve [L]: the 1.5 s
    window with the largest rise x log1p(motion sum / its median), among the windows that raise the
    state by >= .12 if any; cut at 20-80 % of that rise, then widened over motion >= 15 % of the peak
    above the 30th percentile, at most 3 s left and 2 s right. Returns (left, right) inclusive steps,
    the score per window start [L - 1.5 s] and the state's rise from left to right.
    """
    width = TRANSITION_WIDTH
    if len(state) <= width + 2:
        peak = int(np.argmax(motion))
        return peak, peak, np.zeros(0), 0.0
    rise = state[width:] - state[:-width]
    motion_sum = np.convolve(motion, np.ones(width), mode="valid")
    count = min(len(rise), len(motion_sum))
    rise, motion_sum = rise[:count], motion_sum[:count]
    scale = float(np.median(motion_sum)) + 1e-18
    score = rise * np.log1p(motion_sum / scale)
    eligible = rise >= TRANSITION_RISE
    pick = np.where(eligible, score, -np.inf) if np.any(eligible) else score
    index = int(np.argmax(pick))
    span_end = min(len(state) - 1, index + width)
    start_level, end_level = float(state[index]), float(state[span_end])
    delta = end_level - start_level
    left, right = index, span_end
    if delta > 0.05:
        low, high = start_level + 0.2 * delta, start_level + 0.8 * delta
        above_low = np.flatnonzero(state[index:span_end + 1] >= low)
        below_high = np.flatnonzero(state[index:span_end + 1] <= high)
        if above_low.size:
            left = index + int(above_low[0])
        if below_high.size:
            right = index + int(below_high[-1])
        right = max(right, left)
    height = float(np.max(motion[left:right + 1]))
    baseline = float(np.percentile(motion, 30))
    level = baseline + TRANSITION_EDGE * max(height - baseline, 0.0)
    origin = left
    while left > 0 and motion[left - 1] >= level and (origin - left) < TRANSITION_CAPS[0]:
        left -= 1
    while right < len(motion) - 1 and motion[right + 1] >= level and (right - origin) < TRANSITION_CAPS[1]:
        right += 1
    return left, right, score, float(state[right] - state[left])


def match(logit: np.ndarray, logp: np.ndarray, targets: list[tuple[int, int, int]], span_weight_flagged: float = 0.5):
    """
    The lowest-cost assignment of Q predictions (fall logits [Q], start / end log-probabilities
    [Q, L, 2]) to target falls (a, b, kind) in their steps, cost -log P(fall) - (log P_start(a) +
    log P_end(b)) / 2, by enumeration (Q and the targets are a handful). Returns per prediction the
    fall target [Q], its weight [Q], the span [Q, 2] and its weight [Q]:

        matched to SCORED   fall 1 (weight 1), span weight 1
        matched to FLAGGED  no fall loss, span weight `span_weight_flagged`
        matched to CUT      no loss at all (a fall the window or region only partly holds)
        unmatched           fall 0 (weight 1)
    """
    Q, L = logp.shape[0], logp.shape[1]
    target, weight = np.zeros(Q, np.float32), np.ones(Q, np.float32)
    span, span_w = np.zeros((Q, 2), np.int64), np.zeros(Q, np.float32)
    targets = targets[:6]
    if not targets or Q == 0:
        return target, weight, span, span_w
    nll = np.logaddexp(0.0, -logit)                                            # -log sigmoid
    cost = np.zeros((Q, len(targets)))
    for j, (a, b, _) in enumerate(targets):
        ia, ib = min(max(a, 0), L - 1), min(max(b, 0), L - 1)
        cost[:, j] = nll - 0.5 * (logp[:, ia, 0] + logp[:, ib, 1])
    k = min(Q, len(targets))
    best, pairs = np.inf, []
    for ts in itertools.combinations(range(len(targets)), k):
        for qs in itertools.permutations(range(Q), k):
            c = cost[list(qs), list(ts)].sum()
            if c < best:
                best, pairs = c, list(zip(qs, ts))
    for q, j in pairs:
        a, b, kind = targets[j]
        if kind == SCORED:
            target[q], span[q], span_w[q] = 1.0, (a, b), 1.0
        elif kind == FLAGGED:
            weight[q], span[q], span_w[q] = 0.0, (a, b), span_weight_flagged
        else:
            weight[q] = 0.0
    return target, weight, span, span_w


def pad_groups(items: list[dict], groups: int, device=None) -> dict[str, torch.Tensor]:
    """
    One batch for GroupRefiner from per-group dicts, padded to the longest region and the most
    members:
        tokens [R, d], signal [3 groups, H, R s], step_valid [R s] (bool), span [2] (region steps),
        members [P, d], member_features [P, 5], stats [4]
    A token is valid when any of its steps is.
    """
    R = max(it["tokens"].shape[0] for it in items)
    P = max(max(it["members"].shape[0] for it in items), 1)
    G, d = len(items), items[0]["tokens"].shape[1]
    s = items[0]["signal"].shape[-1] // max(items[0]["tokens"].shape[0], 1)
    C, H = items[0]["signal"].shape[:2]
    dev = device or items[0]["tokens"].device
    out = {
        "tokens": torch.zeros(G, R, d, device=dev, dtype=items[0]["tokens"].dtype),
        "signal": torch.zeros(G, C, H, R * s, device=dev),
        "step_valid": torch.zeros(G, R * s, dtype=torch.bool, device=dev),
        "span": torch.zeros(G, 2, dtype=torch.long, device=dev),
        "members": torch.zeros(G, P, d, device=dev, dtype=items[0]["tokens"].dtype),
        "member_features": torch.zeros(G, P, NUM_MEMBER_FEATURES, device=dev),
        "member_valid": torch.zeros(G, P, dtype=torch.bool, device=dev),
        "stats": torch.zeros(G, NUM_GROUP_STATS, device=dev),
        "region_len": torch.zeros(G, dtype=torch.long, device=dev),
    }
    tokens, members = [], []
    for i, it in enumerate(items):
        r, p = it["tokens"].shape[0], it["members"].shape[0]
        tokens.append(F.pad(it["tokens"], (0, 0, 0, R - r)))
        members.append(F.pad(it["members"], (0, 0, 0, P - p)))
        out["signal"][i, :, :, :r * s] = it["signal"]
        out["step_valid"][i, :r * s] = it["step_valid"]
        out["span"][i] = it["span"]
        out["member_features"][i, :p] = it["member_features"]
        out["member_valid"][i, :p] = True
        out["stats"][i] = it["stats"]
        out["region_len"][i] = r * s
    out["tokens"] = torch.stack(tokens)                     # stacked, so gradients reach the trunk and the queries
    out["members"] = torch.stack(members)
    out["token_valid"] = out["step_valid"].view(G, R, s).any(-1)
    out["groups"] = groups
    return out

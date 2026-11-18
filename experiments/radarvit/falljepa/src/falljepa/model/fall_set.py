import numpy as np
import torch
import torch.nn.functional as F

from .fall_jepa import FallJEPA
from .aggregate import mbr_span
from .set_heads import CUT, FLAGGED, SCORED, BoundaryRefiner, FallQueries, GroupRefiner, match, pad_groups
from .sets import (MIN_SPAN, TIE_IOU, group_item, group_region, group_stats, member_features, proposals, stream_groups,
                   tiou)

# JF7b: the error a boundary refiner trains on, in s (current position - truth), per kind (start, end):
# (probability, kind of draw, a, b) with "n" a normal of std a and "u" uniform on [a, b). JF6's starts are
# late on long falls (median +0.84 s at 2.5-3 s), its ends late on fold D; errors of 0.5-2.5 s are common.
BOUNDARY_ERRORS = (((0.45, "n", 0.3, 0.0), (0.35, "u", 0.5, 2.5), (0.20, "u", -2.0, -0.5)),
                   ((0.50, "n", 0.3, 0.0), (0.30, "u", 0.5, 2.0), (0.20, "u", -2.0, -0.5)))


class FallSetJEPA(FallJEPA):
    """
    The join-free model (docs/falljepa_join_free.md): the FallJEPA trunk and dense head, plus

        stage 1     FallQueries on every window: `num_queries` falls per window, each P(fall), a start
                    and an end distribution over the window's centre steps and a hidden state
        stage 2     GroupRefiner on groups of overlapping proposals: `group_slots` falls per group,
                    from the group's members, its region's stream tokens and the event physics

    A batch is either windows (inference, validation) or chunks of consecutive windows `chunk_hop`
    apart (training, data.chunk_windows > 1): the model cuts a chunk into its windows, runs the trunk
    and stage 1 on each, averages their tokens on the chunk's token grid (centre-weighted, as
    sets.TokenAccumulator does over a stream) and runs stage 2 on the groups of the chunk's
    proposals, plus a group around every labelled fall no group covers and `negative_groups` random
    no-fall regions. Training perturbs the groups: each member but the strongest is dropped with
    probability `member_drop`, and the reference span moves by N(0, span_jitter_s).

    Targets (with `falls` in the batch) are matched by set_heads.match: a fall wholly inside the
    window (region) is a target, a flagged one trains only the span, one partly inside is ignored.
    At inference, stage 2 runs on whole streams from the module (`refine`).

    JF2 options (docs/falljepa_join_free.md, section 11): `group_gap_s` also groups proposals that lie
    within that many seconds of a group's extent; `stable_anchors` takes the event physics' profiles
    from the stillest stretches near the span; `raw_drop_p` zeroes the first `raw_channels` input
    channels (the raw CIR, leaving the clutter-removed ones) of a training window with that
    probability, for the trunk only (the event physics always reads the raw signal).

    JF4 options (docs/falljepa_JF4_design.md): the trunk arguments of FallJEPA (`trunk`, `radar_fusion`,
    `stem_channels`, `range_heads`, `contrast_channels`, `local`, `tcn_dilations`, `rope`); with `rope` the
    fall queries' cross-attention keys get the learned frame position embedding the trunk no longer adds.
    With the stem layout `raw_channels` is 6 (raw I and Q).

    JF7 options (docs/falljepa_JF7_design.md): `member_channels` and `quality` (GroupRefiner; the quality
    target is the tIoU of the slot's MBR span with its matched fall, else its best tIoU with any fall of
    the region, else 0); `boundary` adds the BoundaryRefiner, trained on crops centred at a scored or
    flagged fall's matched start and end moved by an error drawn from BOUNDARY_ERRORS, and run at
    inference on each emitted slot's start and end (`refine_bounds`).

    JF3 options (section 12): `ref_shift_p` moves a training group's reference span (members' groups
    and the groups around uncovered falls) by |d| ~ U(ref_shift_s), later than the truth with
    probability `ref_shift_late`, so the stream stage trains on the late references that stage 1 gives
    it at inference; `transition` adds the labeller's transition span as a candidate to every group
    (set_heads.event_physics).
    """
    def __init__(
        self,
        in_channels: int = 7,
        n_bins: int = 368,
        window_frames: int = 125,
        d_model: int = 128,
        nhead: int = 4,
        frame_width: int = 32,
        frame_stride: int = 8,
        frame_patch_size: tuple[int, int] = (23, 8),
        frame_pooling: str = "cls",
        frame_encoder_layers: int = 4,
        context_layers: int = 2,
        bidir_layers: int = 1,
        predictor_layers: int = 2,
        pred_horizons: tuple[int, ...] = (4,),
        detach_target: bool = True,
        pre_projector: bool = True,
        num_states: int = 4,
        regression: bool = False,
        dropout: float = 0.1,
        mlp_ratio: int = 4,
        pre_proj_ratio: int = 4,
        physics_channels: int = 0,
        recording_head: bool = False,
        # Stage 1
        num_queries: int = 3,
        query_layers: int = 2,
        # Stage 2
        group_slots: int = 2,
        group_layers: int = 2,
        event_physics: bool = True,
        proposal_threshold: float = 0.05,
        group_iou: float = 0.2,
        group_margin_s: float = 3.0,
        group_max_s: float = 20.0,
        max_members: int = 12,
        max_groups: int = 6,
        member_drop: float = 0.25,
        span_jitter_s: float = 0.2,
        fall_jitter_s: float = 0.3,
        negative_groups: int = 1,
        group_gap_s: float | None = None,
        stable_anchors: bool = False,
        raw_drop_p: float = 0.0,
        raw_channels: int = 9,
        ref_shift_p: float = 0.0,
        ref_shift_s: tuple[float, float] = (0.8, 3.0),
        ref_shift_late: float = 2 / 3,
        transition: bool = False,
        # JF4 trunk (docs/falljepa_JF4_design.md; FallJEPA)
        trunk: str = "frames",
        radar_fusion: str = "attention",
        stem_channels: tuple[int, ...] = (16, 32, 48, 64),
        range_heads: int = 4,
        contrast_channels: int = 0,
        local: str = "none",
        tcn_dilations: tuple[int, ...] = (1, 2, 4, 8),
        rope: bool = False,
        contrast_scales: list | None = None,
        # JF7
        member_channels: bool = False,
        quality: bool = False,
        boundary: bool = False,
        boundary_half_s: float = 3.0,
        range_attention: bool = False,
    ):
        assert not recording_head, "the join-free model has no recording head (data.units: false)"
        super().__init__(in_channels, n_bins, window_frames, d_model, nhead, frame_width, frame_stride, frame_patch_size,
                         frame_pooling, frame_encoder_layers, context_layers, bidir_layers, predictor_layers,
                         pred_horizons, detach_target, pre_projector, num_states, regression, dropout, mlp_ratio,
                         pre_proj_ratio, physics_channels, trunk=trunk, radar_fusion=radar_fusion,
                         stem_channels=stem_channels, range_heads=range_heads, contrast_channels=contrast_channels,
                         local=local, tcn_dilations=tcn_dilations, rope=rope, contrast_scales=contrast_scales,
                         range_attention=range_attention)
        self.margin = (frame_width - frame_stride) // 2
        self.window = window_frames * frame_stride
        self.signal_groups = 2 if n_bins == 184 else 1          # I and Q, or z
        self.queries = FallQueries(d_model, nhead, num_queries, query_layers, frame_stride, physics_channels,
                                   mlp_ratio, dropout)
        self.refiner = GroupRefiner(d_model, nhead, group_slots, group_layers, frame_stride, mlp_ratio, dropout,
                                    event_physics, stable_anchors, transition, member_channels, quality)
        self.boundary = (BoundaryRefiner(d_model, self.refiner.channels, frame_stride, int(round(boundary_half_s * 50)),
                                         dropout=dropout) if boundary else None)
        self.proposal_threshold, self.group_iou = proposal_threshold, group_iou
        self.group_margin = int(round(group_margin_s * 50))
        self.group_max_tokens = int(round(group_max_s * 50)) // frame_stride
        self.max_members, self.max_groups = max_members, max_groups
        self.member_drop, self.span_jitter, self.fall_jitter = member_drop, span_jitter_s * 50, fall_jitter_s * 50
        self.negative_groups = negative_groups
        self.group_gap = None if group_gap_s is None else int(round(group_gap_s * 50))
        assert raw_drop_p == 0 or raw_channels < in_channels - 1, "raw_drop_p needs clutter-removed channels to keep"
        self.raw_drop_p, self.raw_channels = raw_drop_p, raw_channels
        self.ref_shift_p, self.ref_shift_late = ref_shift_p, ref_shift_late
        self.ref_shift = (ref_shift_s[0] * 50, ref_shift_s[1] * 50)

    # ---------------------------------------------------------------- windows

    def _windows(self, batch: dict, n: int, hop: int) -> dict:
        """The n windows of every chunk, chunk-major: [B n, ...]."""
        span = self.window + 2 * self.margin
        out = {}
        for key in ("x", "valid", "phys", "state", "weight", "fit"):
            if key in batch:
                t = batch[key]
                out[key] = torch.stack([t[..., k * hop:k * hop + span] for k in range(n)], 1).flatten(0, 1)
        return out

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = self.with_contrast(batch)                      # before the chunk is cut into windows
        m, W = self.margin, self.window
        L = batch["x"].size(-1) - 2 * m
        n, hop = 1, 0
        wins = batch
        if L != W:
            hop = int(batch["chunk_hop"][0])
            assert hop > 0 and (L - W) % hop == 0 and hop % self.frame_stride == 0, (L, W, hop)
            n = (L - W) // hop + 1
            wins = self._windows(batch, n, hop)
        trunk_in = {k: wins[k] for k in ("x", "valid", "phys", "state", "weight", "fit") if k in wins}
        if self.training and self.raw_drop_p > 0:
            drop = torch.rand(trunk_in["x"].size(0), device=trunk_in["x"].device) < self.raw_drop_p
            if drop.any():
                x = trunk_in["x"].clone()
                x[drop, :self.raw_channels] = 0.0
                trunk_in["x"] = x
        out = super().forward(trunk_in)
        step_valid = wins["valid"][:, m:m + W] > 0
        steps = wins["phys"][:, :, m:m + W] if self.physics_embed is not None and "phys" in wins else None
        key_pos = self.frame_pos_embed[:out["tokens"].size(1)] if self.rope else None   # the trunk has no positions
        hidden, logit, span = self.queries(out["tokens"], out["frame_valid"], step_valid, steps, key_pos)
        Bn, K = logit.shape
        out.update({"q_hidden": hidden, "q_logits": logit, "q_spans": span, "q_step_valid": step_valid,
                    "q_logit": logit.reshape(-1), "q_span_logits": span.reshape(Bn * K, W, 2),
                    "q_domain": step_valid.repeat_interleave(K, 0)})
        if "falls" not in batch:
            return out
        with torch.no_grad():
            logp = torch.log_softmax(span.float(), dim=2).cpu().numpy()
        lg = logit.detach().float().cpu().numpy()
        falls = self._falls(batch)
        rows = [match(lg[i], logp[i], self._targets(falls[i // n], (i % n) * hop, W)) for i in range(Bn)]
        self._put_targets(out, "q", rows, logit.device)
        if n > 1:
            self._stage2(batch, out, falls, n, hop, lg, logp)
        return out

    @staticmethod
    def _falls(batch) -> list[np.ndarray]:
        """Per item [E, 3] (start, stop, scored) of its labelled falls, in its centre steps."""
        f = batch["falls"].cpu().numpy()
        w = batch["falls_w"].cpu().numpy()
        keep = batch["falls_mask"].cpu().numpy()
        return [np.concatenate([f[i][keep[i]], (w[i][keep[i]] > 0)[:, None].astype(np.int64)], 1) for i in range(len(f))]

    @staticmethod
    def _targets(falls: np.ndarray, lo: int, length: int) -> list[tuple[int, int, int]]:
        """The falls of [lo, lo + length) in its own steps: SCORED / FLAGGED when wholly inside, CUT when
        partly inside."""
        out = []
        for a, b, scored in falls:
            a, b = int(a) - lo, int(b) - lo
            if b <= 0 or a >= length:
                continue
            inside = a >= 0 and b <= length
            out.append((a, b, (SCORED if scored else FLAGGED) if inside else CUT))
        return out

    @staticmethod
    def _put_targets(out: dict, prefix: str, rows: list, device) -> None:
        target, weight, span, span_w = (np.concatenate(x) for x in zip(*rows))
        out[f"{prefix}_target"] = torch.from_numpy(target).to(device)
        out[f"{prefix}_w"] = torch.from_numpy(weight).to(device)
        out[f"{prefix}_span"] = torch.from_numpy(span).to(device)
        out[f"{prefix}_span_w"] = torch.from_numpy(span_w).to(device)

    # ---------------------------------------------------------------- stage 2 in training

    def _chunk_tokens(self, tokens, frame_valid, B: int, n: int, hop: int, L: int) -> torch.Tensor:
        """The windows' tokens averaged on the chunk's token grid [B, L / s, d], centre-weighted."""
        s, Fn = self.frame_stride, tokens.size(1)
        Fc = L // s
        i = torch.arange(Fn, device=tokens.device)
        tri = torch.minimum(i + 1, Fn - i).float()
        tok = tokens.view(B, n, Fn, -1)
        w = tri[None, None] * frame_valid.view(B, n, Fn).float()
        num = sum(F.pad(w[:, k, :, None] * tok[:, k], (0, 0, k * hop // s, Fc - k * hop // s - Fn)) for k in range(n))
        den = sum(F.pad(w[:, k], (k * hop // s, Fc - k * hop // s - Fn)) for k in range(n))
        return num / den.clamp_min(1e-6)[..., None]

    def _stage2(self, batch, out, falls, n, hop, lg, logp) -> None:
        m, s, W = self.margin, self.frame_stride, self.window
        x, valid = batch["x"], batch["valid"]
        B, L = x.size(0), x.size(-1) - 2 * m
        K = lg.shape[1]
        chunk = self._chunk_tokens(out["tokens"], out["frame_valid"], B, n, hop, L)
        hidden = out["q_hidden"].view(B, n, K, -1)
        sv = out["q_step_valid"].cpu().numpy()
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,))))
        items, targets = [], []
        for b in range(B):
            rows = []
            for k in range(n):
                for r in proposals(lg[b * n + k], logp[b * n + k], None, k * hop, W, self.proposal_threshold, sv[b * n + k]):
                    r["window"] = k
                    rows.append(r)
            groups = sorted(stream_groups(rows, self.group_iou, self.group_gap),
                            key=lambda g: -max(r["logit"] for r in g[0]))[:self.max_groups]
            groups = [(self._drop_members(members, rng), span) for members, span in groups]
            extra, n_fall_extra = self._extra_groups(falls[b], groups, L, rng)
            n_fall_groups = len(groups) + n_fall_extra          # the reference shift skips the no-fall groups
            groups += extra
            for i, (members, (a, e)) in enumerate(groups):
                if self.training and members:
                    a += int(round(rng.normal(0, self.span_jitter)))
                    e += int(round(rng.normal(0, self.span_jitter)))
                if self.training and i < n_fall_groups and rng.random() < self.ref_shift_p:
                    d = int(round(rng.uniform(*self.ref_shift))) * (1 if rng.random() < self.ref_shift_late else -1)
                    a, e = a + d, e + d
                a = int(np.clip(a, 0, L - MIN_SPAN))
                e = int(np.clip(max(e, a + MIN_SPAN), a + MIN_SPAN, L))
                starts = np.array([r["start"] for r in members])
                ends = np.array([r["end"] for r in members])
                j0, j1 = group_region(starts, ends, (a, e), self.group_margin, s, self.group_max_tokens, L // s)
                lo, hi = j0 * s, j1 * s
                p = 1.0 / (1.0 + np.exp(-np.array([r["logit"] for r in members])))
                h = (torch.stack([hidden[b, r["window"], r["query"]] for r in members]) if members
                     else chunk.new_zeros(0, chunk.size(-1)))
                items.append(group_item(chunk[b, j0:j1], x[b, :3 * self.signal_groups, :, m + lo:m + hi],
                                        valid[b, m + lo:m + hi] > 0, (a - lo, e - lo), h,
                                        member_features(members, lo, hi - lo, W), group_stats(p, (a, e))))
                targets.append(self._targets(falls[b], lo, hi - lo))
        if not items:
            return
        g = pad_groups(items, self.signal_groups)
        res = self.refiner(g)
        G, Q = res["logit"].shape
        Lr = res["span_logits"].size(2)
        with torch.no_grad():
            glogp = torch.log_softmax(res["span_logits"].float(), dim=2).cpu().numpy()
        glg = res["logit"].detach().float().cpu().numpy()
        rows = [match(glg[i], glogp[i], targets[i]) for i in range(G)]
        self._put_targets(out, "g", rows, x.device)
        out["g_logit"] = res["logit"].reshape(-1)
        out["g_span_logits"] = res["span_logits"].reshape(G * Q, Lr, 2)
        out["g_domain"] = g["step_valid"].repeat_interleave(Q, 0)
        if "quality" in res:
            out["g_quality"] = res["quality"].reshape(-1)
            out["g_qtarget"], out["g_qw"] = self._quality_targets(glogp, g["step_valid"].cpu().numpy(), rows, targets, x.device)
        if self.boundary is not None:
            self._boundary_train(res, g, rows, out, rng)

    @staticmethod
    def _quality_targets(glogp, valid, rows, targets, device):
        """Per slot: tIoU of its MBR span with its matched fall (scored or flagged), else the best tIoU with
        any such fall of the region, else 0; weight 1 (0 for a slot matched to a cut fall)."""
        G, Q = glogp.shape[:2]
        qt, qw = np.zeros((G, Q), np.float32), np.ones((G, Q), np.float32)
        for i in range(G):
            keep = np.flatnonzero(valid[i])
            if len(keep) < 2:
                continue
            falls = [(a, b) for a, b, kind in targets[i] if kind != CUT]
            _, weight, span, span_w = rows[i]
            for q in range(Q):
                if span_w[q] == 0 and weight[q] == 0:
                    qw[i, q] = 0.0                                # matched to a cut fall
                    continue
                a, b = mbr_span(keep, glogp[i, q, keep, 0], glogp[i, q, keep, 1], tie_iou=TIE_IOU)
                if span_w[q] > 0:
                    qt[i, q] = tiou(a, b, *span[q])
                elif falls:
                    qt[i, q] = max(tiou(a, b, fa, fb) for fa, fb in falls)
        return torch.from_numpy(qt.reshape(-1)).to(device), torch.from_numpy(qw.reshape(-1)).to(device)

    def _boundary_train(self, res, g, rows, out, rng) -> None:
        """Crops around the matched falls' starts and ends, moved by a draw from BOUNDARY_ERRORS."""
        L = g["step_valid"].size(1)
        half = self.boundary.half
        group, slot, centre, kind, target, w = [], [], [], [], [], []
        for i, (_, _, span, span_w) in enumerate(rows):
            for q in np.flatnonzero(span_w > 0):
                for k in (0, 1):
                    truth = int(span[q][k])
                    r, acc = rng.random(), 0.0
                    for prob, draw, lo_, hi_ in BOUNDARY_ERRORS[k]:
                        acc += prob
                        if r < acc:
                            break
                    err = rng.normal(0, lo_) if draw == "n" else rng.uniform(lo_, hi_)
                    c = int(np.clip(truth + round(err * 50), 0, L - 1))
                    if abs(truth - c) > half:
                        continue
                    group.append(i), slot.append(int(q)), centre.append(c), kind.append(k)
                    target.append(truth - c + half), w.append(float(span_w[q]))
        if not group:
            return
        dev = g["step_valid"].device
        t = lambda v: torch.tensor(v, dtype=torch.long, device=dev)  # noqa: E731
        logits, mask = self.boundary(res["tokens"], res["steps"], g["step_valid"], res["hidden"],
                                     t(group), t(slot), t(centre), t(kind))
        out["b_logits"] = logits[..., None]
        out["b_domain"] = mask
        out["b_target"] = torch.tensor(target, dtype=torch.float32, device=dev)[:, None]
        out["b_w"] = torch.tensor(w, dtype=torch.float32, device=dev)

    def _drop_members(self, members: list[dict], rng) -> list[dict]:
        members = sorted(members, key=lambda r: -r["logit"])[:self.max_members]
        if not self.training or self.member_drop <= 0:
            return members
        return members[:1] + [r for r in members[1:] if rng.random() >= self.member_drop]

    def _extra_groups(self, falls: np.ndarray, groups: list, L: int, rng) -> list:
        """A group around every labelled fall (wholly in the chunk) that no group's region would cover,
        at its jittered span, and `negative_groups` random no-fall spans after them; neither has members.
        Returns the groups and how many of them are fall groups."""
        extra = []
        covered = [(min([r["start"] for r in mem] + [a]) - self.group_margin,
                    max([r["end"] for r in mem] + [e]) + self.group_margin) for mem, (a, e) in groups]
        for a, b, _ in falls:
            a, b = int(a), int(b)
            if a < 0 or b > L or any(lo <= a and b <= hi for lo, hi in covered):
                continue
            ja = a + int(round(rng.normal(0, self.fall_jitter)))
            jb = b + int(round(rng.normal(0, self.fall_jitter)))
            extra.append(([], (ja, max(jb, ja + MIN_SPAN))))
        n_fall = len(extra)
        for _ in range(self.negative_groups):
            for _ in range(5):
                length = int(rng.integers(50, 151))
                if L <= length:
                    break
                a = int(rng.integers(0, L - length))
                if all(a + length <= int(fa) or a >= int(fb) for fa, fb, _ in falls):
                    extra.append(([], (a, a + length)))
                    break
        return extra, n_fall

    # ---------------------------------------------------------------- stage 2 at inference

    @torch.no_grad()
    def refine(self, items: list[dict]) -> dict[str, torch.Tensor]:
        """GroupRefiner on these groups (sets.group_item dicts): fall logits [G, Q], start / end
        log-probabilities over each region's valid steps [G, Q, L, 2] and the valid steps [G, L]."""
        g = pad_groups(items, self.signal_groups)
        res = self.refiner(g)
        logp = torch.log_softmax(res["span_logits"].float().masked_fill(~g["step_valid"][:, None, :, None], -torch.inf), dim=2)
        out = {"logit": res["logit"].float(), "logp": logp, "step_valid": g["step_valid"]}
        if "quality" in res:
            out["quality"] = res["quality"].float()
        if self.boundary is not None:
            out["_bounds"] = (res["tokens"], res["steps"], g["step_valid"], res["hidden"])
        return out

    @torch.no_grad()
    def refine_bounds(self, res: dict, spans: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
        """The BoundaryRefiner on (group, slot, start, end) spans in region steps from `refine`'s result:
        each boundary moves to the argmax of its crop. Spans unchanged without a boundary refiner."""
        if self.boundary is None or not spans:
            return [(a, b) for _, _, a, b in spans]
        tokens, steps, valid, hidden = res["_bounds"]
        dev = steps.device
        n = len(spans)
        group = torch.tensor([s[0] for s in spans for _ in (0, 1)], dtype=torch.long, device=dev)
        slot = torch.tensor([s[1] for s in spans for _ in (0, 1)], dtype=torch.long, device=dev)
        centre = torch.tensor([v for s in spans for v in (s[2], min(s[3], steps.size(-1) - 1))], dtype=torch.long, device=dev)
        kind = torch.tensor([0, 1] * n, dtype=torch.long, device=dev)
        logits, _ = self.boundary(tokens, steps, valid, hidden, group, slot, centre, kind)
        moved = (centre - self.boundary.half + logits.argmax(1)).view(n, 2).cpu().numpy()
        return [(int(a), int(max(b, a + MIN_SPAN))) for a, b in moved]

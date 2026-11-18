"""
On-the-fly stream re-synthesis ("remix"): a training window rebuilt from segments of the published
training streams, joined the way the organizers' generator joins them (README: "segments are joined
with a 0.2 s crossfade, activity files are cut at random points, and fall files are cropped with
random margins around the annotated event"). It is the dynamic mixing of speech separation
(Wavesplit): every epoch sees new contexts around the same recordings.

Only the published training streams and their `streams_composition.csv` are read; segments are cut
inside their interior (no timestep of an old crossfade), so every join in a remixed window is new.

A sampled window becomes

    [ tail of random segments ] [ the anchor, re-cropped ] [ head of random segments ]

    anchor   with a fall in the window: the recording of that fall, cropped to a fresh lead-in
             U(lead_s) before the fall and U(tail_s) after it (as far as the stream holds them), the
             window placed so that the whole fall lies in it at a random position;
             without one: the recording under the window centre, cut at random points around it
    others   random segments (kind drawn from `kinds`), cut at random points that are not within
             1 s before to 0.5 s after any labelled fall, so no fall is cut

Labels move with their timesteps, the 0.2 s crossfades get loss weight 0, and the physics features
are recomputed on the composed signal with `CONTEXT` timesteps of it on each side, so they are what
a whole stream would give. Remix needs the training targets and no cross-validation masks.
"""
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .labels import Event
from .physics import stream_physics

RATE_HZ = 50.0
FADE = 10              # 0.2 s crossfade, the generator's fade_seconds
CONTEXT = 160          # signal kept on each side of the window for the physics features (profile steps reach 155)
MIN_PIECE = 50         # 1 s: the shortest piece of a segment
CUT_BEFORE_S, CUT_AFTER_S = 1.0, 0.5   # no cut this close to a labelled fall
MAX_PIECES = 16


@dataclass
class Segment:
    stream: int                        # index into store.ids
    a: int                             # interior [a, b) in timesteps: no crossfade timestep
    b: int
    kind: str                          # adl, fall or other
    falls: list[tuple[int, int, bool]] = field(default_factory=list)   # (start, end, flagged) inside it


class RemixPool:
    def __init__(self, composition: str | Path, ids: list[str], lengths: dict[str, int],
                 events: dict[str, list[Event]], p: float = 0.5, lead_s=(1.0, 8.0), tail_s=(2.0, 10.0),
                 kinds: dict[str, float] | None = None):
        self.p = float(p)
        self.lead = tuple(float(v) * RATE_HZ for v in lead_s)
        self.tail = tuple(float(v) * RATE_HZ for v in tail_s)
        kinds = kinds or {"adl": 0.7, "fall": 0.2, "other": 0.1}
        index = {sid: i for i, sid in enumerate(ids)}
        rows: dict[str, list[dict]] = {}
        with Path(composition).open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["stream_id"] in index:
                    rows.setdefault(r["stream_id"], []).append(r)

        self.segments: list[list[Segment]] = [[] for _ in ids]      # per stream, in time order
        self.forbidden: list[np.ndarray] = []                        # per stream: no cut here
        for sid in ids:
            T = lengths[sid]
            no_cut = np.zeros(T + 1, dtype=bool)
            falls = []
            for e in events.get(sid, []):
                a, b = max(int(round(e.t_start * RATE_HZ)), 0), min(int(round(e.t_end * RATE_HZ)), T)
                falls.append((a, b, e.flagged))
                no_cut[max(a - int(CUT_BEFORE_S * RATE_HZ), 0):min(b + int(CUT_AFTER_S * RATE_HZ), T) + 1] = True
            self.forbidden.append(no_cut)
            segs = sorted(rows.get(sid, []), key=lambda r: float(r["t_start"]))
            for i, r in enumerate(segs):
                lo, hi = float(r["t_start"]), float(r["t_end"])
                if i > 0:
                    lo = max(lo, float(segs[i - 1]["t_end"]))
                if i + 1 < len(segs):
                    hi = min(hi, float(segs[i + 1]["t_start"]))
                a = max(int(math.ceil(lo * RATE_HZ - 1e-6)), 0)
                b = min(int(math.floor(hi * RATE_HZ + 1e-6)), T)
                if b - a < 2:
                    continue
                inside = [(fa, fb, fl) for fa, fb, fl in falls if a <= fa and fb <= b]
                self.segments[index[sid]].append(Segment(index[sid], a, b, r["kind"], inside))
        self._starts = [np.array([s.a for s in segs]) for segs in self.segments]

        pool = [s for segs in self.segments for s in segs if s.b - s.a >= 2 * MIN_PIECE]
        self.by_kind = {k: [s for s in pool if s.kind == k] for k in kinds}
        self.kinds = [k for k in kinds if self.by_kind[k] and kinds[k] > 0]
        w = np.array([kinds[k] for k in self.kinds], dtype=np.float64)
        self.kind_p = w / w.sum()
        assert self.kinds, "remix: no segment to draw from"

    # ----------------------------------------------------------------- helpers

    def owner(self, stream: int, t: int) -> Segment | None:
        """The segment whose interior holds timestep t (the next one inside a crossfade)."""
        segs, starts = self.segments[stream], self._starts[stream]
        i = int(np.searchsorted(starts, t, side="right")) - 1
        if i >= 0 and segs[i].a <= t < segs[i].b:
            return segs[i]
        return segs[i + 1] if i + 1 < len(segs) else (segs[i] if i >= 0 else None)

    def _cut(self, stream: int, lo: int, hi: int, rng: np.random.Generator, default: int) -> int:
        """A random cut in [lo, hi] outside the no-cut zones, or `default` if there is none."""
        if hi < lo:
            return default
        ok = np.flatnonzero(~self.forbidden[stream][lo:hi + 1])
        return int(lo + ok[int(rng.integers(len(ok)))]) if len(ok) else default

    def _draw(self, rng: np.random.Generator) -> Segment:
        kind = self.kinds[int(rng.choice(len(self.kinds), p=self.kind_p))]
        segs = self.by_kind[kind]
        return segs[int(rng.integers(len(segs)))]

    def _anchor(self, stream: int, start: int, window: int, margin: int, rng: np.random.Generator):
        """(anchor segment, crop c0, crop c1, window start lo), all in stream timesteps, or None."""
        segs = self.segments[stream]
        falls = [(fa, fb, seg) for seg in segs for fa, fb, flagged in seg.falls
                 if not flagged and fa < start + window and fb > start]
        if falls:
            fa, fb, seg = falls[int(rng.integers(len(falls)))]
            if fb - fa > window:
                return None
            c0 = max(seg.a, int(round(fa - rng.uniform(*self.lead))))
            c1 = min(seg.b, int(round(fb + rng.uniform(*self.tail))))
            lo = int(rng.integers(fb - margin - window, fa - margin + 1))
            return seg, c0, c1, lo
        centre = start + window // 2
        seg = self.owner(stream, centre)
        if seg is None:
            return None
        centre = min(max(centre, seg.a), seg.b - 1)
        c0 = self._cut(stream, seg.a, centre - MIN_PIECE // 2, rng, seg.a)
        c1 = self._cut(stream, centre + MIN_PIECE // 2, seg.b, rng, seg.b)
        return seg, c0, c1, start - margin

    # ----------------------------------------------------------------- compose

    def compose(self, store, targets: dict, stream: int, start: int, window: int, margin: int,
                rng: np.random.Generator) -> dict | None:
        """
        The remixed window [lo, lo + window + 2 margin) of a sampled one: z [3, W, 368] (z-scored),
        diff [3, W, 368] or None, phys [12, W] or None, and state, weight, bounds, inside [W].
        None when the sampled window cannot be remixed (the caller keeps it as it is).
        """
        W = window + 2 * margin
        found = self._anchor(stream, start, window, margin, rng)
        if found is None:
            return None
        seg, c0, c1, lo = found
        if c1 - c0 <= FADE:
            return None
        s = lo - c0                                    # window start relative to the anchor crop

        before, need = [], CONTEXT + 1 - s             # +1: the diff channel reads one more timestep
        while need > 0 and len(before) < MAX_PIECES:
            q = self._draw(rng)
            cut = self._cut(q.stream, q.a, q.b - MIN_PIECE, rng, q.a)
            before.append((q.stream, cut, q.b))
            need -= q.b - cut - FADE
        if need > 0:
            return None
        after, need = [], s + W + CONTEXT - (c1 - c0)
        while need > 0 and len(after) < MAX_PIECES:
            q = self._draw(rng)
            cut = self._cut(q.stream, q.a + MIN_PIECE, q.b, rng, q.b)
            after.append((q.stream, q.a, cut))
            need -= cut - q.a - FADE
        if need > 0:
            return None
        pieces = before[::-1] + [(stream, c0, c1)] + after

        total = sum(b - a for _, a, b in pieces) - FADE * (len(pieces) - 1)
        z = np.zeros((3, total, store.mean.shape[-1]), dtype=np.float32)
        state = np.zeros(total, dtype=np.int64)
        weight = np.zeros(total, dtype=np.float32)
        bounds = np.zeros((total, 2), dtype=np.float32)
        inside = np.zeros(total, dtype=bool)
        pos, anchor_pos = 0, 0
        for i, (k, a, b) in enumerate(pieces):
            sid = store.ids[k]
            zp = store.read_z(sid, a, b)
            st, wt, bd, ins = (arr[a:b] for arr in targets[sid])
            n = b - a
            if i == 0:
                z[:, :n] = zp
                state[:n], weight[:n], bounds[:n], inside[:n] = st, wt, bd, ins
                end = n
            else:
                pos = end - FADE
                fade = ((np.arange(FADE, dtype=np.float32) + 1) / (FADE + 1))[None, :, None]
                z[:, pos:end] = (1 - fade) * z[:, pos:end] + fade * zp[:, :FADE]
                z[:, end:pos + n] = zp[:, FADE:]
                weight[pos:end] = 0.0                  # the new crossfade
                state[end:pos + n], weight[end:pos + n] = st[FADE:], wt[FADE:]
                bounds[end:pos + n], inside[end:pos + n] = bd[FADE:], ins[FADE:]
                end = pos + n
            if i == len(before):
                anchor_pos = 0 if i == 0 else pos
        lo = anchor_pos + s
        assert lo - CONTEXT - 1 >= 0 and lo + W + CONTEXT <= total, "remix: not enough signal around the window"

        diff = None
        if store.diff_channel:
            diff = np.diff(z[:, lo - 1:lo + W], axis=1) / store.diff_std
        phys = None
        if store.physics is not None:
            phys = stream_physics(z[:, lo - CONTEXT:lo + W + CONTEXT])[:, CONTEXT:CONTEXT + W]
        sl = slice(lo, lo + W)
        return {"z": z[:, sl], "diff": diff, "phys": phys, "state": state[sl], "weight": weight[sl],
                "bounds": bounds[sl], "inside": inside[sl]}

"""
Units: the source recordings of a stream, the regions the recording head classifies and localizes
(docs/falljepa_recording_head.md).

A stream is source recordings joined by 0.2 s crossfades, and every label is the official labeller
(`annotate.propose_fall`) run on one recording. So the model decides per recording: does it hold a
fall, and where in it. The recordings are found from the signal alone, with Amir's join detector
(postprocess.join_score / join_peaks on the unnormalized CIR), the same way on training, held-out
and test streams, so the units the model learns on have the errors it meets at test.

    units       the spans between consecutive join peaks, without the crossfade: [peak + 6, next - 5);
                one longer than MAX_UNIT (22 s: Autre/Empty recordings, missed joins) is cut into
                equal non-overlapping chunks of at most CHUNK (20 s); spans under MIN_UNIT are dropped
    channels    per timestep, relative to its unit (the labeller's own quantities, official functions
                on the unnormalized CIR of the unit): per radar the smoothed state against the unit's
                first-1 s and last-1.5 s profiles, log1p(cleaned raw-frame motion / its unit median),
                and the rule's transition score rise * log1p(motion sum / median) over 1.5 s; then the
                distance to the unit's start and end (s / 20, at most 1). Zero outside every unit
    scalars     per unit: log10 profile change per radar (+4.67, so 0 is Amir's gate threshold
                2.12e-5), the same sorted, the top radar's share of the change, the log10 motion
                peak / median and the log10 stillness after (last 1.5 s / peak) on the anchor radar,
                and the chunk flag
    rule        per unit: the official rule's [left, right) (raw-frame motion, anchor radar), or -1
                for units under 2 s or with a rule interval under 0.2 s

A fall belongs to the unit it starts in. Its span may run past the unit's end by up to
SPAN_AFTER (2 s): on the official train and val streams the join detector puts a peak inside 16 of
the 253 clean falls (6 %), always 0.7-3.6 s after the start, at the landing, where all three radars
change at once like at a join. The unit holding such a fall's tail is a no-fall unit.

Everything is computed once per stream file and cached (`unit_cache`).
"""
from pathlib import Path

import h5py
import numpy as np

from .. import postprocess as pp
from .labels import Event

RATE_HZ = 50.0
STEP_S = 0.02
MAX_UNIT = 1100          # 22 s: the longest fall clip is 18.6 s
CHUNK = 1000             # long units are cut into chunks of at most 20 s
MIN_UNIT = 10            # 0.2 s
RULE_MIN = 100           # the rule runs on units of >= 2 s (Amir's MIN_WINDOW)
WIDTH = 75               # the rule's 1.5 s window
SPAN_BEFORE = 25         # a span may start 0.5 s before its unit
SPAN_AFTER = 100         # and end 2 s after it (a join peak at a fall's landing cuts the fall)
EDGE = 100               # a window decodes the units at least 2 s from its inner edges
GATE_LOG10 = 4.67        # -log10(2.12e-5)
NUM_CHANNELS = 3 * 3 + 2
NUM_SCALARS = 3 + 3 + 1 + 1 + 1 + 1
CHANNEL_NAMES = [f"{k}_r{r}" for r in range(3) for k in ("state", "motion", "score")] + ["from_start", "to_end"]


def make_units(peaks: np.ndarray, T: int) -> np.ndarray:
    """[U, 3] int64 (start, stop, chunk) of the units of a stream with these join peaks."""
    peaks = np.asarray(peaks, dtype=np.int64)
    los = np.r_[0, peaks + pp.BLEND_HALF + 1]
    his = np.r_[peaks - pp.BLEND_HALF, T]
    out = []
    for lo, hi in zip(los, his):
        lo, hi = max(int(lo), 0), min(int(hi), T)
        if hi - lo < MIN_UNIT:
            continue
        if hi - lo <= MAX_UNIT:
            out.append((lo, hi, 0))
            continue
        n = int(np.ceil((hi - lo) / CHUNK))
        edges = np.round(np.linspace(lo, hi, n + 1)).astype(np.int64)
        out.extend((int(a), int(b), 1) for a, b in zip(edges[:-1], edges[1:]))
    return np.array(out, dtype=np.int64).reshape(-1, 3)


def unit_physics(A, cir: np.ndarray, chunk: bool = False) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """One unit's unnormalized CIR [3, n, 368] -> channels [9, n] (radar-major: state, motion,
    score), scalars [NUM_SCALARS] and the rule's (left, right) in unit timesteps or (-1, -1)."""
    radars, n, _ = cir.shape
    grid = np.arange(n) * STEP_S
    xs, profiles, motions, states = {}, {}, {}, {}
    for r in range(radars):
        x = cir[r].astype(np.float64)
        before, after = A._profiles(x, grid)
        xs[r], profiles[r] = x, (before, after)
        motions[r] = pp.raw_motion_on_grid(x)
        states[r] = A._state(x, before, after)
    cleaned = A._suppress_spikes(motions, states, STEP_S)
    channels = np.zeros((3 * radars, n), dtype=np.float32)
    change = np.zeros(radars)
    smooth_motion = {}
    for r in range(radars):
        before, after = profiles[r]
        state = A._smooth(states[r], STEP_S, A.SMOOTH_S)
        motion = A._smooth(cleaned[r], STEP_S, A.SMOOTH_S)
        smooth_motion[r] = motion
        change[r] = float(np.sum((after - before) ** 2))
        med = float(np.median(motion)) + 1e-18
        score = np.zeros(n)
        if n > WIDTH + 2:
            rise = state[WIDTH:] - state[:-WIDTH]
            msum = np.convolve(motion, np.ones(WIDTH), mode="valid")
            k = min(len(rise), len(msum))
            score[:k] = rise[:k] * np.log1p(msum[:k] / (float(np.median(msum)) + 1e-18))
        channels[3 * r] = state
        channels[3 * r + 1] = np.log1p(motion / med)
        channels[3 * r + 2] = np.clip(score / 3.0, -5, 5)
    lc = np.log10(change + 1e-12) + GATE_LOG10
    anchor = int(np.argmax(change))
    m = smooth_motion[anchor]
    peak = float(m.max()) if n else 0.0
    tail = m[grid >= grid[-1] - 1.5] if n else m
    scalars = np.concatenate([lc, np.sort(lc)[::-1], [change.max() / (change.sum() + 1e-30)],
                              [np.log10(peak / (np.median(m) + 1e-18) + 1e-12) if n else 0.0],
                              [np.log10(float(tail.mean()) / (peak + 1e-18) + 1e-12) if n else 0.0],
                              [float(chunk)]]).astype(np.float32)
    rule = (-1, -1)
    if n >= RULE_MIN:
        state = A._smooth(states[anchor], STEP_S, A.SMOOTH_S)
        left, right, _ = A._transition_bounds(state, smooth_motion[anchor])
        if right - left >= pp.MIN_REFINED:
            rule = (int(left), int(right))
    return channels, scalars, rule


def stream_units(A, cir, peaks: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """The units of one stream (`cir`: its unnormalized [3, T, 368], array or h5py dataset):
    units [U, 3], channels [NUM_CHANNELS, T] float16, scalars [U, NUM_SCALARS], rule [U, 2], peaks."""
    T = int(cir.shape[1])
    if peaks is None:
        peaks = pp.join_peaks(pp.join_score(np.asarray(cir[:])))
    units = make_units(peaks, T)
    channels = np.zeros((NUM_CHANNELS, T), dtype=np.float16)
    scalars = np.zeros((len(units), NUM_SCALARS), dtype=np.float32)
    rule = np.full((len(units), 2), -1, dtype=np.int64)
    for i, (lo, hi, chunk) in enumerate(units):
        ch, sc, (left, right) = unit_physics(A, np.asarray(cir[:, lo:hi], dtype=np.float64), bool(chunk))
        channels[:9, lo:hi] = ch
        t = np.arange(hi - lo)
        channels[9, lo:hi] = np.minimum(t / RATE_HZ, 20.0) / 20.0
        channels[10, lo:hi] = np.minimum((hi - lo - t) / RATE_HZ, 20.0) / 20.0
        scalars[i] = sc
        if left >= 0:
            rule[i] = (lo + left, lo + right)
    return {"units": units, "channels": channels, "scalars": scalars, "rule": rule,
            "peaks": np.asarray(peaks, dtype=np.int64)}


def unit_cache(A, path: str | Path, cache_dir: str | Path | None) -> dict[str, np.ndarray]:
    """stream_units of one stream file, cached as .npz under cache_dir (keyed by path, size, mtime)."""
    path = Path(path)
    cache = None
    if cache_dir is not None:
        stat = path.stat()
        cache = Path(cache_dir) / f"{path.parent.name}_{path.stem}_{stat.st_size}_{int(stat.st_mtime)}_v1.npz"
        if cache.exists():
            with np.load(cache) as data:
                return {k: data[k] for k in data.files}
    with h5py.File(path, "r") as handle:
        out = stream_units(A, np.asarray(handle["cir"][:], dtype=np.float32))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **out)
    return out


def unit_targets(units: np.ndarray, rule: np.ndarray, events: list[Event], rate_hz: float = RATE_HZ,
                 flagged_span_weight: float = 0.5, min_share: float = 0.4, tail_weight: float = 0.5) -> dict[str, np.ndarray]:
    """
    Per unit: fall (1 when a labelled fall starts in it, holds >= min_share of the fall and ends
    within SPAN_AFTER of its end), fall_w (0 when every such fall is flagged: the scorer ignores
    it), span [2] (the fall's [start, stop), stream timesteps) and span_w (1, flagged_span_weight for
    a flagged fall, 0 for two falls in one unit), rule_w (1 for a non-fall unit with a rule
    interval). A unit that holds only a fall's tail is a no-fall unit of weight tail_weight; one
    that holds a fall it cannot own (starting in it but too little of it, or ending too late) has
    no target.
    """
    U = len(units)
    fall, fall_w = np.zeros(U, np.float32), np.ones(U, np.float32)
    span, span_w = np.zeros((U, 2), np.int64), np.zeros(U, np.float32)
    falls = [(int(round(e.t_start * rate_hz)), int(round(e.t_end * rate_hz)), e.flagged) for e in events]
    los, his = units[:, 0], units[:, 1]
    owned = [[] for _ in range(U)]
    for a, b, flagged in falls:
        if b <= a or U == 0:
            continue
        share = (np.clip(np.minimum(b, his) - np.maximum(a, los), 0, None) / (b - a))
        if share.max() <= 0:
            continue
        start_in = np.flatnonzero((los <= a) & (a < his))
        owner = int(start_in[0]) if len(start_in) else int(np.argmax(share))
        for j in np.flatnonzero(share > 0):
            if j != owner:
                fall_w[j] = min(fall_w[j], tail_weight)          # the tail of a fall that started earlier
        if share[owner] >= min_share and b <= his[owner] + SPAN_AFTER and a >= los[owner] - SPAN_BEFORE:
            owned[owner].append((a, b, flagged))
        else:
            fall_w[owner] = 0.0
    for i, hits in enumerate(owned):
        if not hits or fall_w[i] == 0:
            continue
        fall[i] = 1.0
        fall_w[i] = 0.0 if all(h[2] for h in hits) else 1.0
        if len(hits) == 1:
            a, b, flagged = hits[0]
            span[i] = (a, b)
            span_w[i] = flagged_span_weight if flagged else 1.0
    rule_w = ((fall == 0) & (fall_w > 0) & (rule[:, 0] >= 0)).astype(np.float32)
    return {"fall": fall, "fall_w": fall_w, "span": span, "span_w": span_w, "rule_w": rule_w}


def contained(units: np.ndarray, start: int, window: int, T: int, edge: int = EDGE) -> np.ndarray:
    """Indices of the units that lie inside the window [start, start + window) at least `edge`
    timesteps from each window edge that is not a stream end."""
    if len(units) == 0:
        return np.zeros(0, dtype=np.int64)
    left = start + (edge if start > 0 else 0)
    right = start + window - (edge if start + window < T else 0)
    ok = (units[:, 0] >= max(left, 0)) & (units[:, 1] <= min(right, T))
    return np.flatnonzero(ok)

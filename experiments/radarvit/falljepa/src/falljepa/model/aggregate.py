"""
The two aggregation levels and the event decoder.

1. Frames -> timesteps (inside a window, differentiable): frame i covers the w timesteps
   [i s, i s + w) of the window read with its margins and predicts a state for each of them. Every
   timestep is covered by w / s frames; their logits are averaged with a Hann weight over the
   position inside the frame (a frame knows least about its own edges).
2. Windows -> stream (at inference): the per-timestep log-probabilities of the overlapping windows
   are averaged with a Hann weight over the position inside the window, plus a small floor so the
   stream's first and last timesteps (seen only by a window edge) still get a value. At 50 %
   overlap a Hann window sums to a constant, so with the default 20 s / 10 s grid an interior
   timestep is effectively decided by the window it is central in.
3. Stream states -> events: Viterbi over the grammar none -> begin -> middle -> end -> none
   (begin -> end allowed for very short falls), with a penalty per state change; every run of
   non-none states is one fall. Its score is the mean P(fall) over the run (the evaluator ranks
   predictions by it), its confidence also reported as the frames' disagreement. With 2 states
   (none / fall) the grammar is none <-> fall with the same penalty.

Boundary regression (ActionFormer), when the model has it: every timestep inside a fall predicts
its distances to the fall's start and end. They are averaged over frames like the logits, turned
into absolute stream times and merged over windows like the log-probabilities. The decoder then
either refines each Viterbi run's start and end (`refine`), or proposes one interval per timestep
and keeps the best by greedy NMS (`kind: "proposals"`).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from ..data.labels import BEGIN, END, MIDDLE, NONE


def hann(n: int, floor: float = 0.0, device=None) -> torch.Tensor:
    """Hann weights over n positions sampled at bin centres (never exactly 0), plus a floor."""
    pos = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n
    return 0.5 - 0.5 * torch.cos(2 * math.pi * pos) + floor


def frames_mean(values: torch.Tensor, stride: int, weights: torch.Tensor | None = None) -> torch.Tensor:
    """values [B, F, w, C] (each frame's value for each of its w timesteps) -> their weighted mean
    per timestep [B, (F - 1) s + w, C], Hann over the position in the frame by default."""
    B, Fn, w, C = values.shape
    steps = (Fn - 1) * stride + w
    wt = hann(w, device=values.device) if weights is None else weights
    idx = (torch.arange(Fn, device=values.device)[:, None] * stride
           + torch.arange(w, device=values.device)[None, :]).reshape(-1)
    wsum = torch.zeros(steps, device=values.device).index_add(0, idx, wt.repeat(Fn))
    acc = torch.zeros(B, steps, C, device=values.device).index_add(
        1, idx, (values.float() * wt[None, None, :, None]).reshape(B, Fn * w, C))
    return acc / wsum[None, :, None]


def frames_to_steps(frame_logits: torch.Tensor, stride: int, weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    frame_logits [B, F, w, K] -> per-timestep weighted mean logits [B, (F - 1) s + w, K] and the
    weighted std of P(fall) = 1 - P(none) across the frames that cover each timestep [B, steps].
    """
    B, Fn, w, K = frame_logits.shape
    steps = (Fn - 1) * stride + w
    wt = hann(w, device=frame_logits.device) if weights is None else weights
    idx = (torch.arange(Fn, device=frame_logits.device)[:, None] * stride
           + torch.arange(w, device=frame_logits.device)[None, :]).reshape(-1)
    wsum = torch.zeros(steps, device=frame_logits.device).index_add(0, idx, wt.repeat(Fn))
    mean_logits = frames_mean(frame_logits, stride, wt)

    p_fall = 1 - torch.softmax(frame_logits.float(), dim=-1)[..., NONE]                # [B, F, w]
    p1 = torch.zeros(B, steps, device=frame_logits.device).index_add(1, idx, (p_fall * wt).reshape(B, -1))
    p2 = torch.zeros(B, steps, device=frame_logits.device).index_add(1, idx, (p_fall.square() * wt).reshape(B, -1))
    mean_p = p1 / wsum
    spread = (p2 / wsum - mean_p.square()).clamp_min(0).sqrt()
    return mean_logits, spread


def triangular(n: int) -> np.ndarray:
    """Amir's centre weights over a window of n positions: min(i + 1, n - i) / (n / 2)."""
    i = np.arange(n)
    return np.minimum(i + 1, n - i) / (n / 2)


class StreamAccumulator:
    """Weighted sums of window log-probabilities (and frame disagreement, and the regressed fall
    boundaries in absolute stream seconds when the model has them) per stream.
    weight: "hann" (with `floor`) or "triangular" (Amir); merge: "logp" (a weighted mean of the
    log-probabilities, renormalized) or "prob" (a weighted mean of the probabilities, Amir)."""
    def __init__(self, lengths: dict[str, int], window: int, num_states: int = 4, floor: float = 1e-3,
                 weight: str = "hann", merge: str = "logp"):
        assert weight in ("hann", "triangular") and merge in ("logp", "prob")
        self.lengths = lengths
        self.window = window
        self.merge_kind = merge
        self.w = hann(window, floor=floor).numpy().astype(np.float64) if weight == "hann" else triangular(window)
        self.logp = {sid: np.zeros((T, num_states)) for sid, T in lengths.items()}
        self.spread = {sid: np.zeros(T) for sid, T in lengths.items()}
        self.wsum = {sid: np.zeros(T) for sid, T in lengths.items()}
        self.bounds: dict[str, np.ndarray] | None = None      # created by the first window with bounds

    def add(self, sid: str, start: int, logp: np.ndarray, spread: np.ndarray, bounds: np.ndarray | None = None) -> None:
        """logp [L, K], spread [L] and optionally bounds [L, 2] (absolute start and end, seconds) of
        the window starting at `start` (the part inside the stream is used)."""
        T = self.lengths[sid]
        n = min(len(logp), T - start)
        w = self.w[:n]
        self.logp[sid][start:start + n] += w[:, None] * (np.exp(logp[:n]) if self.merge_kind == "prob" else logp[:n])
        self.spread[sid][start:start + n] += w * spread[:n]
        self.wsum[sid][start:start + n] += w
        if bounds is not None:
            if self.bounds is None:
                self.bounds = {s: np.zeros((L, 2)) for s, L in self.lengths.items()}
            self.bounds[sid][start:start + n] += w[:, None] * bounds[:n]

    def merge(self, other: "StreamAccumulator") -> None:
        for sid in self.lengths:
            self.logp[sid] += other.logp[sid]
            self.spread[sid] += other.spread[sid]
            self.wsum[sid] += other.wsum[sid]
            if other.bounds is not None:
                if self.bounds is None:
                    self.bounds = {s: np.zeros((L, 2)) for s, L in self.lengths.items()}
                self.bounds[sid] += other.bounds[sid]

    def result(self, sid: str) -> tuple[np.ndarray, np.ndarray]:
        """Log-probabilities [T, K] (renormalized) and disagreement [T] of a stream."""
        wsum = self.wsum[sid]
        if (wsum <= 0).any():
            raise ValueError(f"{sid}: {(wsum <= 0).sum()} timesteps are not covered by any window")
        mean = self.logp[sid] / wsum[:, None]
        if self.merge_kind == "prob":
            mean = np.log(np.clip(mean, 1e-12, None))
        logp = torch.log_softmax(torch.from_numpy(mean), dim=-1).numpy()
        return logp, self.spread[sid] / wsum

    def result_bounds(self, sid: str) -> np.ndarray | None:
        """The regressed [start, end] in stream seconds of every timestep [T, 2], or None."""
        return None if self.bounds is None else self.bounds[sid] / self.wsum[sid][:, None]


# Allowed transitions of the fall grammar (row: from, column: to)
ALLOWED = np.zeros((4, 4), dtype=bool)
ALLOWED[NONE, [NONE, BEGIN]] = True
ALLOWED[BEGIN, [BEGIN, MIDDLE, END]] = True
ALLOWED[MIDDLE, [MIDDLE, END]] = True
ALLOWED[END, [END, NONE]] = True


def viterbi(logp: np.ndarray, switch_penalty: float = 2.0) -> np.ndarray:
    """Most likely state path [T] under the fall grammar (4 states) or none <-> fall (2 states);
    every state change costs `switch_penalty` nats."""
    T, K = logp.shape
    assert K in (2, 4), f"the decoder knows 2 or 4 states, not {K}"
    allowed = ALLOWED if K == 4 else np.ones((2, 2), dtype=bool)
    starts = [NONE, BEGIN]                                  # BEGIN is the fall state when K = 2
    trans = np.where(allowed, 0.0, -np.inf)
    trans[~np.eye(K, dtype=bool) & allowed] = -switch_penalty
    score = np.full(K, -np.inf)
    score[starts] = logp[0, starts]
    back = np.zeros((T, K), dtype=np.int8)
    for t in range(1, T):
        cand = score[:, None] + trans                       # [from, to]
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + logp[t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(score.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [a, b) runs of True."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def decode_runs(logp: np.ndarray, rate_hz: float = 50.0, kind: str = "viterbi", switch_penalty: float = 2.0,
                threshold: float = 0.5, merge_gap_s: float = 0.3) -> list[tuple[int, int]]:
    """Half-open timestep runs of decoded falls. `kind`: "viterbi" (the grammar above) or
    "threshold" (P(fall) > threshold, runs closer than merge_gap_s merged)."""
    if kind == "viterbi":
        return _runs(viterbi(logp, switch_penalty) != NONE)
    if kind == "threshold":
        merged = []
        for a, b in _runs(1 - np.exp(logp[:, NONE]) > threshold):
            if merged and (a - merged[-1][1]) / rate_hz <= merge_gap_s:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        return merged
    raise ValueError(f"unknown decoder {kind!r}")


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, cum[-1] / 2)])


def runs_to_events(runs: list[tuple[int, int]], logp: np.ndarray, spread: np.ndarray, rate_hz: float = 50.0,
                   min_duration_s: float = 0.3, spread_penalty: float = 0.0,
                   bounds: np.ndarray | None = None) -> list[dict]:
    """
    Events {t_start, t_end, score, t_alarm, confidence, disagreement}: score = mean P(fall) over
    the event, minus spread_penalty x the mean disagreement of the frames. With bounds (the
    regressed [start, end] of every timestep, stream seconds), each run's start and end are the
    P(fall)-weighted medians of its timesteps' regressed start and end.
    """
    p_fall = 1 - np.exp(logp[:, NONE])
    events = []
    for a, b in runs:
        t_start, t_end = a / rate_hz, b / rate_hz
        if bounds is not None:
            w = p_fall[a:b] + 1e-6
            start, end = weighted_median(bounds[a:b, 0], w), weighted_median(bounds[a:b, 1], w)
            if end > start:
                t_start, t_end = max(start, 0.0), min(end, len(logp) / rate_hz)
        if t_end - t_start < min_duration_s:
            continue
        confidence = float(p_fall[a:b].mean())
        disagreement = float(spread[a:b].mean())
        events.append({"t_start": t_start, "t_end": t_end, "t_alarm": t_start,
                       "score": confidence - spread_penalty * disagreement,
                       "confidence": confidence, "disagreement": disagreement})
    return events


def interval_iou(start: float, end: float, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    inter = np.clip(np.minimum(end, ends) - np.maximum(start, starts), 0, None)
    return inter / np.maximum(np.maximum(end, ends) - np.minimum(start, starts), 1e-9)


def propose_events(logp: np.ndarray, spread: np.ndarray, bounds: np.ndarray, rate_hz: float = 50.0,
                   proposal_threshold: float = 0.5, proposal_stride: int = 5, nms_iou: float = 0.3,
                   min_duration_s: float = 0.3) -> list[dict]:
    """
    ActionFormer-style decoding: every `proposal_stride`-th timestep with P(fall) > threshold
    proposes its regressed [start, end]; a proposal's score is the mean P(fall) over it. Greedy NMS
    (highest score first, drop a proposal whose tIoU with a kept one exceeds nms_iou) and the same
    threshold on the score leave one event per fall. Hard NMS, not Soft-NMS: event F1 counts every
    surviving duplicate as a false alarm.
    """
    p_fall = 1 - np.exp(logp[:, NONE])
    T = len(p_fall)
    t = np.flatnonzero(p_fall > proposal_threshold)
    t = t[t % proposal_stride == 0]
    starts, ends = np.clip(bounds[t, 0], 0, T / rate_hz), np.clip(bounds[t, 1], 0, T / rate_hz)
    ok = ends - starts >= min_duration_s
    starts, ends = starts[ok], ends[ok]
    a = np.clip(np.round(starts * rate_hz).astype(int), 0, T - 1)
    b = np.maximum(np.clip(np.round(ends * rate_hz).astype(int), 0, T), a + 1)
    cum_p = np.concatenate([[0.0], np.cumsum(p_fall)])
    cum_s = np.concatenate([[0.0], np.cumsum(spread)])
    scores = (cum_p[b] - cum_p[a]) / (b - a)
    events, kept_s, kept_e = [], np.empty(0), np.empty(0)
    for i in np.argsort(-scores, kind="stable"):
        if scores[i] <= proposal_threshold:
            break
        if len(kept_s) and interval_iou(starts[i], ends[i], kept_s, kept_e).max() > nms_iou:
            continue
        kept_s, kept_e = np.append(kept_s, starts[i]), np.append(kept_e, ends[i])
        score = float(scores[i])
        events.append({"t_start": float(starts[i]), "t_end": float(ends[i]), "t_alarm": float(starts[i]),
                       "score": score, "confidence": score,
                       "disagreement": float((cum_s[b[i]] - cum_s[a[i]]) / (b[i] - a[i]))})
    return sorted(events, key=lambda e: e["t_start"])


def decode_events(logp: np.ndarray, spread: np.ndarray, rate_hz: float = 50.0, kind: str = "viterbi",
                  switch_penalty: float = 2.0, threshold: float = 0.5, merge_gap_s: float = 0.3,
                  min_duration_s: float = 0.3, spread_penalty: float = 0.0, bounds: np.ndarray | None = None,
                  refine: bool = False, proposal_threshold: float = 0.5, proposal_stride: int = 5,
                  nms_iou: float = 0.3) -> list[dict]:
    """
    Events of one stream from its state log-probabilities: decode_runs, then runs_to_events
    (`refine`: with the regressed boundaries); or `kind: "proposals"`, which needs `bounds`
    (propose_events).
    """
    if kind == "proposals":
        if bounds is None:
            raise ValueError('decoder kind "proposals" needs a model with boundary regression')
        return propose_events(logp, spread, bounds, rate_hz, proposal_threshold, proposal_stride, nms_iou, min_duration_s)
    if refine and bounds is None:
        raise ValueError("decoder refine needs a model with boundary regression")
    runs = decode_runs(logp, rate_hz, kind, switch_penalty, threshold, merge_gap_s)
    return runs_to_events(runs, logp, spread, rate_hz, min_duration_s, spread_penalty, bounds if refine else None)


def state_log_probs(mean_logits: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(mean_logits.float(), dim=-1)


# ------------------------------------------------------------------ the recording head's units

class UnitAccumulator:
    """The recording head's output per unit of each stream, kept from the window the unit is most
    central in (the one with the largest distance from the unit to the nearer window edge): the
    fall logit and the start / end log-probabilities over the unit's domain, in stream timesteps."""
    def __init__(self):
        self.best: dict[tuple[str, int], tuple] = {}

    def add(self, sid: str, unit: int, centrality: int, logit: float, positions: np.ndarray,
            logp_start: np.ndarray, logp_end: np.ndarray) -> None:
        key = (sid, int(unit))
        if key not in self.best or centrality > self.best[key][0]:
            self.best[key] = (int(centrality), float(logit), positions.astype(np.int64),
                              logp_start.astype(np.float32), logp_end.astype(np.float32))

    def merge(self, other: "UnitAccumulator") -> None:
        for (sid, unit), v in other.best.items():
            self.add(sid, unit, *v)

    def units(self, sid: str) -> dict[int, tuple]:
        return {u: v for (s, u), v in self.best.items() if s == sid}


def mbr_span(positions: np.ndarray, logp_start: np.ndarray, logp_end: np.ndarray, k: int = 16,
             min_len: int = 10, max_len: int = 300, thetas=(0.1, 0.3, 0.5), tie_iou: float = 0.0) -> tuple[int, int]:
    """
    The span that maximizes the expected challenge score under the head's start and end
    distributions (taken independent): sum over theta of P(tIoU(span, truth) >= theta), with
    hypotheses and truths both the pairs of the top-k starts and top-k ends (min_len <= end - start
    <= max_len timesteps). Falls back to the most likely valid pair, then to argmax start + 1 s.
    tie_iou > 0 adds that times the expected tIoU, so among spans of equal expected score the
    closest to the distribution wins (a peaked distribution otherwise ties every span within tIoU .5).
    """
    ks = np.argsort(-logp_start)[:k]
    ke = np.argsort(-logp_end)[:k]
    s, e = positions[ks][:, None], positions[ke][None, :]
    ls, le = logp_start[ks][:, None], logp_end[ke][None, :]
    ok = (e - s >= min_len) & (e - s <= max_len)
    if not ok.any():
        ok = e > s
        if not ok.any():
            a = int(positions[int(np.argmax(logp_start))])
            return a, a + 50
    hs, he = np.broadcast_to(s, ok.shape)[ok].astype(np.float64), np.broadcast_to(e, ok.shape)[ok].astype(np.float64)
    lw = (ls + le)[ok]
    w = np.exp(lw - lw.max())
    w /= w.sum()
    inter = np.clip(np.minimum(he[:, None], he[None, :]) - np.maximum(hs[:, None], hs[None, :]), 0, None)
    union = (he - hs)[:, None] + (he - hs)[None, :] - inter
    iou = inter / np.maximum(union, 1e-9)                       # [hypothesis, truth]
    gain = (sum((iou >= th).astype(np.float64) for th in thetas) + tie_iou * iou) @ w
    best = int(np.argmax(gain))
    return int(hs[best]), int(he[best])


def decode_units(acc: UnitAccumulator, sid: str, T: int, threshold: float = 0.5, rate_hz: float = 50.0,
                 mbr: bool = True) -> list[dict]:
    """One event per unit whose P(fall) exceeds `threshold`, its span by mbr_span (or the argmax
    start and the argmax end after it), score P(fall)."""
    events = []
    for unit, (_, logit, pos, ls, le) in sorted(acc.units(sid).items()):
        p = 1.0 / (1.0 + np.exp(-logit))
        if p <= threshold:
            continue
        if mbr:
            a, b = mbr_span(pos, ls, le)
        else:
            i = int(np.argmax(ls))
            a = int(pos[i])
            b = int(pos[i + 1 + int(np.argmax(le[i + 1:]))]) if i + 1 < len(pos) else a + 50
        a, b = max(a, 0), min(b, T)
        if b <= a:
            continue
        events.append({"t_start": a / rate_hz, "t_end": b / rate_hz, "t_alarm": a / rate_hz, "score": p,
                       "confidence": p, "disagreement": 0.0, "unit": unit})
    return events

"""
The stream level of the join-free model (docs/falljepa_join_free.md), shared by training (on chunks)
and inference (on whole streams).

    proposals       every window's queries with P(fall) > threshold: logit, most likely start and end
                    (the end after the start), distance to the nearer window edge, hidden state, and
                    the top start / end log-probabilities (for the non-learned merge)
    group_proposals greedy, strongest first: a proposal joins the first group whose strongest member's
                    span overlaps it at tIoU >= `iou` (or, with `gap`, that lies within `gap` steps of
                    the group's extent), else starts a group
    group_region    the group's members and reference span +- `margin` steps, on the token grid, at most
                    `max_tokens` tokens (centred on the reference span when longer)
    group_item      one GroupRefiner input (set_heads.pad_groups) from a region's tokens and signal
    TokenAccumulator / ProposalAccumulator   the per-stream sums of window outputs at inference
    decode_queries  stage 1 alone: per group, the most central member's P and MBR span
    nms             drops events that overlap a higher-scored one at tIoU >= iou (duplicates from two
                    groups whose margins reach the same fall), or whose centre lies within `centre`
                    steps of its centre
"""
import numpy as np
import torch

from .aggregate import mbr_span, weighted_median
from .set_heads import NUM_GROUP_STATS, NUM_MEMBER_FEATURES

TOP_K = 16
TIE_IOU = 1e-3          # mbr_span's tie-break by expected tIoU
MIN_SPAN = 10              # 0.2 s


def tiou(a0: float, b0: float, a1: float, b1: float) -> float:
    inter = max(0.0, min(b0, b1) - max(a0, a1))
    union = max(b0, b1) - min(a0, a1)
    return inter / union if union > 0 else 0.0


def most_likely_span(logp: np.ndarray) -> tuple[int, int]:
    """logp [L, 2] start / end log-probabilities -> the argmax start and the argmax end after it."""
    s = int(np.argmax(logp[:, 0]))
    if s + 1 >= len(logp):
        return s, len(logp)
    return s, s + 1 + int(np.argmax(logp[s + 1:, 1]))


def proposals(logit: np.ndarray, logp: np.ndarray, hidden: np.ndarray | None, offset: int, window: int,
              threshold: float, valid: np.ndarray | None = None) -> list[dict]:
    """The proposals of one window: logit [K], logp [K, W, 2] (log-softmax over the window's valid
    steps), hidden [K, d] (None in training, where the caller keeps the hidden states with their
    gradients: `query` says which), `offset` the window's first step in the stream (chunk)."""
    out = []
    p = 1.0 / (1.0 + np.exp(-logit))
    for k in np.flatnonzero(p > threshold):
        s, e = most_likely_span(logp[k])
        lp = logp[k] if valid is None else np.where(valid[:, None], logp[k], -np.inf)
        ks, ke = np.argsort(-lp[:, 0])[:TOP_K], np.argsort(-lp[:, 1])[:TOP_K]
        out.append({"logit": float(logit[k]), "start": offset + s, "end": offset + e,
                    "centrality": float(min(s, window - e)), "query": int(k),
                    "hidden": None if hidden is None else hidden[k].astype(np.float32),
                    "top_start": (offset + ks, lp[ks, 0].astype(np.float32)),
                    "top_end": (offset + ke, lp[ke, 1].astype(np.float32))})
    return out


def group_proposals(starts: np.ndarray, ends: np.ndarray, scores: np.ndarray, iou: float = 0.2,
                    gap: int | None = None) -> list[np.ndarray]:
    """Member indices of each group, strongest member first. With `gap`, a proposal also joins a group
    whose extent (the union of its members' spans) it overlaps or comes within `gap` steps of, so
    neighbouring motion bursts of one fall meet in one group."""
    reps: list[tuple[float, float]] = []
    extent: list[list[float]] = []
    groups: list[list[int]] = []
    for i in np.argsort(-scores, kind="stable"):
        for g, (a, b) in enumerate(reps):
            near = gap is not None and max(starts[i], extent[g][0]) - min(ends[i], extent[g][1]) <= gap
            if near or tiou(starts[i], ends[i], a, b) >= iou:
                groups[g].append(int(i))
                extent[g] = [min(extent[g][0], float(starts[i])), max(extent[g][1], float(ends[i]))]
                break
        else:
            reps.append((float(starts[i]), float(ends[i])))
            extent.append([float(starts[i]), float(ends[i])])
            groups.append([int(i)])
    return [np.array(g, dtype=np.int64) for g in groups]


def reference_span(starts: np.ndarray, ends: np.ndarray, p: np.ndarray) -> tuple[int, int]:
    """The P-weighted median start and end of a group's members."""
    w = np.maximum(p, 1e-3)
    a = int(round(weighted_median(starts.astype(np.float64), w)))
    b = int(round(weighted_median(ends.astype(np.float64), w)))
    return a, max(b, a + MIN_SPAN)


def group_region(starts: np.ndarray, ends: np.ndarray, span: tuple[int, int], margin: int, stride: int,
                 max_tokens: int, n_tokens: int) -> tuple[int, int]:
    """[j0, j1) tokens: the members and the reference span +- margin steps, clipped to [0, n_tokens)."""
    lo = min(int(starts.min()) if len(starts) else span[0], span[0]) - margin
    hi = max(int(ends.max()) if len(ends) else span[1], span[1]) + margin
    j0, j1 = max(lo // stride, 0), min(-(-hi // stride), n_tokens)
    if j1 - j0 > max_tokens:
        c = (span[0] + span[1]) / 2 / stride
        j0 = int(np.clip(round(c - max_tokens / 2), 0, max(n_tokens - max_tokens, 0)))
        j1 = min(j0 + max_tokens, n_tokens)
    return j0, j1


def member_features(members: list[dict], lo: int, length: int, window: int) -> np.ndarray:
    rows = [(m["logit"], (m["start"] - lo) / length, (m["end"] - lo) / length, (m["end"] - m["start"]) / 250.0,
             m["centrality"] / (window / 2)) for m in members]
    return np.array(rows, dtype=np.float32).reshape(-1, NUM_MEMBER_FEATURES)


def group_stats(p: np.ndarray, span: tuple[int, int]) -> np.ndarray:
    return np.array([np.log1p(len(p)), p.max() if len(p) else 0.0, p.mean() if len(p) else 0.0,
                     (span[1] - span[0]) / 250.0], dtype=np.float32).reshape(NUM_GROUP_STATS)


def group_item(tokens: torch.Tensor, signal: torch.Tensor, step_valid: torch.Tensor, span_local: tuple[int, int],
               members: torch.Tensor, features: np.ndarray, stats: np.ndarray) -> dict:
    """One set_heads.pad_groups item; tokens [R, d], signal [3 groups, H, R s], step_valid [R s]."""
    dev = tokens.device
    return {"tokens": tokens, "signal": signal, "step_valid": step_valid.bool(),
            "span": torch.tensor(span_local, dtype=torch.long, device=dev), "members": members,
            "member_features": torch.from_numpy(features).to(dev), "stats": torch.from_numpy(stats).to(dev)}


class TokenAccumulator:
    """Per stream, the centre-weighted (triangular) mean of the windows' tokens on the token grid:
    token j owns stream steps [j s, (j + 1) s)."""
    def __init__(self, lengths: dict[str, int], stride: int, d_model: int):
        self.lengths, self.s = lengths, stride
        self.sum = {sid: np.zeros((-(-T // stride), d_model), np.float32) for sid, T in lengths.items()}
        self.wsum = {sid: np.zeros(-(-T // stride), np.float32) for sid, T in lengths.items()}

    def add(self, sid: str, start: int, tokens: np.ndarray, valid: np.ndarray) -> None:
        Fn = len(tokens)
        j0 = start // self.s
        n = max(min(Fn, len(self.wsum[sid]) - j0), 0)
        w = (np.minimum(np.arange(Fn) + 1, Fn - np.arange(Fn)) * valid).astype(np.float32)[:n]
        self.sum[sid][j0:j0 + n] += w[:, None] * tokens[:n]
        self.wsum[sid][j0:j0 + n] += w

    def merge(self, other: "TokenAccumulator") -> None:
        for sid in self.sum:
            self.sum[sid] += other.sum[sid]
            self.wsum[sid] += other.wsum[sid]

    def result(self, sid: str) -> tuple[np.ndarray, np.ndarray]:
        w = self.wsum[sid]
        return self.sum[sid] / np.maximum(w, 1e-6)[:, None], w > 0


class ProposalAccumulator:
    def __init__(self):
        self.rows: dict[str, list[dict]] = {}

    def add(self, sid: str, rows: list[dict]) -> None:
        self.rows.setdefault(sid, []).extend(rows)

    def merge(self, other: "ProposalAccumulator") -> None:
        for sid, rows in other.rows.items():
            self.add(sid, rows)

    def get(self, sid: str) -> list[dict]:
        return self.rows.get(sid, [])


def stream_groups(rows: list[dict], iou: float, gap: int | None = None) -> list[tuple[list[dict], tuple[int, int]]]:
    """The groups of a stream's proposals with their reference spans."""
    if not rows:
        return []
    starts = np.array([r["start"] for r in rows])
    ends = np.array([r["end"] for r in rows])
    p = 1.0 / (1.0 + np.exp(-np.array([r["logit"] for r in rows])))
    out = []
    for idx in group_proposals(starts, ends, p, iou, gap):
        out.append(([rows[i] for i in idx], reference_span(starts[idx], ends[idx], p[idx])))
    return out


def nms(events: list[dict], iou: float = 0.5, centre_s: float | None = None) -> list[dict]:
    kept: list[dict] = []
    mid = lambda e: (e["t_start"] + e["t_end"]) / 2  # noqa: E731
    for e in sorted(events, key=lambda e: -e["score"]):
        if all(tiou(e["t_start"], e["t_end"], k["t_start"], k["t_end"]) < iou
               and (centre_s is None or abs(mid(e) - mid(k)) >= centre_s) for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e["t_start"])


def event(a: int, b: int, p: float, T: int, rate_hz: float = 50.0) -> dict | None:
    a, b = max(int(a), 0), min(int(b), T)
    if b <= a:
        return None
    return {"t_start": a / rate_hz, "t_end": b / rate_hz, "t_alarm": a / rate_hz, "score": float(p),
            "confidence": float(p), "disagreement": 0.0}


def decode_queries(rows: list[dict], T: int, threshold: float = 0.5, iou: float = 0.2, nms_iou: float = 0.5,
                   gap: int | None = None, centre_s: float | None = None) -> list[dict]:
    """Stage 1 with a non-learned merge: per group, the member most central in its window gives P and
    the MBR span of its top starts and ends."""
    events = []
    for members, _ in stream_groups(rows, iou, gap):
        best = max(members, key=lambda m: m["centrality"])
        p = 1.0 / (1.0 + np.exp(-best["logit"]))
        if p <= threshold:
            continue
        ps, ls = best["top_start"]
        pe, le = best["top_end"]
        pos = np.union1d(ps, pe)
        lps = np.full(len(pos), -np.inf)
        lpe = np.full(len(pos), -np.inf)
        lps[np.searchsorted(pos, ps)] = ls
        lpe[np.searchsorted(pos, pe)] = le
        a, b = mbr_span(pos, lps, lpe, tie_iou=TIE_IOU)
        e = event(a, b, p, T)
        if e is not None:
            events.append(e)
    return nms(events, nms_iou, centre_s)

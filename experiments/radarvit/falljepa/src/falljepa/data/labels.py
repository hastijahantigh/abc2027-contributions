"""
Per-timestep supervision for one stream, from the official `streams_events.csv`.

Every timestep (one 20 ms CIR sample) gets one of four states and a loss weight:

    NONE    no fall
    BEGIN   the first `boundary_s` of a fall (capped at a third of the fall)
    MIDDLE  the rest of the fall
    END     the last `boundary_s` of a fall (capped at a third of the fall)

With `num_states = 2` (the binary ablation) there are no bands: NONE or FALL (= 1).

`stream_bounds` gives the boundary-regression targets (ActionFormer): for every timestep inside a
fall, its distance in seconds to the fall's start and end.

A fall covers the half-open timesteps [round(t_start * rate), round(t_end * rate)). Falls the
scorer ignores (`ignore`, `quality = low`, `needs_review`) keep their states but get
`flagged_weight` instead of 1: the scorer neither rewards nor penalizes predictions on them, so
training on them as down-weighted falls costs nothing at test time and doubles the falls the
model sees (173 of the 359 training falls are flagged). `flagged_margin_s` widens that weight to
the no-fall timesteps next to a flagged fall.

`boundary_mask_s` > 0 gives weight 0 to the timesteps within that distance of every fall's start
and end (Amir's loss mask: +-0.2 s; the labeller, not the model, places the boundaries). With
flagged_weight 0 and flagged_margin_s 2 it is Amir's mask of flagged falls (+-2 s).
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NONE, BEGIN, MIDDLE, END = 0, 1, 2, 3
FALL = 1                                    # the fall state of the binary labels (num_states 2)
STATE_NAMES = ("none", "begin", "middle", "end")


@dataclass(frozen=True)
class Event:
    stream_id: str
    t_start: float
    t_end: float
    flagged: bool


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_events(path: str | Path) -> dict[str, list[Event]]:
    """Events per stream id; `flagged` is the scorer's `--ignore-flagged` rule (plus `ignore`)."""
    events: dict[str, list[Event]] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            flagged = (_as_bool(row.get("ignore", "")) or _as_bool(row.get("needs_review", ""))
                       or str(row.get("quality", "")).strip().lower() == "low")
            event = Event(row["stream_id"], float(row["t_start"]), float(row["t_end"]), flagged)
            events.setdefault(event.stream_id, []).append(event)
    return events


def _fall_steps(event: Event, num_steps: int, rate_hz: float) -> tuple[int, int]:
    return max(int(round(event.t_start * rate_hz)), 0), min(int(round(event.t_end * rate_hz)), num_steps)


def stream_targets(events: list[Event], num_steps: int, rate_hz: float = 50.0, boundary_s: float = 0.3,
                   flagged_weight: float = 0.5, flagged_margin_s: float = 0.0,
                   num_states: int = 4, boundary_mask_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """States [T] (int64) and loss weights [T] (float32) of one stream; num_states 2: NONE / FALL."""
    assert num_states in (2, 4), f"num_states must be 2 or 4, not {num_states}"
    state = np.zeros(num_steps, dtype=np.int64)
    weight = np.ones(num_steps, dtype=np.float32)
    boundary = int(round(boundary_s * rate_hz))
    margin = int(round(flagged_margin_s * rate_hz))
    mask = int(round(boundary_mask_s * rate_hz))
    for event in sorted(events, key=lambda e: e.t_start):
        a, b = _fall_steps(event, num_steps, rate_hz)
        if b <= a:
            continue
        if num_states == 2:
            state[a:b] = FALL
        else:
            n = max(min(boundary, (b - a) // 3), 1)
            state[a:b] = MIDDLE
            state[a:a + n] = BEGIN
            state[b - n:b] = END
        if event.flagged:
            lo, hi = max(a - margin, 0), min(b + margin, num_steps)
            weight[lo:hi] = np.minimum(weight[lo:hi], flagged_weight)
        if mask > 0:
            for edge in (a, b):
                weight[max(edge - mask, 0):min(edge + mask, num_steps)] = 0.0
    return state, weight


def stream_bounds(events: list[Event], num_steps: int, rate_hz: float = 50.0,
                  center: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Boundary-regression targets of one stream: distances [T, 2] (float32, seconds) from each
    timestep t to the start and the end of its fall, so [t - d_start, t + d_end] = [a, b) in
    timesteps, and the timesteps [T] (bool) that get a target. `center` < 1 is ActionFormer's
    center sampling: only the central `center` fraction of each fall is regressed.
    """
    bounds = np.zeros((num_steps, 2), dtype=np.float32)
    inside = np.zeros(num_steps, dtype=bool)
    for event in events:
        a, b = _fall_steps(event, num_steps, rate_hz)
        if b <= a:
            continue
        t = np.arange(a, b)
        bounds[a:b, 0] = (t - a) / rate_hz
        bounds[a:b, 1] = (b - t) / rate_hz
        inside[a:b] = np.abs(t + 0.5 - (a + b) / 2) <= max(center * (b - a) / 2, 0.5)
    return bounds, inside

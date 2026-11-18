"""
Recording-grouped cross-validation on the training streams, with the friend's 6 folds.

The windows of this model cross the joins between source recordings, so a fold cannot be a set
of windows. Instead every timestep of a training stream belongs to the source recording(s) that
cover it (`streams_composition.csv`; the 0.2 s crossfade at a join belongs to both sides), and
for held-out fold k:

    fit       normalizer statistics, the state losses and the self-supervised losses (JEPA,
              SIGReg) only use timesteps none of whose recordings is in fold k
    score     predictions and ground-truth events are scored only inside fold k's timesteps, with
              the official evaluator (false alarms per hour over fold k's duration)

The input itself is not cut: a training window still shows fold k's signal around the timesteps
it learns from, as unlabeled context (it never sees fold k's labels, and fold k's frames add
nothing to the self-supervised losses). This keeps joins and context as in the test streams; the
remaining optimism is that exposure to fold k's raw signal.

Fold assignment: the friend's `source_fold_assignment.csv` (Stage04: 6 folds, each one fall
candidate and one ADL candidate; candidate = presumed subject, `Test_P<d>_<n>` / file suffix),
keyed by sha256("abc2027-stage04:" + path)[:16], or any CSV with `path,fold` columns.
"""
import csv
import hashlib
from pathlib import Path

import numpy as np

RATE_HZ = 50.0


def _source_key(path: str) -> str:
    return hashlib.sha256(("abc2027-stage04:" + path).encode()).hexdigest()[:16]


def load_assignment(path: str | Path) -> dict[str, int]:
    """{key: fold}, where key is the hashed source id or the plain recording path."""
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    key = "source_id" if "source_id" in rows[0] else "path"
    return {row[key]: int(row["fold"]) for row in rows}


def stream_fold_bits(composition: str | Path, assignment: dict[str, int], lengths: dict[str, int]) -> dict[str, np.ndarray]:
    """
    {stream_id: int16 [T]}: bit f is set when a recording of fold f covers the timestep. Only the
    streams in `lengths` are mapped; an unassigned recording (or an uncovered timestep) is an error.
    """
    bits = {sid: np.zeros(T, dtype=np.int16) for sid, T in lengths.items()}
    with Path(composition).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = row["stream_id"]
            if sid not in bits:
                continue
            path = row["path"]
            fold = assignment.get(path, assignment.get(_source_key(path)))
            if fold is None:
                raise KeyError(f"{sid}: recording {path!r} has no fold in the assignment")
            a = max(int(round(float(row["t_start"]) * RATE_HZ)), 0)
            b = min(int(round(float(row["t_end"]) * RATE_HZ)), lengths[sid])
            bits[sid][a:b] |= np.int16(1 << fold)
    for sid, b in bits.items():
        if (b == 0).any():
            raise ValueError(f"{sid}: {(b == 0).sum()} timesteps are covered by no recording")
    return bits


def fit_and_score_masks(bits: dict[str, np.ndarray], fold: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """(fit, score) bool masks per stream for held-out `fold`."""
    held = {sid: (b & (1 << fold)) != 0 for sid, b in bits.items()}
    return {sid: ~h for sid, h in held.items()}, held

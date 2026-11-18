"""
Amir's source-balanced batch sampler with hard negatives (his stage31_sampler, section 5.4).

Training windows lie on a fixed grid (complete windows only). Each window belongs to the source
recording that covers its centre timestep, and to that recording's subject group (the Stage04
fold of the recording). A batch has `fall_slots` "fall" slots and `batch_size - fall_slots`
"no-fall" slots in random order:

    fall slot       rotates over the subject groups, takes a source not used yet in this batch,
                    then a random window of that source holding an eligible fall patch
    no-fall slot    the same with an eligible no-fall patch
    hard slot       `hard_slots` of the no-fall slots draw one of the hard-negative sources ("Autre":
                    same room and mat as the falls, no fall), which appear only through these slots
    join slot       `join_slots` more no-fall slots draw a window that starts an ADL clip: its centre
                    lies in an activity recording, at most `join_s` after the join into it (where
                    most of our remaining false alarms are). These windows stay in the no-fall pool too

A patch is 10 aligned timesteps (Amir's 0.2 s token); it is eligible for a class when all of its
timesteps have that label and a non-zero loss weight. The schedule is seeded by (seed, epoch); an
epoch has ceil(windows / batch_size) batches.

`split` > 1 yields every drawn batch in that many consecutive pieces (batch_size must divide), so
gradient accumulation over `split` pieces trains on exactly the batches of the unsplit sampler with
less GPU memory per piece.
"""
import csv
import math
from pathlib import Path

import numpy as np
from torch.utils.data import Sampler

from .folds import _source_key, load_assignment

PATCH = 10
RATE_HZ = 50.0


def eligible_patches(state: np.ndarray, weight: np.ndarray, starts: list[int], window: int) -> tuple[np.ndarray, np.ndarray]:
    """For windows [s, s + window) of one stream: (has an eligible fall patch, has an eligible
    no-fall patch), with patches aligned to the window start."""
    fall, none = np.zeros(len(starts), bool), np.zeros(len(starts), bool)
    ok = weight > 0
    for i, s in enumerate(starts):
        n = window // PATCH
        st = state[s:s + n * PATCH].reshape(n, PATCH)
        w = ok[s:s + n * PATCH].reshape(n, PATCH)
        fall[i] = bool(((st > 0) & w).all(1).any())
        none[i] = bool(((st == 0) & w).all(1).any())
    return fall, none


def window_owners(composition: str | Path, stream_id: str, centres: list[int]) -> list[dict]:
    """The composition row (path, label, kind) of the recording covering each centre timestep
    (the later one inside a crossfade)."""
    with Path(composition).open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["stream_id"] == stream_id]
    rows.sort(key=lambda r: float(r["t_start"]))
    starts = np.array([float(r["t_start"]) for r in rows])
    out = []
    for c in centres:
        i = int(np.searchsorted(starts, c / RATE_HZ, side="right")) - 1
        out.append(rows[max(i, 0)])
    return out


def source_groups(paths: list[str], assignment_path: str | Path | None) -> list[int]:
    """The subject group (Stage04 fold) of each source path; all 0 without an assignment."""
    if not assignment_path:
        return [0] * len(paths)
    assignment = load_assignment(assignment_path)
    return [assignment.get(p, assignment.get(_source_key(p), -1)) for p in paths]


class BalancedBatchSampler(Sampler):
    def __init__(self, sources: list[str], groups: list[int], hard: list[bool], fall_ok: np.ndarray,
                 none_ok: np.ndarray, batch_size: int = 32, fall_slots: int = 16, hard_slots: int = 4,
                 steps: int | None = None, seed: int = 0, join: list[bool] | None = None, join_slots: int = 0,
                 split: int = 1):
        assert batch_size % split == 0, "split must divide batch_size"
        self.batch_size, self.seed, self.epoch, self.split = batch_size, seed, 0, split
        self.steps = steps or math.ceil(len(sources) / batch_size)
        group_of = {}
        pools = {"fall": {}, "none": {}, "hard": {}, "join": {}}   # class -> source -> window indices
        join = join if join is not None else [False] * len(sources)
        for i, (src, g, h) in enumerate(zip(sources, groups, hard)):
            group_of[src] = g
            if join[i] and none_ok[i] and not h:
                pools["join"].setdefault(src, []).append(i)
            if h:
                if none_ok[i]:
                    pools["hard"].setdefault(src, []).append(i)
                continue
            if fall_ok[i]:
                pools["fall"].setdefault(src, []).append(i)
            if none_ok[i]:
                pools["none"].setdefault(src, []).append(i)
        self.pools = {c: {s: np.array(v) for s, v in p.items()} for c, p in pools.items()}
        self.by_group = {c: {} for c in ("fall", "none")}
        for c in ("fall", "none"):
            for src in sorted(self.pools[c]):
                self.by_group[c].setdefault(group_of[src], []).append(src)
        self.groups = {c: sorted(self.by_group[c]) for c in self.by_group}
        # Classes with no window give their slots to the no-fall class
        self.fall_slots = fall_slots if self.pools["fall"] else 0
        self.hard_slots = min(hard_slots, batch_size - self.fall_slots) if self.pools["hard"] else 0
        self.join_slots = min(join_slots, batch_size - self.fall_slots - self.hard_slots) if self.pools["join"] else 0
        assert self.pools["none"], "no window with an eligible no-fall patch"

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.steps * self.split

    def _pick(self, c: str, used: set, pointer: dict, rng: np.random.Generator) -> int:
        if c in ("hard", "join"):
            sources = sorted(self.pools[c])
            cand = [s for s in sources if s not in used] or sources
        else:
            groups = self.groups[c]
            cand = []
            for _ in range(len(groups)):
                g = groups[pointer[c] % len(groups)]
                pointer[c] += 1
                cand = [s for s in self.by_group[c][g] if s not in used]
                if cand:
                    break
            cand = cand or sorted(self.pools[c])
        src = cand[int(rng.integers(len(cand)))]
        used.add(src)
        windows = self.pools[c][src]
        return int(windows[int(rng.integers(len(windows)))])

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        pointer = {c: int(rng.integers(max(len(self.groups[c]), 1))) for c in ("fall", "none")}
        slots = (["fall"] * self.fall_slots + ["hard"] * self.hard_slots + ["join"] * self.join_slots
                 + ["none"] * (self.batch_size - self.fall_slots - self.hard_slots - self.join_slots))
        for _ in range(self.steps):
            used: set = set()
            batch = [self._pick(c, used, pointer, rng) for c in rng.permutation(slots)]
            n = self.batch_size // self.split
            for k in range(self.split):
                yield batch[k * n:(k + 1) * n]

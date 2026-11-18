"""
Amir's post-processing (teammate docs of 2026-10-25: "gates and post-processing to 98 %" and "exact
current pipeline"), recoded as one chain of switchable steps: from P(fall) per 20 ms step, or from a
model's own events, and the unnormalized stream CIR to fall events.

    decode   runs of P >= theta; runs at most `gap` steps apart are joined, THEN runs shorter than
             `min_run` are dropped; score = max P over the run. Each run is a raw parent.
    refine   F: the official labeller (annotate.py internals, unchanged, with the official raw-frame
             motion) on an analysed span around each candidate: with joins, the source recording
             between the nearest joins >= 1 s from the centre when it is 2-20 s long, else -4/+5 s
             around the centre inside it; without joins, always -4/+5 s. The candidate becomes the
             anchor radar's transition (the radar whose profile changed most), kept with every
             radar's transition and rise for `move`.
    merge    strictly overlapping intervals become one; score = max, parents and analyses = union
    nms      strongest first; drop a candidate whose centre is within `nms_s` of a kept one's, and
             with `nms_iou` one overlapping a kept one at tIoU >= nms_iou (the join-free model's rule)
    gate     broad (G): max over radars of |after - before|^2 (official profiles) on the analysed
             span of the candidate's geometry >= 2.12e-5; spans < 2 s pass.
             local (C1): component-wise median profile of [start - 1 s, start) and [stop, stop + 1.5 s),
             clipped to the source recording (joins) or the stream, max over radars >= 1.934e-5;
             either side < 5 steps falls back to G.
    extend   E3: a candidate shorter than 72 steps grows toward its raw parents' envelope [a, b):
             min(25, (s - a + 1) // 2) on the left, min(25, (b - e + 1) // 2) on the right
    move     M: a candidate whose geometry is exactly the anchor transition of one of its analyses
             (with joins, all its analyses on the same span) moves to an earlier, disjoint transition
             of another radar in that analysis when that transition is usable, rises >= .12 and more
             than the anchor's, is supported by the model and overlaps one raw parent by >= 50 steps.
             Ties: larger rise, larger single-parent overlap, lower radar index. Decided on the
             geometry before `extend`; a move replaces the extended interval.
             Support: with P(fall), a run of >= 10 steps with P >= .3 inside the interval (also
             required of the candidate itself); with model events, a raw parent with P >= .3.
             A model event that was not refined is first analysed on its own span; it is eligible
             when the anchor transition overlaps it (it cannot equal it).
    score    P: keep candidates with score >= min_score (the parents' max, never recomputed)

The steps run in this order; refine, merge and the gate are not rerun after extend / move / score.
`Config()` is Amir's selected chain (k3 + C1 + E3 + M + P), `OLD` his earlier chain (broad gate, no
E3 / M / P). `joins=False` makes every step join-free.

The tuple helpers (threshold_runs, refine, merge, nms, gate, one_per_recording, postprocess) keep
the interface of the earlier module for tools/amir_postprocess.py, tools/score_rs_hybrid.py,
tools/tune_amir_pp.py and the `amir` decoder.
"""
import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import numpy as np

RATE = 50.0
STEP_S = 0.02

# Joins (Amir's section 12)
STEP, GAP, BASELINE, GRID, PEAK_RADIUS, BLEND_HALF, JOIN_THRESHOLD = 10, 6, 501, 50, 50, 5, 1.5
# The analysed span and the refinement
PRE, POST, MIN_WINDOW, MIN_REFINED, MAX_SEGMENT, GUARD = 200, 250, 100, 10, 1000, 50
NMS_RADIUS_S, MIN_PROFILE_CHANGE = 5.9, 2.12e-5
# C1, E3, M, P
LOCAL_BEFORE, LOCAL_AFTER, LOCAL_MIN, LOCAL_CHANGE = 50, 75, 5, 1.933638830415486e-5
SHORT, EXTEND_CAP = 72, 25
RISE_WIDTH, RISE_MIN, SUPPORT_P, SUPPORT_RUN, PARENT_OVERLAP = 75, 0.12, 0.3, 10, 50
MIN_SCORE = 0.5

_ANNOTATE = {}


def load_annotate(release: str | Path):
    """The official annotate module of the release, loaded once per path."""
    path = Path(release) / "src" / "abc2027" / "annotate.py"
    if path not in _ANNOTATE:
        spec = importlib.util.spec_from_file_location(f"official_annotate_{len(_ANNOTATE)}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module                  # its dataclasses look the module up
        spec.loader.exec_module(module)
        _ANNOTATE[path] = module
    return _ANNOTATE[path]


# ---------------------------------------------------------------- configuration

@dataclass(frozen=True)
class Config:
    theta: float = 0.3
    min_run: int = 10
    gap: int = 10
    joins: bool = True
    refine: bool = True
    merge: bool = True
    nms_s: float | None = NMS_RADIUS_S
    nms_iou: float | None = None
    gate: str = "local"                     # none | broad | local
    broad_change: float = MIN_PROFILE_CHANGE
    local_change: float = LOCAL_CHANGE
    extend: bool = True
    move: bool = True
    min_score: float | None = MIN_SCORE
    per_recording: bool = False             # one event per detected recording (RS-era option, needs joins)


OLD = Config(gate="broad", extend=False, move=False, min_score=None)


# ---------------------------------------------------------------- candidates

@dataclass(frozen=True)
class Transition:
    """One radar's transition [start, stop) in stream steps, with the rise its search selected."""
    radar: int
    start: int
    stop: int
    rise: float

    @property
    def usable(self) -> bool:
        return self.stop - self.start >= MIN_REFINED and bool(np.isfinite(self.rise))


@dataclass(frozen=True)
class Analysis:
    """The labeller on one analysed span: the anchor radar and every radar's transition."""
    span: tuple[int, int]
    anchor: int
    transitions: tuple[Transition, ...]

    @property
    def bounds(self) -> tuple[int, int]:
        t = self.transitions[self.anchor]
        return t.start, t.stop


@dataclass(frozen=True)
class Candidate:
    start: int
    stop: int
    score: float
    parents: tuple[tuple[int, int, float], ...]     # raw parents (start, stop, P)
    analyses: tuple[Analysis, ...] = ()             # the refinements that made it
    b0: tuple[int, int] | None = None               # the geometry before extend (move decides on it)

    @property
    def interval(self) -> tuple[int, int]:
        return self.start, self.stop


# ---------------------------------------------------------------- joins

def join_score(cir: np.ndarray) -> np.ndarray:
    """J(t) = min over radars of log(step / local median step) on the unnormalized CIR (3, T, 368)."""
    radars, T, _ = cir.shape
    lo, hi = GAP + STEP, T - GAP - STEP
    if hi <= lo:
        return np.full(T, -np.inf)
    t = np.arange(lo, hi)
    grid = np.arange(0, T, GRID)
    half = BASELINE // 2
    rel = np.zeros((radars, T))
    for r in range(radars):
        c = np.zeros((T + 1, cir.shape[2]))
        np.cumsum(cir[r].astype(np.float64), axis=0, out=c[1:])
        after = (c[t + GAP + STEP] - c[t + GAP]) / STEP
        before = (c[t - GAP] - c[t - GAP - STEP]) / STEP
        step = np.full(T, np.nan)
        step[lo:hi] = ((after - before) ** 2).sum(1)
        step[:lo], step[hi:] = step[lo], step[hi - 1]
        base = np.array([np.median(step[max(0, g - half):g + half + 1]) for g in grid])
        base = np.interp(np.arange(T), grid, base)
        rel[r] = np.log(step + 1e-30) - np.log(base + 1e-30)
    return rel.min(0)


def join_peaks(score: np.ndarray, threshold: float = JOIN_THRESHOLD, radius: int = PEAK_RADIUS) -> np.ndarray:
    cand = np.flatnonzero(score > threshold)
    kept = []
    for i in cand[np.argsort(-score[cand], kind="stable")]:
        if all(abs(int(i) - j) > radius for j in kept):
            kept.append(int(i))
    return np.array(sorted(kept), dtype=int)


def join_span(peaks: np.ndarray, start: int, stop: int, T: int) -> tuple[int, int]:
    """The source recording around a detection: between the nearest joins >= 1 s from its centre."""
    centre = (start + stop) // 2
    left = peaks[peaks <= centre - GUARD]
    right = peaks[peaks >= centre + GUARD]
    lo = left[-1] + BLEND_HALF + 1 if len(left) else 0
    hi = right[0] - BLEND_HALF if len(right) else T
    return max(0, int(lo)), min(T, int(hi))


def analysed_span(T: int, peaks: np.ndarray | None, start: int, stop: int, joins: bool = True) -> tuple[int, int]:
    centre = (start + stop) // 2
    if not joins or peaks is None:                  # the generic -4/+5 s window
        return max(0, centre - PRE), min(T, centre + POST)
    lo, hi = join_span(peaks, start, stop, T)
    if MIN_WINDOW <= hi - lo <= MAX_SEGMENT:        # the whole source recording (2-20 s)
        return lo, hi
    return max(lo, centre - PRE), min(hi, centre + POST)


# ---------------------------------------------------------------- the official rule

def raw_motion_on_grid(x: np.ndarray) -> np.ndarray:
    """Official raw-frame motion of one radar (T, 368): a run of identical grid frames is one raw
    frame, timed at the run's centre; ||frame - previous||^2 interpolated onto the 50 Hz grid."""
    T = len(x)
    change = np.ones(T, bool)
    change[1:] = np.any(x[1:] != x[:-1], axis=1)
    starts = np.flatnonzero(change)
    if len(starts) < 3:
        return np.zeros(T)
    stops = np.r_[starts[1:], T]
    times = (starts + stops - 1) / 2.0 * STEP_S
    delta = np.diff(x[starts].astype(np.float64), axis=0)
    motion = np.einsum("ij,ij->i", delta, delta)
    return np.interp(np.arange(T) * STEP_S, times[1:], motion, left=float(motion[0]), right=float(motion[-1]))


def grid_motion(x: np.ndarray) -> np.ndarray:
    d = np.diff(x.astype(np.float64), axis=0)
    m = np.einsum("ij,ij->i", d, d)
    return np.r_[m[0], m]


def selected_rise(state: np.ndarray, motion: np.ndarray) -> float:
    """state[i + 1.5 s] - state[i] at the index annotate._transition_bounds selects."""
    if len(state) <= RISE_WIDTH + 2:
        return float("nan")
    rise = state[RISE_WIDTH:] - state[:-RISE_WIDTH]
    motion_sum = np.convolve(motion, np.ones(RISE_WIDTH), mode="valid")
    n = min(len(rise), len(motion_sum))
    rise, motion_sum = rise[:n], motion_sum[:n]
    score = rise * np.log1p(motion_sum / (float(np.median(motion_sum)) + 1e-18))
    if np.any(rise >= RISE_MIN):
        score = np.where(rise >= RISE_MIN, score, -np.inf)
    return float(rise[int(np.argmax(score))])


def analyse(A, cir: np.ndarray, raw: bool = True) -> tuple[int, list[tuple[int, int, float]]]:
    """propose_fall's internals on an unnormalized (3, L, 368) span -> the anchor radar and every
    radar's (left, right, rise) in span steps."""
    radars, L, _ = cir.shape
    grid = np.arange(L) * STEP_S
    xs, motions, states, profiles = {}, {}, {}, {}
    for r in range(radars):
        x = cir[r].astype(np.float64)
        profiles[r] = A._profiles(x, grid)
        xs[r], motions[r] = x, (raw_motion_on_grid(x) if raw else grid_motion(x))
        states[r] = A._state(x, *profiles[r])
    cleaned = A._suppress_spikes(motions, states, STEP_S)
    changes, out = {}, []
    for r in range(radars):
        before, after = profiles[r]
        state = A._smooth(A._state(xs[r], before, after), STEP_S, A.SMOOTH_S)
        motion = A._smooth(cleaned[r], STEP_S, A.SMOOTH_S)
        changes[r] = float(np.sum((after - before) ** 2))
        left, right, _ = A._transition_bounds(state, motion)
        out.append((int(left), int(right), selected_rise(state, motion)))
    return max(changes, key=changes.get), out


def official_bounds(A, cir: np.ndarray, raw: bool = True) -> tuple[int, int]:
    """The labeller's bounds (anchor radar) on an unnormalized (3, L, 368) span, in span steps."""
    anchor, trans = analyse(A, cir, raw)
    return trans[anchor][0], trans[anchor][1]


def profile_change(A, cir, lo: int, hi: int) -> float:
    """G's statistic: max over radars of |after - before|^2 with the official profiles on [lo, hi)."""
    span = np.asarray(cir[:, lo:hi])
    grid = np.arange(hi - lo) * STEP_S
    best = 0.0
    for r in range(span.shape[0]):
        before, after = A._profiles(span[r].astype(np.float64), grid)
        best = max(best, float(((after - before) ** 2).sum()))
    return best


# ---------------------------------------------------------------- one stream

class Stream:
    """One stream's unnormalized CIR (an array, or an h5py dataset read per span), its join peaks
    (None without joins) and caches of every per-span computation, shared by all configurations."""
    def __init__(self, A, cir, peaks: np.ndarray | None = None, raw: bool = True):
        self.A, self.cir, self.peaks, self.raw = A, cir, peaks, raw
        self.T = int(cir.shape[1])
        self._analysis: dict[tuple[int, int], Analysis | None] = {}
        self._broad: dict[tuple[int, int], float] = {}
        self._local: dict[tuple, float | None] = {}

    def span(self, start: int, stop: int, joins: bool) -> tuple[int, int]:
        return analysed_span(self.T, self.peaks, start, stop, joins)

    def recording(self, start: int, stop: int, joins: bool) -> tuple[int, int]:
        if not joins or self.peaks is None:
            return 0, self.T
        return join_span(self.peaks, start, stop, self.T)

    def analysis(self, lo: int, hi: int) -> Analysis | None:
        if (lo, hi) not in self._analysis:
            an = None
            if hi - lo >= MIN_WINDOW:
                anchor, trans = analyse(self.A, np.asarray(self.cir[:, lo:hi]), self.raw)
                an = Analysis((lo, hi), anchor, tuple(Transition(r, lo + a, lo + b, rise)
                                                      for r, (a, b, rise) in enumerate(trans)))
            self._analysis[(lo, hi)] = an
        return self._analysis[(lo, hi)]

    def broad_change(self, lo: int, hi: int) -> float:
        if (lo, hi) not in self._broad:
            self._broad[(lo, hi)] = profile_change(self.A, self.cir, lo, hi)
        return self._broad[(lo, hi)]

    def local_change(self, start: int, stop: int, rec: tuple[int, int]) -> float | None:
        """C1's statistic; None when either side has fewer than 5 steps."""
        key = (start, stop, rec)
        if key not in self._local:
            b0, b1 = max(rec[0], start - LOCAL_BEFORE), min(rec[1], start)
            a0, a1 = max(rec[0], stop), min(rec[1], stop + LOCAL_AFTER)
            value = None
            if b1 - b0 >= LOCAL_MIN and a1 - a0 >= LOCAL_MIN:
                before = np.median(np.asarray(self.cir[:, b0:b1], dtype=np.float64), axis=1)
                after = np.median(np.asarray(self.cir[:, a0:a1], dtype=np.float64), axis=1)
                value = float(((after - before) ** 2).sum(-1).max())
            self._local[key] = value
        return self._local[key]


# ---------------------------------------------------------------- the steps

def threshold_runs(p: np.ndarray, theta: float = 0.3, min_frames: int = 10, gap_frames: int = 10) -> list[tuple[int, int, float]]:
    edges = np.diff(np.r_[False, p >= theta, False].astype(np.int8))
    merged = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if merged and a - merged[-1][1] <= gap_frames:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(int(a), int(b), float(p[a:b].max())) for a, b in merged if b - a >= min_frames]


def decode(p: np.ndarray, cfg: Config) -> list[Candidate]:
    return [Candidate(a, b, s, ((a, b, s),)) for a, b, s in threshold_runs(p, cfg.theta, cfg.min_run, cfg.gap)]


def from_events(events: list[tuple[int, int, float]], parents: list[list[tuple[int, int, float]]] | None = None) -> list[Candidate]:
    """Model events (start, stop, score) as candidates; `parents[i]` (start, stop, P) are event i's raw
    parents, the event itself when absent or empty."""
    out = []
    for i, (a, b, s) in enumerate(events):
        rows = parents[i] if parents is not None and len(parents[i]) else [(a, b, s)]
        out.append(Candidate(int(a), int(b), float(s), tuple((int(x), int(y), float(q)) for x, y, q in rows)))
    return out


def refine_step(stream: Stream, cands: list[Candidate], cfg: Config) -> list[Candidate]:
    out = []
    for c in cands:
        an = stream.analysis(*stream.span(c.start, c.stop, cfg.joins))
        if an is None:
            out.append(c)
            continue
        a, b = an.bounds
        if b - a < MIN_REFINED:
            a, b = c.start, c.stop
        out.append(replace(c, start=a, stop=b, analyses=c.analyses + (an,)))
    return out


def merge_step(cands: list[Candidate]) -> list[Candidate]:
    out: list[Candidate] = []
    for c in sorted(cands, key=lambda c: (c.start, c.stop)):
        if out and c.start < out[-1].stop:
            o = out[-1]
            out[-1] = Candidate(o.start, max(o.stop, c.stop), max(o.score, c.score), o.parents + c.parents,
                                o.analyses + c.analyses)
        else:
            out.append(c)
    return out


def tiou(a0: float, b0: float, a1: float, b1: float) -> float:
    inter = max(0.0, min(b0, b1) - max(a0, a1))
    union = max(b0, b1) - min(a0, a1)
    return inter / union if union > 0 else 0.0


def nms_step(cands: list[Candidate], radius_s: float | None, iou: float | None = None) -> list[Candidate]:
    if radius_s is None and iou is None:
        return cands
    kept: list[Candidate] = []
    for c in sorted(cands, key=lambda c: -c.score):
        if all((radius_s is None or abs(c.start + c.stop - k.start - k.stop) / 100 >= radius_s)
               and (iou is None or tiou(c.start, c.stop, k.start, k.stop) < iou) for k in kept):
            kept.append(c)
    return sorted(kept, key=lambda c: (c.start, c.stop))


def gate_step(stream: Stream, cands: list[Candidate], cfg: Config) -> list[Candidate]:
    if cfg.gate == "none":
        return cands

    def broad(c):
        lo, hi = stream.span(c.start, c.stop, cfg.joins)
        return hi - lo < MIN_WINDOW or stream.broad_change(lo, hi) >= cfg.broad_change

    def local(c):
        value = stream.local_change(c.start, c.stop, stream.recording(c.start, c.stop, cfg.joins))
        return broad(c) if value is None else value >= cfg.local_change

    keep = broad if cfg.gate == "broad" else local
    return [c for c in cands if keep(c)]


def extend_step(cands: list[Candidate], T: int) -> list[Candidate]:
    out = []
    for c in cands:
        if c.stop - c.start >= SHORT:
            out.append(c)
            continue
        a = min(p[0] for p in c.parents)
        b = max(p[1] for p in c.parents)
        left = min(EXTEND_CAP, max(0, (c.start - a + 1) // 2))
        right = min(EXTEND_CAP, max(0, (b - c.stop + 1) // 2))
        out.append(replace(c, start=max(0, c.start - left), stop=min(T, c.stop + right)))
    return out


def _supported(p: np.ndarray | None, a: int, b: int) -> bool:
    """A run of >= 10 steps with P >= .3 inside [a, b) (always true without P)."""
    if p is None:
        return True
    run = best = 0
    for v in p[max(a, 0):b] >= SUPPORT_P:
        run = run + 1 if v else 0
        best = max(best, run)
    return best >= SUPPORT_RUN


def _parent_overlap(c: Candidate, t: Transition, model_events: bool) -> int:
    rows = [q for q in c.parents if not model_events or q[2] >= SUPPORT_P]
    return max((max(0, min(t.stop, b) - max(t.start, a)) for a, b, _ in rows), default=0)


def move_step(stream: Stream, cands: list[Candidate], cfg: Config, p: np.ndarray | None) -> list[Candidate]:
    out = []
    for c in cands:
        b0 = c.b0 or c.interval
        an = None
        if c.analyses:
            # Amir: every raw parent analysed, and with joins all on the same span
            if (p is not None and len(c.analyses) < len(c.parents)) or (cfg.joins and len({x.span for x in c.analyses}) > 1):
                out.append(c)
                continue
            an = next((x for x in c.analyses if x.bounds == b0), None)
        elif p is None:                                  # an unrefined model event: its own analysis
            x = stream.analysis(*stream.span(*b0, cfg.joins))
            if x is not None and min(x.bounds[1], b0[1]) > max(x.bounds[0], b0[0]):
                an = x
        if an is None:
            out.append(c)
            continue
        anchor = an.transitions[an.anchor]
        if not anchor.usable or not _supported(p, *b0):
            out.append(c)
            continue
        alts = []
        for t in an.transitions:
            if t.radar == an.anchor or not t.usable or t.stop > b0[0]:
                continue
            if t.rise < RISE_MIN or not t.rise > anchor.rise or not _supported(p, t.start, t.stop):
                continue
            overlap = _parent_overlap(c, t, p is None)
            if overlap >= PARENT_OVERLAP:
                alts.append((-t.rise, -overlap, t.radar, t))
        if alts:
            t = min(alts, key=lambda x: x[:3])[3]
            out.append(replace(c, start=t.start, stop=t.stop))
        else:
            out.append(c)
    return out


def run(stream: Stream, cands: list[Candidate], cfg: Config, p: np.ndarray | None = None) -> list[Candidate]:
    """The chain after decode (or from_events) on one stream. `p` is the P(fall) the candidates were
    decoded from (None for model events)."""
    if cfg.refine:
        cands = refine_step(stream, cands, cfg)
    if cfg.merge:
        cands = merge_step(cands)
    cands = nms_step(cands, cfg.nms_s, cfg.nms_iou)
    cands = gate_step(stream, cands, cfg)
    cands = [replace(c, b0=c.interval) for c in cands]
    if cfg.extend:
        cands = extend_step(cands, stream.T)
    if cfg.move:
        cands = move_step(stream, cands, cfg, p)
    if cfg.min_score is not None:
        cands = [c for c in cands if c.score >= cfg.min_score]
    if cfg.per_recording and stream.peaks is not None:
        best = {}
        for c in cands:
            key = int(np.searchsorted(stream.peaks, (c.start + c.stop) // 2))
            if key not in best or c.score > best[key].score:
                best[key] = c
        cands = sorted(best.values(), key=lambda c: c.start)
    return cands


def chain(stream: Stream, p: np.ndarray, cfg: Config = Config()) -> list[tuple[int, int, float]]:
    """The whole chain from P(fall) on one stream: [(start_step, stop_step, score)]."""
    return [(c.start, c.stop, c.score) for c in run(stream, decode(p, cfg), cfg, p)]


# ---------------------------------------------------------------- the earlier interface (tuples)

def refine(A, cir, peaks, start: int, stop: int, joins: bool = True, raw: bool = True) -> tuple[int, int]:
    lo, hi = analysed_span(cir.shape[1], peaks, start, stop, joins)
    if hi - lo < MIN_WINDOW:
        return start, stop
    left, right = official_bounds(A, np.asarray(cir[:, lo:hi]), raw)
    return (lo + left, lo + right) if right - left >= MIN_REFINED else (start, stop)


def merge(intervals):
    return [(c.start, c.stop, c.score) for c in merge_step([Candidate(a, b, s, ()) for a, b, s in intervals])]


def nms(intervals, radius_s: float = NMS_RADIUS_S):
    return [(c.start, c.stop, c.score) for c in nms_step([Candidate(a, b, s, ()) for a, b, s in intervals], radius_s)]


def gate(A, events, cir, peaks, joins: bool = True):
    stream = Stream(A, cir, peaks)
    kept = gate_step(stream, [Candidate(a, b, s, ()) for a, b, s in events], replace(OLD, joins=joins))
    return [(c.start, c.stop, c.score) for c in kept]


def one_per_recording(events, peaks):
    best = {}
    for s, e, c in events:
        key = int(np.searchsorted(peaks, (s + e) // 2))
        if key not in best or c > best[key][2]:
            best[key] = (s, e, c)
    return sorted(best.values())


def postprocess(A, p: np.ndarray, cir, peaks: np.ndarray, theta: float = 0.3, raw: bool = True,
                use_gate: bool = True, per_recording: bool = False, cfg: Config | None = None) -> list[tuple[int, int, float]]:
    """The earlier chain (refine, merge, NMS 5.9 s, broad gate) unless `cfg` is given."""
    cfg = cfg or replace(OLD, theta=theta, gate="broad" if use_gate else "none", per_recording=per_recording)
    return chain(Stream(A, cir, peaks, raw), p, cfg)


class StreamJoins:
    """Join peaks per stream file, computed once and cached in `cache_dir` (keyed by the file's path,
    size and mtime); and the open CIR datasets for refinement and the gate."""
    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self._peaks: dict[str, np.ndarray] = {}
        self._handles: dict[str, h5py.File] = {}

    def cir(self, path: str | Path):
        path = str(path)
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]["cir"]

    def peaks(self, path: str | Path) -> np.ndarray:
        path = Path(path)
        key = str(path)
        if key in self._peaks:
            return self._peaks[key]
        cache = None
        if self.cache_dir is not None:
            stat = path.stat()
            cache = self.cache_dir / f"{path.parent.name}_{path.stem}_{stat.st_size}_{int(stat.st_mtime)}.npy"
            if cache.exists():
                self._peaks[key] = np.load(cache)
                return self._peaks[key]
        peaks = join_peaks(join_score(self.cir(path)[:]))
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, peaks)
        self._peaks[key] = peaks
        return peaks

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

"""
UWB-FallTrack windows for the frame model.

A stream is cut into large overlapping windows (`window_s`, 20 s by default). Each window is one
sample; the model frames it into small overlapping frames and predicts the state of every
timestep. Predictions of overlapping windows are merged back into one per-timestep prediction
for the stream (`src/abc/model/aggregate.py`).

    train   windows at random starts (`train_crop: random`, the default) or on the fixed grid
            0, hop, 2 hop, ... (`train_crop: grid`); `train_hop_s` sets how many windows an epoch has
    val     every stream on the fixed grid with `infer_hop_s`, the last window ending at the
    test    stream end, so every timestep is covered

Each window is read with `frame_margin = (frame_width - frame_stride) / 2` extra timesteps of
real signal on both sides, so the first and last frames of a window see real data rather than
padding; only the stream ends are padded (zeros, and the `valid` channel says so).

Input channels, in this order:
    3   per-(radar, bin) z-scored CIR, statistics from the training streams only
    3   (diff_channel) the temporal difference of those, scaled by its own training std,
        computed on the continuous stream (so a window's first timestep has its real predecessor)
    1   valid: 1 on real timesteps, 0 on padding

The tensor is [C, 368, L]: rows are range bins, columns are timesteps, like a gir line image.

clutter_s (JF2, with complex_input): 9 more channels before valid, the I, Q and amplitude of the
signal minus its centred moving mean over `clutter_s` seconds (the static scene of the room), computed
on the stream with that much context: [I, Q, A, I_c, Q_c, A_c, valid] = 19 channels. A model then sees
the person's motion and the change of the scene without the session's background.

complex_input (README v1.1: the 368 values of a frame are 184 complex range bins stored as
[I_0 ... I_183, Q_0 ... Q_183]): the tensor is [10, 184, L] instead, channels I of the 3 radars, Q of
the 3 radars, the amplitude sqrt(I^2 + Q^2) of the 3 radars, valid. I and Q of one bin then share a
row, so a patch embedding sees both and can learn phase and Doppler filters (in the 368-row layout
they sit 184 rows apart, in different patches). I and Q of a bin are scaled by one common std,
sqrt((std_I^2 + std_Q^2) / 2), so the normalization keeps phase rotations rotations.

amplitude false (JF4, docs/falljepa_JF4_design.md): the |z| channels are left out of the complex
layout (the stem computes its own moduli): [I, Q, I_c, Q_c, valid] = 13 channels with clutter.
contrast (JF4): per (gap, width) in seconds, 3 more channels (one per radar) after the clutter-removed
ones: |mean z[t + g, t + g + w) - mean z[t - g - w, t - g)| per complex bin, the before / after contrast
of the scene (a constant background cancels in it), k-major: [D1 x3, D2 x3, ...]. At the stream ends a
mean is over the steps that exist; with one side empty the contrast is 0.
contrast_in_model (JF4): the contrast channels are left out of x and computed by the model on the GPU
(model/stem.contrast_from_context) from `ctx` [6, 184, L + 2 R] (the normalized raw I x3, Q x3 with
R = the contrast's reach on each side, after the step swap and the radar dropout) and `ctx_valid`; the
data loader then does no contrast work.

fall_sets (the join-free model, docs/falljepa_join_free.md): every item also carries the labelled falls
that overlap it, `falls` [E, 2] (start, stop in timesteps of its centre part), `falls_w` [E] (1 scored,
0 flagged or with a held-out timestep) and `falls_mask` [E], plus `chunk_hop`. With chunk_windows > 1
a training item is a chunk of that many consecutive inference windows (infer_hop_s apart), which the
model cuts into its windows.

units (data/units.py): every stream also carries its units (the source recordings found from the
join detector), their per-timestep channels `uphys` [11, L + 2m], and per window the units that lie
inside it (`units`, `unit_*`, padded to `unit_max`); training targets per unit come from the events.

Amir's preprocessing (configs/uwb/AP.yaml) changes four things:
    normalizer: interior    the statistics come from each training source recording's longest
                            span without crossfade timesteps, once per recording (activity
                            recordings are reused in several streams)
    physics: true           12 more values per timestep (data/physics.py: raw-frame motion and
                            profile steps at 0.2 / 1 / 3 s per radar, on the whole stream), as
                            `phys` [12, L + 2m]; the model adds them to its frame embedding
    train_sampler: balanced training windows on the fixed grid, complete ones only, drawn by
                            data/sampler.py (source-balanced, with hard negatives)
    infer_tail: pad         inference windows start every hop up to the stream end and are
                            padded past it, instead of a last window that ends at the stream end
"""
import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .folds import fit_and_score_masks, load_assignment, stream_fold_bits
from .labels import load_events, stream_bounds, stream_targets
from .physics import NUM_FEATURES, stream_physics
from .units import NUM_CHANNELS as UNIT_CHANNELS, NUM_SCALARS as UNIT_SCALARS, contained, unit_cache, unit_targets
from .remix import RemixPool
from .sampler import RATE_HZ, BalancedBatchSampler, eligible_patches, source_groups, window_owners

RATE_HZ = 50.0
RADAR_ORDER = (29171, 29444, 30722)
HALF_BINS = 184          # README v1.1: 184 complex bins, I block then Q block


def signal_groups(z: np.ndarray, complex_input: bool, amplitude: bool = True) -> np.ndarray:
    """z [3, n, 368] (normalized) -> the signal channels [C, n, H]: z itself, or with complex_input
    I [3], Q [3] and (amplitude) the amplitude [3] over the 184 complex bins."""
    if not complex_input:
        return z
    i, q = z[..., :HALF_BINS], z[..., HALF_BINS:]
    return np.concatenate([i, q, np.sqrt(i * i + q * q)] if amplitude else [i, q], axis=0)


def contrast_maps(z: np.ndarray, scales: list[tuple[int, int]], lo: int = 0, hi: int | None = None) -> np.ndarray:
    """z [3, n, 368] (normalized) -> [3 K, hi - lo, 184] for timesteps [lo, hi) of z: per scale (gap g,
    width w, in timesteps) and radar, |mean z[t + g, t + g + w) - mean z[t - g - w, t - g)| over the complex
    bins, each mean over the steps of z that exist; 0 where a side is empty. Window sums are differences of
    an edge-padded cumulative sum (float64 sum, float32 differences: about 1e-4 of error)."""
    n = z.shape[1]
    hi = n if hi is None else hi
    reach = max(g + w for g, w in scales)
    cs = np.zeros((z.shape[0], n + 1, z.shape[2]), dtype=np.float64)
    # A constant profile cancels in the contrast: removing the mean keeps the float32 differences small
    np.cumsum(z - z.mean(1, keepdims=True), axis=1, dtype=np.float64, out=cs[:, 1:])
    P = np.pad(cs, ((0, 0), (reach, reach), (0, 0)), mode="edge").astype(np.float32)
    t, m, o = np.arange(lo, hi), hi - lo, reach + lo
    out = np.empty((z.shape[0] * len(scales), m, HALF_BINS), dtype=np.float32)
    for k, (g, w) in enumerate(scales):
        na = np.clip(t - g, 0, n) - np.clip(t - g - w, 0, n)
        nb = np.clip(t + g + w, 0, n) - np.clip(t + g, 0, n)
        before = P[:, o - g:o - g + m] - P[:, o - g - w:o - g - w + m]
        after = P[:, o + g + w:o + g + w + m] - P[:, o + g:o + g + m]
        d = (after * (1 / np.maximum(nb, 1)).astype(np.float32)[None, :, None]
             - before * (1 / np.maximum(na, 1)).astype(np.float32)[None, :, None])
        d *= ((na > 0) & (nb > 0))[None, :, None]
        np.sqrt(d[..., :HALF_BINS] ** 2 + d[..., HALF_BINS:] ** 2, out=out[3 * k:3 * k + 3])
    return out


def in_channels(diff_channel: bool, complex_input: bool, clutter_s: float = 0.0, amplitude: bool = True,
                contrast: list | None = None) -> int:
    """The input channels of a layout: 3 radars x (signal groups x (1 or 2 with the clutter) + contrast
    scales) + valid."""
    groups = (3 if amplitude else 2) if complex_input else 2 if diff_channel else 1
    return 3 * (groups * (2 if clutter_s else 1) + len(contrast or [])) + 1


def complex_std(std: np.ndarray) -> np.ndarray:
    """[3, 1, 368] per-value std -> one common std per complex bin, for its I and its Q."""
    common = np.sqrt((std[..., :HALF_BINS] ** 2 + std[..., HALF_BINS:] ** 2) / 2)
    return np.concatenate([common, common], axis=-1)


def moving_mean(z: np.ndarray, half: int) -> np.ndarray:
    """Centred mean of z [3, n, H] over [t - half, t + half] along time, over the timesteps that exist."""
    c = np.cumsum(np.pad(z.astype(np.float64), ((0, 0), (1, 0), (0, 0))), axis=1)
    n = z.shape[1]
    t = np.arange(n)
    lo, hi = np.maximum(t - half, 0), np.minimum(t + half + 1, n)
    return ((c[:, hi] - c[:, lo]) / (hi - lo)[None, :, None]).astype(np.float32)


def stream_files(release: Path, split: str) -> dict[str, Path]:
    """{stream_id: path} for `streams/<split>/stream_XX.h5` (ids as the official evaluator names them)."""
    folder = Path(release) / "streams" / split
    return {f"{split}_{int(p.stem.split('_')[1]):02d}": p for p in sorted(folder.glob("stream_*.h5"))}


def _check_stream(handle: h5py.File, path: Path) -> int:
    cir = handle["cir"]
    if cir.ndim != 3 or cir.shape[0] != 3 or cir.shape[2] != 368:
        raise ValueError(f"{path}: cir has shape {cir.shape}, expected (3, T, 368)")
    if "radar_serials" in handle:
        serials = tuple(int(s) for s in np.asarray(handle["radar_serials"]).reshape(-1))
        if serials != RADAR_ORDER:
            raise ValueError(f"{path}: radar order {serials}, expected {RADAR_ORDER}")
    return int(cir.shape[1])


def fit_normalizer(paths: list[Path], masks: dict[str, np.ndarray] | None = None, chunk: int = 20000) -> dict:
    """
    Mean and std per (radar, bin) over the timesteps of the given (training) streams, and the
    std of the temporal difference of the z-scored signal. Accumulated in float64.
    masks: optional {str(path): bool [T]}, only True timesteps count (cross-validation leaves the
    held-out fold out); a difference counts when both of its timesteps do.
    """
    def keep(path, a, b):
        return None if masks is None else masks[str(path)][a:b]

    n, s1, s2 = 0, np.zeros((3, 368)), np.zeros((3, 368))
    for path in paths:
        with h5py.File(path, "r") as handle:
            T = _check_stream(handle, path)
            for a in range(0, T, chunk):
                x = handle["cir"][:, a:a + chunk, :].astype(np.float64)
                k = keep(path, a, a + chunk)
                if k is not None:
                    x = x[:, k]
                n += x.shape[1]
                s1 += x.sum(axis=1)
                s2 += np.square(x).sum(axis=1)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0.0))
    std = np.maximum(std, 1e-6)

    m, d2 = 0, np.zeros((3, 368))
    for path in paths:
        with h5py.File(path, "r") as handle:
            T = int(handle["cir"].shape[1])
            prev, prev_k = None, None
            for a in range(0, T, chunk):
                z = (handle["cir"][:, a:a + chunk, :].astype(np.float64) - mean[:, None]) / std[:, None]
                k = keep(path, a, a + chunk)
                k = np.ones(z.shape[1], dtype=bool) if k is None else k
                full = z if prev is None else np.concatenate([prev, z], axis=1)
                full_k = k if prev_k is None else np.concatenate([prev_k, k])
                d = np.diff(full, axis=1)[:, full_k[1:] & full_k[:-1]]
                m += d.shape[1]
                d2 += np.square(d).sum(axis=1)
                prev, prev_k = z[:, -1:], k[-1:]
    diff_std = np.maximum(np.sqrt(d2 / max(m, 1)), 1e-6)
    return {"mean": mean.tolist(), "std": std.tolist(), "diff_std": diff_std.tolist(),
            "fit_streams": [str(p) for p in paths], "fit_timesteps": n}


def interior_masks(composition: str | Path, files: dict[str, Path], lengths: dict[str, int]) -> dict[str, np.ndarray]:
    """{str(path): bool [T]}: for every source recording in these streams, its longest span with no
    crossfade timestep (a timestep two segments cover); each recording counts once."""
    rows: dict[str, list] = {}
    with Path(composition).open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["stream_id"] in files:
                rows.setdefault(r["stream_id"], []).append(r)
    best: dict[str, tuple[int, str, int, int]] = {}
    for sid, segs in rows.items():
        segs.sort(key=lambda r: float(r["t_start"]))
        for i, r in enumerate(segs):
            lo, hi = float(r["t_start"]), float(r["t_end"])
            if i > 0:
                lo = max(lo, float(segs[i - 1]["t_end"]))
            if i + 1 < len(segs):
                hi = min(hi, float(segs[i + 1]["t_start"]))
            a = max(int(math.ceil(lo * RATE_HZ - 1e-6)), 0)
            b = min(int(math.floor(hi * RATE_HZ + 1e-6)), lengths[sid])
            if b - a > best.get(r["path"], (0,))[0]:
                best[r["path"]] = (b - a, sid, a, b)
    masks = {str(files[sid]): np.zeros(lengths[sid], dtype=bool) for sid in files}
    for _, sid, a, b in best.values():
        masks[str(files[sid])][a:b] = True
    return masks


def stream_lengths(files: dict[str, Path]) -> dict[str, int]:
    lengths = {}
    for sid, path in files.items():
        with h5py.File(path, "r") as handle:
            lengths[sid] = _check_stream(handle, path)
    return lengths


class StreamStore:
    """Lazy reads of normalized windows; each worker opens its own HDF5 handles. With `physics`,
    the features of every stream are computed once here (cached as .npy under physics_cache)."""
    def __init__(self, files: dict[str, Path], stats: dict, diff_channel: bool, physics: bool = False,
                 physics_cache: str | Path | None = None, complex_input: bool = False, annotate=None,
                 unit_cache_dir: str | Path | None = None, clutter_s: float = 0.0, amplitude: bool = True,
                 contrast: list | None = None, contrast_in_model: bool = False):
        assert not (complex_input and diff_channel), "complex_input has no difference channel"
        assert clutter_s == 0 or complex_input, "the clutter-removed channels need complex_input"
        assert (amplitude and not contrast) or complex_input, "amplitude / contrast are options of complex_input"
        self.files = files
        self.clutter = int(round(clutter_s * RATE_HZ)) // 2 * 2       # moving-mean width in timesteps, even
        self.amplitude = amplitude
        self.contrast = [(int(round(g * RATE_HZ)), int(round(w * RATE_HZ))) for g, w in (contrast or [])]
        self.contrast_in_model = bool(contrast_in_model and self.contrast)
        self.reach = max([g + w for g, w in self.contrast], default=0)
        # Context read on each side: half the moving mean, the contrast's reach, and 1 for the difference
        self.context = max([1, self.clutter // 2] + [g + w for g, w in self.contrast])
        self.ids = list(files)
        self.mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, :]
        self.std = np.asarray(stats["std"], dtype=np.float32)[:, None, :]
        if complex_input:
            self.std = complex_std(self.std).astype(np.float32)
        self.diff_std = np.asarray(stats["diff_std"], dtype=np.float32)[:, None, :]
        self.diff_channel = diff_channel
        self.complex_input = complex_input
        self.lengths = stream_lengths(files)
        self._handles: dict[str, h5py.File] = {}
        self.physics = {sid: self._physics(sid, physics_cache) for sid in self.ids} if physics else None
        # {sid: units, channels, scalars, rule, peaks} (data/units.py), with the official labeller
        self.units = {sid: unit_cache(annotate, files[sid], unit_cache_dir) for sid in self.ids} if annotate else None

    def read_units(self, sid: str, lo: int, hi: int) -> np.ndarray:
        """Unit channels [11, hi - lo] float32; zero past the stream ends and outside units."""
        T = self.lengths[sid]
        a, b = max(lo, 0), min(hi, T)
        out = np.zeros((UNIT_CHANNELS, hi - lo), dtype=np.float32)
        out[:, a - lo:b - lo] = self.units[sid]["channels"][:, a:b]
        return out

    def _physics(self, sid: str, cache_dir) -> np.ndarray:
        path = Path(self.files[sid])
        cache = None
        if cache_dir is not None:
            key = f"{float(self.mean.sum()):.6e}_{float(self.std.sum()):.6e}"
            stat = path.stat()
            cache = Path(cache_dir) / f"{path.parent.name}_{path.stem}_{stat.st_size}_{int(stat.st_mtime)}_{key}.npy"
            if cache.exists():
                return np.load(cache)
        with h5py.File(path, "r") as handle:
            z = (np.asarray(handle["cir"][:], dtype=np.float32) - self.mean) / self.std
        feats = stream_physics(z)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, feats)
        return feats

    def read_physics(self, sid: str, lo: int, hi: int) -> np.ndarray:
        """Physics features [12, hi - lo]; zero past the stream ends."""
        T = self.lengths[sid]
        a, b = max(lo, 0), min(hi, T)
        out = np.zeros((self.physics[sid].shape[0], hi - lo), dtype=np.float32)
        out[:, a - lo:b - lo] = self.physics[sid][:, a:b]
        return out

    def __getstate__(self):  # DataLoader workers re-open the files
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def _cir(self, sid: str):
        if sid not in self._handles:
            self._handles[sid] = h5py.File(self.files[sid], "r")
        return self._handles[sid]["cir"]

    def read_z(self, sid: str, a: int, b: int) -> np.ndarray:
        """The z-scored CIR [3, b - a, 368] of timesteps [a, b), which must lie in the stream."""
        x = np.asarray(self._cir(sid)[:, a:b, :], dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f"{sid}: non-finite values in [{a}, {b})")
        return (x - self.mean) / self.std

    def read(self, sid: str, lo: int, hi: int, step: tuple[int, np.ndarray] | None = None, ctx: bool = False):
        """
        Channels [C, 368, hi - lo] and valid [hi - lo] for timesteps [lo, hi), which may reach past
        the stream: those positions are zero with valid 0.
        step: optional (t, profile [3, 368]) of the step swap (JF4): from stream timestep t on, the
        normalized signal moves from its own background (its mean over the 2 s before t) to `profile`
        over 10 timesteps, before any channel is derived, so the clutter and contrast channels see it.
        ctx (contrast_in_model): also return the context for the model's contrast, the raw I and Q
        [6, 184, hi - lo + 2 reach] over [lo - reach, hi + reach) and its valid mask; x then has no
        contrast channels.
        """
        T = self.lengths[sid]
        a, b = max(lo, 0), min(hi, T)
        h = self.clutter // 2
        ra = max(a - self.context, 0)  # the difference, half the moving mean, the contrast windows
        rb = min(b + self.context, T)
        x = np.asarray(self._cir(sid)[:, ra:rb, :], dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f"{sid}: non-finite values in [{ra}, {rb})")
        z = (x - self.mean) / self.std                                    # [3, n, 368]
        if step is not None and ra < step[0] < rb:
            k = step[0] - ra
            own = z[:, max(k - 100, 0):k].mean(1)
            ramp = np.clip((np.arange(z.shape[1]) - k + 1) / 10.0, 0.0, 1.0).astype(np.float32)
            z = z + ramp[None, :, None] * (step[1] - own)[:, None, :]
        parts = [signal_groups(z[:, a - ra:b - ra], self.complex_input, self.amplitude)]
        if self.clutter:
            parts.append(signal_groups(z[:, a - ra:b - ra] - moving_mean(z, h)[:, a - ra:b - ra], True, self.amplitude))
        context = None
        if self.contrast_in_model and ctx:
            R = self.reach
            ca, cb = max(lo - R, 0), min(hi + R, T)                      # inside [ra, rb): context >= reach
            context = np.zeros((6, hi - lo + 2 * R, HALF_BINS), dtype=np.float32)
            context[:, ca - lo + R:cb - lo + R] = signal_groups(z[:, ca - ra:cb - ra], True, False)
            context_valid = np.zeros(hi - lo + 2 * R, dtype=np.float32)
            context_valid[ca - lo + R:cb - lo + R] = 1.0
        elif self.contrast:
            parts.append(contrast_maps(z, self.contrast, a - ra, b - ra))
        z = z[:, :b - ra]
        if self.diff_channel:
            d = np.diff(z, axis=1, prepend=z[:, :1]) / self.diff_std     # first of the stream: 0
            parts.append(d[:, a - ra:])
        core = np.concatenate(parts, axis=0)                              # [C-1, b - a, H]

        H = core.shape[-1]
        out = np.zeros((core.shape[0], hi - lo, H), dtype=np.float32)
        out[:, a - lo:b - lo] = core
        valid = np.zeros(hi - lo, dtype=np.float32)
        valid[a - lo:b - lo] = 1.0
        x = np.concatenate([out, np.broadcast_to(valid[None, :, None], (1, hi - lo, H))], axis=0)
        x = np.ascontiguousarray(x.transpose(0, 2, 1))                    # [C, H, L]
        if ctx:
            return x, valid, (None if context is None else np.ascontiguousarray(context.transpose(0, 2, 1))), \
                (None if context is None else context_valid)
        return x, valid


def grid_starts(num_steps: int, window: int, hop: int, tail: str = "end") -> list[int]:
    """0, hop, 2 hop, ... and then, by `tail`: "end" a last window that ends at the stream end (every
    timestep covered); "pad" every hop up to the stream end, windows padded past it (Amir's
    inference); "none" complete windows only (Amir's training windows)."""
    if tail == "pad":
        return list(range(0, num_steps, hop))
    last = max(num_steps - window, 0)
    starts = list(range(0, last + 1, hop))
    if tail == "none":
        return starts if num_steps >= window else []
    if starts[-1] != last:
        starts.append(last)
    return starts


MAX_FALLS = 8           # falls per item (a 50 s chunk holds at most 3 or 4)


AUGMENT_KEYS = {"range_shift", "gain_std", "noise_std", "radar_drop_p", "band_gain_db", "band_count",
                "remix_p", "remix_lead_s", "remix_tail_s", "remix_kinds", "bg_swap_p", "bg_swap_s",
                "step_swap_p", "step_swap_s"}
STEP_GUARD = (150, 50)  # step swap: never within 3 s before to 1 s after a labelled fall


class WindowDataset(Dataset):
    """
    One item per window: x [C, 368, L + 2m], valid [L + 2m], state and weight [L + 2m] (the margins
    keep their real labels: every frame is supervised on all of its timesteps), bounds [L + 2m, 2]
    and bound_weight [L + 2m] (boundary-regression targets in seconds, weight 0 outside falls), fit [L + 2m] (the
    timesteps that may train the model: all of them, except a held-out fold in cross-validation),
    the stream index and the window start (in timesteps, of the centre part).
    masks: optional {stream_id: bool [T]}; weight and fit are 0 where it is False.
    remix: optional data/remix.py pool; a share `remix.p` of the windows is rebuilt from segments
    of the training streams (its draws use their own generator, so the other augmentations of a
    window are the same with and without remix).
    """
    def __init__(self, store: StreamStore, targets: dict[str, tuple[np.ndarray, ...]] | None,
                 window: int, margin: int, mode: str, hop: int, samples: int | None = None,
                 augment: dict | None = None, seed: int = 0, masks: dict[str, np.ndarray] | None = None,
                 tail: str = "end", remix: RemixPool | None = None, events: dict | None = None,
                 unit_max: int = 16, fall_sets: bool = False, chunk_hop: int = 0):
        assert mode in ("random", "grid")
        assert remix is None or store.units is None, "remix windows carry no units"
        unknown = set(augment or {}) - AUGMENT_KEYS
        assert not unknown, f"unknown augment keys {sorted(unknown)}"
        assert remix is None or (targets is not None and masks is None), "remix needs targets and no CV masks"
        self.store, self.targets, self.masks = store, targets, masks
        self.window, self.margin, self.mode, self.hop = window, margin, mode, hop
        self.remix = remix
        self.augment = augment or {}
        self.seed = seed
        self.index = [(i, s) for i, sid in enumerate(store.ids) for s in grid_starts(store.lengths[sid], window, hop, tail)]
        self.samples = samples or len(self.index)
        self.epoch = 0
        lengths = np.array([store.lengths[sid] for sid in store.ids], dtype=np.float64)
        self._p = lengths / lengths.sum()
        self.unit_max = unit_max
        # Per stream unit targets (data/units.py), when the store has units and there are labels
        self.unit_targets = None
        if store.units is not None and targets is not None:
            self.unit_targets = {sid: unit_targets(u["units"], u["rule"], (events or {}).get(sid, []))
                                 for sid, u in store.units.items()}
        # Per stream [n, 3] (start, stop, scored) of the labelled falls, for the join-free model
        self.fall_sets, self.chunk_hop = fall_sets, chunk_hop
        self.falls = None
        if fall_sets and events:
            assert remix is None, "remix windows carry no fall sets"
            self.falls = {}
            for sid in store.ids:
                rows = [(max(int(round(e.t_start * RATE_HZ)), 0), int(round(e.t_end * RATE_HZ)), not e.flagged)
                        for e in (events or {}).get(sid, [])]
                rows = [(a, b, ok) for a, b, ok in rows if b > a]
                if masks is not None:                        # a fall with a held-out timestep is never a target
                    rows = [(a, b, ok and bool(masks[sid][a:b].all())) for a, b, ok in rows]
                self.falls[sid] = np.array(rows, dtype=np.int64).reshape(-1, 3)

    def _fall_fields(self, sid: str, start: int) -> dict:
        """The falls overlapping the item's centre part, in its timesteps (0 = `start`); only the hop
        on streams without labels (the test set)."""
        item = {"chunk_hop": torch.tensor(self.chunk_hop)}
        if self.falls is None:
            return item
        bounds = np.zeros((MAX_FALLS, 2), dtype=np.int64)
        weight = np.zeros(MAX_FALLS, dtype=np.float32)
        mask = np.zeros(MAX_FALLS, dtype=bool)
        f = self.falls[sid]
        f = f[(f[:, 1] > start) & (f[:, 0] < start + self.window)][:MAX_FALLS]
        n = len(f)
        bounds[:n] = f[:, :2] - start
        weight[:n] = f[:, 2]
        mask[:n] = True
        item.update({"falls": torch.from_numpy(bounds), "falls_w": torch.from_numpy(weight), "falls_mask": torch.from_numpy(mask)})
        return item

    def window_units(self, sid: str, start: int) -> np.ndarray:
        """The units of `sid` the window starting at `start` holds (at most unit_max, longest first)."""
        units = self.store.units[sid]["units"]
        idx = contained(units, start, self.window, self.store.lengths[sid])
        if len(idx) > self.unit_max:
            idx = np.sort(idx[np.argsort(-(units[idx, 1] - units[idx, 0]), kind="stable")[:self.unit_max]])
        return idx

    def _unit_fields(self, sid: str, start: int, lo: int, hi: int, dropped: int | None) -> dict:
        """The unit tensors of a window, in timesteps of its centre part (0 = `start`)."""
        U, data = self.unit_max, self.store.units[sid]
        idx = self.window_units(sid, start)
        n = len(idx)
        bounds = np.zeros((U, 2), dtype=np.int64)
        mask = np.zeros(U, dtype=bool)
        index = np.full(U, -1, dtype=np.int64)
        scalars = np.zeros((U, UNIT_SCALARS), dtype=np.float32)
        bounds[:n] = data["units"][idx, :2] - start
        mask[:n] = True
        index[:n] = idx
        scalars[:n] = data["scalars"][idx]
        uphys = self.store.read_units(sid, lo, hi)
        if dropped is not None:                              # 3 channels per radar, radar-major
            uphys[3 * dropped:3 * dropped + 3] = 0.0
        item = {"units": torch.from_numpy(bounds), "unit_mask": torch.from_numpy(mask),
                "unit_index": torch.from_numpy(index), "unit_scalars": torch.from_numpy(scalars),
                "uphys": torch.from_numpy(uphys)}
        if self.unit_targets is not None:
            t = self.unit_targets[sid]
            fit = np.ones(len(idx), dtype=np.float32)
            if self.masks is not None:                       # a unit with a held-out timestep trains nothing
                fit = np.array([float(self.masks[sid][a:b].all()) for a, b, _ in data["units"][idx]], dtype=np.float32)
            fields = {"unit_fall": (t["fall"], np.float32), "unit_fall_w": (t["fall_w"], np.float32),
                      "unit_span_w": (t["span_w"], np.float32), "unit_rule_w": (t["rule_w"], np.float32)}
            for key, (values, dtype) in fields.items():
                arr = np.zeros(U, dtype=dtype)
                arr[:n] = values[idx] * (fit if key != "unit_fall" else 1.0)
                item[key] = torch.from_numpy(arr)
            span = np.zeros((U, 2), dtype=np.int64)
            rule = np.zeros((U, 2), dtype=np.int64)
            span[:n] = t["span"][idx] - start
            rule[:n] = np.where(data["rule"][idx] >= 0, data["rule"][idx] - start, 0)
            item["unit_span"], item["unit_rule"] = torch.from_numpy(span), torch.from_numpy(rule)
        return item

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples if self.mode == "random" else len(self.index)

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng((self.seed, self.epoch, idx))

    def __getitem__(self, idx: int) -> dict:
        if self.mode == "random":
            rng = self._rng(idx)
            stream = int(rng.choice(len(self.store.ids), p=self._p))
            T = self.store.lengths[self.store.ids[stream]]
            start = int(rng.integers(0, max(T - self.window, 0) + 1))
        else:
            rng = self._rng(idx)
            stream, start = self.index[idx]
        sid = self.store.ids[stream]
        lo, hi = start - self.margin, start + self.window + self.margin
        mixed = None
        if self.remix is not None:
            remix_rng = np.random.default_rng((self.seed, self.epoch, idx, 1))
            if remix_rng.random() < self.remix.p:
                mixed = self.remix.compose(self.store, self.targets, stream, start, self.window, self.margin, remix_rng)
        if mixed is not None:
            return self._remixed_item(mixed, stream, start, rng)
        x, valid, ctx, ctx_valid = self.store.read(sid, lo, hi, self._step(sid, lo, hi, idx), ctx=True)
        dropped = None
        if self.augment:
            x, dropped = self._augment(x, rng)
            if ctx is not None and dropped is not None:
                ctx[[dropped, 3 + dropped]] = 0.0

        a, b = max(lo, 0), min(hi, self.store.lengths[sid])
        fit = valid.copy()
        if self.masks is not None:
            fit[a - lo:b - lo] *= self.masks[sid][a:b]
        item = {"x": torch.from_numpy(x), "valid": torch.from_numpy(valid), "fit": torch.from_numpy(fit),
                "stream": torch.tensor(stream), "start": torch.tensor(start)}
        if ctx is not None:
            item["ctx"], item["ctx_valid"] = torch.from_numpy(ctx), torch.from_numpy(ctx_valid)
        if self.store.physics is not None:
            phys = self.store.read_physics(sid, lo, hi)
            if dropped is not None:                          # 4 features per radar, radar-major
                n = phys.shape[0] // 3
                phys[dropped * n:(dropped + 1) * n] = 0.0
            item["phys"] = torch.from_numpy(phys)
        if self.store.units is not None:
            item.update(self._unit_fields(sid, start, lo, hi, dropped))
        if self.fall_sets:
            item.update(self._fall_fields(sid, start))
        if self.targets is not None:
            state, weight, bounds, inside = self.targets[sid]
            st = np.zeros(hi - lo, dtype=np.int64)
            wt = np.zeros(hi - lo, dtype=np.float32)
            bd = np.zeros((hi - lo, 2), dtype=np.float32)
            st[a - lo:b - lo] = state[a:b]
            wt[a - lo:b - lo] = weight[a:b]
            bd[a - lo:b - lo] = bounds[a:b]
            wt *= fit
            item["state"] = torch.from_numpy(st)
            item["weight"] = torch.from_numpy(wt)
            item["bounds"] = torch.from_numpy(bd)
            item["bound_weight"] = torch.from_numpy(wt * np.pad(inside[a:b], (a - lo, hi - b)))
        return item

    def _step(self, sid: str, lo: int, hi: int, idx: int) -> tuple[int, np.ndarray] | None:
        """step_swap_p (JF4): a decoy join for StreamStore.read, at a random timestep of the item that is
        not within STEP_GUARD of a labelled fall, towards the mean profile of `step_swap_s` seconds
        (default 2) of a random training stream. Own generator, so the other draws do not change."""
        p = self.augment.get("step_swap_p", 0)
        if p <= 0 or self.falls is None:
            return None
        rng = np.random.default_rng((self.seed, self.epoch, idx, 2))
        if rng.random() >= p:
            return None
        a, b = max(lo, 0) + 100, min(hi, self.store.lengths[sid])   # 2 s of the item before t: its own background
        ok = np.ones(max(b - a, 0), dtype=bool)
        for fa, fb, _ in self.falls[sid]:
            ok[max(fa - STEP_GUARD[0] - a, 0):max(min(fb + STEP_GUARD[1] - a, len(ok)), 0)] = False
        free = np.flatnonzero(ok)
        if len(free) == 0:
            return None
        t = int(a + free[int(rng.integers(len(free)))])
        other = self.store.ids[int(rng.integers(len(self.store.ids)))]
        n = int(round(float(self.augment.get("step_swap_s", 2.0)) * RATE_HZ))
        T = self.store.lengths[other]
        u = int(rng.integers(0, max(T - n, 0) + 1))
        return t, self.store.read_z(other, u, min(u + n, T)).mean(1)

    def _remixed_item(self, mixed: dict, stream: int, start: int, rng: np.random.Generator) -> dict:
        """The item of a remixed window (data/remix.py): every timestep is real signal."""
        W = self.window + 2 * self.margin
        valid = np.ones(W, dtype=np.float32)
        parts = [signal_groups(mixed["z"], self.store.complex_input, self.store.amplitude)] + ([mixed["diff"]] if mixed["diff"] is not None else [])
        x = np.concatenate(parts + [np.ones((1, W, parts[0].shape[-1]), dtype=np.float32)], axis=0)
        x = np.ascontiguousarray(x.transpose(0, 2, 1))
        dropped = None
        if self.augment:
            x, dropped = self._augment(x, rng)
        item = {"x": torch.from_numpy(x), "valid": torch.from_numpy(valid), "fit": torch.from_numpy(valid.copy()),
                "stream": torch.tensor(stream), "start": torch.tensor(start)}
        if mixed["phys"] is not None:
            phys = np.ascontiguousarray(mixed["phys"], dtype=np.float32)
            if dropped is not None:
                n = phys.shape[0] // 3
                phys[dropped * n:(dropped + 1) * n] = 0.0
            item["phys"] = torch.from_numpy(phys)
        item["state"] = torch.from_numpy(mixed["state"].copy())
        item["weight"] = torch.from_numpy(mixed["weight"].copy())
        item["bounds"] = torch.from_numpy(mixed["bounds"].copy())
        item["bound_weight"] = torch.from_numpy(mixed["weight"] * mixed["inside"])
        return item

    def _swap_background(self, x: np.ndarray, rng: np.random.Generator) -> None:
        """bg_swap_p (see _augment), in place on x [C, 184, L] (complex layout)."""
        valid = x[-1, 0] > 0
        if not valid.any():
            return
        sid = self.store.ids[int(rng.integers(len(self.store.ids)))]
        n = int(round(float(self.augment.get("bg_swap_s", 2.0)) * RATE_HZ))
        T = self.store.lengths[sid]
        t = int(rng.integers(0, max(T - n, 0) + 1))
        other = self.store.read_z(sid, t, min(t + n, T)).mean(1)               # [3, 368]
        for k, half in ((0, slice(0, HALF_BINS)), (3, slice(HALF_BINS, 2 * HALF_BINS))):
            own = x[k:k + 3][:, :, valid].mean(-1)                             # [3, 184]
            x[k:k + 3, :, valid] += (other[:, half] - own)[..., None]
        if self.store.amplitude:
            x[6:9] = np.sqrt(x[0:3] ** 2 + x[3:6] ** 2) * valid

    def _augment(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, int | None]:
        """Range shift (all radars by the same number of bins), per-radar gain, range-band gain,
        additive noise and dropping a whole radar, on the signal channels only. Returns the window
        and the dropped radar (None if none), whose physics features the caller zeroes too.

        band_gain_db: FilterAugment (linear type) along range: per radar, `band_count` = [lo, hi]
        random boundaries split the 368 bins into bands, each band gets a gain U(-db, +db) dB, and
        the gain is interpolated linearly between band centres, so the mat positions stay where
        they are (unlike range_shift).

        bg_swap_p (complex input): with this probability the raw I and Q channels get another
        session's static scene instead of their own: the window's mean profile over its valid timesteps
        is replaced by the mean profile of `bg_swap_s` seconds (default 2) of a random training stream,
        and the amplitude is recomputed. The clutter-removed channels do not change (a constant profile
        cancels in them)."""
        x = x.copy()
        if self.store.complex_input and rng.random() < self.augment.get("bg_swap_p", 0):
            self._swap_background(x, rng)
        signal = x[:-1].reshape(-1, 3, *x.shape[1:])        # [groups, radar, H, L]
        valid = x[-1:]
        k = int(self.augment.get("range_shift", 0))
        if k > 0:
            shift = int(rng.integers(-k, k + 1))
            if shift:                                       # bins shifted in from outside are 0 (the mean)
                signal[:] = np.roll(signal, shift, axis=2)
                if shift > 0:
                    signal[:, :, :shift] = 0.0
                else:
                    signal[:, :, shift:] = 0.0
        if self.augment.get("gain_std", 0) > 0:
            signal *= np.exp(rng.normal(0, self.augment["gain_std"], size=(1, 3, 1, 1))).astype(np.float32)
        if self.augment.get("band_gain_db", 0) > 0:
            signal *= band_gains(rng, signal.shape[2], float(self.augment["band_gain_db"]),
                                 tuple(self.augment.get("band_count", (2, 5))))[None, :, :, None]
        if self.augment.get("noise_std", 0) > 0:
            signal += rng.normal(0, self.augment["noise_std"], size=signal.shape).astype(np.float32) * valid
        dropped = None
        if rng.random() < self.augment.get("radar_drop_p", 0):
            dropped = int(rng.integers(3))
            signal[:, dropped] = 0.0
        return x, dropped


def band_gains(rng: np.random.Generator, bins: int, db: float, count: tuple[int, int]) -> np.ndarray:
    """[3, bins] linear gains: per radar, k ~ U{count} random boundaries make k + 1 bands, each with a
    gain of U(-db, db) dB at its centre, interpolated linearly in dB between centres."""
    out = np.empty((3, bins), dtype=np.float32)
    grid = np.arange(bins)
    for r in range(3):
        k = int(rng.integers(count[0], count[1] + 1))
        edges = np.sort(np.concatenate([[0, bins], rng.integers(1, bins, size=k)]))
        centres = (edges[:-1] + edges[1:]) / 2
        gains = rng.uniform(-db, db, size=len(centres))
        out[r] = 10 ** (np.interp(grid, centres, gains) / 20)
    return out


class UWBDataModule(LightningDataModule):
    """
    release:          the extracted official package (streams/, streams_events.csv)
    stats_path:       where the training-only normalizer is cached (built on first use)
    train_splits / val_splits / test_splits: folders under streams/; validation runs on whole
                      streams and is scored with the official evaluator
    window_s, train_hop_s, infer_hop_s: the large windows (20 s, 10 s by default, as planned)
    frame_width / frame_stride: the model's frames, linked to it by the CLI; they set the margin
    num_states:       4 (none / begin / middle / end) or 2 (none / fall, the binary ablation); set
                      the model's num_states and the class weights to match
    regress_center:   the central fraction of each fall that gets boundary-regression targets
                      (1: all of it; ActionFormer's center sampling: 0.5)
    normalizer:       "all" (every training timestep) or "interior" (Amir: each training recording's
                      longest span without crossfade, once per recording)
    physics:          Amir's physics features as `phys` [12, L + 2m] (data/physics.py)
    boundary_mask_s:  loss weight 0 within this distance of every fall boundary (Amir: 0.2)
    train_sampler:    "random" (train_crop decides) or "balanced" (Amir: complete windows on the
                      train_hop_s grid, batches from data/sampler.py; `sampler` holds fall_slots,
                      hard_slots, hard_label and groups, the CSV of subject groups per recording;
                      join_slots / join_s add no-fall slots from ADL windows just after a join;
                      steps_scale scales the steps of an epoch, e.g. 0.25 to keep the epoch of 2
                      training stitchings on folds built with 8; split yields each batch in that many
                      pieces, for gradient accumulation with less GPU memory)
    infer_tail:       "end" (a last window ending at the stream end) or "pad" (Amir)
    augment:          training augmentations (WindowDataset._augment): range_shift, gain_std,
                      noise_std, radar_drop_p, band_gain_db / band_count; and remix_p (with
                      remix_lead_s, remix_tail_s, remix_kinds): the share of training windows rebuilt
                      from segments of the training streams (data/remix.py). Unknown keys are an error
    complex_input:    README v1.1's complex layout: I, Q and amplitude of 184 bins (see the top)
    units:            the recording head's units (data/units.py) in every window; needs the
                      official labeller, from `annotate_release` (default: release), whose
                      src/abc2027/annotate.py is loaded (the CV fold folders do not hold it)
    unit_max:         at most this many units per window
    fall_sets:        the join-free model's targets in every item (falls, falls_w, falls_mask, chunk_hop)
    chunk_windows:    training items are chunks of this many inference windows, infer_hop_s apart (1:
                      plain windows); with the balanced sampler, batch_size counts chunks
    clutter_s:        with complex_input, 9 clutter-removed channels (the signal minus its centred
                      moving mean over this many seconds); 0: none
    amplitude:        with complex_input, the |z| channels (false: JF4's stem layout, no |z|)
    contrast:         with complex_input, [gap, width] in seconds per scale: 3 before / after contrast
                      channels each (JF4); augment.step_swap_p / step_swap_s add a decoy join per item
    contrast_in_model: the model computes the contrast channels on the GPU from `ctx` (see the top); the
                      model's input channels are the same
    cv_assignment / cv_fold: recording-grouped cross-validation on the training streams (see
                      data/folds.py): with a fold, validation scores that fold's timesteps of the
                      training streams instead of the val streams; test is unchanged
    """
    def __init__(
        self,
        release: str,
        stats_path: str = "runs/_cache/normalizer.json",
        train_splits: list[str] = ["train"],
        val_splits: list[str] = ["val"],
        test_splits: list[str] = ["val"],
        window_s: float = 20.0,
        train_hop_s: float = 10.0,
        infer_hop_s: float = 10.0,
        train_crop: str = "random",
        frame_width: int = 32,
        frame_stride: int = 8,
        diff_channel: bool = True,
        boundary_s: float = 0.3,
        flagged_weight: float = 0.5,
        flagged_margin_s: float = 0.0,
        num_states: int = 4,
        regress_center: float = 1.0,
        augment: dict | None = None,
        normalizer: str = "all",
        physics: bool = False,
        boundary_mask_s: float = 0.0,
        train_sampler: str = "random",
        sampler: dict | None = None,
        infer_tail: str = "end",
        batch_size: int = 16,
        eval_batch_size: int | None = None,
        num_workers: int = 4,
        seed: int = 0,
        cv_assignment: str | None = None,
        cv_fold: int | None = None,
        complex_input: bool = False,
        units: bool = False,
        annotate_release: str | None = None,
        unit_max: int = 16,
        fall_sets: bool = False,
        chunk_windows: int = 1,
        clutter_s: float = 0.0,
        amplitude: bool = True,
        contrast: list[list[float]] | None = None,
        contrast_in_model: bool = False,
    ):
        super().__init__()
        assert clutter_s == 0 or complex_input, "the clutter-removed channels need complex_input"
        assert (amplitude and not contrast) or complex_input, "amplitude / contrast are options of complex_input"
        assert not (augment or {}).get("remix_p") or (amplitude and not contrast), "remix builds the JF2 layout only"
        self.clutter_s, self.amplitude, self.contrast = clutter_s, amplitude, [list(s) for s in (contrast or [])]
        self.contrast_in_model = contrast_in_model
        assert not (complex_input and diff_channel), "complex_input has no difference channel"
        self.complex_input, self.use_units, self.unit_max = complex_input, units, unit_max
        self.annotate_release = Path(annotate_release) if annotate_release else Path(release)
        assert (frame_width - frame_stride) % 2 == 0, "frame_width - frame_stride must be even"
        window = int(round(window_s * RATE_HZ))
        assert window % frame_stride == 0, "the window must be a whole number of frame strides"
        self.release = Path(release)
        self.stats_path = Path(stats_path)
        self.train_splits, self.val_splits, self.test_splits = train_splits, val_splits, test_splits
        self.window = window
        self.train_hop = int(round(train_hop_s * RATE_HZ))
        self.infer_hop = int(round(infer_hop_s * RATE_HZ))
        assert chunk_windows >= 1 and (chunk_windows == 1 or fall_sets), "chunks are for the join-free model"
        assert chunk_windows == 1 or self.infer_hop % frame_stride == 0, "the inference hop must be a whole number of frames"
        self.fall_sets, self.chunk_windows = fall_sets, chunk_windows
        self.chunk = window + (chunk_windows - 1) * self.infer_hop     # the training item, in timesteps
        self.train_crop = train_crop
        self.margin = (frame_width - frame_stride) // 2
        self.diff_channel = diff_channel
        self.label_args = {"boundary_s": boundary_s, "flagged_weight": flagged_weight,
                           "flagged_margin_s": flagged_margin_s, "rate_hz": RATE_HZ, "num_states": num_states,
                           "boundary_mask_s": boundary_mask_s}
        assert normalizer in ("all", "interior") and train_sampler in ("random", "balanced") and infer_tail in ("end", "pad")
        self.normalizer, self.physics, self.train_sampler, self.infer_tail = normalizer, physics, train_sampler, infer_tail
        self.sampler_args = {"fall_slots": 16, "hard_slots": 4, "hard_label": "Autre", "groups": None,
                             "join_slots": 0, "join_s": 4.0, "steps_scale": 1.0, "split": 1, **(sampler or {})}
        self.physics_cache = self.stats_path.parent / "physics"
        self.regress_center = regress_center
        self.augment = augment if augment is not None else {"gain_std": 0.1, "noise_std": 0.05, "radar_drop_p": 0.1}
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.events_path = self.release / "streams_events.csv"
        self.composition_path = self.release / "streams_composition.csv"
        assert (cv_fold is None) or cv_assignment, "cv_fold needs cv_assignment"
        self.cv_assignment, self.cv_fold = cv_assignment, cv_fold
        self.datasets: dict[str, WindowDataset] = {}
        # {stage: {stream_id: bool [T]}}: where predictions and events are scored (all of a stream if absent)
        self.score_masks: dict[str, dict[str, np.ndarray]] = {}

    @property
    def in_channels(self) -> int:
        return in_channels(self.diff_channel, self.complex_input, self.clutter_s, self.amplitude, self.contrast)

    @property
    def n_bins(self) -> int:
        return HALF_BINS if self.complex_input else 368

    def _annotate(self):
        if not self.use_units:
            return None
        from ..postprocess import load_annotate
        return load_annotate(self.annotate_release)

    @property
    def physics_channels(self) -> int:
        return 3 * NUM_FEATURES if self.physics else 0

    def _files(self, splits: list[str]) -> dict[str, Path]:
        files = {}
        for split in splits:
            files.update(stream_files(self.release, split))
        if not files:
            raise FileNotFoundError(f"no streams under {self.release}/streams/{{{','.join(splits)}}}")
        return files

    def _cv_masks(self, lengths: dict[str, int]):
        """(fit, score) masks of the held-out fold over the training streams."""
        bits = stream_fold_bits(self.composition_path, load_assignment(self.cv_assignment), lengths)
        return fit_and_score_masks(bits, self.cv_fold)

    def _stats(self, fit_masks=None) -> dict:
        train_files = self._files(self.train_splits)
        path = self.stats_path
        if self.cv_fold is not None:
            path = path.with_name(f"{path.stem}_cv{self.cv_fold}{path.suffix}")
        if path.exists():
            stats = json.loads(path.read_text())
            if (sorted(stats["fit_streams"]) == sorted(str(p) for p in train_files.values()) and stats.get("cv_fold") == self.cv_fold
                    and stats.get("normalizer", "all") == self.normalizer):
                return stats
        masks = None if fit_masks is None else {str(train_files[sid]): m for sid, m in fit_masks.items()}
        if self.normalizer == "interior":
            inner = interior_masks(self.composition_path, train_files, stream_lengths(train_files))
            masks = inner if masks is None else {k: m & inner[k] for k, m in masks.items()}
        stats = fit_normalizer(list(train_files.values()), masks)
        stats["cv_fold"] = self.cv_fold
        stats["normalizer"] = self.normalizer
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats))
        return stats

    def _targets(self, store: StreamStore) -> dict:
        events = load_events(self.events_path) if self.events_path.exists() else {}
        if not any(sid in events for sid in store.ids):
            return None
        return {sid: (*stream_targets(events.get(sid, []), store.lengths[sid], **self.label_args),
                      *stream_bounds(events.get(sid, []), store.lengths[sid], RATE_HZ, self.regress_center))
                for sid in store.ids}

    def setup(self, stage: str | None = None) -> None:
        fit_masks = score_masks = None
        if self.cv_fold is not None:
            fit_masks, score_masks = self._cv_masks(stream_lengths(self._files(self.train_splits)))
        stats = self._stats(fit_masks)
        if stage in ("fit", None):
            store = self._store(self.train_splits, stats)
            targets = self._targets(store)
            remix = self._remix_pool(store) if self.augment.get("remix_p", 0) > 0 else None
            if self.train_sampler == "balanced":
                self.datasets["train"] = WindowDataset(store, targets, self.chunk, self.margin, "grid", self.train_hop,
                                                       augment=self.augment, seed=self.seed, masks=fit_masks, tail="none",
                                                       remix=remix, events=self._events(), unit_max=self.unit_max,
                                                       fall_sets=self.fall_sets, chunk_hop=self.infer_hop)
                self.batch_sampler = self._balanced_sampler(self.datasets["train"], targets)
            else:
                n = sum(math.ceil(T / self.train_hop) for T in store.lengths.values())
                self.datasets["train"] = WindowDataset(store, targets, self.chunk, self.margin,
                                                       self.train_crop, self.train_hop, samples=n,
                                                       augment=self.augment, seed=self.seed, masks=fit_masks,
                                                       remix=remix, events=self._events(), unit_max=self.unit_max,
                                                       fall_sets=self.fall_sets, chunk_hop=self.infer_hop)
        if stage in ("fit", "validate", None):
            if self.cv_fold is None:
                self.datasets["val"] = self._eval_dataset(self.val_splits, stats)
            else:
                self.datasets["val"] = self._eval_dataset(self.train_splits, stats, masks=score_masks)
                self.score_masks["val"] = score_masks
        if stage in ("test", "predict"):
            self.datasets["test"] = self._eval_dataset(self.test_splits, stats)

    def _remix_pool(self, store: StreamStore) -> RemixPool:
        a = self.augment
        return RemixPool(self.composition_path, store.ids, store.lengths, load_events(self.events_path),
                         p=float(a["remix_p"]), lead_s=tuple(a.get("remix_lead_s", (1.0, 8.0))),
                         tail_s=tuple(a.get("remix_tail_s", (2.0, 10.0))), kinds=a.get("remix_kinds"))

    def _events(self) -> dict:
        return load_events(self.events_path) if self.events_path.exists() else {}

    def _store(self, splits, stats) -> StreamStore:
        return StreamStore(self._files(splits), stats, self.diff_channel, self.physics, self.physics_cache,
                           self.complex_input, self._annotate(), self.stats_path.parent / "units", self.clutter_s,
                           self.amplitude, self.contrast, self.contrast_in_model)

    def _eval_dataset(self, splits, stats, masks=None) -> WindowDataset:
        store = self._store(splits, stats)
        return WindowDataset(store, self._targets(store), self.window, self.margin, "grid", self.infer_hop, masks=masks,
                             tail=self.infer_tail, events=self._events(), unit_max=self.unit_max,
                             fall_sets=self.fall_sets, chunk_hop=self.infer_hop)

    def _balanced_sampler(self, ds: WindowDataset, targets: dict) -> BalancedBatchSampler:
        """Amir's sampler over the grid windows: owner recording (centre timestep), its subject group,
        hard negatives, and which windows hold an eligible fall / no-fall patch."""
        n = len(ds.index)
        sources, hard, join = [""] * n, [False] * n, [False] * n
        fall_ok, none_ok = np.zeros(n, bool), np.zeros(n, bool)
        by_stream: dict[int, list[int]] = {}
        for i, (stream, _) in enumerate(ds.index):
            by_stream.setdefault(stream, []).append(i)
        label = self.sampler_args["hard_label"]
        join_s = float(self.sampler_args["join_s"])
        for stream, idx in by_stream.items():
            sid = ds.store.ids[stream]
            starts = [ds.index[i][1] for i in idx]
            centres = [s + ds.window // 2 for s in starts]
            owners = window_owners(self.composition_path, sid, centres)
            state, weight = targets[sid][0], targets[sid][1]
            if ds.masks is not None:
                weight = weight * ds.masks[sid]
            f, nf = eligible_patches(state, weight, starts, ds.window)
            if ds.unit_targets is not None:   # a fall slot must hold a whole, scored fall unit
                t = ds.unit_targets[sid]
                f = np.array([bool(((t["fall"] * t["fall_w"])[ds.window_units(sid, s)] > 0).any()) for s in starts])
            for k, i in enumerate(idx):
                sources[i], hard[i] = owners[k]["path"], owners[k]["label"] == label
                after = centres[k] / RATE_HZ - float(owners[k]["t_start"])   # seconds since the join into it
                join[i] = owners[k]["kind"] == "adl" and float(owners[k]["t_start"]) > 0 and after <= join_s
                fall_ok[i], none_ok[i] = f[k], nf[k]
        groups = source_groups(sources, self.sampler_args["groups"])
        steps = math.ceil(n * float(self.sampler_args["steps_scale"]) / self.batch_size)
        return BalancedBatchSampler(sources, groups, hard, fall_ok, none_ok, batch_size=self.batch_size,
                                    fall_slots=int(self.sampler_args["fall_slots"]), hard_slots=int(self.sampler_args["hard_slots"]),
                                    steps=steps, seed=self.seed, join=join, join_slots=int(self.sampler_args["join_slots"]),
                                    split=int(self.sampler_args["split"]))

    def train_dataloader(self):
        ds = self.datasets["train"]
        if self.trainer is not None:
            ds.set_epoch(self.trainer.current_epoch)
        if self.train_sampler == "balanced":
            if self.trainer is not None:
                self.batch_sampler.set_epoch(self.trainer.current_epoch)
            return DataLoader(ds, batch_sampler=self.batch_sampler, num_workers=self.num_workers,
                              persistent_workers=False, pin_memory=torch.cuda.is_available())
        return DataLoader(ds, batch_size=self.batch_size, shuffle=ds.mode == "grid", drop_last=True,
                          num_workers=self.num_workers, persistent_workers=False,
                          pin_memory=torch.cuda.is_available())

    def _eval_loader(self, name):
        return DataLoader(self.datasets[name], batch_size=self.eval_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        return self._eval_loader("val")

    def test_dataloader(self):
        return self._eval_loader("test")

    def predict_dataloader(self):
        return self._eval_loader("test")

#!/usr/bin/env python3
"""ABC2027 Test 3: four matched TCN runs, with standard/dense inference.

Run with the existing server Python environment. No installation is performed.
The dataset and previous experiment directories are opened read-only.
Each invocation requires a new output directory. Run --help for options.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
import types
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Do not create bytecode files in the dataset's supplied evaluator directory.
sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import h5py
import numpy as np
import torch
from scipy import signal
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    rate_hz: float = 50.0
    radars: int = 3
    bins: int = 368
    background_tau_s: float = 3.0
    epochs: int = 20
    steps_per_epoch: int = 250
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 2027
    crop_frames: int = 800
    dilations: tuple = (1, 2, 4, 8, 16, 32, 64)
    hidden: int = 128
    train_hop: int = 80
    event_sample_prob: float = 0.5
    max_eligible_duration_s: float = 5.0
    ignore_margin_s: float = 2.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    loc_weight: float = 1.0
    max_grad_norm: float = 5.0
    norm_blocks_per_stream: int = 16
    norm_block_frames: int = 64
    norm_floor_percentile: float = 5.0
    inference_batch_size: int = 4
    local_peak_distance_frames: int = 10
    candidate_score_floor: float = 0.005
    score_grid: tuple = (.005, .01, .02, .03, .05, .075, .1, .15,
                         .2, .3, .4, .5, .6, .7, .8, .9)
    nms_grid: tuple = (.1, .3, .5, .7)
    fixed_score: float = .8
    fixed_nms: float = .1

    @property
    def receptive_field_frames(self):
        return 1 + 4 * sum(self.dilations)

    @property
    def context_frames(self):
        return (self.receptive_field_frames - 1) // 2

    @property
    def core_frames(self):
        return self.crop_frames - 2 * self.context_frames


def jdump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_true(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def inside(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def discover(root, split):
    folder = root / "streams" / split
    found = {}
    for path in sorted(folder.glob("*.h5")):
        suffix = path.stem.removeprefix("stream_").removeprefix(split + "_")
        sid = f"{split}_{suffix}"
        if sid in found:
            raise ValueError(f"Duplicate stream identifier: {sid}")
        found[sid] = path
    if not found:
        raise FileNotFoundError(f"No .h5 streams in {folder}")
    return found


def inspect_streams(streams, cfg):
    times, records = {}, []
    for sid, path in streams.items():
        with h5py.File(path, "r") as handle:
            shape = handle["cir"].shape
            if len(shape) != 3 or shape[0] != cfg.radars or shape[2] != cfg.bins:
                raise ValueError(f"{sid}: expected (3,T,368), received {shape}")
            if np.issubdtype(handle["cir"].dtype, np.complexfloating):
                raise TypeError("Expected the challenge's real float CIR representation")
            t = np.asarray(handle["t"][:], dtype=np.float64)
            if t.ndim != 1 or len(t) != shape[1] or len(t) < 2:
                raise ValueError(f"{sid}: timestamps and CIR length disagree")
            if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
                raise ValueError(f"{sid}: invalid timestamps")
            if abs(t[0]) > 1e-6 or not np.allclose(np.diff(t), 1 / cfg.rate_hz, atol=1e-6, rtol=1e-4):
                raise ValueError(f"{sid}: expected the published uniform 50 Hz timeline starting at zero")
            times[sid] = t
            info = path.stat()
            records.append(dict(stream_id=sid, path=str(path), shape=list(shape),
                                bytes=info.st_size, mtime_ns=info.st_mtime_ns,
                                duration_s=len(t) / cfg.rate_hz))
    return times, records


def load_events(root):
    with (root / "streams_events.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["t_start"], row["t_end"] = float(row["t_start"]), float(row["t_end"])
        if not np.isfinite([row["t_start"], row["t_end"]]).all():
            raise ValueError("Non-finite annotated event")
        row["flagged"] = (is_true(row.get("ignore", "")) or
                          is_true(row.get("needs_review", "")) or
                          str(row.get("quality", "")).strip().lower() == "low")
    return rows


def build_targets(times, rows, cfg):
    """Train only. Mask flagged zones; evaluable positives take precedence."""
    targets, eligible = {}, []
    by_sid = {sid: [] for sid in times}
    for row in rows:
        if row["stream_id"] in by_sid:
            by_sid[row["stream_id"]].append(row)
    for sid, t in times.items():
        count = len(t)
        y, valid, left, right = (np.zeros(count, dtype=np.float32),
                                 np.ones(count, dtype=np.float32),
                                 np.zeros(count, dtype=np.float32),
                                 np.zeros(count, dtype=np.float32))
        good = []
        for row in by_sid[sid]:
            duration = row["t_end"] - row["t_start"]
            ok = not row["flagged"] and 0 < duration <= cfg.max_eligible_duration_s
            if ok:
                good.append(row)
            else:
                margin = cfg.ignore_margin_s if row["flagged"] else 0.0
                a = np.searchsorted(t, row["t_start"] - margin, side="left")
                b = np.searchsorted(t, row["t_end"] + margin, side="right")
                valid[a:b] = 0
        for row in good:
            a = int(np.searchsorted(t, row["t_start"], side="left"))
            b = int(np.searchsorted(t, row["t_end"], side="right"))
            if a >= b:
                continue
            y[a:b] = valid[a:b] = 1
            left[a:b] = np.maximum(0, t[a:b] - row["t_start"])
            right[a:b] = np.maximum(0, row["t_end"] - t[a:b])
            eligible.append(dict(stream_id=sid, first=a, last=b - 1,
                                 center=int(np.searchsorted(t, (row["t_start"] + row["t_end"]) / 2))))
        targets[sid] = dict(y=y, valid=valid, left=left, right=right)
    if not eligible:
        raise ValueError("No eligible training events; check stream IDs and event flags")
    return targets, eligible


def prepare_ema(streams, cache_dir, cfg):
    """Continuous per-stream filter; state is never reset at crop boundaries."""
    paths = {}
    beta = math.exp(-1 / (cfg.rate_hz * cfg.background_tau_s))
    for sid, source in streams.items():
        destination = cache_dir / f"{sid}.npy"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to reuse a cache from an earlier run: {destination}")
        print(f"[prepare EMA] {sid}", flush=True)
        temporary = destination.with_suffix(".partial.npy")
        with h5py.File(source, "r") as handle:
            cir = handle["cir"]
            array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=cir.shape)
            first = np.asarray(cir[:, 0:1, :], dtype=np.float64)
            state = -beta * first
            for start in range(0, cir.shape[1], 4096):
                stop = min(cir.shape[1], start + 4096)
                block = np.asarray(cir[:, start:stop, :], dtype=np.float64)
                if not np.isfinite(block).all():
                    raise ValueError(f"Non-finite CIR in {sid}")
                residual, state = signal.lfilter([beta, -beta], [1, -beta], block, axis=1, zi=state)
                array[:, start:stop, :] = residual.astype(np.float32)
            array.flush()
            del array
        # Verify full payload length, not merely the NPY header.
        with temporary.open("rb") as handle:
            version = np.lib.format.read_magic(handle)
            reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                      else np.lib.format.read_array_header_2_0)
            shape, _, dtype = reader(handle)
            expected = handle.tell() + math.prod(shape) * dtype.itemsize
        if temporary.stat().st_size != expected:
            raise IOError(f"Incomplete newly generated cache: {temporary}")
        os.replace(temporary, destination)
        paths[sid] = destination
    return paths


class Store:
    """Raw HDF5 or residual NPY access; always opens source arrays read-only."""
    def __init__(self, paths, mode):
        self.paths, self.mode = paths, mode
        self.handles, self.arrays = {}, {}

    def get(self, sid):
        if sid not in self.arrays:
            if self.mode == "raw":
                self.handles[sid] = h5py.File(self.paths[sid], "r")
                self.arrays[sid] = self.handles[sid]["cir"]
            else:
                self.arrays[sid] = np.load(self.paths[sid], mmap_mode="r", allow_pickle=False)
        return self.arrays[sid]

    def block(self, sid, start, length):
        x = self.get(sid)
        stop = start + length
        lo, hi = max(0, start), min(x.shape[1], stop)
        if hi <= lo:
            raise ValueError("Requested crop has no real frames")
        arr = np.asarray(x[:, lo:hi, :], dtype=np.float32)
        if start < 0 or stop > x.shape[1]:
            arr = np.pad(arr, ((0, 0), (max(0, -start), max(0, stop - x.shape[1])), (0, 0)), mode="edge")
        if arr.shape[1] != length or not np.isfinite(arr).all():
            raise ValueError(f"Invalid CIR crop in {sid}")
        return arr

    def close(self):
        self.arrays.clear()
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def fit_normalization(store, train_times, cfg, out):
    total = 0
    sum_x = np.zeros((cfg.radars, cfg.bins), dtype=np.float64)
    sum_sq = np.zeros_like(sum_x)
    coordinates = {}
    for sid, t in train_times.items():
        width = min(len(t), cfg.norm_block_frames)
        starts = np.linspace(0, len(t) - width, cfg.norm_blocks_per_stream).round().astype(int)
        coordinates[sid] = starts.tolist()
        for start in starts:
            x = store.block(sid, int(start), width).astype(np.float64)
            sum_x += x.sum(axis=1)
            sum_sq += np.square(x).sum(axis=1)
            total += width
    mean = sum_x / total
    std = np.sqrt(np.maximum(sum_sq / total - mean * mean, 0))
    floor = np.maximum(np.percentile(std, cfg.norm_floor_percentile, axis=1), 1e-8)
    scale = np.maximum(std, floor[:, None])
    mean, scale = mean.astype(np.float32), scale.astype(np.float32)
    np.savez(out / "normalization.npz", mean=mean, scale=scale)
    jdump(out / "normalization_summary.json", dict(
        input_mode=store.mode, split="train", sampled_frames_per_radar=total,
        sampled_block_starts=coordinates, floor=floor.tolist(),
        mean_min=float(mean.min()), mean_max=float(mean.max()),
        scale_min=float(scale.min()), scale_max=float(scale.max())))
    return mean, scale


class CropSampler:
    def __init__(self, store, times, targets, events, mean, scale, cfg, sampling):
        self.store, self.times, self.targets, self.events = store, times, targets, events
        self.mean, self.scale, self.cfg, self.sampling = mean, scale, cfg, sampling
        self.rng = np.random.default_rng(cfg.seed)
        self.sids = sorted(times)
        self.draws = self.event_draws = 0
        self.exposures = np.zeros(len(events), dtype=np.int64)
        self.positions = set()
        self.sequence_digest = hashlib.sha256()
        self.event_starts = []
        for ev in events:
            # Input start = retained-core start minus left context. Keep the
            # complete selected event inside the supervised core when possible.
            low = max(0, ev["last"] - cfg.core_frames + 1)
            high = min(ev["first"], len(times[ev["stream_id"]]) - 1)
            grid = np.arange(math.ceil(low / cfg.train_hop) * cfg.train_hop,
                             high + 1, cfg.train_hop, dtype=np.int64)
            if not len(grid):
                grid = np.array([min(high, max(low, ev["center"] - cfg.core_frames // 2))])
            self.event_starts.append(grid)

    def extract(self, sid, start):
        cfg = self.cfg
        x = self.store.block(sid, start, cfg.crop_frames)
        x = (x - self.mean[:, None, :]) / self.scale[:, None, :]
        indices = start + cfg.context_frames + np.arange(cfg.core_frames)
        inside_mask = (indices >= 0) & (indices < len(self.times[sid]))
        clipped = np.clip(indices, 0, len(self.times[sid]) - 1)
        tar = self.targets[sid]
        y = tar["y"][clipped].copy()
        valid = tar["valid"][clipped].copy() * inside_mask
        left, right = tar["left"][clipped].copy(), tar["right"][clipped].copy()
        y[~inside_mask] = left[~inside_mask] = right[~inside_mask] = 0
        return x, y, valid, left, right

    def sample_one(self):
        cfg = self.cfg
        if self.rng.random() < cfg.event_sample_prob:
            event_index = int(self.rng.integers(len(self.events)))
            event = self.events[event_index]
            sid = event["stream_id"]
            if self.sampling == "random":
                start = event["center"] - cfg.crop_frames // 2 + int(self.rng.integers(-20, 21))
            else:
                choices = self.event_starts[event_index]
                core_start = int(choices[int(self.rng.integers(len(choices)))])
                start = core_start - cfg.context_frames
            self.exposures[event_index] += 1
            self.event_draws += 1
        else:
            sid = self.sids[int(self.rng.integers(len(self.sids)))]
            if self.sampling == "random":
                start = int(self.rng.integers(len(self.times[sid]))) - cfg.crop_frames // 2
            else:
                count = (len(self.times[sid]) - 1) // cfg.train_hop + 1
                start = int(self.rng.integers(count)) * cfg.train_hop - cfg.context_frames
        self.positions.add((sid, start))
        self.sequence_digest.update(f"{sid}:{start}\n".encode())
        self.draws += 1
        return self.extract(sid, start)

    def batch(self):
        items = [self.sample_one() for _ in range(self.cfg.batch_size)]
        return [torch.from_numpy(np.stack(x)) for x in zip(*items)]

    def summary(self):
        return dict(sampling=self.sampling, draws=self.draws,
                    event_targeted_draws=self.event_draws,
                    unique_crop_positions=len(self.positions),
                    crop_sequence_sha256=self.sequence_digest.hexdigest(),
                    event_exposures=self.exposures.tolist(),
                    overlapping_grid_hop=self.cfg.train_hop if self.sampling == "overlap" else None,
                    note="Uniform-time draws can also contain falls; targeted fraction is not frame prevalence.")


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x + y)


class AnchorTCN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.radar_mlp = nn.Sequential(nn.Linear(cfg.bins, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.fuse = nn.Linear(cfg.radars * 64, cfg.hidden)
        self.fuse_norm = nn.LayerNorm(cfg.hidden)
        self.blocks = nn.ModuleList([ResidualTCNBlock(cfg.hidden, d) for d in cfg.dilations])
        self.conf_head = nn.Conv1d(cfg.hidden, 1, 1)
        self.dist_head = nn.Conv1d(cfg.hidden, 2, 1)
        nn.init.normal_(self.conf_head.weight, mean=0, std=.005)
        nn.init.constant_(self.conf_head.bias, math.log(.01 / .99))
        nn.init.constant_(self.dist_head.bias, math.log(math.expm1(1)))

    def forward(self, x):
        batch, radars, frames, bins = x.shape
        z = self.radar_mlp(x.reshape(batch * radars * frames, bins)).reshape(batch, radars, frames, 64)
        z = z.permute(0, 2, 1, 3).reshape(batch, frames, radars * 64)
        z = F.relu(self.fuse_norm(self.fuse(z))).transpose(1, 2)
        for block in self.blocks:
            z = block(z)
        return self.conf_head(z).squeeze(1), F.softplus(self.dist_head(z)).transpose(1, 2)


def detection_loss(logits, dist, y, valid, left, right, cfg):
    probability = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = probability * y + (1 - probability) * (1 - y)
    alpha = cfg.focal_alpha * y + (1 - cfg.focal_alpha) * (1 - y)
    positive = y * valid
    classification = (alpha * (1 - pt).pow(cfg.focal_gamma) * ce * valid).sum() / positive.sum().clamp_min(1)
    intersection = torch.minimum(dist[..., 0], left) + torch.minimum(dist[..., 1], right)
    union = torch.maximum(dist[..., 0], left) + torch.maximum(dist[..., 1], right)
    localization = ((1 - intersection / union.clamp_min(1e-6)) * positive).sum() / positive.sum().clamp_min(1)
    return classification + cfg.loc_weight * localization, classification, localization


def amp_context(device):
    return torch.autocast(device_type="cuda", enabled=True) if device.type == "cuda" else contextlib.nullcontext()


def train_model(store, times, targets, events, mean, scale, cfg, sampling, device, out):
    seed_everything(cfg.seed)
    model = AnchorTCN(cfg).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params != 774339:
        raise ValueError(f"Unexpected architecture: {params} parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    sampler = CropSampler(store, times, targets, events, mean, scale, cfg, sampling)
    rows, started = [], time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    positive_frames, valid_frames = 0, 0
    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for step in range(cfg.steps_per_epoch):
            x, y, valid, left, right = [a.to(device) for a in sampler.batch()]
            positive_frames += int((y * valid).sum().item())
            valid_frames += int(valid.sum().item())
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                logits, dist = model(x)
                a, b = cfg.context_frames, cfg.context_frames + cfg.core_frames
                loss, cls, loc = detection_loss(logits[:, a:b], dist[:, a:b], y, valid, left, right, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}, step {step + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            if device.type != "cuda" and not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite gradients; run stopped without modifying sources")
            # On CUDA, GradScaler has recorded overflow during unscale_ and
            # skips that optimizer step safely. Early FP16 scale calibration
            # must not abort an otherwise valid experiment.
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            applied_update = scaler.get_scale() >= old_scale
            row = dict(epoch=epoch + 1, step=step + 1, loss=float(loss.detach().cpu()),
                       classification=float(cls.detach().cpu()), localization=float(loc.detach().cpu()),
                       optimizer_update_applied=bool(applied_update))
            rows.append(row)
            losses.append(row["loss"])
        write_csv(out / "training_log.csv", rows)
        print(f"  epoch {epoch + 1:02d}/{cfg.epochs}: loss={np.mean(losses):.5f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    summary = dict(parameter_count=params, global_steps=len(rows),
                   optimizer_updates=sum(r["optimizer_update_applied"] for r in rows),
                   amp_skipped_steps=sum(not r["optimizer_update_applied"] for r in rows),
                   training_seconds=time.perf_counter() - started,
                   positive_valid_frame_fraction=positive_frames / max(valid_frames, 1),
                   peak_gpu_allocated_gib=(torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0),
                   sampler=sampler.summary())
    # Save CPU tensors to make the checkpoint portable; reload before inference.
    checkpoint = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                      mean=mean, scale=scale, config=asdict(cfg), sampling=sampling, input_mode=store.mode)
    torch.save(checkpoint, out / "last.pt")
    jdump(out / "training_summary.json", summary)
    del optimizer, scaler, model
    model = AnchorTCN(cfg)
    # The checkpoint was created by this script in this new run directory.
    restored = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(restored["model"])
    model.to(device).eval()
    return model, summary


def infer_both(model, store, sid, times, mean, scale, cfg, device):
    """One dense pass yields both original tiling and averaged overlapping cores.

    Standard: 800 input frames, hop 292, retain the central 292 frames.
    Dense: same input/core, hop 146, average predictions at shared timestamps.
    GroupNorm is intentionally preserved, so shifted windows can disagree.
    """
    count, core, ctx = len(times), cfg.core_frames, cfg.context_frames
    hop = core // 2
    if core % 2:
        raise ValueError("This comparison expects an even retained-core size")
    sums = np.zeros((count, 3), dtype=np.float64)
    weights = np.zeros(count, dtype=np.uint32)
    standard = np.full((count, 3), np.nan, dtype=np.float32)
    starts = list(range(0, count, hop))
    difference_sum, compared = 0.0, 0
    with torch.inference_mode():
        for batch_start in range(0, len(starts), cfg.inference_batch_size):
            selected = starts[batch_start:batch_start + cfg.inference_batch_size]
            windows = [(store.block(sid, start - ctx, cfg.crop_frames) - mean[:, None, :]) /
                       scale[:, None, :] for start in selected]
            x = torch.from_numpy(np.stack(windows)).to(device)
            with amp_context(device):
                logits, dist = model(x)
                probs = torch.sigmoid(logits[:, ctx:ctx + core]).float().cpu().numpy()
                distances = dist[:, ctx:ctx + core].float().cpu().numpy()
            for index, start in enumerate(selected):
                end = min(count, start + core)
                values = np.column_stack((probs[index, :end-start], distances[index, :end-start]))
                if not np.isfinite(values).all():
                    raise FloatingPointError(f"Non-finite inference output in {sid}")
                already = weights[start:end] > 0
                if already.any():
                    previous = sums[start:end][already, 0] / weights[start:end][already]
                    difference_sum += float(np.abs(previous - values[already, 0]).sum())
                    compared += int(already.sum())
                sums[start:end] += values
                weights[start:end] += 1
                if start % core == 0:
                    standard[start:end] = values
    if not weights.all() or not np.isfinite(standard).all():
        raise RuntimeError(f"Incomplete stream coverage: {sid}")
    dense = (sums / weights[:, None]).astype(np.float32)
    return {"standard": standard, "dense": dense}, dict(
        dense_core_hop=hop, standard_core_hop=core,
        mean_absolute_confidence_difference_between_shifted_views=(difference_sum / compared if compared else 0),
        minimum_views=int(weights.min()), maximum_views=int(weights.max()))


def decode(sid, times, values, cfg):
    confidence, left, right = values.T
    peaks, _ = signal.find_peaks(confidence, height=cfg.candidate_score_floor,
                                 distance=cfg.local_peak_distance_frames)
    proposals = []
    for index in peaks:
        start = max(float(times[0]), float(times[index] - left[index]))
        end = min(float(times[-1]), float(times[index] + right[index]))
        if end > start:
            proposals.append(dict(stream_id=sid, t_start=start, t_end=end,
                                  score=float(confidence[index]), t_alarm=None))
    return proposals


def temporal_nms(proposals, threshold):
    kept = []
    for sid in sorted({p["stream_id"] for p in proposals}):
        group = [p for p in proposals if p["stream_id"] == sid]
        scores = np.array([p["score"] for p in group])
        starts = np.array([p["t_start"] for p in group])
        ends = np.array([p["t_end"] for p in group])
        order = np.argsort(-scores, kind="stable")
        while len(order):
            index = int(order[0])
            kept.append(group[index])
            remaining = order[1:]
            intersection = np.maximum(0, np.minimum(ends[index], ends[remaining]) -
                                      np.maximum(starts[index], starts[remaining]))
            union = ends[index] - starts[index] + ends[remaining] - starts[remaining] - intersection
            overlap = intersection / np.maximum(union, 1e-12)
            order = remaining[overlap < threshold]
    return sorted(kept, key=lambda p: (p["stream_id"], p["t_start"]))


def load_official_evaluator(root):
    path = root / "src" / "abc2027" / "evaluate.py"
    if not path.is_file():
        raise FileNotFoundError(f"Expected the supplied official evaluator at {path}")
    source = path.read_bytes()
    module = types.ModuleType("_abc2027_official_evaluator")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    # Execute the supplied evaluator in memory; do not write into the dataset.
    exec(compile(source, str(path), "exec"), module.__dict__)
    for name in ("evaluate", "load_ground_truth", "validate_predictions"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"Official evaluator API is missing {name}; no training was started")
    probe = module.evaluate([], [], {"val_probe": 1.0})
    if "ranking_score" not in probe or "by_theta" not in probe:
        raise RuntimeError("Unexpected official report schema; no training was started")
    return module, dict(path=str(path), sha256=hashlib.sha256(source).hexdigest())


def metric_row(arm, mode, report, score, nms, train_seconds, inference_seconds):
    primary = report["primary"]
    return dict(arm=arm, inference=mode,
                F1_mean=report["ranking_score"]["f1_mean"],
                F1_01=report["by_theta"]["0.1"]["f1"],
                F1_03=report["by_theta"]["0.3"]["f1"],
                F1_05=report["by_theta"]["0.5"]["f1"],
                F1_07=report["by_theta"]["0.7"]["f1"],
                TP_03=primary["tp"], FP_03=primary["fp"], FN_03=primary["fn"],
                false_alarms_per_hour=primary["false_alarms_per_hour"],
                median_start_error_s=primary["start_error_median_abs"],
                median_end_error_s=primary["end_error_median_abs"],
                mean_matched_tIoU=primary["mean_tiou"],
                score_threshold=score, nms_threshold=nms,
                training_seconds=train_seconds, joint_inference_seconds=inference_seconds)


def select_key(report):
    ranking = report["ranking_score"]
    start = ranking.get("start_error_median_abs")
    alarms = ranking.get("false_alarms_per_hour")
    return (float(ranking["f1_mean"]), -(float(start) if start is not None else math.inf),
            -(float(alarms) if alarms is not None else math.inf))


def evaluate_proposals(evaluator, ground, durations, proposals, cfg, out):
    out.mkdir(parents=True, exist_ok=True)
    # Each NMS result is computed once and shared by every confidence threshold.
    nms_cache = {n: temporal_nms(proposals, n) for n in cfg.nms_grid}
    def score_at(score, nms):
        predictions = [p for p in nms_cache[nms] if p["score"] >= score]
        return predictions, evaluator.evaluate(ground, predictions, durations)

    def save(name, predictions, report):
        path = out / f"{name}_submission.csv"
        write_csv(path, predictions, ["stream_id", "t_start", "t_end", "score"])
        # Use the organizer's format and range checks before recording any result.
        checked = evaluator.validate_predictions(path, durations)
        verified = evaluator.evaluate(ground, checked, durations)
        if select_key(verified) != select_key(report):
            raise RuntimeError("Submission round-trip changed its official score")
        jdump(out / f"{name}_official_report.json", verified)
        return verified

    fixed_pred, fixed = score_at(cfg.fixed_score, cfg.fixed_nms)
    fixed = save("fixed", fixed_pred, fixed)
    default_pred, default = score_at(.5, .5)
    save("default", default_pred, default)
    sweep, best = [], None
    for nms in cfg.nms_grid:
        for score in cfg.score_grid:
            predictions, report = score_at(score, nms)
            sweep.append(dict(score_threshold=score, nms_threshold=nms,
                              predictions=len(predictions),
                              F1_mean=report["ranking_score"]["f1_mean"],
                              F1_05=report["by_theta"]["0.5"]["f1"],
                              F1_07=report["by_theta"]["0.7"]["f1"]))
            if best is None or select_key(report) > select_key(best["report"]):
                best = dict(score=score, nms=nms, report=report, predictions=predictions)
    best["report"] = save("tuned", best["predictions"], best["report"])
    write_csv(out / "postprocessing_sweep.csv", sweep)
    jdump(out / "selection.json", dict(
        fixed=dict(score=cfg.fixed_score, nms=cfg.fixed_nms),
        tuned=dict(score=best["score"], nms=best["nms"]),
        note="Tuned using validation labels; development results, not held-out test estimates. Equal official keys keep the first setting."))
    return fixed, best


def format_value(value):
    return "n/a" if value is None else f"{value:.3f}"


def write_comparison(out, fixed_rows, tuned_rows, complete):
    write_csv(out / "comparison_fixed.csv", fixed_rows)
    write_csv(out / "comparison_tuned.csv", tuned_rows)
    def table(rows):
        lines = ["| Run | Inference | F1 mean | F1 @ .5 | F1 @ .7 | Start error s | End error s | FA/hour |",
                 "|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            lines.append("| " + " | ".join([
                row["arm"], row["inference"], format_value(row["F1_mean"]),
                format_value(row["F1_05"]), format_value(row["F1_07"]),
                format_value(row["median_start_error_s"]), format_value(row["median_end_error_s"]),
                format_value(row["false_alarms_per_hour"])]) + " |")
        return lines
    baseline = fixed_rows[0]
    best = max(fixed_rows, key=lambda r: r["F1_mean"])
    strict = max(fixed_rows, key=lambda r: (r["F1_05"], r["F1_07"]))
    text = ["# ABC2027 Anchor-TCN comparison", "",
            f"Status: {'COMPLETE' if complete else 'PARTIAL'} ({len(fixed_rows)}/8 comparison rows).", "",
            "Four trained models; each evaluated using standard and denser inference.",
            "A_ema_random: corrected EMA control with the previous random crop sampling.",
            "B_raw_random: remove EMA only.",
            "C_ema_overlap: keep EMA, use denser training crop positions.",
            "D_raw_overlap: raw CIR and denser training crop positions.", "",
            "Standard inference tiles 292-frame output cores. Dense inference averages overlapping 292-frame cores with hop 146.",
            "All models retain the original 800-frame input and 774,339 parameters.", "",
            "## Main comparison: same fixed score 0.8 and NMS 0.1", "", *table(fixed_rows), "",
            "These settings were chosen before these runs, from the previous experiment.", "",
            "## Additional comparison: same validation sweep for every method", "", *table(tuned_rows), "",
            "Tuned results use validation labels and are development-set results.", "",
            "## What changed", "",
            f"Highest fixed-setting F1 mean so far: {best['arm']} / {best['inference']} = {best['F1_mean']:.4f} ",
            f"(difference from A_ema_random/standard: {best['F1_mean'] - baseline['F1_mean']:+.4f}).", "",
            f"Highest fixed-setting F1 @ tIoU .5 so far: {strict['arm']} / {strict['inference']} = {strict['F1_05']:.4f} ",
            f"(difference from control: {strict['F1_05'] - baseline['F1_05']:+.4f}).", "",
            "F1 @ .5 and .7 require stricter interval overlap. Mean matched tIoU and boundary errors are in the CSV files.",
            "Errors and mean matched tIoU are computed on each method's matched events; their match populations can differ.",
            "A larger classification score or more windows alone does not demonstrate better localization.", "",
            "## Presentation notes", "",
            "This is a one-seed controlled screen, not proof of improvement across subjects, rooms or repeated runs.",
            "The new control uses the corrected low-quality/ignore mask, so it need not reproduce the historical 0.672 exactly.",
            "GroupNorm is preserved in all arms. The 509-frame figure is the convolution-only receptive field; crop statistics span time.",
            "The model uses future context. Report event localization, not real-time alarm latency.",
            "Dense inference does not retrain the model. Its extra views test whether averaging reduces window-position sensitivity.",
            "Training grid positions are sampled on demand with the same update budget; they are not new independent recordings.",
            "A transformer, PCA, handcrafted motion detector and additional boundary loss are not part of this experiment.", "",
            "Share RESULTS_TO_SHARE.zip. It contains reports and training logs, and excludes caches and checkpoints.", ""]
    (out / "RESULTS.md").write_text("\n".join(text), encoding="utf-8")


def share_results(out):
    target = out / "RESULTS_TO_SHARE.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if not path.is_file() or path == target or "ema_cache" in path.parts:
                continue
            if path.suffix.lower() in {".json", ".csv", ".md", ".txt", ".log", ".py"}:
                archive.write(path, arcname=str(path.relative_to(out)))
    return target


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.log.flush()

    def flush(self):
        self.console.flush()
        self.log.flush()


def setup_device(requested, memory_fraction):
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment. Use the server's existing CUDA Python environment.")
        # CUDA memory APIs require an explicit index on the server's PyTorch.
        # Resolve bare 'cuda' before passing this device to any CUDA API.
        index = device.index if device.index is not None else torch.cuda.current_device()
        device = torch.device("cuda", index)
        torch.cuda.set_per_process_memory_fraction(memory_fraction, index)
    elif device.type != "cpu":
        raise ValueError("Use --device cuda, cuda:N or cpu")
    return device


def run(args, cfg, out):
    root = args.root.expanduser().resolve()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = setup_device(args.device, args.gpu_memory_fraction)
    if device.type == "cuda":
        print(f"[device] {device}: {torch.cuda.get_device_name(device)}", flush=True)
    else:
        print("[device] CPU (intended for a small smoke test; full training is slower)", flush=True)
    print(f"[output] {out}", flush=True)
    evaluator, evaluator_info = load_official_evaluator(root)
    train_paths, val_paths = discover(root, "train"), discover(root, "val")
    streams = {**train_paths, **val_paths}
    times, records = inspect_streams(streams, cfg)
    train_times = {sid: times[sid] for sid in train_paths}
    val_times = {sid: times[sid] for sid in val_paths}
    durations = {sid: len(t) / cfg.rate_hz for sid, t in val_times.items()}
    rows = load_events(root)
    unknown = {r["stream_id"] for r in rows if r["stream_id"].startswith(("train_", "val_"))} - set(streams)
    if unknown:
        raise ValueError(f"Event table references missing streams: {sorted(unknown)}")
    targets, eligible = build_targets(train_times, rows, cfg)
    ground = evaluator.load_ground_truth(root / "streams_events.csv", ignore_flagged=True)
    evaluable_val = [r for r in ground if r["stream_id"] in val_times and not r["ignore"]]
    if not evaluable_val:
        raise ValueError("No evaluable validation falls; check stream/event ID mapping")
    cache_bytes = sum(cfg.radars * len(t) * cfg.bins * 4 + 128 for t in times.values())
    required = cache_bytes + max(512 * 1024**2, int(.1 * cache_bytes))
    free = shutil.disk_usage(out).free
    if free < required:
        raise OSError(f"New EMA cache and outputs need about {required/1024**3:.2f} GiB free; available {free/1024**3:.2f} GiB")
    provenance = dict(config=asdict(cfg), dataset=str(root), output=str(out),
                      started_utc=datetime.now(timezone.utc).isoformat(),
                      python=sys.version, torch=torch.__version__, numpy=np.__version__,
                      h5py=h5py.__version__, device=str(device), threads=args.threads,
                      evaluator=evaluator_info,
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      event_table_sha256=hashlib.sha256((root / "streams_events.csv").read_bytes()).hexdigest(),
                      note="Four equal-budget, matched-seed training arms; no architecture/loss change.")
    jdump(out / "run_config.json", provenance)
    jdump(out / "dataset_summary.json", dict(
        train_streams=len(train_paths), validation_streams=len(val_paths),
        eligible_training_events=len(eligible), evaluated_validation_events=len(evaluable_val),
        input_frames=cfg.crop_frames, convolution_only_receptive_field=cfg.receptive_field_frames,
        retained_core_frames=cfg.core_frames, training_grid_hop=cfg.train_hop,
        training_grid_input_overlap=1 - cfg.train_hop / cfg.crop_frames,
        standard_inference_input_overlap=1 - cfg.core_frames / cfg.crop_frames,
        dense_inference_input_overlap=1 - (cfg.core_frames // 2) / cfg.crop_frames,
        ignored_training_zone_margin_s=cfg.ignore_margin_s,
        streams=records))
    print(f"[data] {len(train_paths)} train streams; {len(val_paths)} validation streams; "
          f"{len(eligible)} eligible training falls; {len(evaluable_val)} evaluated validation falls", flush=True)
    print(f"[storage] fresh EMA cache about {cache_bytes / 1024**3:.2f} GiB; raw CIR is read directly", flush=True)
    jdump(out / "progress.json", dict(stage="preparing fresh EMA cache"))
    ema_paths = prepare_ema(streams, out / "ema_cache", cfg)
    fixed_rows, tuned_rows, digests = [], [], {}
    arms = [("A_ema_random", "ema", "random"), ("B_raw_random", "raw", "random"),
            ("C_ema_overlap", "ema", "overlap"), ("D_raw_overlap", "raw", "overlap")]
    overall = time.perf_counter()
    for arm, input_mode, sampling in arms:
        arm_out = out / arm
        arm_out.mkdir()
        print(f"\n[run] {arm}: input={input_mode}, training sampling={sampling}", flush=True)
        jdump(out / "progress.json", dict(stage="training", arm=arm, completed_rows=len(fixed_rows)))
        store = Store(streams if input_mode == "raw" else ema_paths, input_mode)
        try:
            mean, scale = fit_normalization(store, train_times, cfg, arm_out)
            model, summary = train_model(store, train_times, targets, eligible, mean, scale,
                                         cfg, sampling, device, arm_out)
            digest = summary["sampler"]["crop_sequence_sha256"]
            if sampling in digests and digest != digests[sampling]:
                raise RuntimeError("Raw/EMA crop schedules were not identical; comparison stopped")
            digests[sampling] = digest
            proposals = {"standard": [], "dense": []}
            diagnostics = {}
            infer_started = time.perf_counter()
            jdump(out / "progress.json", dict(stage="inference", arm=arm, completed_rows=len(fixed_rows)))
            for sid, t in val_times.items():
                print(f"  [infer standard + dense] {sid}", flush=True)
                outputs, checks = infer_both(model, store, sid, t, mean, scale, cfg, device)
                diagnostics[sid] = checks
                for mode, values in outputs.items():
                    mode_out = arm_out / mode
                    mode_out.mkdir(exist_ok=True)
                    np.savez_compressed(mode_out / f"{sid}_frame_predictions.npz", t=t,
                                        confidence=values[:, 0], d_start=values[:, 1], d_end=values[:, 2])
                    proposals[mode].extend(decode(sid, t, values, cfg))
            inference_seconds = time.perf_counter() - infer_started
            jdump(arm_out / "inference_summary.json", dict(joint_inference_seconds=inference_seconds,
                                                          shifted_window_checks=diagnostics))
            for mode in ("standard", "dense"):
                jdump(out / "progress.json", dict(stage="official evaluation", arm=arm, inference=mode))
                mode_out = arm_out / mode
                jdump(mode_out / "proposals.json", proposals[mode])
                fixed, best = evaluate_proposals(evaluator, ground, durations, proposals[mode], cfg, mode_out)
                # Also retain per-stream official results at the frozen operating point.
                checked = evaluator.validate_predictions(mode_out / "fixed_submission.csv", durations)
                per_stream = {sid: evaluator.evaluate(ground, checked, {sid: seconds})
                              for sid, seconds in durations.items()}
                jdump(mode_out / "fixed_per_stream_reports.json", per_stream)
                fixed_rows.append(metric_row(arm, mode, fixed, cfg.fixed_score, cfg.fixed_nms,
                                             summary["training_seconds"], inference_seconds))
                tuned_rows.append(metric_row(arm, mode, best["report"], best["score"], best["nms"],
                                             summary["training_seconds"], inference_seconds))
                print(f"  [{mode}] fixed F1mean={fixed_rows[-1]['F1_mean']:.4f}, "
                      f"F1@.5={fixed_rows[-1]['F1_05']:.4f}, "
                      f"F1@.7={fixed_rows[-1]['F1_07']:.4f}; "
                      f"tuned F1mean={tuned_rows[-1]['F1_mean']:.4f}", flush=True)
                write_comparison(out, fixed_rows, tuned_rows, complete=False)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        finally:
            store.close()
    write_comparison(out, fixed_rows, tuned_rows, complete=True)
    jdump(out / "progress.json", dict(stage="COMPLETE", completed_rows=8,
                                      experiment_seconds=time.perf_counter() - overall))
    package = share_results(out)
    print("\nCOMPLETE: four models, eight evaluation rows.", flush=True)
    print(f"Read: {out / 'RESULTS.md'}", flush=True)
    print(f"Send back: {package}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("/home/abc2027/dataset/abc2027_uwb_train_v1"),
                        help="Existing challenge dataset; opened read-only")
    parser.add_argument("--out", type=Path, help="NEW output directory; must not already exist")
    parser.add_argument("--device", default="cuda", help="cuda (default), cuda:N or cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-hop", type=int, default=80, help="80 = 90%% input overlap; 40 = 95%%")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.5,
                        help="Upper allocation limit for this process on the selected GPU (default .5)")
    args = parser.parse_args(argv)
    if any(x <= 0 for x in (args.epochs, args.steps_per_epoch, args.batch_size, args.train_hop, args.threads)):
        parser.error("Epochs, steps, batch size, hop and thread count must be positive")
    if args.train_hop > 292:
        parser.error("Training hop must be <= 292 to cover the retained output core")
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error("GPU memory fraction must lie in (0, 1]")
    return args


def main(argv=None):
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (args.out or Path(__file__).resolve().parent / f"anchor_tcn_comparison_{stamp}_{os.getpid()}").expanduser().resolve()
    if inside(out, root) or inside(root, out):
        raise ValueError("Output must be outside the dataset and must not be its parent")
    if out.exists():
        raise FileExistsError(f"Output already exists. Choose a new --out directory: {out}")
    # Creating a unique directory is the only write before preflight checks.
    out.mkdir(parents=True, exist_ok=False)
    cfg = Config(epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
                 batch_size=args.batch_size, train_hop=args.train_hop, seed=args.seed)
    shutil.copyfile(Path(__file__).resolve(), out / "experiment_source.py")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    log_path = out / "console.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        sys.stdout, sys.stderr = Tee(original_stdout, log), Tee(original_stderr, log)
        try:
            run(args, cfg, out)
        except BaseException as error:
            jdump(out / "progress.json", dict(stage="FAILED", error=str(error), error_type=type(error).__name__))
            (out / "FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"\nSTOPPED: {error}\nDetails saved in {out / 'FAILED.txt'}", file=sys.stderr, flush=True)
            share_results(out)
            raise
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    main()

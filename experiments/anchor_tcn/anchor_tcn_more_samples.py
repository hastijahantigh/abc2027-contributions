#!/usr/bin/env python3
"""ABC2027: generate all overlapping training windows; visit each once per epoch.

Standalone server script. Retains EMA and the original Anchor-TCN. Compares
training hops 292, 80 and 40 with the same number of full dataset passes.
Reads the dataset only; writes into a fresh experiment directory.
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

sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import h5py
import numpy as np
import torch
from scipy import signal
from torch import nn
from torch.nn import functional as F

sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

@dataclass
class Config:
    rate_hz: float = 50.0
    radars: int = 3
    bins: int = 368
    background_tau_s: float = 3.0
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 2027
    crop_frames: int = 800
    dilations: tuple = (1, 2, 4, 8, 16, 32, 64)
    hidden: int = 128
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
    score_grid: tuple = (.005,) + tuple(i / 100 for i in range(1, 100))
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


def build_window_manifest(times, targets, hop, cfg):
    """Enumerate ALL grid positions with usable supervised frames.

    The grid is defined on the retained 292-frame core; input starts 254 frames
    earlier. Windows at true stream edges use the established padding policy.
    A window with no valid training labels is skipped and counted explicitly.
    """
    manifest, records = [], []
    for sid in sorted(times):
        size = len(times[sid])
        valid = targets[sid]["valid"] > .5
        positive = (targets[sid]["y"] > .5) & valid
        valid_sum = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
        positive_sum = np.concatenate(([0], np.cumsum(positive, dtype=np.int64)))
        all_count, skipped = 0, 0
        coverage = np.zeros(size, dtype=np.int32)
        for start in range(0, size, hop):
            all_count += 1
            end = min(size, start + cfg.core_frames)
            valid_count = int(valid_sum[end] - valid_sum[start])
            if not valid_count:
                skipped += 1
                continue
            manifest.append(dict(sample_id=len(manifest), stream_id=sid,
                core_start_frame=start, input_start_frame=start - cfg.context_frames,
                valid_frames=valid_count,
                positive_frames=int(positive_sum[end] - positive_sum[start])))
            coverage[start:end] += 1
        if (coverage[valid] == 0).any():
            raise RuntimeError(f"Some valid training references are uncovered: {sid}, hop={hop}")
        records.append(dict(stream_id=sid, all_grid_positions=all_count,
            wholly_masked_positions_skipped=skipped, training_windows=all_count-skipped,
            min_valid_reference_coverage=int(coverage[valid].min()) if valid.any() else 0,
            max_valid_reference_coverage=int(coverage[valid].max()) if valid.any() else 0))
    if not manifest:
        raise ValueError(f"No trainable windows for hop {hop}")
    summary = dict(hop_frames=hop, hop_seconds=hop / cfg.rate_hz,
        input_window_frames=cfg.crop_frames,
        input_overlap_fraction=1 - hop / cfg.crop_frames,
        retained_core_overlap_fraction=1 - hop / cfg.core_frames,
        training_windows=len(manifest),
        positive_windows=sum(row["positive_frames"] > 0 for row in manifest),
        background_windows=sum(row["positive_frames"] == 0 for row in manifest),
        valid_reference_exposures_per_epoch=sum(row["valid_frames"] for row in manifest),
        positive_reference_exposures_per_epoch=sum(row["positive_frames"] for row in manifest),
        steps_per_full_epoch=math.ceil(len(manifest) / cfg.batch_size),
        full_epochs=cfg.epochs, total_training_draws=len(manifest) * cfg.epochs,
        total_training_batches=math.ceil(len(manifest) / cfg.batch_size) * cfg.epochs,
        streams=records)
    summary["positive_reference_fraction"] = summary["positive_reference_exposures_per_epoch"] / summary["valid_reference_exposures_per_epoch"]
    return manifest, summary


class FullWindowDataset:
    def __init__(self, store, times, targets, mean, scale, manifest, cfg):
        self.store, self.times, self.targets = store, times, targets
        self.mean, self.scale, self.manifest, self.cfg = mean, scale, manifest, cfg

    def __len__(self):
        return len(self.manifest)

    def get(self, sample_id):
        row, cfg = self.manifest[int(sample_id)], self.cfg
        sid, start = row["stream_id"], row["input_start_frame"]
        x = (self.store.block(sid, start, cfg.crop_frames) - self.mean[:, None, :]) / self.scale[:, None, :]
        indices = row["core_start_frame"] + np.arange(cfg.core_frames)
        inside_mask = indices < len(self.times[sid])
        clipped = np.minimum(indices, len(self.times[sid]) - 1)
        tar = self.targets[sid]
        y, valid = tar["y"][clipped].copy(), tar["valid"][clipped].copy() * inside_mask
        left, right = tar["left"][clipped].copy(), tar["right"][clipped].copy()
        y[~inside_mask] = left[~inside_mask] = right[~inside_mask] = 0
        return x, y, valid, left, right

    def batch(self, sample_ids):
        items = [self.get(i) for i in sample_ids]
        return [torch.from_numpy(np.stack(a)) for a in zip(*items)]


def epoch_batches(size, batch_size, rng):
    """One shuffled visit per manifest row, including the final partial batch."""
    order = rng.permutation(size)
    for start in range(0, size, batch_size):
        yield order[start:start + batch_size]


def train_full_epochs(dataset, cfg, device, out, plan):
    seed_everything(cfg.seed)
    model = AnchorTCN(cfg).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params != 774339:
        raise RuntimeError(f"Unexpected architecture: {params} parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(cfg.seed)
    rows, epochs = [], []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        seen = np.zeros(len(dataset), dtype=np.uint8)
        order_digest = hashlib.sha256()
        total_loss, sample_count = 0.0, 0
        epoch_start = time.perf_counter()
        positive_count, valid_count = 0, 0
        for step, sample_ids in enumerate(epoch_batches(len(dataset), cfg.batch_size, rng), start=1):
            if seen[sample_ids].any():
                raise RuntimeError("A window was repeated within a full epoch")
            seen[sample_ids] = 1
            order_digest.update(sample_ids.astype("<i8").tobytes())
            x, y, valid, left, right = [a.to(device) for a in dataset.batch(sample_ids)]
            positive_count += int((y * valid).sum().item())
            valid_count += int(valid.sum().item())
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                logits, dist = model(x)
                a, b = cfg.context_frames, cfg.context_frames + cfg.core_frames
                loss, cls, loc = detection_loss(logits[:, a:b], dist[:, a:b], y, valid, left, right, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in epoch {epoch}, batch {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            if device.type != "cuda" and not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite CPU gradients")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            applied = scaler.get_scale() >= old_scale
            batch_count = len(sample_ids)
            lv = float(loss.detach().cpu())
            total_loss += lv * batch_count
            sample_count += batch_count
            rows.append(dict(epoch=epoch, batch=step, global_batch=len(rows) + 1,
                batch_samples=batch_count, loss=lv,
                classification=float(cls.detach().cpu()), localization=float(loc.detach().cpu()),
                optimizer_update_applied=bool(applied)))
            if step % 250 == 0:
                print(f"    epoch {epoch}/{cfg.epochs}, batch {step}/{plan['steps_per_full_epoch']}", flush=True)
        if not (seen == 1).all() or sample_count != len(dataset):
            raise RuntimeError("Epoch coverage audit failed: some training windows were omitted")
        if positive_count != plan["positive_reference_exposures_per_epoch"] or valid_count != plan["valid_reference_exposures_per_epoch"]:
            raise RuntimeError("Manifest label counts disagree with the extracted training targets")
        epoch_summary = dict(epoch=epoch, samples_seen=sample_count, unique_windows_seen=int(seen.sum()),
            expected_windows=len(dataset), every_window_seen_exactly_once=True,
            order_sha256=order_digest.hexdigest(),
            mean_loss_per_sample=total_loss / sample_count,
            positive_valid_frames=positive_count, valid_frames=valid_count,
            seconds=time.perf_counter() - epoch_start)
        epochs.append(epoch_summary)
        write_csv(out / "training_log.csv", rows)
        jdump(out / "epoch_coverage.json", epochs)
        print(f"  epoch {epoch}/{cfg.epochs}: {sample_count:,}/{len(dataset):,} windows, "
              f"loss={total_loss/sample_count:.5f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    summary = dict(parameter_count=params, full_epochs=cfg.epochs, training_windows=len(dataset),
        training_draws=sum(r["batch_samples"] for r in rows), global_steps=len(rows),
        optimizer_updates=sum(r["optimizer_update_applied"] for r in rows),
        amp_skipped_steps=sum(not r["optimizer_update_applied"] for r in rows),
        training_seconds=time.perf_counter() - started,
        peak_gpu_allocated_gib=(torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0),
        note="Every indexed window visited once per full epoch. Smaller hops increase samples AND optimizer iterations.")
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    mean=dataset.mean, scale=dataset.scale, config=asdict(cfg), window_plan=plan), out / "last.pt")
    jdump(out / "training_summary.json", summary)
    del optimizer, scaler, model
    restored = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    model = AnchorTCN(cfg)
    model.load_state_dict(restored["model"])
    return model.to(device).eval(), summary


def infer_standard(model, store, sid, times, mean, scale, cfg, device):
    count, core, ctx = len(times), cfg.core_frames, cfg.context_frames
    result = np.full((count, 3), np.nan, dtype=np.float32)
    starts = list(range(0, count, core))
    with torch.inference_mode():
        for offset in range(0, len(starts), cfg.inference_batch_size):
            selected = starts[offset:offset + cfg.inference_batch_size]
            windows = [(store.block(sid, gs-ctx, cfg.crop_frames) - mean[:,None,:]) / scale[:,None,:] for gs in selected]
            with amp_context(device):
                logits, dist = model(torch.from_numpy(np.stack(windows)).to(device))
                probability = torch.sigmoid(logits[:,ctx:ctx+core]).float().cpu().numpy()
                distance = dist[:,ctx:ctx+core].float().cpu().numpy()
            for index, start in enumerate(selected):
                end = min(count,start+core)
                result[start:end] = np.column_stack((probability[index,:end-start],distance[index,:end-start]))
    if not np.isfinite(result).all():
        raise RuntimeError(f"Invalid or uncovered inference references in {sid}")
    return result


def result_row(arm, hop, plan, summary, report, score, nms):
    primary = report["primary"]
    return dict(arm=arm, hop_frames=hop, input_overlap=plan["input_overlap_fraction"],
        unique_training_windows=plan["training_windows"], full_epochs=plan["full_epochs"],
        total_training_draws=summary["training_draws"], training_batches=summary["global_steps"],
        optimizer_updates=summary["optimizer_updates"], F1_mean=report["ranking_score"]["f1_mean"],
        F1_01=report["by_theta"]["0.1"]["f1"], F1_03=report["by_theta"]["0.3"]["f1"],
        F1_05=report["by_theta"]["0.5"]["f1"],
        TP_03=primary["tp"],FP_03=primary["fp"],FN_03=primary["fn"],
        false_alarms_per_hour=primary["false_alarms_per_hour"],
        median_start_error_s=primary["start_error_median_abs"],
        median_end_error_s=primary["end_error_median_abs"], mean_matched_tIoU=primary["mean_tiou"],
        score_threshold=score,nms_threshold=nms,training_seconds=summary["training_seconds"])


def write_results(out, fixed_rows, tuned_rows, planned_runs, complete):
    write_csv(out / "comparison_fixed.csv", fixed_rows)
    write_csv(out / "comparison_tuned.csv", tuned_rows)
    def table(rows):
        lines=["| Training hop | Windows/epoch | Total crop visits | Training batches | F1 @ .1 | F1 @ .3 | F1 @ .5 | Mean F1 | FP @ .3 |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(f"| {r['hop_frames']} frames | {r['unique_training_windows']} | {r['total_training_draws']} | {r['training_batches']} | "
                         f"{r['F1_01']:.3f} | {r['F1_03']:.3f} | {r['F1_05']:.3f} | **{r['F1_mean']:.3f}** | {r['FP_03']} |")
        return lines
    lines=["# Full-window training experiment", "",
        f"Status: {'COMPLETE' if complete else 'PARTIAL'} ({len(fixed_rows)}/{planned_runs} models).", "",
        "EMA and the original 774,339-parameter TCN are retained in every run.",
        "Every trainable indexed window is visited exactly once in each full epoch, including partial final minibatches.",
        "All runs use standard inference with the same 292-frame retained-core hop.", "",
        "## Main challenge objective", "",
        "Ranking mean = (event F1 at tIoU 0.1 + 0.3 + 0.5) / 3. No model or threshold is selected using tIoU 0.7.", "",
        "## Same fixed confidence 0.8 and NMS 0.1", "", *table(fixed_rows), "",
        "## Same validation confidence/NMS sweep for every model", "", *table(tuned_rows), "",
        "The sweep checks confidence 0.005 and 0.01 through 0.99 in steps of 0.01 with NMS 0.1, 0.3, 0.5, 0.7.",
        "These are validation-selected development results, not hidden-test estimates.", "",
        "## What was tested", "",
        "Inputs remain 800 frames (16 seconds). Hop 292 is the context-preserving reference: its retained cores tile time, while input windows already overlap 63.5%.",
        "Hop 80 gives 90% input overlap. Hop 40 gives 95% input overlap.",
        "The denser grids create larger indexed datasets. They are not randomly capped at 20,000 crop draws.",
        "Only windows whose entire retained core is masked are excluded; their counts are recorded in the window plans.",
        "CSV manifests store window coordinates and label counts; CIR is loaded on demand without saving duplicate windows.",
        "Sampling is exhaustive and shuffled without replacement. There is no additional 50% event oversampling.", "",
        "## Interpretation limits", "",
        "The number of full epochs is fixed. Denser overlap therefore increases BOTH temporal training positions and optimization iterations.",
        "This tests the supervisor's more-samples proposal; it does not isolate overlap from extra compute.",
        "All grid models share this full-coverage training policy. Compare them to one another; the historical random-sampler score is not a matched baseline for this experiment.",
        "The class proportions are measured in every manifest; full-stream training has more background exposure than the previous event-biased sampler.",
        "There is one seed. Small differences need repeat runs before a general improvement is claimed.",
        "The original GroupNorm and bidirectional context remain. No real-time latency claim is made.", "",
        "For audit, inspect window_manifest.csv, window_plan.json, epoch_coverage.json and training_summary.json inside each run.",
        "Return RESULTS_TO_SHARE.zip after completion.", ""]
    if complete:
        best=max(tuned_rows,key=lambda r:r["F1_mean"])
        baseline=tuned_rows[0]
        lines.extend(["## Observed mean F1", "",
            f"Best development mean: {best['arm']} = {best['F1_mean']:.4f}.",
            f"Difference from hop {baseline['hop_frames']} reference: {best['F1_mean']-baseline['F1_mean']:+.4f}.", ""])
    (out / "RESULTS.md").write_text("\n".join(lines),encoding="utf-8")


def run_full_experiment(args,cfg,out):
    root=args.root.expanduser().resolve()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    evaluator,evaluator_info=load_official_evaluator(root)
    train_paths,val_paths=discover(root,"train"),discover(root,"val")
    streams={**train_paths,**val_paths}
    times,records=inspect_streams(streams,cfg)
    train_times={sid:times[sid] for sid in train_paths}
    val_times={sid:times[sid] for sid in val_paths}
    durations={sid:len(t)/cfg.rate_hz for sid,t in val_times.items()}
    rows=load_events(root)
    unknown={r["stream_id"] for r in rows if r["stream_id"].startswith(("train_","val_"))}-set(streams)
    if unknown:
        raise ValueError(f"Event table refers to missing streams: {sorted(unknown)}")
    targets,eligible=build_targets(train_times,rows,cfg)
    ground=evaluator.load_ground_truth(root/"streams_events.csv",ignore_flagged=True)
    if not any(g["stream_id"] in val_times and not g["ignore"] for g in ground):
        raise ValueError("No evaluable validation events")
    manifests,plans={},{}
    for hop in args.hops:
        manifest,plan=build_window_manifest(train_times,targets,hop,cfg)
        manifests[hop],plans[hop]=manifest,plan
        arm_out=out/f"hop_{hop:03d}"
        arm_out.mkdir()
        write_csv(arm_out/"window_manifest.csv",manifest)
        jdump(arm_out/"window_plan.json",plan)
        print(f"[plan hop {hop}] {len(manifest):,} windows per full epoch; "
              f"{plan['total_training_draws']:,} total crop visits; "
              f"{plan['total_training_batches']:,} training batches",flush=True)
    plan_rows=[{k:p[k] for k in ("hop_frames","hop_seconds","input_overlap_fraction","training_windows",
               "positive_windows","background_windows","positive_reference_fraction","steps_per_full_epoch",
               "full_epochs","total_training_draws","total_training_batches")} for p in plans.values()]
    write_csv(out/"training_plan.csv",plan_rows)
    jdump(out/"dataset_summary.json",dict(train_streams=len(train_paths),validation_streams=len(val_paths),
        eligible_training_events=len(eligible),streams=records,window_plans=plan_rows))
    jdump(out/"run_config.json",dict(config=asdict(cfg),hops=args.hops,dataset=str(root),output=str(out),
        python=sys.version,torch=torch.__version__,numpy=np.__version__,h5py=h5py.__version__,
        started_utc=datetime.now(timezone.utc).isoformat(),evaluator=evaluator_info,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        event_table_sha256=hashlib.sha256((root/"streams_events.csv").read_bytes()).hexdigest(),
        note="All-window training: equal full epochs, increasing samples and training iterations as hop decreases."))
    if args.plan_only:
        jdump(out/"progress.json",dict(stage="PLAN_ONLY_COMPLETE",training_started=False))
        share_results(out)
        print(f"[plan only] No cache or model created. Read {out/'training_plan.csv'}",flush=True)
        return
    device=setup_device(args.device,args.gpu_memory_fraction)
    print(f"[device] {device}",flush=True)
    if device.type=="cuda":
        print(f"[gpu] {torch.cuda.get_device_name(device)}",flush=True)
    cache_bytes=sum(cfg.radars*len(t)*cfg.bins*4+128 for t in times.values())
    needed=cache_bytes+max(512*1024**2,int(.1*cache_bytes))
    free=shutil.disk_usage(out).free
    if free<needed:
        raise OSError(f"Need about {needed/1024**3:.2f} GiB free for the new cache and outputs; available {free/1024**3:.2f} GiB")
    print(f"[storage] Fresh shared EMA cache about {cache_bytes/1024**3:.2f} GiB; windows are coordinates, not duplicated CIR arrays.",flush=True)
    jdump(out/"progress.json",dict(stage="preparing fresh shared EMA cache"))
    ema_paths=prepare_ema(streams,out/"ema_cache",cfg)
    preprocessing=out/"preprocessing"
    preprocessing.mkdir()
    store=Store(ema_paths,"ema")
    fixed_rows,tuned_rows=[],[]
    try:
        mean,scale=fit_normalization(store,train_times,cfg,preprocessing)
        for hop in args.hops:
            name=f"hop_{hop:03d}"
            arm_out=out/name
            plan=plans[hop]
            print(f"\n[train {name}] {cfg.epochs} FULL epochs; {plan['training_windows']:,} windows per epoch",flush=True)
            jdump(out/"progress.json",dict(stage="training",arm=name,windows_per_epoch=plan["training_windows"]))
            dataset=FullWindowDataset(store,train_times,targets,mean,scale,manifests[hop],cfg)
            model,summary=train_full_epochs(dataset,cfg,device,arm_out,plan)
            jdump(out/"progress.json",dict(stage="standard inference",arm=name))
            inference_out=arm_out/"inference"
            inference_out.mkdir()
            proposals=[]
            started=time.perf_counter()
            for sid,t in val_times.items():
                print(f"  [infer] {sid}",flush=True)
                values=infer_standard(model,store,sid,t,mean,scale,cfg,device)
                np.savez_compressed(inference_out/f"{sid}_frame_predictions.npz",t=t,
                    confidence=values[:,0],d_start=values[:,1],d_end=values[:,2])
                proposals.extend(decode(sid,t,values,cfg))
            jdump(inference_out/"summary.json",dict(seconds=time.perf_counter()-started,
                input_frames=cfg.crop_frames,retained_core_frames=cfg.core_frames,
                inference_hop=cfg.core_frames,proposals=len(proposals)))
            jdump(inference_out/"proposals.json",proposals)
            jdump(out/"progress.json",dict(stage="official evaluation",arm=name))
            evaluation_out=arm_out/"evaluation"
            fixed,best=evaluate_proposals(evaluator,ground,durations,proposals,cfg,evaluation_out)
            fixed_rows.append(result_row(name,hop,plan,summary,fixed,cfg.fixed_score,cfg.fixed_nms))
            tuned_rows.append(result_row(name,hop,plan,summary,best["report"],best["score"],best["nms"]))
            for label in ("fixed","tuned"):
                predictions=evaluator.validate_predictions(evaluation_out/f"{label}_submission.csv",durations)
                per_stream={sid:evaluator.evaluate(ground,predictions,{sid:duration}) for sid,duration in durations.items()}
                jdump(evaluation_out/f"{label}_per_stream_reports.json",per_stream)
            print(f"  [result] fixed mean F1={fixed_rows[-1]['F1_mean']:.4f}; "
                  f"tuned mean F1={tuned_rows[-1]['F1_mean']:.4f} "
                  f"(F1 .1/.3/.5 = {tuned_rows[-1]['F1_01']:.3f}/{tuned_rows[-1]['F1_03']:.3f}/{tuned_rows[-1]['F1_05']:.3f})",flush=True)
            write_results(out,fixed_rows,tuned_rows,len(args.hops),False)
            del model,dataset
            if device.type=="cuda":
                torch.cuda.empty_cache()
    finally:
        store.close()
    write_results(out,fixed_rows,tuned_rows,len(args.hops),True)
    jdump(out/"progress.json",dict(stage="COMPLETE",completed_models=len(args.hops)))
    package=share_results(out)
    print(f"\nCOMPLETE. Read: {out/'RESULTS.md'}\nSend back: {package}",flush=True)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root",type=Path,default=Path("/home/abc2027/dataset/abc2027_uwb_train_v1"))
    parser.add_argument("--out",type=Path,help="New output directory; must not already exist")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--epochs",type=int,default=5,help="Complete passes through every indexed training window")
    parser.add_argument("--hops",nargs="+",type=int,default=[292,80,40],help="Training grid hops in frames")
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--seed",type=int,default=2027)
    parser.add_argument("--gpu-memory-fraction",type=float,default=.5)
    parser.add_argument("--plan-only",action="store_true",help="Write sample counts and manifests without caching or training")
    args=parser.parse_args(argv)
    if any(v<=0 for v in (args.epochs,args.batch_size,args.threads)):
        parser.error("Epochs, batch size and threads must be positive")
    if any(h<=0 or h>292 for h in args.hops) or len(set(args.hops))!=len(args.hops):
        parser.error("Hops must be distinct positive integers <=292")
    args.hops=sorted(args.hops,reverse=True)
    if not 0<args.gpu_memory_fraction<=1:
        parser.error("GPU memory fraction must be in (0,1]")
    return args


def main(argv=None):
    args=parse_args(argv)
    root=args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    out=(args.out or Path(__file__).resolve().parent/f"anchor_tcn_more_samples_{stamp}_{os.getpid()}").expanduser().resolve()
    if inside(out,root) or inside(root,out):
        raise ValueError("Output must be outside the dataset and must not be its parent")
    if out.exists():
        raise FileExistsError(f"Choose a new output directory; this one exists: {out}")
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(Path(__file__).resolve(),out/"experiment_source.py")
    cfg=Config(epochs=args.epochs,batch_size=args.batch_size,seed=args.seed)
    stdout,stderr=sys.stdout,sys.stderr
    with (out/"console.log").open("w",encoding="utf-8",buffering=1) as log:
        sys.stdout,sys.stderr=Tee(stdout,log),Tee(stderr,log)
        try:
            run_full_experiment(args,cfg,out)
        except BaseException as error:
            jdump(out/"progress.json",dict(stage="FAILED",error=str(error),error_type=type(error).__name__))
            (out/"FAILED.txt").write_text(traceback.format_exc(),encoding="utf-8")
            print(f"\nSTOPPED: {error}\nDetails: {out/'FAILED.txt'}",file=sys.stderr,flush=True)
            share_results(out)
            raise
        finally:
            sys.stdout,sys.stderr=stdout,stderr


if __name__=="__main__":
    main()

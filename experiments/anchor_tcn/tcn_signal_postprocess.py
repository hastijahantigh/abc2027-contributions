#!/usr/bin/env python3
"""Postprocess a FROZEN TCN's saved predictions; no training or GPU required.

Example (replace YOUR_RUN with your actual existing experiment directory):
  python -B tcn_signal_postprocess.py --run /home/abc2027/hasti/YOUR_RUN/hop_080

--run accepts the model directory or its inference directory. It must contain
*_frame_predictions.npz (t, confidence, d_start, d_end). The script reads original
CIR from --root and uses the organizer's annotate.py and evaluate.py in memory.
No installation, checkpoint modification, or dataset modification is performed.
All reports go to a NEW directory next to this script, or a new --out directory.

The signal-processing design follows the inspected Vision-TriRadar pipeline:
join detection, repeat-aware motion, organizer boundary helpers, and profile
persistence. This independent integration keeps the original TCN decoder as its
control, adds ablations, bounds raw-data memory, and evaluates through the official
evaluator. The 5.9-second center-suppression reference is diagnostic only.

For subsequent frozen deployment on saved TEST frame outputs:
  python -B tcn_signal_postprocess.py --run TEST_INFERENCE --split test \
      --selection VALIDATION_OUTPUT/selection.json --root RELEASE_ROOT
Test mode does not read ground-truth labels or tune settings.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "2")
import argparse
import csv
import hashlib
import json
import math
import time
import traceback
import types
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy import signal

VERSION = "1.0"
RATE = 50.0
PROFILE_GATE = 2.12e-5
SCORES = (0.005,) + tuple(i / 100 for i in range(1, 100))
NMSES = (0.1, 0.3, 0.5, 0.7)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_save(path, value):
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=convert) + "\n", encoding="utf-8")


def csv_save(path, records, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)


def inside(path, parent):
    return Path(path).resolve().is_relative_to(Path(parent).resolve())


def official_module(root, name):
    path = root / "src" / "abc2027" / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"Required organizer file is missing: {path}")
    module = types.ModuleType(f"_tcn_signal_official_{name}")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module, digest(path)


def resolve_inference(run):
    choices = [p for p in (run, run / "inference", run / "standard")
               if p.is_dir() and any(p.glob("*_frame_predictions.npz"))]
    if len(choices) != 1:
        raise ValueError("--run must identify one model's inference folder (or its parent). "
                         "Expected *_frame_predictions.npz; the compact RESULTS_TO_SHARE.zip does not contain them.")
    return choices[0]


def load_inputs(root, folder, split):
    paths = {f"{split}_{int(p.stem.split('_')[-1]):02d}": p
             for p in (root / "streams" / split).glob("stream_*.h5")}
    predictions = {p.name.removesuffix("_frame_predictions.npz"): p
                   for p in folder.glob("*_frame_predictions.npz")}
    if not paths or set(paths) != set(predictions):
        raise ValueError(f"Complete {split} coverage required. Missing predictions: {sorted(set(paths)-set(predictions))}; "
                         f"unexpected predictions: {sorted(set(predictions)-set(paths))}")
    frames, manifest = {}, []
    for sid in sorted(paths):
        path = predictions[sid]
        with np.load(path, allow_pickle=False) as archive:
            needed = ("t", "confidence", "d_start", "d_end")
            if not set(needed).issubset(archive.files):
                raise ValueError(f"Missing frame arrays in {path}")
            arrays = {k: np.asarray(archive[k], dtype=np.float64) for k in needed}
        n = len(arrays["t"])
        if n < 2 or any(x.shape != (n,) or not np.isfinite(x).all() for x in arrays.values()):
            raise ValueError(f"Invalid frame arrays: {path}")
        if not np.allclose(arrays["t"], np.arange(n) / RATE, atol=1e-6, rtol=0):
            raise ValueError(f"Expected continuous zero-origin 50 Hz timestamps: {sid}")
        if np.any((arrays["confidence"] < 0) | (arrays["confidence"] > 1)) or any(np.any(arrays[k] < 0) for k in ("d_start", "d_end")):
            raise ValueError(f"Invalid probability or negative boundary distance: {sid}")
        with h5py.File(paths[sid], "r") as h:
            if h["cir"].shape != (3, n, 368) or np.issubdtype(h["cir"].dtype, np.complexfloating):
                raise ValueError(f"Expected real CIR [3,{n},368]: {paths[sid]}")
            if not np.allclose(h["t"][:], arrays["t"], atol=1e-6, rtol=0):
                raise ValueError(f"Raw CIR and prediction timelines differ: {sid}")
        frames[sid] = arrays
        stat = paths[sid].stat()
        manifest.append(dict(stream_id=sid, frames=n, prediction_path=str(path),
                             prediction_sha256=digest(path), cir_path=str(paths[sid]),
                             cir_bytes=stat.st_size, cir_mtime_ns=stat.st_mtime_ns))
    return frames, paths, manifest


def event(sid, start, end, score):
    return dict(stream_id=sid, t_start=float(start), t_end=float(end), score=float(score), t_alarm=None)


def anchors(sid, data):
    p = data["confidence"]
    indices, _ = signal.find_peaks(p, height=0.005, distance=10)
    result = []
    for i in indices:
        start = max(0.0, data["t"][i] - data["d_start"][i])
        end = min(data["t"][-1], data["t"][i] + data["d_end"][i])
        if end > start:
            result.append(event(sid, start, end, p[i]))
    return result


def suppress(proposals, threshold, mode="iou"):
    kept = []
    for sid in sorted({p["stream_id"] for p in proposals}):
        group = [p for p in proposals if p["stream_id"] == sid]
        starts = np.array([p["t_start"] for p in group])
        ends = np.array([p["t_end"] for p in group])
        order = np.argsort(-np.array([p["score"] for p in group]), kind="stable")
        while order.size:
            i = order[0]
            kept.append(group[i].copy())
            rest = order[1:]
            if mode == "center":
                good = np.abs((starts[i] + ends[i] - starts[rest] - ends[rest]) / 2) >= threshold
            else:
                overlap = np.maximum(0, np.minimum(ends[i], ends[rest]) - np.maximum(starts[i], starts[rest]))
                union = ends[i] - starts[i] + ends[rest] - starts[rest] - overlap
                good = overlap / np.maximum(union, 1e-12) < threshold
            order = rest[good]
    return sorted(kept, key=lambda x: (x["stream_id"], x["t_start"]))


def probability_runs(sid, probability, threshold):
    changes = np.diff(np.r_[0, (probability >= threshold).astype(np.int8), 0])
    spans = []
    for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        if spans and a - spans[-1][1] <= 10:
            spans[-1][1] = int(b)
        else:
            spans.append([int(a), int(b)])
    return [event(sid, a / RATE, b / RATE, np.max(probability[a:b]))
            for a, b in spans if b - a >= 10]


def join_strength(dataset, chunk=2048):
    """Exact block-mean profile step, with bounded CIR working memory."""
    n = dataset.shape[1]
    if n <= 32:
        return np.full(n, -np.inf)
    combined = np.full(n, np.inf)
    for radar in range(3):
        step = np.empty(n, dtype=np.float64)
        for a in range(16, n - 16, chunk):
            b = min(n - 16, a + chunk)
            x = np.asarray(dataset[radar, a - 16:b + 16, :], dtype=np.float64)
            if not np.isfinite(x).all():
                raise ValueError("Non-finite raw CIR encountered")
            prefix = np.concatenate((np.zeros((1, 368)), np.cumsum(x, axis=0)), axis=0)
            i = np.arange(16, 16 + b - a)
            difference = (prefix[i + 16] - prefix[i + 6] - prefix[i - 6] + prefix[i - 16]) / 10
            step[a:b] = np.einsum("ij,ij->i", difference, difference)
        step[:16] = step[16]
        step[n - 16:] = step[n - 17]
        grid = np.arange(0, n, 50)
        medians = np.array([np.median(step[max(0, j-250):min(n, j+251)]) for j in grid])
        baseline = np.interp(np.arange(n), grid, medians)
        np.minimum(combined, np.log(step + 1e-30) - np.log(baseline + 1e-30), out=combined)
    return combined


def locate_joins(strength):
    candidates = np.flatnonzero(strength > 1.5)
    order = candidates[np.argsort(-strength[candidates], kind="stable")]
    blocked = np.zeros(len(strength), dtype=bool)
    accepted = []
    for i in order:
        if not blocked[i]:
            accepted.append(int(i))
            blocked[max(0, i-50):min(len(strength), i+51)] = True
    return np.array(sorted(accepted), dtype=int)


def analysis_span(joins, start, stop, n):
    center = (start + stop) // 2
    before = np.searchsorted(joins, center - 50, side="right")
    after = np.searchsorted(joins, center + 50, side="left")
    a = int(joins[before-1]) + 6 if before else 0
    b = int(joins[after]) - 5 if after < len(joins) else n
    a, b = max(0, a), min(n, b)
    if not 100 <= b-a <= 1000:
        a, b = max(a, center-200), min(b, center+250)
    return a, b


def restored_motion(x):
    changes = np.r_[True, np.any(x[1:] != x[:-1], axis=1)]
    first = np.flatnonzero(changes)
    if len(first) < 3:
        return np.zeros(len(x))
    last = np.r_[first[1:], len(x)]
    times = (first + last - 1) / (2 * RATE)
    delta = np.diff(x[first].astype(np.float64), axis=0)
    energy = np.einsum("ij,ij->i", delta, delta)
    return np.interp(np.arange(len(x)) / RATE, times[1:], energy)


class SignalRefiner:
    def __init__(self, paths, annotate):
        self.paths, self.annotate = paths, annotate
        self.joins, self.cache = {}, {}
        self.seconds = dict(join_scan=0.0, refinement=0.0)

    def prepare(self, sid, h):
        if sid not in self.joins:
            began = time.perf_counter()
            print(f"  [raw CIR] {sid}: scanning joins in bounded blocks", flush=True)
            self.joins[sid] = locate_joins(join_strength(h["cir"]))
            self.seconds["join_scan"] += time.perf_counter() - began

    def features(self, sid, h, a, b):
        key = (sid, a, b)
        if key in self.cache:
            return self.cache[key]
        began = time.perf_counter()
        x = np.asarray(h["cir"][:, a:b, :], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite raw CIR in {sid}:{a}:{b}")
        ann = self.annotate
        grid = np.arange(b-a) / RATE
        states, motion, changes = {}, {}, []
        gate_change = 0.0
        for r in range(3):
            initial, final = ann._profiles(x[r], grid)
            states[r] = ann._state(x[r], initial, final)
            motion[r] = restored_motion(x[r])
            changes.append(float(np.sum((final-initial)**2)))
            before = np.median(x[r, :min(50, b-a)], axis=0)
            after = np.median(x[r, -min(75, b-a):], axis=0)
            gate_change = max(gate_change, float(np.sum((after-before)**2)))
        cleaned = ann._suppress_spikes(motion, states, 1/RATE)
        anchor = int(np.argmax(changes))
        state = ann._smooth(states[anchor], 1/RATE, ann.SMOOTH_S)
        energy = ann._smooth(cleaned[anchor], 1/RATE, ann.SMOOTH_S)
        left, right, _ = ann._transition_bounds(state, energy)
        result = (int(a+left), int(a+right), gate_change)
        self.cache[key] = result
        self.seconds["refinement"] += time.perf_counter() - began
        return result

    def refine(self, proposals):
        result = []
        for sid in sorted({p["stream_id"] for p in proposals}):
            with h5py.File(self.paths[sid], "r") as h:
                self.prepare(sid, h)
                n = h["cir"].shape[1]
                for p in (q for q in proposals if q["stream_id"] == sid):
                    q = p.copy()
                    a, b = analysis_span(self.joins[sid], int(round(p["t_start"]*RATE)), int(round(p["t_end"]*RATE)), n)
                    if b-a >= 100:
                        left, right, _ = self.features(sid, h, a, b)
                        if 0 <= left < right <= n and right-left >= 10:
                            q["t_start"], q["t_end"] = left/RATE, right/RATE
                    result.append(q)
        return result

    def gate(self, proposals):
        result = []
        for sid in sorted({p["stream_id"] for p in proposals}):
            with h5py.File(self.paths[sid], "r") as h:
                self.prepare(sid, h)
                n = h["cir"].shape[1]
                for p in (q for q in proposals if q["stream_id"] == sid):
                    a, b = analysis_span(self.joins[sid], int(round(p["t_start"]*RATE)), int(round(p["t_end"]*RATE)), n)
                    if b-a < 100 or self.features(sid, h, a, b)[2] >= PROFILE_GATE:
                        result.append(p.copy())
        return result


def union_intervals(proposals):
    result = []
    for p in sorted(proposals, key=lambda x: (x["stream_id"], x["t_start"], x["t_end"], x["score"])):
        if result and result[-1]["stream_id"] == p["stream_id"] and p["t_start"] < result[-1]["t_end"]:
            result[-1]["t_end"] = max(result[-1]["t_end"], p["t_end"])
            result[-1]["score"] = max(result[-1]["score"], p["score"])
        else:
            result.append(p.copy())
    return result


def execute_config(config, frames, all_anchors, refiner):
    if config["candidates"] == "anchors":
        result = [p for p in suppress(all_anchors, config["anchor_nms"]) if p["score"] >= config["threshold"]]
    else:
        result = [p for sid, data in frames.items() for p in probability_runs(sid, data["confidence"], config["threshold"])]
    if config["refine"]:
        result = refiner.refine(result)
    if config["merge"]:
        result = union_intervals(result)
    if config["center_nms"]:
        result = suppress(result, config["center_nms"], "center")
    if config["gate"]:
        result = refiner.gate(result)
    if config["final_iou"] is not None:
        result = suppress(result, config["final_iou"])
    return result


def config(candidates="anchors", threshold=0.8, anchor_nms=0.1, refine=False, gate=False,
           merge=False, final_iou=None, center_nms=None):
    return dict(candidates=candidates, threshold=threshold, anchor_nms=anchor_nms,
                refine=refine, gate=gate, merge=merge, final_iou=final_iou, center_nms=center_nms)


def ranking(report):
    r = report["ranking_score"]
    return (float(r["f1_mean"]), -(float(r["start_error_median_abs"]) if r["start_error_median_abs"] is not None else math.inf),
            -(float(r["false_alarms_per_hour"]) if r["false_alarms_per_hour"] is not None else math.inf))


def summary_row(name, report, eligible=True):
    main = report["by_theta"]["0.3"]
    return dict(method=name, selection_eligible=eligible, mean_f1=report["ranking_score"]["f1_mean"],
                **{f"f1_{k}": report["by_theta"][k]["f1"] for k in ("0.1", "0.3", "0.5", "0.7")},
                precision_03=main["precision"], recall_03=main["recall"],
                tp_03=main["tp"], fp_03=main["fp"], fn_03=main["fn"],
                false_alarms_per_hour=main["false_alarms_per_hour"],
                start_error_median_abs=main["start_error_median_abs"],
                end_error_median_abs=main["end_error_median_abs"], mean_tiou=main["mean_tiou"])


def save_scored(out, name, predictions, evaluator, durations, ground):
    path = out / f"{name}_submission.csv"
    csv_save(path, predictions, ("stream_id", "t_start", "t_end", "score"))
    checked = evaluator.validate_predictions(path, durations)
    if ground is None:
        return None
    report = evaluator.evaluate(ground, checked, durations)
    json_save(out / f"{name}_official_report.json", report)
    return report


def validate(frames, paths, folder, root, out, evaluator, annotate, hashes):
    durations = {sid: len(data["t"])/RATE for sid, data in frames.items()}
    ground = evaluator.load_ground_truth(root / "streams_events.csv", ignore_flagged=True)
    proposals = [p for sid, data in frames.items() for p in anchors(sid, data)]
    refiner = SignalRefiner(paths, annotate)
    rows, selections, reports, predictions_by_name = [], {}, {}, {}
    def record(name, cfg, predictions, eligible=True):
        report = save_scored(out, name, predictions, evaluator, durations, ground)
        rows.append(summary_row(name, report, eligible))
        selections[name], reports[name], predictions_by_name[name] = cfg, report, predictions
        print(f"[result] {name}: mean F1={report['ranking_score']['f1_mean']:.4f}; "
              f"TP/FP/FN={report['primary']['tp']}/{report['primary']['fp']}/{report['primary']['fn']}", flush=True)
        return report

    fixed = config()
    fixed_predictions = execute_config(fixed, frames, proposals, refiner)
    record("tcn_fixed", fixed, fixed_predictions)
    print("[control] Repeating the original confidence/NMS validation sweep", flush=True)
    sweep, best = [], None
    for nms in NMSES:
        kept = suppress(proposals, nms)
        for threshold in SCORES:
            predictions = [p for p in kept if p["score"] >= threshold]
            report = evaluator.evaluate(ground, predictions, durations)
            sweep.append(dict(threshold=threshold, nms=nms, mean_f1=report["ranking_score"]["f1_mean"]))
            if best is None or ranking(report) > ranking(best[0]):
                best = (report, threshold, nms, predictions)
    csv_save(out / "control_sweep.csv", sweep, tuple(sweep[0]))
    tuned = config(threshold=best[1], anchor_nms=best[2])
    record("tcn_tuned", tuned, best[3])
    historical = folder.parent / "evaluation" / "tuned_official_report.json"
    if historical.is_file():
        expected = json.loads(historical.read_text(encoding="utf-8"))["ranking_score"]["f1_mean"]
        if abs(float(expected)-best[0]["ranking_score"]["f1_mean"]) > 1e-9:
            raise RuntimeError("Original TCN score does not reproduce the saved report. Stop and check the run, dataset and evaluator.")

    for name, cfg in (
        ("tcn_fixed_refined", dict(fixed, refine=True)),
        ("tcn_fixed_refined_gate", dict(fixed, refine=True, gate=True)),
        ("tcn_tuned_refined", dict(tuned, refine=True)),
        ("tcn_tuned_refined_gate", dict(tuned, refine=True, gate=True)),
        ("tcn_tuned_refined_gate_nms", dict(tuned, refine=True, gate=True, final_iou=0.1)),
    ):
        print(f"[postprocess] {name}", flush=True)
        record(name, cfg, execute_config(cfg, frames, proposals, refiner))

    # A small declared grid for the alternative probability-run candidate route.
    # The core signal stages have recovered constants, not a large fitted search.
    run_sweep, run_best = [], None
    for threshold in (0.1, 0.3, 0.5, 0.7, 0.9):
        cfg = config(candidates="runs", threshold=threshold, refine=True, gate=True, merge=True)
        candidates = execute_config(cfg, frames, proposals, refiner)
        for nms in (0.1, 0.3, 0.5):
            predictions = suppress(candidates, nms)
            report = evaluator.evaluate(ground, predictions, durations)
            run_sweep.append(dict(threshold=threshold, nms=nms, mean_f1=report["ranking_score"]["f1_mean"]))
            if run_best is None or ranking(report) > ranking(run_best[0]):
                run_best = (report, dict(cfg, final_iou=nms), predictions)
    csv_save(out / "signal_candidate_sweep.csv", run_sweep, tuple(run_sweep[0]))
    record("signal_runs_tuned", run_best[1], run_best[2])
    reference = config(candidates="runs", threshold=0.3, refine=True, gate=True, merge=True, center_nms=5.9)
    record("reference_center_5p9", reference, execute_config(reference, frames, proposals, refiner), eligible=False)

    eligible = [r["method"] for r in rows if r["selection_eligible"]]
    selected = max(eligible, key=lambda name: ranking(reports[name]))
    chosen = dict(version=VERSION, method=selected, config=selections[selected], helper_hashes=hashes,
                  script_sha256=digest(Path(__file__)),
                  note="Selected on validation labels. Freeze before hidden test; no hidden-test score claimed.")
    json_save(out / "selection.json", chosen)
    json_save(out / "all_method_configs.json", selections)
    csv_save(out / "comparison.csv", rows, tuple(rows[0]))
    save_scored(out, "selected", predictions_by_name[selected], evaluator, durations, ground)
    json_save(out / "selected_per_stream_reports.json", {sid: evaluator.evaluate(ground, predictions_by_name[selected], {sid:duration}) for sid,duration in durations.items()})
    json_save(out / "signal_processing_audit.json", dict(join_count={sid:len(x) for sid,x in refiner.joins.items()},
              distinct_spans_analyzed=len(refiner.cache), seconds=refiner.seconds))
    text = ["# Frozen TCN: signal postprocessing", "",
            "Same saved TCN predictions; no model training or checkpoint modification.", "",
            "| Method | F1 .1 | F1 .3 | F1 .5 | Mean F1 | Precision .3 | Recall .3 | TP/FP/FN .3 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        text.append(f"| {row['method']} | {row['f1_0.1']:.4f} | {row['f1_0.3']:.4f} | {row['f1_0.5']:.4f} | "
                    f"{row['mean_f1']:.4f} | {row['precision_03']:.4f} | {row['recall_03']:.4f} | {row['tp_03']}/{row['fp_03']}/{row['fn_03']} |")
    delta = reports[selected]["ranking_score"]["f1_mean"] - reports["tcn_tuned"]["ranking_score"]["f1_mean"]
    text += ["", f"Selected: **{selected}**; difference from tuned original TCN: **{delta:+.4f}** mean F1.", "",
             "The 5.9-second center-suppression row is diagnostic and is excluded from automatic selection.",
             "Refined anchor rows begin with the original selected detections and confidence scores; boundaries, profile gating and, in the final-nms row, duplicate suppression change.",
             "The signal-runs row also changes candidate generation and merging. It does not use learned distance heads.",
             "All reported values are validation development results. Tuning on these same streams makes them optimistic estimates of unseen performance.",
             "The official JSON reports include precision, recall, F1, counts, boundary errors and false alarms/hour.",
             "No measured alarm time is supplied. Any evaluator latency field is not measured runtime or a real-time guarantee.",
             "Raw-CIR signal processing uses future context and reconstructed stream joins; performance on differently constructed streams is unmeasured.", ""]
    (out / "RESULTS.md").write_text("\n".join(text), encoding="utf-8")


def package(out):
    target = out / "RESULTS_TO_SHARE.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and p.suffix in {".json", ".csv", ".md", ".log"}:
                z.write(p, p.name)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/home/abc2027/dataset/abc2027_uwb_train_v1"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--selection", type=Path, help="Apply frozen settings without reading labels or tuning")
    parser.add_argument("--check-only", action="store_true", help="Check inputs and dependencies; create no outputs")
    args = parser.parse_args(argv)
    root, run = args.root.expanduser().resolve(), args.run.expanduser().resolve()
    folder = resolve_inference(run)
    if args.split == "test" and args.selection is None:
        parser.error("--split test requires a frozen --selection from validation")
    evaluator, evaluator_hash = official_module(root, "evaluate")
    annotate, annotate_hash = official_module(root, "annotate")
    for name in ("_profiles", "_state", "_suppress_spikes", "_smooth", "_transition_bounds", "SMOOTH_S"):
        if not hasattr(annotate, name):
            raise RuntimeError(f"Organizer annotation helper is missing {name}")
    hashes = dict(annotate=annotate_hash, evaluate=evaluator_hash)
    frames, paths, manifest = load_inputs(root, folder, args.split)
    chosen = None
    if args.selection:
        chosen = json.loads(args.selection.expanduser().resolve().read_text(encoding="utf-8"))
        if chosen["version"] != VERSION or chosen["helper_hashes"] != hashes or chosen["script_sha256"] != digest(Path(__file__)):
            raise ValueError("Frozen selection requires the same script and organizer helper versions")
    elif not (root / "streams_events.csv").is_file():
        raise FileNotFoundError("Validation requires streams_events.csv")
    print(f"[inputs checked] {len(frames)} complete {args.split} streams; {sum(len(x['t']) for x in frames.values()):,} frames", flush=True)
    if args.check_only:
        print("CHECK COMPLETE. No output directory created; no training or signal scan performed.")
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (args.out or Path(__file__).resolve().parent / f"tcn_signal_{stamp}_{os.getpid()}").expanduser().resolve()
    if out.exists() or any(inside(out, p) or inside(p, out) for p in (root, run)):
        raise ValueError("Output must be a new directory separate from the dataset and original run")
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    old_out, old_err = sys.stdout, sys.stderr
    failed = False
    with (out / "console.log").open("w", encoding="utf-8", buffering=1) as log:
        sys.stdout, sys.stderr = Tee(old_out, log), Tee(old_err, log)
        try:
            json_save(out / "run_manifest.json", dict(version=VERSION, started_utc=datetime.now(timezone.utc).isoformat(),
                source_inference=str(folder), root=str(root), split=args.split, input_files=manifest,
                helper_hashes=hashes, script_sha256=digest(Path(__file__)), numpy=np.__version__, h5py=h5py.__version__,
                design_reference="Supplied Vision-TriRadar experiment-suite a8157b4 postprocessing; independently integrated with the frozen Anchor-TCN",
                fixed_signal_settings=dict(rate_hz=RATE, profile_gate=PROFILE_GATE, run_min_frames=10,
                    run_gap_frames=10, join_threshold=1.5, join_peak_radius_frames=50,
                    join_block_frames=10, join_gap_frames=6, join_baseline_frames=501,
                    span_min_frames=100, span_max_frames=1000, fallback_before_frames=200,
                    fallback_after_frames=250, minimum_refined_frames=10),
                training_performed=False, gpu_used=False, labels_used=chosen is None))
            if chosen:
                proposals = [p for sid,data in frames.items() for p in anchors(sid,data)]
                refiner = SignalRefiner(paths, annotate)
                predictions = execute_config(chosen["config"], frames, proposals, refiner)
                save_scored(out, "frozen", predictions, evaluator, {sid:len(x["t"])/RATE for sid,x in frames.items()}, None)
                json_save(out / "applied_selection.json", chosen)
            else:
                validate(frames, paths, folder, root, out, evaluator, annotate, hashes)
            json_save(out / "completion.json", dict(status="COMPLETE", wall_seconds=time.perf_counter()-start))
            print(f"COMPLETE. Results: {out}", flush=True)
        except BaseException:
            failed = True
            (out / "FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
            json_save(out / "completion.json", dict(status="FAILED", wall_seconds=time.perf_counter()-start))
            raise
        finally:
            sys.stdout, sys.stderr = old_out, old_err
    if not failed:
        print(f"Send back: {package(out)}", flush=True)


if __name__ == "__main__":
    main()

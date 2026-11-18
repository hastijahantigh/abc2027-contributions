#!/usr/bin/env python3
"""Apply the previously frozen TCN signal postprocessing, then score validation.
Standalone CPU-only script. No training, checkpoint selection, threshold sweep,
package installation, or writes to the dataset or original experiment.
The original signal-processing functions are retained unchanged.
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



FROZEN_SELECTION = {'version': '1.0', 'method': 'tcn_tuned_refined_gate_nms', 'config': {'candidates': 'anchors', 'threshold': 0.51, 'anchor_nms': 0.1, 'refine': True, 'gate': True, 'merge': False, 'final_iou': 0.1, 'center_nms': None}, 'helper_hashes': {'annotate': 'ffc4d613a643ee9d627d585cc70dbdff6041c2390e90ca20209f47af13044958', 'evaluate': 'df23749689ac2a41861f400059b18a575ada32ca91ea49733d9d219c3d9d4daa'}, 'script_sha256': '466ee418bd46ed28cb2af6afe253f5486e0e6fbc471feb7172c3ce479927eac4', 'note': 'Selected on validation labels. Freeze before hidden test; no hidden-test score claimed.'}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Already selected epoch directory")
    parser.add_argument("--root", type=Path, default=Path("/home/abc2027/dataset/abc2027_uwb_train_v1"))
    parser.add_argument("--out", type=Path, help="New directory outside dataset and training experiment")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    root, run = args.root.expanduser().resolve(), args.run.expanduser().resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    experiment = run.parent.parent
    best_path = experiment / "best_checkpoint.json"
    best_record = None
    if best_path.is_file():
        best_record = json.loads(best_path.read_text(encoding="utf-8"))
        if (experiment / best_record["run_directory"]).resolve() != run:
            raise ValueError("--run must match the previously selected best_checkpoint.json")
    folder = resolve_inference(run)
    evaluator, evaluator_hash = official_module(root, "evaluate")
    annotate, annotate_hash = official_module(root, "annotate")
    hashes = dict(annotate=annotate_hash, evaluate=evaluator_hash)
    if hashes != FROZEN_SELECTION["helper_hashes"]:
        raise ValueError("Organizer helpers differ from the frozen postprocessing run; no parameters were changed")
    label_path = root / "streams_events.csv"
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    frames, paths, manifest = load_inputs(root, folder, "val")
    durations = {sid:len(x["t"])/RATE for sid,x in frames.items()}
    cfg = FROZEN_SELECTION["config"]
    print(f"[checked] {len(frames)} complete validation streams; confidence={cfg['threshold']}, "
          f"anchor NMS={cfg['anchor_nms']}, final NMS={cfg['final_iou']}", flush=True)
    if args.check_only:
        print("CHECK COMPLETE. No outputs created or signal processing performed.")
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (args.out or Path(__file__).resolve().parent / f"tcn_frozen_eval_{stamp}_{os.getpid()}").expanduser().resolve()
    protected = [root, run] + ([experiment] if best_record is not None else [])
    if out.exists() or any(inside(out,p) or inside(p,out) for p in protected):
        raise ValueError("Output must be a new directory separate from the dataset and original experiment")
    out.mkdir(parents=True, exist_ok=False)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    started = time.perf_counter()
    with (out / "console.log").open("w", encoding="utf-8", buffering=1) as log:
        sys.stdout, sys.stderr = Tee(old_stdout,log), Tee(old_stderr,log)
        try:
            json_save(out / "applied_selection.json", FROZEN_SELECTION)
            json_save(out / "run_manifest.json", dict(
                source_inference=str(folder), root=str(root), split="val",
                started_utc=datetime.now(timezone.utc).isoformat(), input_files=manifest,
                helper_hashes=hashes, script_sha256=digest(Path(__file__)),
                original_postprocessor_sha256=FROZEN_SELECTION["script_sha256"],
                ground_truth_sha256=digest(label_path), numpy=np.__version__, h5py=h5py.__version__,
                checkpoint_selection=best_record,
                training_performed=False, gpu_used=False, parameter_search_performed=False,
                labels_used_for_scoring_only=True,
                note="Frozen settings previously selected on validation. This is development evaluation, not hidden test."))
            # Generate BOTH prediction sets before loading ground-truth labels.
            proposals = [p for sid,data in frames.items() for p in anchors(sid,data)]
            before = [p for p in suppress(proposals,cfg["anchor_nms"]) if p["score"] >= cfg["threshold"]]
            refiner = SignalRefiner(paths,annotate)
            after = execute_config(cfg,frames,proposals,refiner)
            ground = evaluator.load_ground_truth(label_path,ignore_flagged=True)
            if not any(g["stream_id"] in durations and not g["ignore"] for g in ground):
                raise ValueError("No evaluable validation events")
            rows = []
            reports = {}
            for name,predictions in (("before_postprocess",before),("after_postprocess",after)):
                report = save_scored(out,name,predictions,evaluator,durations,ground)
                reports[name] = report
                row = summary_row(name,report,eligible=False)
                rows.append(row)
                checked = evaluator.validate_predictions(out/f"{name}_submission.csv",durations)
                json_save(out/f"{name}_per_stream_reports.json",{
                    sid:evaluator.evaluate(ground,checked,{sid:duration}) for sid,duration in durations.items()})
                print(f"[result] {name}: mean F1={row['mean_f1']:.6f}; "
                      f"precision@.3={row['precision_03']:.6f}; recall@.3={row['recall_03']:.6f}; "
                      f"TP/FP/FN={row['tp_03']}/{row['fp_03']}/{row['fn_03']}",flush=True)
            csv_save(out/"comparison.csv",rows,list(rows[0]))
            delta = rows[1]["mean_f1"]-rows[0]["mean_f1"]
            text = ["# Selected TCN checkpoint: frozen postprocessing evaluation", "",
                f"Input checkpoint directory: `{run}`", "",
                "No training or parameter search was performed. Predictions were computed before loading labels.",
                "Confidence 0.51; initial NMS 0.1; signal boundary refinement; profile gate 2.12e-5; final NMS 0.1.",
                "No interval merging or center-distance suppression.", "",
                "| Method | F1 .1 | F1 .3 | F1 .5 | Mean F1 | Precision .3 | Recall .3 | TP/FP/FN .3 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for r in rows:
                text.append(f"| {r['method']} | {r['f1_0.1']:.4f} | {r['f1_0.3']:.4f} | {r['f1_0.5']:.4f} | "
                            f"{r['mean_f1']:.4f} | {r['precision_03']:.4f} | {r['recall_03']:.4f} | "
                            f"{r['tp_03']}/{r['fp_03']}/{r['fn_03']} |")
            text += ["", f"Mean F1 change: {100*delta:+.2f} percentage points.", "",
                "Full official reports include all reported tIoUs, boundary errors and false alarms per hour.",
                "This validation set previously informed checkpoint and postprocessing selection; these are development scores.",
                "Hidden-test performance is unmeasured. Signal processing uses future context; no real-time latency claim is made.", ""]
            (out/"RESULTS.md").write_text("\n".join(text),encoding="utf-8")
            json_save(out/"completion.json",dict(status="COMPLETE",wall_seconds=time.perf_counter()-started))
        except BaseException:
            (out/"FAILED.txt").write_text(traceback.format_exc(),encoding="utf-8")
            json_save(out/"completion.json",dict(status="FAILED",wall_seconds=time.perf_counter()-started))
            raise
        finally:
            sys.stdout,sys.stderr = old_stdout,old_stderr
            package(out)
    print(f"COMPLETE. Read: {out/'RESULTS.md'}\nSend back: {out/'RESULTS_TO_SHARE.zip'}",flush=True)


if __name__ == "__main__":
    main()

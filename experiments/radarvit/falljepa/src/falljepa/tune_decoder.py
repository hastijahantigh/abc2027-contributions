"""
Tunes the event decoder on saved per-timestep probabilities, scored with the official evaluator.

    uv run python -m falljepa.tune_decoder --events DATA/abc2027_uwb_train_v1/streams_events.csv \\
        --probs runs/UWB/CV/U1-cv*/*/predictions/val-validate.probs.npz --out decoder.json

Pass the `val-validate.probs.npz` of the 6 cross-validation runs (each scored only on its own
held-out fold, so together they are out-of-fold predictions for every training timestep): the
decoder is then tuned without touching the official val streams. Each (file, stream) is scored as
its own pseudo-stream, restricted to the file's mask. The best setting by the ranking score (mean
F1 at theta 0.1/0.3/0.5, then the median start error, then false alarms per hour) is printed and
written as a `model.decoder` block for the config.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .model.aggregate import decode_runs, runs_to_events
from .module import in_mask
from .official import evaluate as official

# The class weights raise P(fall) = 1 - P(none) above its frequency, so thresholds go up to 0.9
GRID = {
    "viterbi": {"switch_penalty": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]},
    "threshold": {"threshold": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "merge_gap_s": [0.2, 0.5, 1.0]},
}
POST = {"min_duration_s": [0.2, 0.3, 0.4, 0.6], "spread_penalty": [0.0, 0.5, 1.0]}


def load_probs(paths: list[Path]) -> list[dict]:
    """[{key, sid, logp, spread, mask}] for every stream of every file."""
    streams = []
    for i, path in enumerate(paths):
        data = np.load(path)
        for name in data.files:
            if not name.startswith("logp/"):
                continue
            sid = name.split("/", 1)[1]
            mask = data[f"mask/{sid}"] if f"mask/{sid}" in data.files else np.ones(len(data[name]), dtype=bool)
            streams.append({"key": f"{sid}@{i}", "sid": sid, "logp": data[name].astype(np.float64),
                            "spread": data[f"spread/{sid}"].astype(np.float64), "mask": mask})
    return streams


def rank_key(report: dict) -> tuple:
    r = report["ranking_score"]
    return (r["f1_mean"], -(r["start_error_median_abs"] or 1e9), -(r["false_alarms_per_hour"] or 1e9))


def tune(streams: list[dict], ground: list[dict]) -> list[tuple[dict, dict]]:
    """[(decoder settings, official report)] for the whole grid, best first."""
    pseudo_ground, durations = [], {}
    for s in streams:
        durations[s["key"]] = float(s["mask"].sum()) / 50.0
        for e in ground:
            if e["stream_id"] == s["sid"] and in_mask(s["mask"], e["t_start"], e["t_end"]):
                pseudo_ground.append({**e, "stream_id": s["key"]})

    results = []
    for kind, grid in GRID.items():
        for values in itertools.product(*grid.values()):
            run_args = {"kind": kind, **dict(zip(grid, values))}
            runs = {s["key"]: decode_runs(s["logp"], **run_args) for s in streams}
            for post in itertools.product(*POST.values()):
                post_args = dict(zip(POST, post))
                records = []
                for s in streams:
                    for ev in runs_to_events(runs[s["key"]], s["logp"], s["spread"], **post_args):
                        if in_mask(s["mask"], ev["t_start"], ev["t_end"]):
                            records.append({"stream_id": s["key"], **{k: ev[k] for k in ("t_start", "t_end", "score", "t_alarm")}})
                report = official.evaluate(pseudo_ground, records, durations)
                results.append(({**run_args, **post_args}, report))
    results.sort(key=lambda item: rank_key(item[1]), reverse=True)
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--probs", required=True, type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ignore-flagged", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    ground = official.load_ground_truth(args.events, ignore_flagged=args.ignore_flagged)
    results = tune(load_probs(args.probs), ground)
    print(f"{'F1_mean':>8} {'F1@.1':>6} {'F1@.3':>6} {'F1@.5':>6} {'start':>6} {'FA/h':>6}  decoder")
    for settings, report in results[:10]:
        by, r = report["by_theta"], report["ranking_score"]
        print(f"{r['f1_mean']:8.3f} {by['0.1']['f1']:6.3f} {by['0.3']['f1']:6.3f} {by['0.5']['f1']:6.3f} "
              f"{r['start_error_median_abs'] or 0:6.2f} {r['false_alarms_per_hour'] or 0:6.1f}  {settings}")
    if args.out:
        args.out.write_text(json.dumps({"decoder": results[0][0], "report": results[0][1]}, indent=2))


if __name__ == "__main__":
    main()

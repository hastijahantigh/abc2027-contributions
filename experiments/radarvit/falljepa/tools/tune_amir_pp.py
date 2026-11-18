"""
Tune Amir's post-processing (src/falljepa/postprocess.py) on the subject-held-out CV folds, from the
saved dense probabilities of fold runs (predictions/val-validate.probs.npz), the fold's held-out
streams (raw CIR, for joins, the official rule and the gate) and its labels. Nothing is retrained.

    uv run python tools/tune_amir_pp.py --release /path/to/abc2027_uwb_train_v1 --out runs/_analysis/amirpp_cv \\
        --fold A=FOLDS/A:runs/UWB/RS-A/RS-s0/<TAG>/predictions/val-validate.probs.npz --fold B=... --fold C=... --fold D=...

Each --fold is NAME=FOLD_DIR:PROBS. FOLD_DIR holds streams/val/stream_XX.h5 and streams_events.csv;
streams in the probs file but not in FOLD_DIR are skipped (so the official release works as a fold
directory for the official val streams). --release gives the official labeller and evaluator.

The grid (every combination; Amir's values marked *):
    theta      P(fall) threshold of the detector            .1 .2 .3* .4 .5 .6 .7 .8
    min_run    shortest run kept, timesteps                 5 10* 20
    gap        runs this close are merged, timesteps        10* 25
    join       join-peak threshold on J(t)                  1.0 1.25 1.5* 2.0
    nms        seconds between kept event centres           3 4 5 5.9* 7 9
    gate       minimum profile change (0 = no gate)         0 5e-6 1e-5 1.5e-5 2.12e-5* 3e-5 5e-5 8e-5
    per_rec    one event per detected recording             no* yes

The rule's bounds and the gate's profile change depend only on the analysed span, so both are cached
per span; each stream is processed once in a worker and returns, for every configuration and each
tIoU threshold .1/.3/.5, its TP / FP / FN. A fold's score (the official ranking score: mean F1 at
.1/.3/.5) and the pooled score over all folds come from summed counts, exactly as the official
evaluator computes them (it matches stream by stream).

Reported:
    amir       Amir's values on every fold and pooled (should match the runs' val-validate.amir.csv)
    per fold   the best configuration of each fold, scored on that fold: optimistic (in-sample)
    LOFO       for each fold, the configuration best pooled over the OTHER folds, scored on it: the
               honest estimate of what tuning buys on unseen subjects
    all folds  the configuration best pooled over all folds: the values to use on the test set,
               with the score of its neighbours (one value changed at a time), to see whether it
               sits on a plateau
Writes <out>.json (all of the above) and <out>.npz (the counts of every configuration and stream).
"""
import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from falljepa import postprocess as pp  # noqa: E402

THETAS = (0.1, 0.3, 0.5)
GRID = {
    "theta": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "min_run": [5, 10, 20],
    "gap": [10, 25],
    "join": [1.0, 1.25, 1.5, 2.0],
    "nms": [3.0, 4.0, 5.0, 5.9, 7.0, 9.0],
    "gate": [0.0, 5e-6, 1e-5, 1.5e-5, 2.12e-5, 3e-5, 5e-5, 8e-5],
    "per_rec": [False, True],
}
AMIR = {"theta": 0.3, "min_run": 10, "gap": 10, "join": 1.5, "nms": 5.9, "gate": 2.12e-5, "per_rec": False}

_A = _EVAL = None


def _init(release: str):
    global _A, _EVAL
    _A = pp.load_annotate(release)
    sys.path.insert(0, str(Path(release) / "src"))
    from abc2027 import evaluate
    _EVAL = evaluate


def configs(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def stream_counts(job) -> np.ndarray:
    """[configs, 3 thetas, 3 (tp, fp, fn)] of one stream."""
    path, p, ground, grid = job
    with h5py.File(path, "r") as h:
        cir = np.asarray(h["cir"][:], dtype=np.float32)
    T = cir.shape[1]
    duration = {"s": T / 50.0}
    ground = [{"stream_id": "s", **g} for g in ground]
    score = pp.join_score(cir)
    bounds_cache, change_cache = {}, {}
    all_cfg = configs(grid)
    index = {tuple(c.values()): i for i, c in enumerate(all_cfg)}
    out = np.zeros((len(all_cfg), len(THETAS), 3), dtype=np.int32)

    def rule(lo, hi):
        if (lo, hi) not in bounds_cache:
            bounds_cache[(lo, hi)] = pp.official_bounds(_A, cir[:, lo:hi], raw=True)
        return bounds_cache[(lo, hi)]

    def change(lo, hi):
        if (lo, hi) not in change_cache:
            change_cache[(lo, hi)] = pp.profile_change(_A, cir, lo, hi)
        return change_cache[(lo, hi)]

    for join in grid["join"]:
        peaks = pp.join_peaks(score, threshold=join)
        for theta, min_run, gap in itertools.product(grid["theta"], grid["min_run"], grid["gap"]):
            refined = []
            for a, b, c in pp.threshold_runs(p, theta, min_frames=min_run, gap_frames=gap):
                lo, hi = pp.analysed_span(T, peaks, a, b, True)
                if hi - lo >= pp.MIN_WINDOW:
                    left, right = rule(lo, hi)
                    if right - left >= pp.MIN_REFINED:
                        a, b = lo + left, lo + right
                refined.append((a, b, c))
            merged = pp.merge(refined)
            for nms in grid["nms"]:
                kept = pp.nms(merged, nms)
                spans = [pp.analysed_span(T, peaks, s, e, True) for s, e, _ in kept]
                changes = [change(lo, hi) if hi - lo >= pp.MIN_WINDOW else np.inf for lo, hi in spans]
                for gate in grid["gate"]:
                    gated = [ev for ev, ch in zip(kept, changes) if ch >= gate]
                    for per_rec in grid["per_rec"]:
                        events = pp.one_per_recording(gated, peaks) if per_rec else gated
                        preds = [{"stream_id": "s", "t_start": s / 50.0, "t_end": e / 50.0, "score": c, "t_alarm": None}
                                 for s, e, c in events]
                        i = index[(theta, min_run, gap, join, nms, gate, per_rec)]
                        for k, th in enumerate(THETAS):
                            r = _EVAL.score_events(ground, preds, duration, th)
                            out[i, k] = (r["tp"], r["fp"], r["fn"])
    return out


def ranking(counts: np.ndarray) -> np.ndarray:
    """Summed counts [..., 3 thetas, 3] -> mean F1 over the thetas [...]."""
    tp, fp, fn = counts[..., 0].astype(float), counts[..., 1].astype(float), counts[..., 2].astype(float)
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    return f1.mean(-1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--fold", action="append", required=True, help="NAME=FOLD_DIR:PROBS")
    ap.add_argument("--out", required=True, type=Path, help="output prefix (.json and .npz)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    _init(str(args.release))

    jobs, owner = [], []
    for spec in args.fold:
        name, rest = spec.split("=", 1)
        folder, probs = rest.rsplit(":", 1)
        folder = Path(folder)
        ground = _EVAL.load_ground_truth(folder / "streams_events.csv", ignore_flagged=True)
        data = np.load(probs)
        for key in sorted(k for k in data.files if k.startswith("logp/")):
            sid = key.split("/", 1)[1]
            path = folder / "streams" / "val" / f"stream_{sid.split('_')[1]}.h5"
            if not sid.startswith("val_") or not path.exists():
                continue
            p = 1 - np.exp(data[key].astype(np.float64)[:, 0])
            g = [{k: e[k] for k in ("t_start", "t_end", "ignore")} for e in ground if e["stream_id"] == sid]
            jobs.append((str(path), p, g, GRID))
            owner.append((name, sid))
        print(f"fold {name}: {sum(o[0] == name for o in owner)} streams", flush=True)
    all_cfg = configs(GRID)
    print(f"{len(all_cfg)} configurations x {len(jobs)} streams, {args.workers} workers", flush=True)

    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(args.release),)) as pool:
        counts = []
        for i, c in enumerate(pool.map(stream_counts, jobs)):
            counts.append(c)
            print(f"  {owner[i][0]}/{owner[i][1]} done ({i + 1}/{len(jobs)})", flush=True)
    counts = np.stack(counts)                                   # [streams, configs, 3, 3]
    folds = sorted({o[0] for o in owner})
    per_fold = {f: counts[[o[0] == f for o in owner]].sum(0) for f in folds}
    fold_score = {f: ranking(c) for f, c in per_fold.items()}  # [configs]
    pooled = ranking(sum(per_fold.values()))

    keys = list(GRID)
    amir = all_cfg.index(AMIR)
    describe = lambda i: {k: all_cfg[i][k] for k in keys}  # noqa: E731

    def at3(c, i):
        tp, fp, fn = c[i, 1]
        return {"tp": int(tp), "fp": int(fp), "fn": int(fn)}

    result = {"grid": GRID, "amir": {"config": AMIR, "pooled": float(pooled[amir]),
                                    "per_fold": {f: float(fold_score[f][amir]) for f in folds}}}
    print("\nscore = mean F1 at tIoU .1/.3/.5, --ignore-flagged; counts at .3")
    print(f"Amir's values        pooled {pooled[amir]:.3f}   " + "  ".join(f"{f} {fold_score[f][amir]:.3f}" for f in folds))

    best_fold = {}
    print("\nper fold (in-sample, optimistic):")
    for f in folds:
        i = int(np.argmax(fold_score[f]))
        best_fold[f] = {"config": describe(i), "score": float(fold_score[f][i]), "amir": float(fold_score[f][amir]), **at3(per_fold[f], i)}
        print(f"  {f}: {fold_score[f][i]:.3f} (Amir {fold_score[f][amir]:.3f})  {describe(i)}")
    result["per_fold_best"] = best_fold

    lofo, lofo_counts, amir_counts = {}, 0, 0
    if len(folds) > 1:
        print("\nleave one fold out (tuned on the other folds, scored on this one):")
        for f in folds:
            others = ranking(sum(per_fold[g] for g in folds if g != f))
            i = int(np.argmax(others))
            lofo[f] = {"config": describe(i), "score": float(fold_score[f][i]), "amir": float(fold_score[f][amir]),
                       "tuned_on_others": float(others[i])}
            lofo_counts = lofo_counts + per_fold[f][i]
            amir_counts = amir_counts + per_fold[f][amir]
            print(f"  {f}: {fold_score[f][i]:.3f} (Amir {fold_score[f][amir]:.3f})  {describe(i)}")
        result["lofo"] = {"per_fold": lofo, "pooled": float(ranking(lofo_counts)), "amir_pooled": float(ranking(amir_counts))}
        print(f"  pooled LOFO {ranking(lofo_counts):.3f} vs Amir {ranking(amir_counts):.3f}")

    i = int(np.argmax(pooled))
    neighbours = {}
    for k in keys:
        row = {}
        for v in GRID[k]:
            j = all_cfg.index({**all_cfg[i], k: v})
            row[str(v)] = float(pooled[j])
        neighbours[k] = row
    result["all_folds_best"] = {"config": describe(i), "pooled": float(pooled[i]),
                                "per_fold": {f: float(fold_score[f][i]) for f in folds},
                                **at3(sum(per_fold.values()), i), "neighbours": neighbours}
    top = np.argsort(-pooled)[:10]
    result["top10"] = [{"config": describe(int(j)), "pooled": float(pooled[j])} for j in top]
    print(f"\nall folds: pooled {pooled[i]:.3f} (Amir {pooled[amir]:.3f})  {describe(i)}")
    print("  per fold " + "  ".join(f"{f} {fold_score[f][i]:.3f}" for f in folds))
    print("  one value changed at a time (pooled):")
    for k, row in neighbours.items():
        print(f"    {k:8s} " + "  ".join(f"{v}: {s:.3f}" for v, s in row.items()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(args.out.with_suffix(".npz"), counts=counts, folds=np.array([o[0] for o in owner]),
                        streams=np.array([o[1] for o in owner]), configs=json.dumps(all_cfg))
    print(f"\nwrote {args.out.with_suffix('.json')} and {args.out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()

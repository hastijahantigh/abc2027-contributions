"""
Decode the recording head's saved outputs (predictions/val-validate.probs.npz of configs/uwb/RS.yaml runs)
three ways and score them on the subject-held-out folds, like tools/score_folds.py. Nothing is retrained.

    uv run python tools/score_rs_hybrid.py --release /path/to/abc2027_uwb_train_v1 --out runs/_analysis/rs_hybrid \\
        --fold A=FOLDS/A:runs/UWB/RS-A/RS-s0/<TAG>/predictions/val-validate.probs.npz --fold B=... --fold C=... --fold D=...

Each --fold is NAME=FOLD_DIR:PROBS. Streams of the probs file that FOLD_DIR does not hold are skipped
(so the official release works as the fold directory of the official val streams).

    head         the run's own events: units with P(fall) > threshold, span by MBR (recomputed; equals
                 the run's val-validate.csv)
    head + rule  the same units, but each span is the official rule (annotate.propose_fall internals,
                 raw-frame motion: postprocess.official_bounds) on the unit's raw CIR; units under 2 s,
                 or a rule interval under 0.2 s, keep the head's span
    + gate       head + rule, without units whose profile change is under Amir's 2.12e-5
    rule         no model: the official rule on every unit of >= 2 s with a rule interval, scored by the
                 unit's profile change (the scorer only uses the score to order matches)
    rule + gate  no model: the same, only units whose profile change passes Amir's gate

The rule is deterministic (a fixed function of the unit's raw CIR) and so are the units (Amir's join
detector), so the two model-free variants are the floor any model has to beat on these streams.

Writes <out>/<fold>_<variant>.csv (submission format) and <out>/scores.json; prints pooled, per-fold
and per-stitching scores (official evaluator, --ignore-flagged; counts at tIoU .3).
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_folds  # noqa: E402

from falljepa import postprocess as pp  # noqa: E402
from falljepa.model.aggregate import mbr_span  # noqa: E402

RATE = 50.0
VARIANTS = ("head", "head + rule", "head + rule + gate", "rule", "rule + gate")


def decode_fold(A, folder: Path, probs: Path, threshold: float) -> dict[str, list[dict]]:
    data = np.load(probs)
    out = {v: [] for v in VARIANTS}
    for key in sorted(k for k in data.files if k.startswith("units/")):
        sid = key.split("/", 1)[1]
        path = folder / "streams" / "val" / f"stream_{sid.split('_')[1]}.h5"
        if not path.exists():
            continue
        spans = data[f"span/{sid}"]
        with h5py.File(path, "r") as h:
            cir = h["cir"]
            T = cir.shape[1]
            for u, lo, hi, logit in data[key]:
                p = float(1 / (1 + np.exp(-logit)))
                lo, hi = int(lo), int(hi)
                rule, change = None, np.inf
                if hi - lo >= pp.MIN_WINDOW:
                    x = np.asarray(cir[:, lo:hi], dtype=np.float32)
                    left, right = pp.official_bounds(A, x, raw=True)
                    if right - left >= pp.MIN_REFINED:
                        rule = (lo + left, lo + right)
                    change = pp.profile_change(A, x, 0, hi - lo)
                if rule is not None:                          # the model-free variants
                    ev = {"stream_id": sid, "t_start": rule[0] / RATE, "t_end": rule[1] / RATE, "score": float(change)}
                    out["rule"].append(ev)
                    if change >= pp.MIN_PROFILE_CHANGE:
                        out["rule + gate"].append(ev)
                if p <= threshold:
                    continue
                sel = spans[spans[:, 0] == u]
                a, b = mbr_span(sel[:, 1].astype(int), sel[:, 2].astype(np.float64), sel[:, 3].astype(np.float64))
                a, b = max(a, 0), min(b, T)
                ra, rb = rule if rule is not None else (a, b)
                for variant, (s, e) in (("head", (a, b)), ("head + rule", (ra, rb))):
                    out[variant].append({"stream_id": sid, "t_start": s / RATE, "t_end": e / RATE, "score": p})
                if change >= pp.MIN_PROFILE_CHANGE:
                    out["head + rule + gate"].append({"stream_id": sid, "t_start": ra / RATE, "t_end": rb / RATE, "score": p})
        print(f"  {folder.name}/{sid}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", required=True, type=Path, help="the official package (labeller and evaluator)")
    ap.add_argument("--fold", action="append", required=True, help="NAME=FOLD_DIR:PROBS")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)
    A = pp.load_annotate(args.release)
    sys.path.insert(0, str(args.release / "src"))
    from abc2027 import evaluate as official
    score_folds.official = official
    args.out.mkdir(parents=True, exist_ok=True)

    folds = []
    for spec in args.fold:
        name, rest = spec.split("=", 1)
        folder, probs = rest.rsplit(":", 1)
        folds.append((name, Path(folder), Path(probs)))
    files = {v: {} for v in VARIANTS}
    for name, folder, probs in folds:
        print(f"fold {name}", flush=True)
        for variant, preds in decode_fold(A, folder, probs, args.threshold).items():
            path = args.out / f"{name}_{variant.replace(' + ', '+')}.csv"
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["stream_id", "t_start", "t_end", "score"])
                w.writeheader()
                w.writerows({k: (f"{v:.3f}" if k.startswith("t_") else (f"{v:.6g}" if k == "score" else v)) for k, v in r.items()} for r in preds)
            files[variant][name] = path

    line = lambda n, s: (f"{n:<10} {s['score']:.3f}  F1 {s['f1_0.1']:.3f}/{s['f1_0.3']:.3f}/{s['f1_0.5']:.3f}  "  # noqa: E731
                         f"tp {s['tp']:3d} fp {s['fp']:3d} fn {s['fn']:3d}  FA/h {s['fa_per_h']:.2f}")
    result = {}
    for variant in VARIANTS:
        durations, ground, preds, per_fold, per_stitch = {}, [], [], {}, {}
        for name, folder, _ in folds:
            d, g, p, stitchings = score_folds.load_fold(name, folder, files[variant][name])
            durations.update(d)
            ground += g
            preds += p
            per_fold[name] = score_folds.summary(official.evaluate(g, p, d))
            for k, ids in stitchings:
                sub = {sid: d[sid] for sid in ids if sid in d}
                if sub:
                    per_stitch[f"{name}#{k}"] = score_folds.summary(official.evaluate(g, p, sub))
        pooled = score_folds.summary(official.evaluate(ground, preds, durations))
        scores = [v["score"] for v in per_fold.values()]
        result[variant] = {"pooled": pooled, "per_fold": per_fold, "per_stitching": per_stitch,
                           "fold_mean": statistics.mean(scores), "fold_sd": statistics.stdev(scores) if len(scores) > 1 else 0.0}
        print(f"\n{variant}   (score = mean F1 at tIoU .1/.3/.5; counts at .3)")
        print(line("pooled", pooled))
        for name, s in per_fold.items():
            print(line(f"fold {name}", s))
        for name, s in per_stitch.items():
            print(line(f"  {name}", s))
    (args.out / "scores.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}/scores.json and the prediction CSVs")


if __name__ == "__main__":
    main()

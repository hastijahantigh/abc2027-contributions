"""
Score the subject-held-out folds (tools/build_fold_streams.py) with the official evaluator: pooled over
every fold's held-out streams, per fold, and per held-out stitching, all under --ignore-flagged.

    uv run python tools/score_folds.py --release /path/to/abc2027_uwb_train_v1 --out runs/_analysis/cv_<name>.json \
        --fold A=DATA/folds/A:runs/UWB/CV-A/<spec>-s0/<TAG>/predictions/val-validate.csv \
        --fold B=DATA/folds/B:... --fold C=DATA/folds/C:... --fold D=DATA/folds/D:...

Each --fold is NAME=FOLD_DIR:PREDICTIONS, the predictions being a submission CSV on that fold's
`val_XX` streams (what `run.py validate` writes). Stream ids are prefixed with the fold name
(`A/val_00`) so the folds pool into one score: one F1 over all held-out falls, false alarms per hour
over all held-out hours. The robustness numbers are the per-fold scores, their spread and the worst fold.
The evaluator is the release's own `src/abc2027/evaluate.py` (--release), so this file runs outside
the falljepa repo; without --release it uses falljepa's identical copy.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

official = None


def load_fold(name: str, folder: Path, predictions: Path):
    durations = official.stream_durations(folder / "streams" / "val")
    ground = [e for e in official.load_ground_truth(folder / "streams_events.csv", ignore_flagged=True)
              if e["stream_id"] in durations]
    preds = official.validate_predictions(predictions, durations)
    prefix = lambda sid: f"{name}/{sid}"  # noqa: E731
    for e in ground:
        e["stream_id"] = prefix(e["stream_id"])
    for p in preds:
        p["stream_id"] = prefix(p["stream_id"])
    stitchings = []
    meta = folder / "fold.json"
    if meta.exists():
        for s in json.loads(meta.read_text())["stitchings"]:
            if s["role"] == "val":
                stitchings.append((s["stitching"], [prefix(sid) for sid in s["streams"]]))
    return {prefix(k): v for k, v in durations.items()}, ground, preds, stitchings


def summary(report: dict) -> dict:
    p = report["by_theta"]["0.3"]
    return {"score": report["ranking_score"]["f1_mean"], "f1_0.1": report["by_theta"]["0.1"]["f1"],
            "f1_0.3": p["f1"], "f1_0.5": report["by_theta"]["0.5"]["f1"], "tp": p["tp"], "fp": p["fp"],
            "fn": p["fn"], "fa_per_h": p["false_alarms_per_hour"], "start_err_med": p["start_error_median_abs"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--fold", action="append", required=True, help="NAME=FOLD_DIR:PREDICTIONS")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--release", type=Path, help="the official package; its src/abc2027/evaluate.py scores")
    args = ap.parse_args()
    global official
    if args.release:
        sys.path.insert(0, str(args.release / "src"))
        from abc2027 import evaluate as official
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from falljepa.official import evaluate as official

    durations, ground, preds, rows = {}, [], [], {}
    per_fold = {}
    for spec in args.fold:
        name, rest = spec.split("=", 1)
        folder, predictions = rest.rsplit(":", 1)
        d, g, p, stitchings = load_fold(name, Path(folder), Path(predictions))
        durations.update(d)
        ground += g
        preds += p
        per_fold[name] = summary(official.evaluate(g, p, d))
        for k, ids in stitchings:
            sub = {sid: d[sid] for sid in ids}
            rows[f"{name}#{k}"] = summary(official.evaluate(g, p, sub))
    pooled = summary(official.evaluate(ground, preds, durations))
    scores = [v["score"] for v in per_fold.values()]
    result = {"pooled": pooled, "per_fold": per_fold, "per_stitching": rows,
              "fold_mean": statistics.mean(scores), "fold_sd": statistics.stdev(scores) if len(scores) > 1 else 0.0,
              "worst_fold": min(per_fold, key=lambda k: per_fold[k]["score"])}

    line = lambda n, s: (f"{n:<10} {s['score']:.3f}  F1 {s['f1_0.1']:.3f}/{s['f1_0.3']:.3f}/{s['f1_0.5']:.3f}  "  # noqa: E731
                         f"tp {s['tp']:3d} fp {s['fp']:3d} fn {s['fn']:3d}  FA/h {s['fa_per_h']:.2f}")
    print("score = mean F1 at theta 0.1/0.3/0.5, --ignore-flagged; counts at theta 0.3")
    print(line("pooled", pooled))
    for name, s in per_fold.items():
        print(line(f"fold {name}", s))
    for name, s in rows.items():
        print(line(f"  {name}", s))
    print(f"folds: mean {result['fold_mean']:.3f}, SD {result['fold_sd']:.3f}, worst {result['worst_fold']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

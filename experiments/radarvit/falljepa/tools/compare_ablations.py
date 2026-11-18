"""
Compare ablation runs (tools/run_ablations.sh) on the official val streams, seeds pooled:

    uv run python tools/compare_ablations.py [--tag <TAG>] [--ref A1] [--last 30] [--csv out.csv]
    uv run python tools/compare_ablations.py --tag <TAG> --specs A0 A1 A1b --best   # prints the winner

Reads every runs/UWB/Ablation/<spec>-s<seed>/<id>/ (only ids equal to --tag, if given): its
csv/fit-*/metrics.csv (resumed runs have several; a later file wins for an epoch) and its
predictions/val-validate.probs.npz (last.ckpt: the latest epoch). Runs that stopped before
trainer.max_epochs of their config.yaml are left out (their last N epochs are earlier ones than
everybody else's); --incomplete keeps them, marked with * after their epoch count. Per run:

    F1_mean@lastN   mean val/event/F1_mean over the last N epochs: the primary metric (nothing selected on val)
    F1_0.5@lastN, StartErr@lastN, FA/h@lastN   the same mean for the boundary and false-alarm values
    F1_last         val/event/F1_mean of the last epoch
    F1_best (ep)    the best epoch: selected on val, optimistic, for reference only
    AP              average precision of per-timestep P(fall) of the last checkpoint against the
                    labels; timesteps inside flagged falls are left out (the scorer ignores them)

Per spec, the mean and SD over seeds, and the difference of the means to --ref. A difference counts
as real if it is larger than twice the seed SD and every seed agrees in sign (see
docs/falljepa_ablation_plan.md). --best prints only the spec with the highest mean F1_mean@lastN
(tools/run_stages.sh builds stage 2 on it).
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from falljepa.data.labels import NONE, load_events, stream_targets  # noqa: E402

COLUMNS = {                          # name -> metrics.csv column, averaged over the last N epochs
    "F1_mean": "val/event/F1_mean",
    "F1_0.5": "val/event/F1_0.5",
    "StartErr": "val/event/StartErrorMedian_0.3",
    "FA/h": "val/event/FalseAlarmsPerHour_0.3",
}


def epoch_values(run: Path) -> dict[int, dict[str, float]]:
    """{epoch: {column: value}} of the validation rows of every fit-*/metrics.csv, in time order."""
    values: dict[int, dict[str, float]] = {}
    for path in sorted(run.glob("csv/fit-*/metrics.csv")):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if not row.get(COLUMNS["F1_mean"]):
                    continue
                values[int(row["epoch"])] = {k: float(row[c]) for k, c in COLUMNS.items() if row.get(c)}
    return values


def max_epochs(run: Path) -> int | None:
    """trainer.max_epochs of the run's saved config (None if it has none)."""
    config = run / "config.yaml"
    if not config.exists():
        return None
    return (yaml.safe_load(config.read_text()).get("trainer") or {}).get("max_epochs")


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    hits = label[order]
    if not hits.any():
        return float("nan")
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].mean())


def timestep_ap(probs: Path, events: dict) -> float:
    data = np.load(probs)
    scores, labels = [], []
    for key in data.files:
        if not key.startswith("logp/"):
            continue
        sid = key[len("logp/"):]
        logp = data[key].astype(np.float64)
        # flagged_weight 0: timesteps of flagged falls get weight 0 and are left out
        state, weight = stream_targets(events.get(sid, []), len(logp), flagged_weight=0.0)
        keep = weight > 0
        scores.append(1 - np.exp(logp[keep, NONE]))
        labels.append(state[keep] != NONE)
    return average_precision(np.concatenate(scores), np.concatenate(labels))


def summarize(run: Path, last: int, events: dict | None) -> dict | None:
    values = epoch_values(run)
    if not values:
        return None
    epochs = sorted(values)
    tail = epochs[-last:]
    expected = max_epochs(run)
    row = {"epochs": len(epochs), "max_epochs": expected,
           "complete": expected is None or set(range(expected)) <= set(epochs)}
    for k in COLUMNS:
        row[f"{k}@last"] = float(np.mean([values[e][k] for e in tail if k in values[e]]))
    row["F1_last"] = values[epochs[-1]]["F1_mean"]
    best = max(epochs, key=lambda e: values[e]["F1_mean"])
    row["F1_best"], row["best_epoch"] = values[best]["F1_mean"], best
    probs = run / "predictions" / "val-validate.probs.npz"
    row["AP"] = timestep_ap(probs, events) if events is not None and probs.exists() else float("nan")
    return row


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("runs/UWB/Ablation"))
    parser.add_argument("--tag", help="only runs with this id (the TAG of run_ablations.sh)")
    parser.add_argument("--ref", help="the reference spec the differences are taken to")
    parser.add_argument("--last", type=int, default=30, help="epochs averaged for the primary metric")
    parser.add_argument("--events", type=Path, default=Path("runs/_dataset/streams_events.csv"),
                        help="streams_events.csv for the timestep AP (skipped if missing)")
    parser.add_argument("--specs", nargs="+", help="only these specs")
    parser.add_argument("--best", action="store_true", help="print only the spec with the highest mean F1_mean@lastN")
    parser.add_argument("--incomplete", action="store_true",
                        help="keep runs with fewer validated epochs than trainer.max_epochs (marked *)")
    parser.add_argument("--csv", type=Path, help="also write the per-run rows here")
    args = parser.parse_args(argv)

    events = load_events(args.events) if args.events.exists() else None
    if events is None:
        print(f"[compare] {args.events} not found: no timestep AP", file=sys.stderr)

    runs: dict[str, list[dict]] = defaultdict(list)
    for run in sorted(args.root.glob("*/*")):
        if not run.is_dir() or (args.tag and run.name != args.tag):
            continue
        match = re.fullmatch(r"(.+)-s(\d+)", run.parent.name)
        if not match or (args.specs and match[1] not in args.specs):
            continue
        row = summarize(run, args.last, events)
        if row is None:
            print(f"[compare] {run}: no validation epochs yet", file=sys.stderr)
            continue
        if not row["complete"]:
            print(f"[compare] {run}: {row['epochs']} of {row['max_epochs']} epochs"
                  + (", kept (--incomplete)" if args.incomplete else ", left out (--incomplete keeps it)"), file=sys.stderr)
            if not args.incomplete:
                continue
        runs[match[1]].append({"spec": match[1], "seed": int(match[2]), "run": str(run), **row})
    if not runs:
        sys.exit(f"no runs under {args.root}" + (f" with id {args.tag}" if args.tag else ""))
    if args.best:
        print(max(runs, key=lambda spec: np.mean([r["F1_mean@last"] for r in runs[spec]])))
        return

    keys = ["F1_mean@last", "F1_0.5@last", "StartErr@last", "FA/h@last", "F1_last", "F1_best", "AP"]
    ref = runs.get(args.ref) if args.ref else None
    if args.ref and ref is None:
        print(f"[compare] reference {args.ref!r} not found", file=sys.stderr)

    def cell(rows, k):
        v = np.array([r[k] for r in rows], dtype=float)
        return f"{np.nanmean(v):.3f}" + (f"±{np.nanstd(v, ddof=1):.3f}" if len(v) > 1 else "")

    header = ["spec", "seeds", "epochs", *[k.replace("@last", f"@{args.last}") for k in keys]]
    if ref:
        header += ["Δ F1_mean", "real?"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for spec, rows in sorted(runs.items()):
        line = [spec, str(len(rows)), "/".join(f"{r['epochs']}{'' if r['complete'] else '*'}" for r in rows), *[cell(rows, k) for k in keys]]
        if ref:
            mine = np.array([r["F1_mean@last"] for r in rows])
            base = np.array([r["F1_mean@last"] for r in ref])
            delta = mine.mean() - base.mean()
            sds = [np.std(a, ddof=1) for a in (mine, base) if len(a) > 1]
            sd = max(sds) if sds else np.nan
            agree = bool(np.all(np.sign(mine - base.mean()) == np.sign(delta)))
            real = "—" if spec == args.ref else ("?" if np.isnan(sd) else ("yes" if abs(delta) > 2 * sd and agree else "no"))
            line += [f"{delta:+.3f}", real]
        print("| " + " | ".join(line) + " |")
    print(f"\nF1_best is selected on val (optimistic); compare by F1_mean@{args.last}.")
    if any(not r["complete"] for rs in runs.values() for r in rs):
        print("* fewer epochs than trainer.max_epochs: its @N mean covers earlier epochs than the others'.")

    if args.csv:
        rows = [r for rs in runs.values() for r in rs]
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

"""
Load the ABC2027 subject-held-out CV folds and score predictions on them (see README.md).

    import cv_folds
    fold = cv_folds.Fold("A")
    for sid in fold.train_ids:
        x = fold.read(sid)                 # float32 [3 radars, T timesteps (50 Hz), 368]
        y, w = fold.labels(sid)            # [T] 1 inside a fall; [T] 0 on flagged falls, else 1
    ...
    cv_folds.write_predictions("predA.csv", rows)       # rows: (stream_id, t_start, t_end, score)

    python cv_folds.py info [--root FOLDS] [--release PACKAGE]
    python cv_folds.py score A=predA.csv B=predB.csv C=predC.csv D=predD.csv [--out cv.json]

Needs numpy and h5py. The folds are read from --root (or ABC_CV_FOLDS; default: the ABC27 server
copy). Scoring uses the organizers' evaluator from the official package (--release or ABC_RELEASE),
unchanged, under the test rule (--ignore-flagged). In Python, pass root= and release= instead.
"""
import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

import h5py
import numpy as np

FOLDS_ROOT = Path(os.environ.get("ABC_CV_FOLDS", "/home/abc2027/dataset/abc2027_uwb_cv_folds"))
RELEASE = Path(os.environ.get("ABC_RELEASE", "/home/abc2027/dataset/abc2027_uwb_train_v1"))
FOLDS = ("A", "B", "C", "D")
RATE_HZ = 50


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class Fold:
    """One fold folder, laid out like the official package:

        streams/train/stream_XX.h5   training streams  -> ids train_00, train_01, ...
        streams/val/stream_XX.h5     held-out streams  -> ids val_00, val_01, ...
        streams_events.csv           fall labels of both, official format
        streams_composition.csv      the clips of every stream
        fold.json                    held-out subjects, seeds, the streams of every stitching

    Stream ids repeat between folds (every fold has a train_00): keep them per fold.
    """

    def __init__(self, name: str, root: Path = FOLDS_ROOT):
        self.name = name
        self.folder = Path(root) / name
        if not self.folder.is_dir():
            raise FileNotFoundError(f"no fold folder {self.folder}")
        meta = self.folder / "fold.json"
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}
        self._events = None

    def _paths(self, role: str) -> dict[str, Path]:
        return {f"{role}_{int(p.stem.split('_')[1]):02d}": p
                for p in sorted((self.folder / "streams" / role).glob("stream_*.h5"))}

    @property
    def train_ids(self) -> list[str]:
        return list(self._paths("train"))

    @property
    def heldout_ids(self) -> list[str]:
        return list(self._paths("val"))

    def path(self, sid: str) -> Path:
        return self._paths(sid.split("_")[0])[sid]

    def open(self, sid: str) -> h5py.File:
        """The stream's HDF5 file (keep it open for many window reads); `cir` is [3, T, 368] float32."""
        return h5py.File(self.path(sid), "r")

    def length(self, sid: str) -> int:
        with self.open(sid) as h:
            return int(h["cir"].shape[1])

    def read(self, sid: str, a: int = 0, b: int | None = None) -> np.ndarray:
        """CIR of timesteps [a, b) (all of it by default): float32 [3 radars, b - a, 368], 50 Hz.
        The 368 values of a frame are 184 complex range bins, I then Q (README v1.1)."""
        with self.open(sid) as h:
            x = np.asarray(h["cir"][:, a:b, :], dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f"{self.name}/{sid}: non-finite values; is segments.h5 next to the fold folders?")
        return x

    def events(self, sid: str | None = None) -> list[dict]:
        """Falls (of one stream, or all): t_start, t_end in seconds, fall_type, flagged, source_path.
        flagged = quality low or needs_review: the scorer ignores these falls, as on the test."""
        if self._events is None:
            self._events = [{"stream_id": r["stream_id"], "t_start": float(r["t_start"]), "t_end": float(r["t_end"]),
                             "fall_type": r["fall_type"], "source_path": r["source_path"],
                             "flagged": (r["quality"].strip().lower() == "low" or _truthy(r["needs_review"])
                                         or _truthy(r["ignore"]))}
                            for r in _read_csv(self.folder / "streams_events.csv")]
        return [e for e in self._events if sid is None or e["stream_id"] == sid]

    def labels(self, sid: str, flagged_weight: float = 0.0, flagged_margin_s: float = 0.0):
        """Per-timestep targets of a stream: y uint8 [T] (1 inside a fall) and w float32 [T]
        (flagged_weight inside flagged falls, widened by flagged_margin_s; 1 elsewhere)."""
        T = self.length(sid)
        y = np.zeros(T, dtype=np.uint8)
        w = np.ones(T, dtype=np.float32)
        margin = int(round(flagged_margin_s * RATE_HZ))
        for e in self.events(sid):
            a, b = int(round(e["t_start"] * RATE_HZ)), int(round(e["t_end"] * RATE_HZ))
            y[a:b] = 1
            if e["flagged"]:
                w[max(a - margin, 0):b + margin] = flagged_weight
        return y, w

    def composition(self, sid: str | None = None) -> list[dict]:
        """The clips of a stream in order: kind (fall / adl / other), path, label, t_start, t_end."""
        return [r for r in _read_csv(self.folder / "streams_composition.csv") if sid is None or r["stream_id"] == sid]

    def __repr__(self) -> str:
        held = f"fall {self.meta.get('fall')} adl {self.meta.get('adl')}" if self.meta else "?"
        return f"Fold({self.name}: {len(self.train_ids)} train, {len(self.heldout_ids)} held-out streams; held out {held})"


def write_predictions(path, rows) -> None:
    """A submission CSV: rows of (stream_id, t_start, t_end, score), times in seconds, t_start < t_end."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        out = csv.writer(f)
        out.writerow(["stream_id", "t_start", "t_end", "score"])
        for sid, a, b, s in rows:
            out.writerow([sid, f"{float(a):.3f}", f"{float(b):.3f}", f"{float(s):.6f}"])


# ------------------------------------------------------------------ scoring


def _evaluator(release: Path):
    sys.path.insert(0, str(Path(release) / "src"))
    from abc2027 import evaluate
    return evaluate


def _summary(report: dict) -> dict:
    p = report["by_theta"]["0.3"]
    return {"score": report["ranking_score"]["f1_mean"], "f1_0.1": report["by_theta"]["0.1"]["f1"],
            "f1_0.3": p["f1"], "f1_0.5": report["by_theta"]["0.5"]["f1"], "tp": p["tp"], "fp": p["fp"],
            "fn": p["fn"], "fa_per_h": p["false_alarms_per_hour"], "start_err_med": p["start_error_median_abs"]}


def score(predictions: dict, root: Path = FOLDS_ROOT, release: Path = RELEASE, verbose: bool = True) -> dict:
    """Official score of held-out predictions {fold name: CSV path}: pooled over all given folds (one F1
    over all their held-out falls), per fold, per held-out stitching, and the folds' mean, SD and worst."""
    ev = _evaluator(release)
    durations, ground, preds, per_fold, per_stitching = {}, [], [], {}, {}
    for name, csv_path in predictions.items():
        fold = Fold(name, root)
        d = ev.stream_durations(fold.folder / "streams" / "val")
        g = [e for e in ev.load_ground_truth(fold.folder / "streams_events.csv", ignore_flagged=True)
             if e["stream_id"] in d]
        p = ev.validate_predictions(Path(csv_path), d)
        for item in g + p:
            item["stream_id"] = f"{name}/{item['stream_id']}"
        d = {f"{name}/{k}": v for k, v in d.items()}
        durations.update(d)
        ground += g
        preds += p
        per_fold[name] = _summary(ev.evaluate(g, p, d))
        for s in fold.meta.get("stitchings", []):
            if s["role"] == "val":
                sub = {f"{name}/{sid}": d[f"{name}/{sid}"] for sid in s["streams"]}
                per_stitching[f"{name}#{s['stitching']}"] = _summary(ev.evaluate(g, p, sub))
    scores = [v["score"] for v in per_fold.values()]
    result = {"pooled": _summary(ev.evaluate(ground, preds, durations)), "per_fold": per_fold,
              "per_stitching": per_stitching, "fold_mean": statistics.mean(scores),
              "fold_sd": statistics.stdev(scores) if len(scores) > 1 else 0.0,
              "worst_fold": min(per_fold, key=lambda k: per_fold[k]["score"])}
    if verbose:
        line = lambda n, s: (f"{n:<10} {s['score']:.3f}  F1 {s['f1_0.1']:.3f}/{s['f1_0.3']:.3f}/{s['f1_0.5']:.3f}  "  # noqa: E731
                             f"tp {s['tp']:3d} fp {s['fp']:3d} fn {s['fn']:3d}  FA/h {s['fa_per_h']:.2f}")
        print("score = mean F1 at theta 0.1/0.3/0.5 (official ranking), flagged falls ignored; counts at theta 0.3")
        print(line("pooled", result["pooled"]))
        for name, s in per_fold.items():
            print(line(f"fold {name}", s))
        for name, s in per_stitching.items():
            print(line(f"  {name}", s))
        print(f"folds: mean {result['fold_mean']:.3f}, SD {result['fold_sd']:.3f}, worst {result['worst_fold']}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="ABC2027 subject-held-out CV folds")
    ap.add_argument("--root", type=Path, default=FOLDS_ROOT, help="the folder holding A/ B/ C/ D/ and segments.h5")
    ap.add_argument("--release", type=Path, default=RELEASE, help="the official package (for its evaluator)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="list the folds")
    sc = sub.add_parser("score", help="score held-out predictions")
    sc.add_argument("pred", nargs="+", help="FOLD=predictions.csv, e.g. A=predA.csv")
    sc.add_argument("--out", type=Path, help="write the full result as JSON")
    args = ap.parse_args()
    if args.cmd == "info":
        for name in FOLDS:
            fold = Fold(name, args.root)
            hours = lambda ids: sum(fold.length(s) for s in ids) / RATE_HZ / 3600  # noqa: E731
            falls = [e for e in fold.events() if e["stream_id"].startswith("val")]
            print(f"{fold}\n  train {hours(fold.train_ids):.2f} h, held-out {hours(fold.heldout_ids):.2f} h, "
                  f"held-out falls {len(falls)} ({sum(not e['flagged'] for e in falls)} scored, counted per stitching)")
    else:
        result = score(dict(p.split("=", 1) for p in args.pred), args.root, args.release)
        if args.out:
            args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

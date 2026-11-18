"""
Ensemble the dense head of several runs on the same fold: average P(fall) per timestep (and the
disagreement), decode with the dense head's Viterbi decoder (as module._decode_dense) and score with
the official evaluator (as tools/score_folds.py, --ignore-flagged), pooled and per held-out stitching.

    uv run python tools/ensemble_dense.py --fold-dir FOLD_DIR --out runs/_analysis/jf4_ens/ens.json \\
        --set Se0+Se1=RUN_A/predictions/val-validate.probs.npz,RUN_B/predictions/val-validate.probs.npz --set ...

FOLD_DIR holds streams_events.csv and fold.json. Stream durations are the saved logp lengths / 50 Hz
(a single-run set reproduces the run's own dense score, which checks this).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from falljepa.model.aggregate import decode_events  # noqa: E402
from falljepa.official import evaluate as official  # noqa: E402


def load(path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    d = np.load(path)
    return {k[5:]: (d[k].astype(np.float32), d[f"spread/{k[5:]}"].astype(np.float32))
            for k in d.files if k.startswith("logp/")}


def ensemble(runs: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for sid in runs[0]:
        T = min(len(r[sid][0]) for r in runs)
        p = np.mean([np.exp(r[sid][0][:T]) for r in runs], 0)
        p /= p.sum(1, keepdims=True)
        out[sid] = (np.log(np.clip(p, 1e-12, None)), np.mean([r[sid][1][:T] for r in runs], 0))
    return out


def summary(report: dict) -> dict:
    p = report["by_theta"]["0.3"]
    return {"score": report["ranking_score"]["f1_mean"], "f1_0.1": report["by_theta"]["0.1"]["f1"],
            "f1_0.3": p["f1"], "f1_0.5": report["by_theta"]["0.5"]["f1"], "tp": p["tp"], "fp": p["fp"],
            "fn": p["fn"], "fa_per_h": p["false_alarms_per_hour"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-dir", type=Path, required=True)
    ap.add_argument("--set", action="append", required=True, help="NAME=PROBS[,PROBS...]")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    ground_all = official.load_ground_truth(args.fold_dir / "streams_events.csv", ignore_flagged=True)
    stitchings = [(s["stitching"], s["streams"]) for s in json.loads((args.fold_dir / "fold.json").read_text())["stitchings"]
                  if s["role"] == "val"]
    cache, result = {}, {}
    for spec in args.set:
        name, paths = spec.split("=", 1)
        runs = [cache.setdefault(p, load(p)) for p in paths.split(",")]
        probs = ensemble(runs)
        durations = {sid: len(lp) / 50.0 for sid, (lp, _) in probs.items()}
        ground = [e for e in ground_all if e["stream_id"] in durations]
        preds = [{"stream_id": sid, **e} for sid, (lp, sp) in probs.items() for e in decode_events(lp, sp)]
        preds = [{k: p[k] for k in ("stream_id", "t_start", "t_end", "score", "t_alarm")} for p in preds]
        row = {"pooled": summary(official.evaluate(ground, preds, durations))}
        for k, ids in stitchings:
            row[f"D#{k}"] = summary(official.evaluate(ground, preds, {sid: durations[sid] for sid in ids}))
        result[name] = row
        s = row["pooled"]
        print(f"{name:<14} {s['score']:.3f}  F1 {s['f1_0.1']:.3f}/{s['f1_0.3']:.3f}/{s['f1_0.5']:.3f}  "
              f"tp {s['tp']:3d} fp {s['fp']:3d} fn {s['fn']:3d}  FA/h {s['fa_per_h']:.2f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

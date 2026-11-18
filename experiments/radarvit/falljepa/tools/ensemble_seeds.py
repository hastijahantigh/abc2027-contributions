"""
Ensemble several runs (seeds or models) on the CV folds and score the result as tools/score_folds.py does:
official evaluator, --ignore-flagged, pooled over the folds' held-out streams and per fold.

    uv run python tools/ensemble_seeds.py --release /path/to/abc2027_uwb_train_v1 --folds DATA/cv_folds \\
        --out runs/_analysis/ens_seeds.json \\
        --set JF6=runs/UWB/JF5T-{fold}/JF5T-s0/TAG0/predictions/val-validate.csv,runs/UWB/JF5T-{fold}/...s1.../val-validate.csv \\
        --set JF6-dense=.../val-validate.probs.npz,.../val-validate.probs.npz

Each --set is NAME=PATH[,PATH...], `{fold}` standing for the fold name (A B C D, folders under --folds).

Event sets (.csv, submission files): the members' events are fused per stream. Taking events by
descending score, each joins the fused event it overlaps best at tIoU >= --iou, unless that fused event
already holds one from the same member; otherwise it opens a new one. A fused event spans the
score-weighted mean start and end, its score is the members' summed score / N (a member without an event
there counts 0) and its votes the number of members in it. Each set is scored under every keep rule:
votes >= k for k = 1..N, and fused score >= each --thr.

Dense sets (.npz, the dense head's probabilities): P(state) is averaged per timestep and decoded with the
dense head's Viterbi decoder (as tools/ensemble_dense.py).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from falljepa.model.aggregate import decode_events  # noqa: E402

official = None
KEYS = ("stream_id", "t_start", "t_end", "score", "t_alarm")


def tiou(a: dict, b: dict) -> float:
    inter = min(a["t_end"], b["t_end"]) - max(a["t_start"], b["t_start"])
    union = max(a["t_end"], b["t_end"]) - min(a["t_start"], b["t_start"])
    return max(inter, 0.0) / union if union > 0 else 0.0


def fuse(members: list[list[dict]], iou: float) -> list[dict]:
    """Fused events of one stream; each carries `votes` and the fused `score`."""
    n = len(members)
    events = sorted(((m, e) for m, evs in enumerate(members) for e in evs), key=lambda me: -me[1]["score"])
    clusters = []  # [fused event, {member: event}]
    for m, e in events:
        best, best_iou = None, iou
        for c in clusters:
            if m not in c[1] and (v := tiou(c[0], e)) >= best_iou:
                best, best_iou = c, v
        if best is None:
            clusters.append([dict(e), {m: e}])
            continue
        best[1][m] = e
        w = np.array([x["score"] for x in best[1].values()])
        w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
        for k in ("t_start", "t_end", "t_alarm"):
            best[0][k] = float(sum(wi * x[k] for wi, x in zip(w, best[1].values())))
    out = []
    for fused, got in clusters:
        fused["score"] = sum(x["score"] for x in got.values()) / n
        fused["votes"] = len(got)
        fused["t_alarm"] = min(max(fused["t_alarm"], fused["t_start"]), fused["t_end"])
        out.append(fused)
    return sorted(out, key=lambda e: e["t_start"])


def dense_events(paths: list[Path], durations: dict[str, float]) -> list[dict]:
    runs = [np.load(p) for p in paths]
    preds = []
    for sid in durations:
        T = min(len(r[f"logp/{sid}"]) for r in runs)
        p = np.mean([np.exp(r[f"logp/{sid}"][:T].astype(np.float32)) for r in runs], 0)
        p /= p.sum(1, keepdims=True)
        spread = np.mean([r[f"spread/{sid}"][:T].astype(np.float32) for r in runs], 0)
        preds += [{"stream_id": sid, **e} for e in decode_events(np.log(np.clip(p, 1e-12, None)), spread)]
    return [{k: p[k] for k in KEYS} for p in preds]


def summary(report: dict) -> dict:
    p = report["by_theta"]["0.3"]
    return {"score": report["ranking_score"]["f1_mean"], "tp": p["tp"], "fp": p["fp"], "fn": p["fn"],
            "fa_per_h": p["false_alarms_per_hour"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True, help="the official package (its evaluator scores)")
    ap.add_argument("--folds", type=Path, required=True, help="folder holding one fold folder per fold name")
    ap.add_argument("--fold-names", default="A,B,C,D")
    ap.add_argument("--set", action="append", required=True, help="NAME=PATH[,PATH...] with {fold}")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--thr", default="0.3,0.4,0.5,0.6,0.7")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    global official
    sys.path.insert(0, str(args.release / "src"))
    from abc2027 import evaluate as official

    folds = {}
    for f in args.fold_names.split(","):
        d = official.stream_durations(args.folds / f / "streams" / "val")
        g = [e for e in official.load_ground_truth(args.folds / f / "streams_events.csv", ignore_flagged=True)
             if e["stream_id"] in d]
        folds[f] = (d, g)
    thrs = [float(t) for t in args.thr.split(",")]
    result = {}
    for spec in args.set:
        name, paths = spec.split("=", 1)
        paths = paths.split(",")
        rules = {}  # rule -> fold -> predictions
        for f, (d, _) in folds.items():
            files = [Path(p.format(fold=f)) for p in paths]
            if files[0].suffix == ".npz":
                rules.setdefault("dense-avg", {})[f] = dense_events(files, d)
                continue
            members = [official.validate_predictions(p, d) for p in files]
            fused = []
            for sid in d:
                fused += [{"stream_id": sid, **e}
                          for e in fuse([[e for e in m if e["stream_id"] == sid] for m in members], args.iou)]
            for k in range(1, len(files) + 1):
                rules.setdefault(f"votes>={k}", {})[f] = [e for e in fused if e["votes"] >= k]
            for t in thrs:
                rules.setdefault(f"score>={t:g}", {})[f] = [e for e in fused if e["score"] >= t]
        result[name] = {}
        for rule, per in rules.items():
            row, ground, preds, durations = {}, [], [], {}
            for f, p in per.items():
                d, g = folds[f]
                p = [{k: e[k] for k in KEYS} for e in p]
                row[f] = summary(official.evaluate(g, p, d))
                ground += [{**e, "stream_id": f"{f}/{e['stream_id']}"} for e in g]
                preds += [{**e, "stream_id": f"{f}/{e['stream_id']}"} for e in p]
                durations.update({f"{f}/{k}": v for k, v in d.items()})
            row["pooled"] = summary(official.evaluate(ground, preds, durations))
            row["fold_mean"] = float(np.mean([row[f]["score"] for f in per]))
            result[name][rule] = row
            s = row["pooled"]
            print(f"{name:<12} {rule:<11} " + "  ".join(f"{f} {row[f]['score']:.3f}" for f in per)
                  + f"  | mean {row['fold_mean']:.3f}  pooled {s['score']:.3f}  tp {s['tp']} fp {s['fp']} fn {s['fn']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

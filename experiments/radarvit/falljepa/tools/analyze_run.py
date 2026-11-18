"""
Error analysis of one run's decoded streams, for the report:

    uv run python tools/analyze_run.py --run runs/UWB/Frame/U0/<id> [--stage val-validate] [--tune]

Reads the run's predictions/<stage>.csv, .report.json and .probs.npz, its metrics.csv files, and the
dataset's streams_events.csv / streams_composition.csv (runs/_dataset/ by default, where
tools/pull_sync.sh puts them). Writes <run>/analysis/<stage>.json:

    official        the official report (ranking score, every theta)
    matching        every prediction (TP / FP / ignored at theta 0.3) and every scored fall (hit or
                    missed), with the stream context of each: recording kind, activity, distance to
                    the nearest join, dataset, duration, overlap
    false_alarms    false alarms by cause; misses by cause; recall by dataset, fall type, duration
    boundaries      signed start/end errors and tIoU of the true positives
    scores          score of TP vs FP, and the ranking score when only predictions above a score are kept
    timesteps       per-timestep P(fall) against the labels: AUROC, average precision, calibration
    profiles        mean state probabilities around fall onsets, fall ends and joins between activities
    examples        short traces (10 Hz) around some hits, false alarms and misses
    curves          per-epoch training and validation values
    decoder         (--tune) the decoder grid on these probabilities: optimistic, it is tuned on them
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from falljepa.official import evaluate as official  # noqa: E402

RATE = 50.0
THETA = 0.3
JOIN_S = 1.0          # a false alarm within this distance of a join counts as "at a join"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def flagged(row: dict) -> bool:
    return (row.get("ignore", "").lower() == "true" or row.get("needs_review", "").lower() == "true"
            or row.get("quality", "").lower() == "low")


# ------------------------------------------------------------------ context


class StreamContext:
    """Which recording covers a time of a stream, and where its joins are."""
    def __init__(self, segments: list[dict]):
        self.segments = sorted(segments, key=lambda r: float(r["t_start"]))
        self.joins = np.array([float(s["t_start"]) for s in self.segments[1:]])

    def at(self, t: float) -> dict:
        covering = [s for s in self.segments if float(s["t_start"]) <= t < float(s["t_end"])]
        seg = covering[-1] if covering else self.segments[-1]
        return {"kind": seg["kind"], "label": seg["label"] or seg["kind"], "dataset": seg.get("dataset", ""),
                "join_distance_s": float(np.abs(self.joins - t).min()) if len(self.joins) else None}


def false_alarm_cause(pred: dict, ctx: dict, events: list[dict]) -> str:
    overlapping = [e for e in events if min(pred["t_end"], e["t_end"]) > max(pred["t_start"], e["t_start"])]
    if overlapping:
        return "overlaps a fall, tIoU too low or a duplicate"
    if ctx["join_distance_s"] is not None and ctx["join_distance_s"] <= JOIN_S:
        return "at a join between recordings"
    if ctx["kind"] == "fall":
        return "inside a fall recording, outside the fall"
    if ctx["kind"] == "adl":
        return "during a daily activity"
    return f"during '{ctx['kind']}' (empty room or other)"


# --------------------------------------------------------------- matching


def match(stage_preds: list[dict], ground: list[dict], events_meta: dict, contexts: dict[str, StreamContext]) -> dict:
    """Every prediction and scored fall at theta 0.3, with its context and cause."""
    by_gt, by_pred = defaultdict(list), defaultdict(list)
    for e in ground:
        by_gt[e["stream_id"]].append(e)
    for p in stage_preds:
        by_pred[p["stream_id"]].append(p)

    preds_out, falls_out, per_stream = [], [], {}
    for sid in sorted(set(by_gt) | set(by_pred)):
        gt, preds = by_gt[sid], by_pred[sid]
        result = official._match_stream(gt, preds, THETA)
        scored = result["scored"]
        ignored = [e for e in gt if e["ignore"]]
        matched_pred = {pi: (gi, iou) for pi, gi, iou in result["pairs"]}
        matched_gt = {gi: (pi, iou) for pi, gi, iou in result["pairs"]}
        ctx = contexts.get(sid)

        for pi, p in enumerate(preds):
            mid = (p["t_start"] + p["t_end"]) / 2
            c = ctx.at(mid) if ctx else {"kind": "?", "label": "?", "dataset": "", "join_distance_s": None}
            row = {"stream_id": sid, "t_start": p["t_start"], "t_end": p["t_end"], "score": p["score"],
                   "disagreement": p.get("disagreement"), **c}
            if pi in matched_pred:
                gi, iou = matched_pred[pi]
                row.update(status="TP", tiou=iou, start_error=p["t_start"] - scored[gi]["t_start"],
                           end_error=p["t_end"] - scored[gi]["t_end"])
            elif any(official._intervals_overlap(p["t_start"], p["t_end"], e["t_start"] - 2, e["t_end"] + 2) for e in ignored):
                row.update(status="ignored")
            else:
                row.update(status="FP", cause=false_alarm_cause(p, c, scored))
            preds_out.append(row)

        for gi, e in enumerate(scored):
            meta = events_meta.get((sid, round(e["t_start"], 3)), {})
            ious = [official.temporal_iou(p["t_start"], p["t_end"], e["t_start"], e["t_end"]) for p in preds]
            best = max(ious, default=0.0)
            row = {"stream_id": sid, "t_start": e["t_start"], "t_end": e["t_end"],
                   "duration": e["t_end"] - e["t_start"], "dataset": meta.get("dataset", ""),
                   "fall_type": meta.get("fall_type", ""), "best_tiou": best}
            if gi in matched_gt:
                row.update(status="hit", tiou=matched_gt[gi][1])
            else:
                row.update(status="missed", cause="detected, but tIoU < 0.3" if best > 0 else "no prediction overlaps it")
            falls_out.append(row)

        tp = len(result["pairs"])
        fp, fn = result["false_positive"], result["false_negative"]
        per_stream[sid] = {"tp": tp, "fp": fp, "fn": fn, "ignored": result["ignored_predictions"],
                           "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None}
    return {"predictions": preds_out, "falls": falls_out, "per_stream": per_stream}


def breakdowns(m: dict) -> dict:
    fps = [p for p in m["predictions"] if p["status"] == "FP"]
    falls = m["falls"]

    def recall_by(key, values=None):
        groups = defaultdict(list)
        for f in falls:
            groups[values(f) if values else f[key]].append(f["status"] == "hit")
        return {k: {"n": len(v), "recall": float(np.mean(v))} for k, v in sorted(groups.items())}

    def duration_bin(f):
        d = f["duration"]
        return "< 1 s" if d < 1 else "1-1.5 s" if d < 1.5 else "1.5-2 s" if d < 2 else "2-3 s" if d < 3 else ">= 3 s"

    return {
        "false_alarm_causes": dict(Counter(p["cause"] for p in fps).most_common()),
        "false_alarm_activities": dict(Counter(p["label"] for p in fps if p["cause"] == "during a daily activity").most_common()),
        "miss_causes": dict(Counter(f["cause"] for f in falls if f["status"] == "missed").most_common()),
        "recall_by_dataset": recall_by("dataset"),
        "recall_by_fall_type": recall_by("fall_type"),
        "recall_by_duration": recall_by(None, duration_bin),
        "counts": dict(Counter(p["status"] for p in m["predictions"])) | {"falls": len(falls),
                                                                         "missed": sum(f["status"] == "missed" for f in falls)},
    }


def boundary_stats(m: dict) -> dict:
    tps = [p for p in m["predictions"] if p["status"] == "TP"]
    if not tps:
        return {}
    s = np.array([p["start_error"] for p in tps])
    e = np.array([p["end_error"] for p in tps])
    q = lambda a: {k: float(v) for k, v in zip(("p10", "p25", "median", "p75", "p90"), np.percentile(a, [10, 25, 50, 75, 90]))}  # noqa: E731
    return {"n": len(tps), "start_error": q(s), "end_error": q(e),
            "start_abs_median": float(np.median(np.abs(s))), "end_abs_median": float(np.median(np.abs(e))),
            "start_early_fraction": float((s < 0).mean()), "end_late_fraction": float((e > 0).mean()),
            "tiou": q(np.array([p["tiou"] for p in tps])),
            "start_errors": s.round(3).tolist(), "end_errors": e.round(3).tolist()}


def score_curve(preds: list[dict], ground: list[dict], durations: dict) -> dict:
    """Ranking score when only predictions with score >= s are kept (on these streams: optimistic)."""
    scores = np.array([p["score"] for p in preds])
    out = []
    for s in np.unique(np.quantile(scores, np.linspace(0, 0.9, 19))) if len(scores) else []:
        kept = [p for p in preds if p["score"] >= s]
        r = official.evaluate(ground, kept, durations)
        out.append({"min_score": float(s), "kept": len(kept), "f1_mean": r["ranking_score"]["f1_mean"],
                    "precision": r["by_theta"]["0.3"]["precision"], "recall": r["by_theta"]["0.3"]["recall"]})
    return out


# -------------------------------------------------------------- timesteps


def labels_for(sid: str, T: int, events: list[dict]) -> np.ndarray:
    """1 inside scored falls, 0 elsewhere, -1 inside flagged falls widened by 2 s (the scorer ignores them)."""
    y = np.zeros(T, dtype=np.int8)
    for e in events:
        if e["stream_id"] != sid or not e["ignore"]:
            continue
        y[max(int(round((e["t_start"] - 2) * RATE)), 0):int(round((e["t_end"] + 2) * RATE))] = -1
    for e in events:
        if e["stream_id"] == sid and not e["ignore"]:
            y[int(round(e["t_start"] * RATE)):int(round(e["t_end"] * RATE))] = 1
    return y


def auroc_ap(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-p, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    P, N = tp[-1], fp[-1]
    if P == 0 or N == 0:
        return float("nan"), float("nan")
    auroc = float(np.trapezoid(np.concatenate([[0], tp / P]), np.concatenate([[0], fp / N])))
    ap = float(np.sum((tp / (tp + fp))[y == 1]) / P)
    return auroc, ap


def timestep_stats(probs, ground) -> dict:
    ps, ys = [], []
    for sid, (logp, _) in probs.items():
        y = labels_for(sid, len(logp), ground)
        keep = y >= 0
        ps.append(1 - np.exp(logp[keep, 0]))
        ys.append(y[keep])
    p, y = np.concatenate(ps), np.concatenate(ys).astype(np.int64)
    auroc, ap = auroc_ap(p, y)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    calib = [{"bin": f"{bins[i]:.1f}-{bins[i + 1]:.1f}", "n": int((idx == i).sum()),
              "mean_p": float(p[idx == i].mean()) if (idx == i).any() else None,
              "fall_fraction": float(y[idx == i].mean()) if (idx == i).any() else None} for i in range(10)]
    return {"auroc": auroc, "average_precision": ap, "fall_fraction": float(y.mean()),
            "mean_p_fall": float(p.mean()), "calibration": calib, "timesteps": int(len(y))}


def profiles(probs, ground, contexts) -> dict:
    """Mean state probabilities around onsets, ends of scored falls and joins between two activities."""
    def locked(times_by_stream, before, after):
        acc, n = None, 0
        for sid, times in times_by_stream.items():
            if sid not in probs:
                continue
            P = np.exp(probs[sid][0])
            for t in times:
                a, b = int(round((t - before) * RATE)), int(round((t + after) * RATE))
                if a < 0 or b > len(P):
                    continue
                acc = P[a:b] if acc is None else acc + P[a:b]
                n += 1
        if acc is None:
            return None
        mean = acc / n
        step = 5  # 10 Hz
        return {"n": n, "t": (np.arange(0, len(mean), step) / RATE - before).round(2).tolist(),
                **{name: mean[::step, k].round(4).tolist() for k, name in enumerate(("none", "begin", "middle", "end"))}}

    onsets, ends, joins = defaultdict(list), defaultdict(list), defaultdict(list)
    for e in ground:
        if not e["ignore"]:
            onsets[e["stream_id"]].append(e["t_start"])
            ends[e["stream_id"]].append(e["t_end"])
    for sid, ctx in contexts.items():
        falls = [(e["t_start"], e["t_end"]) for e in ground if e["stream_id"] == sid]
        for prev, seg in zip(ctx.segments[:-1], ctx.segments[1:]):
            t = float(seg["t_start"])
            if prev["kind"] == "adl" and seg["kind"] == "adl" and all(abs(t - a) > 5 and abs(t - b) > 5 for a, b in falls):
                joins[sid].append(t)

    join_hits = []
    for sid, times in joins.items():
        if sid in probs:
            pf = 1 - np.exp(probs[sid][0][:, 0])
            for t in times:
                a, b = max(int((t - 1) * RATE), 0), int((t + 1) * RATE)
                join_hits.append(float(pf[a:b].max()))
    return {"onset": locked(onsets, 3.0, 5.0), "end": locked(ends, 5.0, 3.0), "activity_join": locked(joins, 3.0, 3.0),
            "activity_joins_with_p_fall_over_0.5": float(np.mean(np.array(join_hits) > 0.5)) if join_hits else None,
            "activity_joins": len(join_hits)}


def examples(probs, m, ground, contexts, per_kind: int = 3) -> list[dict]:
    """10 Hz traces of 16 s around a few hits, false alarms and misses (the clearest of each)."""
    picks = []
    ranked = sorted([f for f in m["falls"] if f["status"] == "hit"], key=lambda f: f["tiou"])
    hits = ranked[-1:] + [f for f in ranked[:per_kind - 1] if f is not ranked[-1]]   # the best and the worst
    fps = sorted([p for p in m["predictions"] if p["status"] == "FP"], key=lambda p: -p["score"])[:per_kind]
    misses = [f for f in m["falls"] if f["status"] == "missed"][:per_kind]
    for kind, items in (("hit", hits), ("false alarm", fps), ("miss", misses)):
        for it in items:
            sid = it["stream_id"]
            if sid not in probs:
                continue
            centre = (it["t_start"] + it["t_end"]) / 2
            a, b = max(centre - 8, 0), centre + 8
            P = np.exp(probs[sid][0])
            ia, ib = int(a * RATE), min(int(b * RATE), len(P))
            ctx = contexts.get(sid)
            picks.append({
                "kind": kind, "stream_id": sid, "t0": a, "detail": it.get("cause") or it.get("status"),
                "t": (np.arange(ia, ib, 5) / RATE).round(2).tolist(),
                "p_fall": (1 - P[ia:ib:5, 0]).round(3).tolist(),
                "begin": P[ia:ib:5, 1].round(3).tolist(), "end": P[ia:ib:5, 3].round(3).tolist(),
                "falls": [[e["t_start"], e["t_end"], e["ignore"]] for e in ground
                          if e["stream_id"] == sid and e["t_end"] > a and e["t_start"] < b],
                "predictions": [[p["t_start"], p["t_end"], p["status"]] for p in m["predictions"]
                                if p["stream_id"] == sid and p["t_end"] > a and p["t_start"] < b],
                "joins": [float(j) for j in (ctx.joins if ctx else []) if a < j < b],
                "segments": [[float(s["t_start"]), float(s["t_end"]), s["kind"], s["label"]] for s in (ctx.segments if ctx else [])
                             if float(s["t_end"]) > a and float(s["t_start"]) < b],
            })
    return picks


# ------------------------------------------------------------------ curves


def curves(run: Path) -> dict:
    """Per-epoch values from every csv/fit-*/metrics.csv (a resumed run has several)."""
    per_epoch: dict[int, dict] = defaultdict(dict)
    for path in sorted(run.glob("csv/fit-*/metrics.csv")):
        for row in read_csv(path):
            if not row.get("epoch"):
                continue
            ep = int(float(row["epoch"]))
            for k, v in row.items():
                if v not in ("", None) and (k.endswith("_epoch") or k.startswith(("val/", "lr-"))):
                    per_epoch[ep][k.removesuffix("_epoch")] = float(v)
    keys = sorted({k for d in per_epoch.values() for k in d})
    epochs = sorted(per_epoch)
    return {"epoch": epochs, **{k: [per_epoch[e].get(k) for e in epochs] for k in keys}}


# -------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--stage", default="val-validate")
    parser.add_argument("--dataset", type=Path, default=Path("runs/_dataset"))
    parser.add_argument("--tune", action="store_true", help="also run the decoder grid on these probabilities")
    args = parser.parse_args()

    pred_dir = args.run / "predictions"
    report = json.loads((pred_dir / f"{args.stage}.report.json").read_text())
    preds = [{"stream_id": r["stream_id"], "t_start": float(r["t_start"]), "t_end": float(r["t_end"]),
              "score": float(r["score"]), "t_alarm": float(r["t_alarm"]) if r.get("t_alarm") else None,
              "disagreement": float(r["disagreement"]) if r.get("disagreement") else None}
             for r in read_csv(pred_dir / f"{args.stage}.csv")]
    npz = np.load(pred_dir / f"{args.stage}.probs.npz")
    probs = {n.split("/", 1)[1]: (npz[n].astype(np.float64), npz[f"spread/{n.split('/', 1)[1]}"].astype(np.float64))
             for n in npz.files if n.startswith("logp/")}
    if any(n.startswith("mask/") for n in npz.files):
        sys.exit("cross-validation probabilities (with masks): use tune_decoder for those, this analyses whole streams")

    events_path = args.dataset / "streams_events.csv"
    ground = [e for e in official.load_ground_truth(events_path, ignore_flagged=True) if e["stream_id"] in probs]
    events_meta = {(r["stream_id"], round(float(r["t_start"]), 3)): r for r in read_csv(events_path)}
    comp = defaultdict(list)
    for r in read_csv(args.dataset / "streams_composition.csv"):
        if r["stream_id"] in probs:
            comp[r["stream_id"]].append(r)
    contexts = {sid: StreamContext(rows) for sid, rows in comp.items()}
    durations = {sid: len(v[0]) / RATE for sid, v in probs.items()}

    m = match(preds, ground, events_meta, contexts)
    out = {
        "run": str(args.run), "stage": args.stage, "official": report,
        "flagged_falls": sum(e["ignore"] for e in ground), "scored_falls": sum(not e["ignore"] for e in ground),
        "hours": sum(durations.values()) / 3600,
        "matching": m, "breakdowns": breakdowns(m), "boundaries": boundary_stats(m),
        "scores": {"tp": [p["score"] for p in m["predictions"] if p["status"] == "TP"],
                   "fp": [p["score"] for p in m["predictions"] if p["status"] == "FP"],
                   "tp_disagreement": [p["disagreement"] for p in m["predictions"] if p["status"] == "TP"],
                   "fp_disagreement": [p["disagreement"] for p in m["predictions"] if p["status"] == "FP"],
                   "min_score_curve": score_curve(preds, ground, durations)},
        "timesteps": timestep_stats(probs, ground),
        "profiles": profiles(probs, ground, contexts),
        "examples": examples(probs, m, ground, contexts),
        "curves": curves(args.run),
    }
    if args.tune:
        from falljepa.tune_decoder import load_probs, tune
        results = tune(load_probs([pred_dir / f"{args.stage}.probs.npz"]), ground)
        out["decoder"] = [{"settings": s, "f1_mean": r["ranking_score"]["f1_mean"],
                           "f1": {t: r["by_theta"][t]["f1"] for t in ("0.1", "0.3", "0.5", "0.7")},
                           "fa_per_hour": r["by_theta"]["0.3"]["false_alarms_per_hour"],
                           "start_error_median": r["ranking_score"]["start_error_median_abs"]} for s, r in results[:15]]

    dest = args.run / "analysis" / f"{args.stage}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=float))
    b = out["breakdowns"]
    print(f"ranking score {report['ranking_score']['f1_mean']:.3f}; {b['counts']}")
    print(f"false alarms: {b['false_alarm_causes']}")
    print(f"misses: {b['miss_causes']}")
    print(f"timesteps: AUROC {out['timesteps']['auroc']:.3f}, AP {out['timesteps']['average_precision']:.3f}")
    print(f"written {dest}")


if __name__ == "__main__":
    main()

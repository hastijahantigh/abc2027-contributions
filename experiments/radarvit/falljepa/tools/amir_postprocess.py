"""
Decodes saved FallJEPA probabilities with Amir's post-processing (teammate_pipeline_20261013,
Appendix A) and scores every step with the official evaluator.

    uv run python tools/amir_postprocess.py --release /path/to/abc2027_uwb_train_v1 \\
        --probs runs/UWB/Ablation/A1-s*/*/predictions/val-validate.probs.npz --out runs/_analysis/amir_pp.csv

The chain is Amir's: detect (P(fall) >= theta, or our Viterbi runs) -> refine each detection with
the official labeller (annotate.propose_fall internals) on its source recording, found from the
signal (join peaks) -> merge overlaps -> NMS 5.9 s -> persistence gate -> optionally one event per
recording. Each step is scored on its own so its gain is visible. Refinement and the gate read the
unnormalized stream CIR; the labeller is loaded unchanged from the release's src/abc2027/annotate.py.

--ensemble also scores the mean P(fall) of all --probs files as one extra model.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from falljepa.model.aggregate import NONE, decode_runs, propose_events, runs_to_events, weighted_median  # noqa: E402
from falljepa.module import in_mask  # noqa: E402
from falljepa.official import evaluate as official  # noqa: E402

from falljepa import postprocess as pp  # noqa: E402
from falljepa.postprocess import (BLEND_HALF, MAX_SEGMENT, MIN_REFINED, MIN_WINDOW, join_peaks, join_score,  # noqa: E402
                                  join_span, merge, nms, one_per_recording, threshold_runs)

RATE = 50.0
A = None                                   # the official annotate module, set in main()


def viterbi_runs(logp, p, switch_penalty=2.0, min_frames=20):
    """Our U0 decoder (min 0.4 s), scored by the maximum P(fall) like Amir's events."""
    return [(int(a), int(b), float(p[a:b].max())) for a, b in decode_runs(logp, kind="viterbi", switch_penalty=switch_penalty)
            if b - a >= min_frames]


# The package's functions with the official labeller bound
def official_bounds(cir, raw=True):
    return pp.official_bounds(A, cir, raw)


def refine(cir, peaks, start, stop, joins=True, raw=True):
    return pp.refine(A, cir, peaks, start, stop, joins, raw)


def gate(events, cir, peaks, joins=True):
    return pp.gate(A, events, cir, peaks, joins)


# ---------------------------------------------------------------- the chain, step by step

STEPS = ["detect+NMS", "+ window refine (grid motion)", "+ join-aware refine (grid motion)",
         "+ raw-frame motion", "+ persistence gate", "+ one per recording"]


def chain(p, logp, cir, peaks, detector, theta):
    """{step: [(start, stop, score)]} for one stream; each step adds one change."""
    det = threshold_runs(p, theta) if detector == "threshold" else viterbi_runs(logp, p)
    out = {STEPS[0]: nms(merge(det))}
    variants = {STEPS[1]: (False, False), STEPS[2]: (True, False), STEPS[3]: (True, True)}
    for name, (joins, raw) in variants.items():
        out[name] = nms(merge([(*refine(cir, peaks, a, b, joins, raw), c) for a, b, c in det]))
    out[STEPS[4]] = gate(out[STEPS[3]], cir, peaks)
    out[STEPS[5]] = one_per_recording(out[STEPS[4]], peaks)
    return out


def regressed(bounds, p, a, b):
    """The A4 boundaries of run [a, b): P(fall)-weighted medians of its regressed start and end."""
    w = p[a:b] + 1e-6
    s, e = weighted_median(bounds[a:b, 0], w), weighted_median(bounds[a:b, 1], w)
    if e <= s:
        return a, b
    return max(0, int(round(s * RATE))), min(len(p), int(round(e * RATE)))


def refine_a4(cir, peaks, bounds, p, a, b, mode):
    """The rule with the regression head as a fallback. 'fallback': regressed when the span is not a
    whole recording (2-20 s) or the rule gives < 0.2 s. 'agree': also regressed when the rule's
    interval and the regressed one have tIoU < 0.1 (the rule picked another transition)."""
    T = cir.shape[1]
    ra, rb = regressed(bounds, p, a, b)
    lo, hi = join_span(peaks, a, b, T)
    if not MIN_WINDOW <= hi - lo <= MAX_SEGMENT:
        return ra, rb
    left, right = official_bounds(cir[:, lo:hi], raw=True)
    if right - left < MIN_REFINED:
        return ra, rb
    s, e = lo + left, lo + right
    if mode == "agree" and tiou(s, e, ra, rb) < 0.1:
        return ra, rb
    return s, e


A4_STEPS = ["regressed bounds + NMS", "regressed bounds + NMS + gate", "rule, regressed fallback + NMS + gate",
            "rule unless it disagrees + NMS + gate"]


def chain_a4(p, logp, bounds, cir, peaks, detector, theta):
    if detector == "proposals":
        det = [(int(round(e["t_start"] * RATE)), int(round(e["t_end"] * RATE)), e["score"])
               for e in propose_events(logp, np.zeros(len(p)), bounds, proposal_threshold=0.5, proposal_stride=5, nms_iou=0.3)]
        out = {"A4b decoder (proposals + NMS .3)": sorted(det)}
        out["+ NMS 5.9 s + gate"] = gate(nms(merge(det)), cir, peaks)
        out["+ rule on the recording (raw) + NMS + gate"] = gate(nms(merge(
            [(*refine(cir, peaks, a, b, True, True), c) for a, b, c in det])), cir, peaks)
        out["+ rule, regressed fallback + NMS + gate"] = gate(nms(merge(
            [(*refine_a4(cir, peaks, bounds, p, a, b, "fallback"), c) for a, b, c in det])), cir, peaks)
        return out
    det = threshold_runs(p, theta) if detector == "threshold" else viterbi_runs(logp, p)
    reg = nms(merge([(*regressed(bounds, p, a, b), c) for a, b, c in det]))
    out = {A4_STEPS[0]: reg, A4_STEPS[1]: gate(reg, cir, peaks)}
    for name, mode in ((A4_STEPS[2], "fallback"), (A4_STEPS[3], "agree")):
        out[name] = gate(nms(merge([(*refine_a4(cir, peaks, bounds, p, a, b, mode), c) for a, b, c in det])), cir, peaks)
    return out


def score(events_by_sid, ground, masks):
    durations = {sid: float(m.sum()) / RATE for sid, m in masks.items()}
    gt = [g for g in ground if g["stream_id"] in masks and in_mask(masks[g["stream_id"]], g["t_start"], g["t_end"])]
    rows = [{"stream_id": sid, "t_start": a / RATE, "t_end": b / RATE, "score": c, "t_alarm": a / RATE}
            for sid, evs in events_by_sid.items() for a, b, c in evs if in_mask(masks[sid], a / RATE, b / RATE)]
    report = official.evaluate(gt, rows, durations)
    r, at3 = report["ranking_score"], report["by_theta"]["0.3"]
    return {"F1_mean": r["f1_mean"], "F1@0.1": report["by_theta"]["0.1"]["f1"], "F1@0.3": at3["f1"],
            "F1@0.5": report["by_theta"]["0.5"]["f1"], "FA/h": r["false_alarms_per_hour"],
            "TP": at3["tp"], "FP": at3["fp"], "FN": at3["fn"], "start_err": r["start_error_median_abs"]}


def tiou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def check_port(release, split, ground):
    """Amir's section 6.2/6.3 check: the rule on each scored fall's recording against its label.
    His numbers on the 186 training falls: mean tIoU .766 (grid motion) and .840 (raw), >= .5 for
    85.5 % and 93.5 %. 'joins' takes the recording from the detected join peaks instead of the
    composition file, with the true fall as the detection."""
    comp = list(csv.DictReader(open(release / "streams_composition.csv")))
    falls = [g for g in ground if g["stream_id"].startswith(split + "_") and not g["ignore"]]
    res = {k: [] for k in ("oracle grid", "oracle raw", "joins raw")}
    for sid in sorted({g["stream_id"] for g in falls}):
        with h5py.File(release / "streams" / split / f"stream_{sid.split('_')[1]}.h5", "r") as h:
            cir = h["cir"][:]
        peaks = join_peaks(join_score(cir))
        segs = [c for c in comp if c["stream_id"] == sid and c["kind"] == "fall"]
        for g in (g for g in falls if g["stream_id"] == sid):
            mid = (g["t_start"] + g["t_end"]) / 2
            seg = [c for c in segs if float(c["t_start"]) <= mid < float(c["t_end"])]
            if not seg:
                continue
            lo = int(round(float(seg[0]["t_start"]) * RATE)) + BLEND_HALF + 1
            hi = int(round(float(seg[0]["t_end"]) * RATE)) - BLEND_HALF
            for raw, key in ((False, "oracle grid"), (True, "oracle raw")):
                left, right = official_bounds(cir[:, lo:hi], raw)
                res[key].append(tiou((lo + left) / RATE, (lo + right) / RATE, g["t_start"], g["t_end"]))
            a, b = int(g["t_start"] * RATE), int(g["t_end"] * RATE)
            s, e = refine(cir, peaks, a, b, joins=True, raw=True)
            res["joins raw"].append(tiou(s / RATE, e / RATE, g["t_start"], g["t_end"]))
        print(f"{sid}: {len(res['oracle raw'])} falls so far", flush=True)
    for key, v in res.items():
        v = np.array(v)
        print(f"{key:12s} n={len(v)}  mean tIoU {v.mean():.3f}  >=.3 {np.mean(v >= .3):.3f}  "
              f">=.5 {np.mean(v >= .5):.3f}  >=.7 {np.mean(v >= .7):.3f}")


def load_model(path):
    data = np.load(path)
    model = {}
    for name in data.files:
        if name.startswith("logp/"):
            sid = name.split("/", 1)[1]
            logp = data[name].astype(np.float64)
            mask = data[f"mask/{sid}"].astype(bool) if f"mask/{sid}" in data.files else np.ones(len(logp), bool)
            model[sid] = {"logp": logp, "p": 1 - np.exp(logp[:, NONE]), "mask": mask,
                          "bounds": data[f"bounds/{sid}"].astype(np.float64) if f"bounds/{sid}" in data.files else None}
    return model


def label_of(path):
    parts = Path(path).parts
    i = parts.index("predictions")
    return f"{parts[i - 2]}/{Path(path).name.replace('.probs.npz', '')}"


def main(argv=None):
    global A
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--probs", nargs="+", type=Path, default=[])
    ap.add_argument("--check", metavar="SPLIT", help="only verify the port on SPLIT's labelled falls (train: Amir's table)")
    ap.add_argument("--events", type=Path, help="default: <release>/streams_events.csv")
    ap.add_argument("--detectors", nargs="+", default=["threshold", "viterbi"], choices=["threshold", "viterbi"])
    ap.add_argument("--theta", type=float, nargs="+", default=[0.3])
    ap.add_argument("--ensemble", action="store_true", help="also score the mean P(fall) of all files")
    ap.add_argument("--no-single", action="store_true", help="score only the ensemble")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    A = pp.load_annotate(args.release)
    ground = official.load_ground_truth(args.events or args.release / "streams_events.csv", ignore_flagged=True)
    if args.check:
        return check_port(args.release, args.check, ground)

    models = {} if args.no_single else {label_of(p): load_model(p) for p in args.probs}
    sources = {}                                         # the F1 of the decoder each run used
    for p in args.probs:
        rep = Path(str(p).replace(".probs.npz", ".report.json"))
        if rep.exists():
            sources[label_of(p)] = round(json.loads(rep.read_text())["ranking_score"]["f1_mean"], 4)
    if args.ensemble:
        parts = [load_model(p) for p in args.probs]
        ens = {}
        for sid in parts[0]:
            p = np.mean([m[sid]["p"] for m in parts], axis=0)
            logp = np.log(np.clip(np.mean([np.exp(m[sid]["logp"]) for m in parts], axis=0), 1e-12, None))
            ens[sid] = {"logp": logp, "p": p, "mask": np.all([m[sid]["mask"] for m in parts], axis=0), "bounds": None}
        models[f"ensemble of {len(parts)}"] = ens

    sids = sorted({sid for m in models.values() for sid in m})
    cirs, peaks = {}, {}
    for sid in sids:
        split, idx = sid.split("_")
        with h5py.File(args.release / "streams" / split / f"stream_{idx}.h5", "r") as h:
            cirs[sid] = h["cir"][:]
        peaks[sid] = join_peaks(join_score(cirs[sid]))
        print(f"{sid}: {cirs[sid].shape[1]} steps, {len(peaks[sid])} join peaks", flush=True)

    rows = []
    for label, model in models.items():
        masks = {sid: s["mask"] for sid, s in model.items()}
        own = {}                                         # our U0 decoder, nothing of Amir's
        for sid, s in model.items():
            evs = runs_to_events(decode_runs(s["logp"], kind="viterbi", switch_penalty=2.0), s["logp"],
                                 np.zeros(len(s["p"])), min_duration_s=0.4)
            own[sid] = [(round(e["t_start"] * RATE), round(e["t_end"] * RATE), e["score"]) for e in evs]
        row = {"model": label, "detector": "FallJEPA", "step": "U0 decoder (Viterbi 2, min 0.4 s)", **score(own, ground, masks)}
        report = sources.get(label)
        row["as_run_F1"] = report
        rows.append(row)
        print(f"{label:45s} {'FallJEPA':14s} {row['step']:36s} F1 {row['F1_mean']:.3f}  @.5 {row['F1@0.5']:.3f}  "
              f"FA/h {row['FA/h']:5.1f}  TP/FP/FN {row['TP']}/{row['FP']}/{row['FN']}  as run {report}", flush=True)
        for detector in args.detectors:
            for theta in (args.theta if detector == "threshold" else [None]):
                per_step = {step: {} for step in STEPS}
                for sid, s in model.items():
                    for step, evs in chain(s["p"], s["logp"], cirs[sid], peaks[sid], detector, theta).items():
                        per_step[step][sid] = evs
                for step in STEPS:
                    row = {"model": label, "detector": detector if theta is None else f"threshold {theta}",
                           "step": step, **score(per_step[step], ground, masks)}
                    rows.append(row)
                    print(f"{label:45s} {row['detector']:14s} {step:36s} F1 {row['F1_mean']:.3f}  "
                          f"@.5 {row['F1@0.5']:.3f}  FA/h {row['FA/h']:5.1f}  TP/FP/FN {row['TP']}/{row['FP']}/{row['FN']}", flush=True)
        if any(s["bounds"] is not None for s in model.values()):
            for detector in ["threshold", "viterbi", "proposals"]:
                theta = args.theta[0] if detector == "threshold" else None
                per_step = {}
                for sid, s in model.items():
                    for step, evs in chain_a4(s["p"], s["logp"], s["bounds"], cirs[sid], peaks[sid], detector, theta).items():
                        per_step.setdefault(step, {})[sid] = evs
                for step, evs in per_step.items():
                    row = {"model": label, "detector": "A4 " + (detector if theta is None else f"threshold {theta}"),
                           "step": step, **score(evs, ground, masks)}
                    rows.append(row)
                    print(f"{label:45s} {row['detector']:16s} {step:44s} F1 {row['F1_mean']:.3f}  "
                          f"@.5 {row['F1@0.5']:.3f}  FA/h {row['FA/h']:5.1f}  TP/FP/FN {row['TP']}/{row['FP']}/{row['FN']}", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), restval="")
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()

"""
Amir's post-processing (src/falljepa/postprocess.py) on the join-free model's saved outputs
(predictions/val-validate.probs.npz), step by step, on the held-out streams of every run. Nothing is
retrained.

    uv run python tools/sweep_pp.py --release /path/to/abc2027_uwb_train_v1 --out runs/_analysis/pp_jf2/sweep \\
        --run cv/s0/A=FOLDS/A:runs/UWB/JF2-A/JF2-s0/<TAG>/predictions/val-validate.probs.npz --run ...

Each --run is SET/SEED/FOLD=FOLD_DIR:PROBS; FOLD_DIR holds streams/val/stream_XX.h5 and
streams_events.csv. A stream file shared by several runs is read once.

Two paths:
    dense   the dense head's P(fall) through the whole chain (decode, F, merge, NMS, gate, E3, M, P)
    event   the model's own events: every slot with P > .05 (its MBR span), whose raw parents are its
            group's member proposals (regrouped from the saved proposals exactly as the model groups
            them); then F, merge, NMS, gate, E3, M and the slot threshold P. With every step off and
            the model's NMS (tIoU .5, centres 3 s) and P > .5 it is the model's own output.

Configurations, each with and without joins (joins only matter to F, the gates and M):
    ladder  the steps added one at a time in the chain's order
    loo     the full chain with one step left out (or swapped: broad gate for C1)
    grid    a few values per step, for leave-one-fold-out tuning (tools/report_pp.py)

Writes <out>.npz: counts [streams x runs, configs, 3 tIoU (.1 .3 .5), 3 (tp fp fn)], the run and
stream of each row and the configurations (JSON).
"""
import argparse
import dataclasses
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
from falljepa.model import sets  # noqa: E402

THETAS = (0.1, 0.3, 0.5)
GROUP_IOU, GROUP_GAP = 0.2, 150                  # the model's grouping (JF2: tIoU .2, 3 s)
SLOT_FLOOR = 0.05

C = pp.Config
OFF = dict(refine=False, merge=False, nms_s=None, nms_iou=None, gate="none", extend=False, move=False, min_score=None)


def dense_configs() -> dict[str, pp.Config]:
    out = {}
    for joins in (True, False):
        j = "J" if joins else "N"
        steps = [("raw", {}), ("+F", dict(refine=True)), ("+merge", dict(merge=True)), ("+NMS", dict(nms_s=pp.NMS_RADIUS_S)),
                 ("+G", dict(gate="broad")), ("C1", dict(gate="local")), ("+E3", dict(extend=True)),
                 ("+M", dict(move=True)), ("+P", dict(min_score=pp.MIN_SCORE))]
        kw = dict(OFF, joins=joins)
        for name, change in steps:
            kw.update(change)
            out[f"dense/{j}/ladder/{name}"] = C(**kw)
        full = C(joins=joins)
        for name, change in [("-F", dict(refine=False)), ("-merge", dict(merge=False)), ("-NMS", dict(nms_s=None)),
                             ("-gate", dict(gate="none")), ("G", dict(gate="broad")), ("-E3", dict(extend=False)),
                             ("-M", dict(move=False)), ("-P", dict(min_score=None))]:
            out[f"dense/{j}/loo/{name}"] = dataclasses.replace(full, **change)
        for theta, nms_s, gate, ms in itertools.product((0.2, 0.3, 0.4, 0.5), (3.0, 4.0, 5.0, 5.9, 7.0),
                                                        ("none", "broad", "local"), (None, 0.4, 0.5, 0.6, 0.7)):
            out[f"dense/{j}/grid/theta={theta},nms={nms_s},gate={gate},P={ms}"] = dataclasses.replace(
                full, theta=theta, nms_s=nms_s, gate=gate, min_score=ms)
    return out


def event_configs() -> dict[str, pp.Config]:
    out = {}
    base = dict(OFF, nms_s=3.0, nms_iou=0.5, min_score=0.5)
    for joins in (True, False):
        j = "J" if joins else "N"
        kw = dict(base, joins=joins)
        out[f"event/{j}/ladder/model"] = C(**kw)
        for name, change in [("+F", dict(refine=True)), ("+merge", dict(merge=True)), ("NMS5.9", dict(nms_s=pp.NMS_RADIUS_S)),
                             ("+C1", dict(gate="local")), ("+E3", dict(extend=True)), ("+M", dict(move=True))]:
            kw.update(change)
            out[f"event/{j}/ladder/{name}"] = C(**kw)
        full = C(**kw)
        for name, change in [("-F", dict(refine=False)), ("-merge", dict(merge=False)), ("NMS3", dict(nms_s=3.0)),
                             ("-gate", dict(gate="none")), ("G", dict(gate="broad")), ("-E3", dict(extend=False)),
                             ("-M", dict(move=False))]:
            out[f"event/{j}/loo/{name}"] = dataclasses.replace(full, **change)
        model = C(**dict(base, joins=joins))
        for name, change in [("+F", dict(refine=True, merge=True)), ("+C1", dict(gate="local")), ("+G", dict(gate="broad")),
                             ("+E3", dict(extend=True)), ("+M", dict(move=True)), ("NMS5.9", dict(nms_s=pp.NMS_RADIUS_S))]:
            out[f"event/{j}/alone/{name}"] = dataclasses.replace(model, **change)
        for ms, nms_s, refine, gate, extend, move in itertools.product((0.3, 0.4, 0.5, 0.6, 0.7, 0.8), (3.0, 4.0, 5.0, 5.9),
                                                                       (False, True), ("none", "local"), (False, True), (False, True)):
            out[f"event/{j}/grid/P={ms},nms={nms_s},F={int(refine)},gate={gate},E3={int(extend)},M={int(move)}"] = C(
                **dict(base, joins=joins, min_score=ms, nms_s=nms_s, refine=refine, merge=refine, gate=gate,
                       extend=extend, move=move))
    # the event path without joins and without F / gate / M does not depend on joins: keep both anyway
    return out


CONFIGS = {**dense_configs(), **event_configs()}
NAMES = list(CONFIGS)

_A = _EVAL = None


def _init(release: str):
    global _A, _EVAL
    _A = pp.load_annotate(release)
    sys.path.insert(0, str(Path(release) / "src"))
    from abc2027 import evaluate
    _EVAL = evaluate


def model_events(slots: np.ndarray, proposals: np.ndarray) -> tuple[list, list]:
    """Slot events (start, stop, P) with P > .05 and, for each, its group's members (start, stop, P)."""
    rows = [{"logit": float(l), "start": int(s), "end": int(e)} for l, s, e, _ in proposals]
    groups = sets.stream_groups(rows, GROUP_IOU, GROUP_GAP)
    events, parents = [], []
    for g, _, logit, a, b in slots:
        p = 1.0 / (1.0 + np.exp(-logit))
        if p <= SLOT_FLOOR or b <= a:
            continue
        members = groups[int(g)][0] if int(g) < len(groups) else []
        events.append((int(a), int(b), float(p)))
        parents.append([(m["start"], m["end"], 1.0 / (1.0 + np.exp(-m["logit"]))) for m in members])
    return events, parents


def stream_counts(job) -> list[np.ndarray]:
    """For each run on this stream: [configs, 3 thetas, 3 (tp, fp, fn)]."""
    path, ground, runs = job
    with h5py.File(path, "r") as h:
        cir = np.asarray(h["cir"][:], dtype=np.float32)
    T = cir.shape[1]
    peaks = pp.join_peaks(pp.join_score(cir))
    stream = pp.Stream(_A, cir, peaks)
    ground = [{"stream_id": "s", **g} for g in ground]
    duration = {"s": T / 50.0}
    out = []
    for p, events, parents in runs:
        counts = np.zeros((len(NAMES), len(THETAS), 3), np.int32)
        dense_cands = {}
        for i, name in enumerate(NAMES):
            cfg = CONFIGS[name]
            if name.startswith("dense/"):
                key = (cfg.theta, cfg.min_run, cfg.gap)
                if key not in dense_cands:
                    dense_cands[key] = pp.decode(p, cfg)
                res = pp.run(stream, dense_cands[key], cfg, p)
            else:
                res = pp.run(stream, pp.from_events(events, parents), cfg, None)
            preds = [{"stream_id": "s", "t_start": c.start / 50.0, "t_end": c.stop / 50.0, "score": c.score, "t_alarm": None}
                     for c in res]
            for k, th in enumerate(THETAS):
                r = _EVAL.score_events(ground, preds, duration, th)
                counts[i, k] = (r["tp"], r["fp"], r["fn"])
        out.append(counts)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--run", action="append", required=True, help="SET/SEED/FOLD=FOLD_DIR:PROBS")
    ap.add_argument("--out", required=True, type=Path, help="output prefix (.npz)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    _init(str(args.release))

    by_path: dict[str, dict] = {}
    for spec in args.run:
        name, rest = spec.split("=", 1)
        folder, probs = rest.rsplit(":", 1)
        folder = Path(folder)
        ground = _EVAL.load_ground_truth(folder / "streams_events.csv", ignore_flagged=True)
        data = np.load(probs)
        n = 0
        for key in sorted(k for k in data.files if k.startswith("logp/")):
            sid = key.split("/", 1)[1]
            path = folder / "streams" / "val" / f"stream_{sid.split('_')[1]}.h5"
            if not sid.startswith("val_") or not path.exists():
                continue
            p = 1 - np.exp(data[key].astype(np.float64)[:, 0])
            events, parents = model_events(data[f"slots/{sid}"], data[f"proposals/{sid}"])
            g = [{k: e[k] for k in ("t_start", "t_end", "ignore")} for e in ground if e["stream_id"] == sid]
            item = by_path.setdefault(str(path), {"ground": g, "runs": [], "owners": []})
            item["runs"].append((p, events, parents))
            item["owners"].append((name, sid))
            n += 1
        print(f"{name}: {n} streams", flush=True)
    jobs = [(path, it["ground"], it["runs"]) for path, it in by_path.items()]
    owners = [it["owners"] for it in by_path.values()]
    print(f"{len(NAMES)} configurations x {sum(len(o) for o in owners)} run-streams ({len(jobs)} files), "
          f"{args.workers} workers", flush=True)

    rows, who = [], []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(args.release),)) as pool:
        for i, res in enumerate(pool.map(stream_counts, jobs)):
            rows.extend(res)
            who.extend(owners[i])
            print(f"  {Path(jobs[i][0]).parent.parent.parent.name}/{Path(jobs[i][0]).stem} done ({i + 1}/{len(jobs)})", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.with_suffix(".npz"), counts=np.stack(rows), runs=np.array([w[0] for w in who]),
                        streams=np.array([w[1] for w in who]), names=np.array(NAMES),
                        configs=json.dumps({n: dataclasses.asdict(c) for n, c in CONFIGS.items()}))
    print(f"wrote {args.out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()

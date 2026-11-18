"""
Radar signal around every annotated fall, for tools/analyze_falls.py. Needs the streams, so run it
on the server (CPU, a few minutes):

    uv run python tools/fall_signal.py --release /home/abc2027/dataset/abc2027_uwb_train_v1 \
        [--stats runs/_cache/normalizer_train.json] [--splits train val] [--out runs/_analysis/fall_signal.npz]

The signal is z-scored per (radar, bin) with the training normalizer, as the model sees it. For
each fall (aligned on its annotated start, from -BEFORE_S to +AFTER_S, 50 Hz):

    motion      [N, 3, L]   per radar: RMS over bins of the temporal difference / its training std,
                            smoothed over 0.2 s (how much the scene changes, ~1 when typical)
    state       [N, 3, L]   per radar: d_before / (d_before + d_after), the distance of each timestep's
                            profile to the static profile before the fall and after it: 0 before, 1 lying
                            still (the organizers' annotator places the fall on this step)
    change      [N, 3]      per radar: mean squared difference of the after and before profiles
    bin_motion  [N, 3, 368] mean |difference| / std per bin inside the label (where in range it moves)
    bin_change  [N, 3, 368] after - before profile (where the static scene changed)
    valid       [N, L]      the timestep exists in the stream

Static profiles: before = median over [start - 2.8, start - 0.5] s, after = median over
[end + 0.5, end + 3] s; a fall recording is cropped to at least 3 s before and 5 s after its fall,
so both lie inside the fall's own recording. Events are in streams_events.csv order (`event_id`).
"""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

RATE = 50
BEFORE_S, AFTER_S = 6.0, 11.0
SMOOTH_S = 0.2


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--release", required=True, type=Path)
    p.add_argument("--stats", type=Path, default=Path("runs/_cache/normalizer_train.json"))
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--out", type=Path, default=Path("runs/_analysis/fall_signal.npz"))
    args = p.parse_args(argv)

    stats = json.loads(args.stats.read_text())
    mean = np.asarray(stats["mean"], np.float32)[:, None, :]
    std = np.asarray(stats["std"], np.float32)[:, None, :]
    diff_std = np.asarray(stats["diff_std"], np.float32)[:, None, :]
    with (args.release / "streams_events.csv").open(newline="") as f:
        events = [r for r in csv.DictReader(f) if r["stream_id"].split("_")[0] in args.splits]

    pre, post = int(BEFORE_S * RATE), int(AFTER_S * RATE)
    L = pre + post
    N = len(events)
    out = {k: np.full(shape, np.nan, np.float32) for k, shape in
           (("motion", (N, 3, L)), ("state", (N, 3, L)), ("change", (N, 3)),
            ("bin_motion", (N, 3, 368)), ("bin_change", (N, 3, 368)))}
    valid = np.zeros((N, L), bool)
    kernel = np.ones(int(SMOOTH_S * RATE)) / int(SMOOTH_S * RATE)

    by_stream: dict[str, list[int]] = {}
    for i, e in enumerate(events):
        by_stream.setdefault(e["stream_id"], []).append(i)
    for sid, idx in sorted(by_stream.items()):
        split, num = sid.split("_")
        with h5py.File(args.release / "streams" / split / f"stream_{num}.h5", "r") as h:
            cir = h["cir"]
            T = cir.shape[1]
            for i in idx:
                e = events[i]
                a, b = round(float(e["t_start"]) * RATE), round(float(e["t_end"]) * RATE)
                lo, hi = a - pre - 1, a + post             # one extra step for the first difference
                rlo, rhi = max(lo, 0), min(hi, T)
                z = (cir[:, rlo:rhi, :].astype(np.float32) - mean) / std
                d = np.diff(z, axis=1) / diff_std          # [3, n-1, 368], d[:, k] = z[k+1] - z[k]
                m = np.sqrt(np.mean(d ** 2, axis=2))
                m = np.stack([np.convolve(r, kernel, mode="same") for r in m])
                off = rlo + 1 - (lo + 1)                    # where d[:, 0] sits in the output
                n = m.shape[1]
                out["motion"][i, :, off:off + n] = m
                valid[i, off:off + n] = True

                def seg(t0, t1):
                    return z[:, max(t0 - rlo, 0):max(t1 - rlo, 0)]
                before = np.median(seg(a - int(2.8 * RATE), a - int(0.5 * RATE)), axis=1)    # [3, 368]
                after = np.median(seg(b + int(0.5 * RATE), b + int(3.0 * RATE)), axis=1)
                db = np.sum((z - before[:, None]) ** 2, axis=2)
                da = np.sum((z - after[:, None]) ** 2, axis=2)
                s = db / (db + da + 1e-12)
                start = off - 1                             # z[:, 0] is one step before d[:, 0]
                s = s[:, max(-start, 0):]
                start = max(start, 0)
                out["state"][i, :, start:start + s.shape[1]] = s[:, :L - start]
                out["change"][i] = np.mean((after - before) ** 2, axis=1)
                out["bin_change"][i] = after - before
                inside = d[:, max(a - rlo - 1, 0):max(b - rlo - 1, 0)]
                if inside.shape[1]:
                    out["bin_motion"][i] = np.abs(inside).mean(axis=1)
        print(f"{sid}: {len(idx)} falls")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, event_id=np.array([e["event_id"] for e in events]), valid=valid,
                        t=(np.arange(L) - pre) / RATE, **out)
    print(f"written {args.out}: {N} falls, traces {-BEFORE_S}..{AFTER_S} s around the start")


if __name__ == "__main__":
    main()

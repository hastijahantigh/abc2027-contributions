"""
A tiny release with the official layout: streams/{train,val}/stream_XX.h5, streams_events.csv and
streams_composition.csv (8 s segments joined with a 0.2 s overlap), plus folds.csv (path,fold),
which puts the segments of the training streams in folds 0 and 1 alternately.
"""
import csv
from pathlib import Path

import h5py
import numpy as np

RATE = 50


def _stream(rng: np.random.Generator, seconds: float, falls: list[tuple[float, float]]) -> np.ndarray:
    T = int(seconds * RATE)
    cir = rng.normal(0, 0.05, size=(3, T, 368)).astype(np.float32)
    # Piecewise-constant background (segment joins every ~8 s), like the concatenated streams
    for a in range(0, T, 8 * RATE):
        cir[:, a:a + 8 * RATE] += rng.normal(0, 1, size=(3, 1, 368)).astype(np.float32)
    bins = np.arange(368)
    for t0, t1 in falls:
        a, b = int(t0 * RATE), int(t1 * RATE)
        for t in range(a, b):
            centre = 60 + 120 * (t - a) / max(b - a, 1)          # a reflector sweeping in range
            cir[:, t] += 3 * np.exp(-0.5 * ((bins - centre) / 6) ** 2)
    return cir


def make_release(root: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    plans = {
        ("train", 0): (40.0, [(5.0, 6.6), (21.0, 23.0)]),
        ("train", 1): (36.0, [(9.0, 10.2), (27.0, 29.4)]),
        ("val", 0): (30.0, [(7.0, 8.8), (19.0, 20.4)]),
    }
    rows, composition, folds = [], [], []
    for (split, index), (seconds, falls) in plans.items():
        folder = root / "streams" / split
        folder.mkdir(parents=True, exist_ok=True)
        cir = _stream(rng, seconds, falls)
        with h5py.File(folder / f"stream_{index:02d}.h5", "w") as handle:
            handle["cir"] = cir
            handle["t"] = np.arange(cir.shape[1]) / RATE
            handle["radar_serials"] = np.array([29171, 29444, 30722])
        sid = f"{split}_{index:02d}"
        for j, a in enumerate(range(0, int(seconds), 8)):
            path = f"activities/Synth_{sid}_{j}.json"
            composition.append({"stream_id": sid, "segment_id": f"{sid}_{j:04d}", "kind": "adl", "path": path,
                                "label": "Synth", "dataset": "", "t_start": f"{max(a - 0.2, 0):.3f}",
                                "t_end": f"{min(a + 8, seconds):.3f}", "source_t0": "0", "source_t1": "8"})
            if split == "train":
                folds.append({"path": path, "fold": j % 2})
        for k, (t0, t1) in enumerate(falls):
            rows.append({"stream_id": sid, "event_id": f"{sid}_{k:04d}", "t_start": f"{t0:.3f}", "t_end": f"{t1:.3f}",
                         "fall_type": "Chute01", "ignore": "false", "quality": "low" if k == 1 and split == "train" else "normal",
                         "needs_review": "false", "source_path": "", "source_t_start": "", "source_t_end": "", "dataset": "Dataset 1"})
    for name, table in (("streams_events.csv", rows), ("streams_composition.csv", composition), ("folds.csv", folds)):
        with (root / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    return root

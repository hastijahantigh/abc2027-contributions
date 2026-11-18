"""
Write the events of one post-processing configuration of tools/sweep_pp.py as submission files, one per run,
so they can be analysed like the model's own predictions/val-validate.csv.

    uv run python tools/dump_pp_events.py --release /path/to/abc2027_uwb_train_v1 --config event/J/ladder/+M \\
        --run OUT.csv=FOLD_DIR:PROBS [--run ...]
"""
import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

from sweep_pp import CONFIGS, model_events, pp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--config", default="event/J/ladder/+M")
    ap.add_argument("--run", action="append", required=True, help="OUT.csv=FOLD_DIR:PROBS")
    args = ap.parse_args(argv)
    cfg = CONFIGS[args.config]
    if args.config.startswith("dense/"):
        raise SystemExit("only event configurations")
    annotate = pp.load_annotate(str(args.release))
    for spec in args.run:
        out, rest = spec.split("=", 1)
        folder, probs = rest.rsplit(":", 1)
        data = np.load(probs)
        rows = []
        for key in sorted(k for k in data.files if k.startswith("logp/")):
            sid = key.split("/", 1)[1]
            path = Path(folder) / "streams" / "val" / f"stream_{sid.split('_')[1]}.h5"
            if not sid.startswith("val_") or not path.exists():
                continue
            with h5py.File(path, "r") as h:
                cir = np.asarray(h["cir"][:], dtype=np.float32)
            stream = pp.Stream(annotate, cir, pp.join_peaks(pp.join_score(cir)))
            events, parents = model_events(data[f"slots/{sid}"], data[f"proposals/{sid}"])
            for c in pp.run(stream, pp.from_events(events, parents), cfg, None):
                rows.append([sid, f"{c.start / 50.0:.3f}", f"{c.stop / 50.0:.3f}", f"{c.score:.6f}", f"{c.start / 50.0:.3f}"])
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as h:
            w = csv.writer(h)
            w.writerow(["stream_id", "t_start", "t_end", "score", "t_alarm"])
            w.writerows(rows)
        print(f"{out}: {len(rows)} events", flush=True)


if __name__ == "__main__":
    main()

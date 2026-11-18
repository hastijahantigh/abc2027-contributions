"""
Per-fold scores of chosen post-processing configurations from tools/sweep_pp.py outputs.

    uv run python tools/pp_fold_table.py runs/_analysis/pp_jf2/sweep.npz [--config NAME ...]

For each configuration and seed: the official ranking score (mean F1 at tIoU .1/.3/.5) of each CV fold,
the fold mean and the CV pooled over folds (counts summed, as tools/score_folds.py pools them).
"""
import argparse
from pathlib import Path

import numpy as np

DEFAULT = ["event/J/ladder/model",                                  # the model's own output
           "event/J/ladder/+M",                                     # full chain with joins: F, merge, NMS 5.9, C1, E3, M
           "event/N/grid/P=0.3,nms=3.0,F=0,gate=local,E3=1,M=0"]    # best join-free: C1 + E3, slot P >= .3


def ranking(counts: np.ndarray) -> float:
    tp, fp, fn = (counts[..., k].astype(float) for k in range(3))
    return float(np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", type=Path)
    ap.add_argument("--config", action="append")
    args = ap.parse_args()
    z = np.load(args.sweep)
    counts, runs, names = z["counts"], z["runs"], list(z["names"])
    for name in args.config or DEFAULT:
        c = counts[:, names.index(name)]
        print(name)
        for seed in sorted({r.split("/")[1] for r in runs if r.startswith("cv/")}):
            folds = sorted({r.split("/")[2] for r in runs if r.startswith(f"cv/{seed}/")})
            per = {f: ranking(c[runs == f"cv/{seed}/{f}"].sum(0)) for f in folds}
            pooled = c[np.char.startswith(runs.astype(str), f"cv/{seed}/")].sum(0)
            print(f"  {seed}  " + "  ".join(f"{f} {v:.3f}" for f, v in per.items())
                  + f"  | fold mean {np.mean(list(per.values())):.3f}  pooled {ranking(pooled):.3f}")


if __name__ == "__main__":
    main()

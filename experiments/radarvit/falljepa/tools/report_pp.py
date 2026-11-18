"""
Tables from tools/sweep_pp.py: what each post-processing step adds to the join-free model's outputs.

    uv run python tools/report_pp.py runs/_analysis/pp_jf2/sweep.npz [--out runs/_analysis/pp_jf2/report.json]

Run names are SET/SEED/FOLD (cv/s0/A, off/x1/D, ...); a "set" is SET/SEED (cv/s0, cv/s1, off/x1, off/x8)
and its score is the official ranking score (mean F1 at tIoU .1/.3/.5) of the counts summed over its
folds, as tools/score_folds.py pools them.

    ladder   steps added one at a time, with the change each brings
    loo      the full chain with one step left out: its drop is the step's share in the full chain
    alone    (event path) one step on top of the model's own output
    tuned    per path, with and without joins, leave-one-fold-out on the CV (both seeds): the grid
             point best on the other three folds,
             scored on the held-out fold; then the point best on all four folds, its plateau (one value
             changed at a time) and its score on the official-split runs, which take no part in tuning
"""
import argparse
import json
from pathlib import Path

import numpy as np


def ranking(counts: np.ndarray) -> np.ndarray:
    """Summed counts [..., 3 thetas, 3] -> mean F1 over the thetas [...]."""
    tp, fp, fn = (counts[..., k].astype(float) for k in range(3))
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    return f1.mean(-1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sweep", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    z = np.load(args.sweep)
    counts, runs, names = z["counts"], z["runs"], list(z["names"])
    col = {n: i for i, n in enumerate(names)}
    sets = sorted({"/".join(r.split("/")[:2]) for r in runs})
    folds = sorted({r.split("/")[2] for r in runs if r.startswith("cv/")})

    def summed(prefix: str, fold: str | None = None) -> np.ndarray:
        m = np.array([r.startswith(prefix + "/") and (fold is None or r.endswith("/" + fold)) for r in runs])
        return counts[m].sum(0)                                   # [configs, 3, 3]

    per_set = {s: summed(s) for s in sets}
    score = {s: ranking(c) for s, c in per_set.items()}
    result = {"sets": sets, "tables": {}}

    def line(name: str) -> dict:
        i = col[name]
        row = {s: round(float(score[s][i]), 4) for s in sets}
        row.update({f"{s}@.3": "/".join(str(int(v)) for v in per_set[s][i, 1]) for s in sets})
        return row

    print("score = mean F1 at tIoU .1/.3/.5 (ignore flagged); TP/FP/FN at .3; sets: " + ", ".join(sets))
    for path in ("dense", "event"):
        for j, jname in (("J", "with joins"), ("N", "join-free")):
            for kind in ("ladder", "loo", "alone"):
                rows = [n for n in names if n.startswith(f"{path}/{j}/{kind}/")]
                if not rows:
                    continue
                print(f"\n{path} path, {jname}, {kind}")
                print(f"  {'':12s}" + "".join(f"{s:>26s}" for s in sets))
                ref = col[f"{path}/{j}/ladder/{'+P' if path == 'dense' else '+M'}"] if kind == "loo" else \
                    col[f"{path}/{j}/ladder/model"] if kind == "alone" else None
                table = {}
                prev = None
                for n in rows:
                    i = col[n]
                    base = prev if kind == "ladder" else ref
                    cells = []
                    for s in sets:
                        v = score[s][i]
                        d = "" if base is None else f" {v - score[s][base]:+.3f}"
                        cells.append(f"{v:.3f}{d} {'/'.join(str(int(x)) for x in per_set[s][i, 1]):>9s}")
                    print(f"  {n.split('/')[-1]:12s}" + "".join(f"{c:>26s}" for c in cells))
                    table[n] = line(n)
                    prev = i
                result["tables"][f"{path}/{j}/{kind}"] = table

    # Leave-one-fold-out tuning on the CV grids (both seeds pooled per fold)
    result["tuned"] = {}
    cv_fold = {f: summed("cv", f) for f in folds}
    for path in ("dense/J", "dense/N", "event/J", "event/N"):
        grid = [n for n in names if n.startswith(f"{path}/grid/")]
        if not grid or len(folds) < 2:
            continue
        gi = np.array([col[n] for n in grid])
        fold_scores = {f: ranking(cv_fold[f][gi]) for f in folds}
        lofo_counts, rows = 0, {}
        for f in folds:
            others = ranking(sum(cv_fold[g][gi] for g in folds if g != f))
            k = int(np.argmax(others))
            lofo_counts = lofo_counts + cv_fold[f][gi[k]]
            rows[f] = {"config": grid[k], "score": float(fold_scores[f][k]), "on_others": float(others[k])}
        pooled = ranking(sum(cv_fold[f][gi] for f in folds))
        k = int(np.argmax(pooled))
        best = grid[k]
        # the plateau: one value changed at a time
        head, values = best.rsplit("/", 1)
        parts = dict(kv.split("=") for kv in values.split(","))
        plateau = {}
        for key in parts:
            alts = {}
            for n in grid:
                h, v = n.rsplit("/", 1)
                d = dict(kv.split("=") for kv in v.split(","))
                if h == head and all(d[x] == parts[x] for x in parts if x != key):
                    alts[d[key]] = round(float(pooled[grid.index(n)]), 4)
            plateau[key] = alts
        transfer = {s: round(float(score[s][col[best]]), 4) for s in sets}
        result["tuned"][path] = {"lofo": rows, "lofo_pooled": float(ranking(lofo_counts)), "best": best,
                                 "best_cv_pooled": float(pooled[k]), "best_on_sets": transfer, "plateau": plateau}
        print(f"\n{path} tuned on the CV grid (both seeds):")
        for f, r in rows.items():
            print(f"  LOFO {f}: {r['score']:.3f}  {r['config']}")
        print(f"  LOFO pooled {ranking(lofo_counts):.3f}")
        print(f"  best on all folds: {best}  CV pooled {pooled[k]:.3f}")
        print("  on each set: " + "  ".join(f"{s} {v:.3f}" for s, v in transfer.items()))
        for key, alts in plateau.items():
            print(f"    {key:6s} " + "  ".join(f"{v}: {s:.3f}" for v, s in alts.items()))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

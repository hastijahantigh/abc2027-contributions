# ABC2027 subject-held-out CV: a quick guide

A cross-validation setup for the ABC2027 UWB fall-detection challenge. It is built so the score
predicts the **test** score: each fold holds out people the model has never seen, in streams made
exactly the way the organizers made val and test. The fold data is already built on the server.
You only need `cv_folds.py` (numpy and h5py) and this file.

## The data

```
/home/abc2027/dataset/abc2027_uwb_cv_folds/
    segments.h5            shared by all folds; do not move or delete it
    A/  B/  C/  D/         one folder per fold, laid out like the official package
```

| Fold | Held-out fall subjects | Held-out ADL subjects | Held-out falls (scored) |
|---|---|---|---|
| A | 2, 3 | 4, 9 | 120 (69) |
| B | 1, 7 | 5, 12 | 120 (58) |
| C | 5, 6 | 6, 10 | 119 (59) |
| D (= official split) | 8, 9 | 0, 11 | 119 (67) |

**What a fold is.** All official train and val clips are pooled: 8 fall subjects, 8 ADL subjects,
478 falls. Each fold holds out 2 of each, like the test. The test has new people in both groups,
and fall people and ADL people are different persons (README v1.1).

**Streams of a fold:**

- **Training, `train_00…train_25`:** 26 streams, about 12.3 h. These are the fold's training
  clips stitched twice by the organizers' generator.
- **Held-out, `val_00…val_09`:** 10 streams, about 4.2 h. These are the held-out clips stitched
  twice with the official val/test settings. Every held-out fall appears once per stitching, so
  each is scored in 2 contexts. Read `fold.json` for which streams form each stitching.

**Fold D** is the official split. Its `train_00–12` and `val_00–04` are the official streams
themselves, so fold D, stitching 0, is exactly the official val.

The new streams were checked against the originals:

- **Same recipe:** rebuilding the 18 official streams this way gives the official files bit for bit.
- **Exact labels:** under every fall label, the signal is an exact copy of that fall's recording.
- **Same format as the official streams:** `cir` is float32 `[3 radars, T, 368]` at 50 Hz; the 368
  values are 184 complex range bins, I then Q.

The stream files are HDF5 virtual datasets that read their frames from `segments.h5`. h5py reads
them like ordinary files, so slicing `cir[:, a:b]` works as usual.

## Quick start

```python
import sys; sys.path.insert(0, "/path/to/cv_kit")
import cv_folds

fold = cv_folds.Fold("A")                # elsewhere: cv_folds.Fold("A", root="/path/to/folds")
print(fold)                               # streams and held-out subjects

for sid in fold.train_ids:                # 'train_00' ...
    x = fold.read(sid)                    # float32 [3, T, 368]; fold.read(sid, a, b) for timesteps [a, b)
    y, w = fold.labels(sid)               # y [T]: 1 inside a fall; w [T]: 0 on flagged falls, 1 elsewhere
    events = fold.events(sid)             # falls: t_start, t_end (s), fall_type, flagged, source_path

with fold.open("train_00") as h:          # for many window reads, keep the file open
    window = h["cir"][:, 1000:1800]

# after training on fold A's train streams: predict every held-out stream
rows = [("val_00", 12.30, 14.10, 0.93), ...]          # (stream_id, t_start s, t_end s, score)
cv_folds.write_predictions("predA.csv", rows)
```

Scoring uses the organizers' evaluator, unchanged, under the test rule (flagged falls ignored):

```bash
python cv_folds.py info                                                    # the four folds
# elsewhere: python cv_folds.py --root /path/to/folds --release /path/to/abc2027_uwb_train_v1 score ...
python cv_folds.py score A=predA.csv B=predB.csv C=predC.csv D=predD.csv --out cv.json
```

It prints the **pooled** score (all folds' held-out streams as one set), each fold, each held-out
stitching, and the folds' mean, SD and worst fold. The score is the official ranking score: mean
event F1 at tIoU 0.1, 0.3 and 0.5.

**Labels.** A fall is flagged if it has `quality = low` or `needs_review`. Flagged falls are
neither rewarded nor penalized by the scorer: an unmatched prediction within ±2 s of one is dropped, not
counted as a false alarm. So giving
them weight 0 in training (the default of `labels()`) matches the test rule.
`labels(sid, flagged_weight=0.5)` or `flagged_margin_s=2` change that.

## Rules: your score is comparable only if you follow them

1. **Fit everything on the fold's training streams only.** That includes the network and anything
   else that is fitted: normalizer or feature statistics, location or scene priors,
   post-processing parameters, thresholds. Never fit anything on the fold's `val_*` streams, on the
   official package, or on other folds.
2. **No selection on the held-out streams.** Use a fixed number of epochs and the final or a
   weight-averaged model, not the best epoch on held-out. If you tune a decoder or
   post-processing, tune it on 3 folds' held-out predictions and apply it to the 4th.
3. **Predict all held-out streams of every fold.** That is 10 per fold, one CSV per fold.
4. **Score with `cv_folds.py score`.** Report the pooled score and the per-fold scores.
5. **Run at least 2 seeds.** GPU training isn't reproducible per seed, and runs vary by about
   0.02–0.03 F1. Pooled differences smaller than about 0.03 are noise.
6. **Work on stream signals only.** Don't try to detect repeated activity pieces across streams;
   that is against the challenge rules.

## Good to know

- **Same ids in every fold.** Stream ids repeat between folds (each fold has a `train_00`), so key
  caches by fold and stream.
- **Don't write into the fold folders.** Put caches, normalizers and outputs in your own directory.
- **Validation cost.** Decoding all held-out streams is about 4.2 h of signal per fold, twice the official val. Validating
  every few epochs is fine, since nothing should be selected on it.
- **Subjects are inferred, not given.** A fall subject is the suffix of the session folder:
  `Test_P1_3`, `Test_P2_3` and `Test_P3_3` are person 3 at the 3 mat positions. An ADL subject is
  the activity file's suffix (`Boire_5.json` → 5). The organizers don't state this; if it were wrong,
  the folds would still separate whole recordings.
- **Permission errors.** If reading a file fails with a permission error, ask Mehdi to open read
  access for your account.

Full design and assumptions: `docs/stream_cv.md` on the `stream-cv` branch of Mehdi's repo.

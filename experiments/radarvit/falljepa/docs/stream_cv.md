# Subject-held-out cross-validation from re-stitched streams

Branch `stream-cv`, 2026-10-23. This describes the new evaluation setup for the ABC2027 UWB fall
model: what it is, why, the assumptions behind it, how to build its data on the server, how to
train and score it, and how to decide between models with it. No fold data has been built yet.

## 1. Why a new CV

- **Official val alone is too small and used up.** It has 5 streams, about 2 h and 67 clean falls.
  We have chosen epochs, decoders and variants on it many times; picking the epoch on it alone
  added about +0.08.
- **The old 6-fold CV leaked.** It held out recordings inside the training streams, so the
  held-out signal stayed in the training windows as unlabeled context. Fold F1 fell to about 0.1
  by the last epoch while official val stayed at 0.4–0.7.
- **Leave-one-stream-out leaks people.** Every official training stream contains every training
  fall subject (each subject's sessions are spread over 4–12 streams), and identical ADL pieces
  are reused across streams.

The new CV holds out **people**, not streams. A fold's held-out clips are stitched into their own
streams, exactly the way the organizers built val and test.

## 2. How the official streams are made (from the organizers' generator)

`scripts/build_streams.py` and `src/abc2027/streams.py` of the release, seed 2027, `configs/streams.yaml`:

1. **Fall clips.** Each fall recording is used once: the labeled fall plus a random margin from
   the same recording, U(3, 8) s before and U(5, 10) s after (shorter if the recording ends).
2. **ADL clips.** Each ADL recording is cut into random U(8, 20) s pieces, repeated with fresh
   offsets 3 times for train and 2 times for val and test. That is why overlapping pieces of one ADL
   file appear in several streams.
3. **Other clips.** Each "other" (Autre/Empty, from the fall corpus) recording is used whole.
4. **Fall rate.** ADL pieces are drawn until the split reaches 57.19 falls per hour.
5. **Mixing.** All pieces of a split are shuffled into one list and cut in order into 30-minute
   streams. The clip before a fall is a random recording, so continuity across joins is not real
   continuity.
6. **Joins.** Each join is a 0.2 s linear crossfade that never touches a labeled fall.
7. **Test flags.** On test only, `quality=low` and `needs_review` falls get `ignore=true`. That
   is what `--ignore-flagged` reproduces on our streams.

## 3. Assumptions

| # | Assumption | Basis | If wrong |
|---|---|---|---|
| A1 | Test fall and ADL subjects are unseen in train and val | README v1.1: "Training, validation, and the held-out test use different subjects"; test built "with a private seed and held-out subjects"; subject splits are "disjoint inside each corpus" | The CV is pessimistic, not wrong |
| A2 | A fall subject is the suffix of the session folder: `Test_P1_3`, `Test_P2_3`, `Test_P3_3` are person 3 at mat positions 1–3 | The same 8 suffixes appear in all three datasets; train holds 1, 2, 3, 5, 6, 7 and val 8, 9, which is consistent with a subject split. Not stated by the organizers | Folds still separate whole recordings, but a "person" could leak across positions |
| A3 | An ADL subject is the file suffix (`Boire_5.json` → 5) | Train ADL holds 4, 5, 6, 9, 10, 12 and val 0, 11, each appearing across all 15 activities | As A2 |
| A4 | Fall subjects and ADL subjects are different people (F5 ≠ A5) | README v1.1 | — |
| A5 | Autre/Empty recordings belong to their fall session's subject | They sit in the session folders | Minor |
| A6 | ADL and Autre/Empty clips contain no fall | Organizers' generator and README | Same as for the official streams |
| A7 | Labels are as good as the official ones (±0.5–1 s bounds; the 2.00 s labels are suspect) | README | Same for every fold |
| A8 | Generating streams from `segments/` is allowed | README v1.1 Rules: "Additional training streams are allowed" | — |

The suffixes missing from train and val (fall 4; ADL 1, 2, 3, 7, 8) hint at which IDs the test
uses. That is only a hint.

## 4. The folds

`configs/folds/subject_folds.yaml`. All train and val clips are pooled (8 fall subjects, 8 ADL
subjects, 478 falls). Each fold holds out 2 fall subjects and 2 ADL subjects, like the test.

| Fold | Held-out fall | Held-out ADL | Held-out falls (clean) | Training falls (clean) |
|---|---|---|---|---|
| A | 2, 3 | 4, 9 | 120 (69) | 358 (184) |
| B | 1, 7 | 5, 12 | 120 (58) | 358 (195) |
| C | 5, 6 | 6, 10 | 119 (59) | 359 (194) |
| D (official split) | 8, 9 | 0, 11 | 119 (67) | 359 (186) |

Each fall subject has about 60 falls (15 Chute04) and 5–6 min of Autre/Empty; each ADL subject
has about 40–50 min. Clean falls are the ones not flagged `quality=low` or `needs_review`; only they are scored
under `--ignore-flagged`. Fall pairs were chosen to balance clean falls, and the ADL pairings in
A–C are arbitrary but disjoint. Every clip's fold is listed in `metadata/fold_clips.csv`.

## 5. The streams of a fold

| | Training streams | Held-out streams |
|---|---|---|
| Clips | the fold's training clips | the fold's held-out clips |
| Stitchings | 2 (`--train-stitchings`), about 12.5 h | 2 (`--heldout-stitchings`), 10 streams, about 4.2 h |
| ADL piece uses | 3 (official train setting) | 2 (official val/test setting) |
| Fall rate | 57.19 /h | 57.19 /h |
| Stream ids | `train_00…train_25` | `val_00…val_09` |

- **What a stitching is.** One run of the official generator over the set with its own seed
  `[2027, fold index, 0 train / 1 held-out, stitching]`: every fall once with fresh margins, ADL cut at
  fresh points, a new shuffle, 30-minute streams. Stitchings differ only by seed.
- **Built once.** All streams are built before training and never re-stitched during training.
- **Two training stitchings.** They give every fall in two contexts, which is more context, not more
  falls.
- **Two held-out stitchings (10 streams, about 4.2 h).** Each one uses every held-out fall once,
  so each fall is scored in 2 arrangements (neighbours, joins, margins). This reduces the luck of
  one arrangement at inference cost only. It adds contexts, not falls: a fold still has about 60
  clean held-out falls.
- **Fold D.** Its first training stitching is the 13 official training streams and its first
  held-out stitching is the 5 official val streams (symlinked). So fold D, stitching 0, is exactly
  the official val.

## 6. Reproducing the official recipe exactly

The generator as shipped does **not** reproduce the official streams from `segments/`: it plans
56.71 falls/h instead of 57.19, and every stream differs. It takes a clip's duration as n frames /
50, while the official plan used:

- **ADL and Autre/Empty clips:** `floor((n − 1) / 50)` whole seconds. The radar clocks are integer
  seconds. This holds for all 144 such clips: the 54 used whole (30 val ADL files, 24 Autre/Empty)
  show their duration in the CSV, and the RNG replay recovers the other 90 (training ADL).
- **Fall clips (`t_last`, only matters when the after-margin is cut short):** `(n − 1) / 50`, the last
  sample time. One exception is `falls/Dataset 3/Test_P3_8/Chute02_270.json`, which used 774 frames
  (n = 776).

With these rules, verified on 2026-10-22 with a full rebuild:

- all 18 official train and val `.h5` files come out **bit for bit identical** (`cir`, `t`,
  `radar_serials`, attributes);
- `streams_events.csv` is identical;
- `streams_composition.csv` differs only by a blank `dataset` column on the 24 "other" rows.

A build with a different seed (12345) was then checked fall by fall, for all 478 falls:

- the labeled interval is an exact copy of the fall's frames in `segments/`;
- it sits at the right offset in its clip;
- its source bounds, `quality`, `needs_review` and fall type are the official ones.

So the labels of our new streams are exact in the same sense as the official ones.

## 7. Storage: HDF5 virtual datasets

Raw streams cost 3 × 368 × 50 × 4 B ≈ 0.8 GB per hour. Written as ordinary files, 4 folds would take
about 70 GB. But a stream is only copies of segment frames plus the 0.2 s crossfades, so:

- **One copy of the frames.** `segments/*.npy` is copied once into `<out>/segments.h5` (4.9 GB,
  one contiguous dataset per clip).
- **Virtual `cir`.** Each stream's `cir` is an **HDF5 virtual dataset** that maps frame ranges of
  `segments.h5`. Only the crossfaded frames are stored in the stream file (`crossfades`), so a
  30-minute stream file is about 5–8 MB instead of about 400 MB.
- **No loader change.** h5py reads it like any other file. This was tested: the project's
  `StreamStore` reads windows from a virtual `val_04` identical to the official file.
- **Total disk** is about 5 GB for all 4 folds.
- **Keep `<out>` together.** Sources are stored as paths relative to the stream file. A missing
  `segments.h5` reads as NaN, and `StreamStore.read_z` raises on non-finite values.
- LMDB would not save space: it is a container, not compression. Compressing float32 radar noise
  gains little, and float16 loses precision near the 5.12e-6 low-quality threshold the README uses.
- `--storage full` writes ordinary files with the organizers' writer instead.

## 8. Building the data (on the server)

```bash
cd /home/abc2027/mehdi            # the repo, branch stream-cv
REL=/home/abc2027/dataset/abc2027_uwb_train_v1

# 1. Optional: the duration rules reproduce the official plan (reads array shapes only, seconds)
uv run python tools/build_fold_streams.py --release $REL --check-official

# 2. Optional: print every fold's plan, write nothing
uv run python tools/build_fold_streams.py --release $REL --dry-run

# 3. Build all folds
uv run python tools/build_fold_streams.py --release $REL --out DATA/folds
```

Step 3 does, in order, and stops at the first failure:

1. Replans the official train and val and requires an identical plan.
2. Writes `DATA/folds/segments.h5` once (4.9 GB).
3. Rebuilds all 18 official streams virtually and requires them **bit for bit identical** to the
   official files.
4. Builds each fold.
5. Checks every fall label of every fold against `segments/`.
6. Computes the SHA-256 of every stream's content and every label file, and compares them with
   `configs/folds/stream_checksums.csv` (156 files: 36 streams and 3 CSVs per fold). The build
   fails on any difference. The checksums hash the content, not the file, so a build with
   `--storage full` or with fold D's symlinks gives the same values. They come from a reference
   build on 2026-10-25, which was checked in three ways:
   - its 18 rebuilt official streams have content identical to the official files, and their
     478 events rows and 2346 composition rows equal the official CSVs;
   - a second build into a fresh folder was identical in all 156 files, and `segments.h5` was
     byte-identical;
   - fold D built with `--storage full`, written by the organizers' own `_write_stream`, gave
     the same checksums.

   They apply with the default fold file, seed and stitchings; otherwise none are compared
   (`--checksums-out FILE` writes your own). To check an existing build without rebuilding:
   `--out DIR --verify-only --checksums-ref configs/folds/stream_checksums.csv`.

Per fold it writes:

```
DATA/folds/<fold>/streams/train/stream_XX.h5   virtual (fold D: train_00-12 symlinked to the official files)
DATA/folds/<fold>/streams/val/stream_XX.h5     virtual (fold D: val_00-04 symlinked)
DATA/folds/<fold>/streams_events.csv           labels, official format
DATA/folds/<fold>/streams_composition.csv      clips per stream, official format
DATA/folds/<fold>/groups.csv                   path, fold (= subject index), subject: for the balanced sampler
DATA/folds/<fold>/fold.json                    held-out subjects, seeds, the streams of every stitching
```

Options: `--folds A` builds one fold, `--train-stitchings N` changes the training amount,
`--seed` changes the stitching seeds, and `--storage full` writes ordinary files. The release must
include `segments/`. The organizers' `build_streams.py` imports matplotlib only for a figure; if
it is missing, the tool stubs it.

## 9. Training a fold

Use each fold folder as the release. Each fold needs its **own** `stats_path` folder, because the
normalizer and the physics-feature cache are stored next to it:

```bash
F=A
RELEASE=DATA/folds/$F BASE=configs/uwb/AP.yaml GROUP=UWB/CV-$F CKPT=swa SEEDS="0" GPUS="0" \
  tools/run_ablations.sh rdrop+dswa -- \
  --data.stats_path runs/_cache/fold_$F/normalizer.json --data.sampler.groups DATA/folds/$F/groups.csv
```

- **No checkpoint selection on the held-out fold.** Use the weight-averaged final model
  (`CKPT=swa`, dense SWA from AP3). The held-out score is computed once, after training.
- **Epoch budget.** With 2 training stitchings an epoch has about twice the steps (about 690
  instead of 345 at batch 32). To keep the AP recipe's number of updates, use
  `--trainer.max_epochs 15` and start dense SWA around epoch 2. This is a suggestion; it has not
  been run yet. Set the budget once for all folds, never per fold.
- **Validation cost.** Each validation pass decodes the fold's 10 held-out streams (about 4.2 h),
  twice the official val. Since nothing is selected on it, validating less often is fine.

## 10. Scoring

After every fold has run `validate` (which writes `predictions/val-validate.csv`):

```bash
uv run python tools/score_folds.py --out runs/_analysis/cv_rdrop+dswa.json \
  --fold A=DATA/folds/A:runs/UWB/CV-A/rdrop+dswa-s0/<TAG>/predictions/val-validate.csv \
  --fold B=DATA/folds/B:... --fold C=DATA/folds/C:... --fold D=DATA/folds/D:...
```

It reports, under the official rule (mean F1 at θ 0.1, 0.3 and 0.5, `--ignore-flagged`):

- **pooled:** all held-out streams of all folds scored as one set (one F1 over about 480 falls,
  about 250 of them clean);
- **per fold** and **per held-out stitching**;
- **fold mean, SD and worst fold.**

It uses the official evaluator code (checked: 0.315 on the sample submission, as in the README).

## 11. Deciding between models

Noise levels to keep in mind:

- one fold has about 60 clean falls, so its F1 moves by about ±0.06;
- the pooled score moves by about ±0.03;
- training seeds add 0.02–0.03 (GPU runs are not reproducible per seed).

A variant beats the reference only if:

1. its **pooled** score is higher by at least about 0.02, averaged over 2 seeds;
2. it is better on at least **3 of 4 folds**;
3. its **worst fold** is not worse.

Between similar variants, prefer the one with the smaller fold SD. Cost: screening is 4 folds × 1 seed
(4 runs per variant); finalists get 4 folds × 3 seeds.

## 12. Decoder and post-processing tuning

Tune decoder thresholds, the minimum score and similar settings **nested**:

1. Tune on 3 folds' held-out probabilities.
2. Apply the result to the 4th fold.
3. Rotate through all 4 folds.

This uses only saved `val-validate.probs.npz`, so it is cheap. The final decoder is tuned on all
4 folds pooled. `src/falljepa/tune_decoder.py` was written for the old in-stream CV and has to be
pointed at the fold folders; that adaptation is **not done yet**.

## 13. The final model

1. Retrain the chosen recipe on **all** clips (8 + 8 subjects; for example the official streams
   plus 1 more stitching of all clips), with the fixed epoch budget and SWA.
2. Train 3 or more seeds and ensemble them. The 4 fold models can join the ensemble.
3. Use the decoder tuned on all folds pooled.

The CV slightly underestimates this model, because each fold model saw only 6 of the 8 fall
subjects.

## 14. Compliance (README v1.1 Rules)

- **Allowed:** generating training streams from `segments/`.
- **Not allowed:** matching repeated ADL content within or across test streams. The repeated pieces
  matter here only as a leak to avoid in the CV, never as a feature.
- **Required:** each test stream processed independently. A non-causal, whole-stream model is
  allowed.

## 15. Limits and open items

- **Not built yet.** The fold data has not been built (the server was down). The builder's writing
  path was tested on official streams (bit for bit identical) but not on a whole fold.
- **Decoder tuning.** `tune_decoder.py` still needs adapting for the nested tuning (section 12).
- **Epoch budget.** The budget for 2 stitchings (section 9) is a suggestion, not tested.
- **Same apartment.** All folds use the same apartment and the same 3 mat positions, as the test
  does. The CV cannot measure robustness to a new room.
- **Official val is no longer untouched.** It has been scored many times. The pooled CV is now
  the honest estimate; the test set is the only untouched data.

## 17. Using this setup on another machine (comparable results)

The fold data is deterministic: the same release, fold file and seed give the same streams and
labels on any machine. What makes two people's numbers comparable is building the identical data
and following the same protocol.

**Needs:**

- **The release.** The official `abc2027_uwb_train_v1` package **with `segments/`**. The tools use
  its own `src/abc2027` (the generator and the evaluator).
- **Python packages:** `numpy`, `h5py`, `pyyaml`. `tools/build_fold_streams.py` and
  `tools/score_folds.py --release …` do not need the falljepa repo.
- **Disk:** about 5 GB (`segments.h5` plus the small virtual stream files).

**1. Build and confirm the data is identical:**

```bash
python tools/build_fold_streams.py --release /path/to/abc2027_uwb_train_v1 --check-official   # must print identical: True twice
python tools/build_fold_streams.py --release /path/to/abc2027_uwb_train_v1 --dry-run          # compare the fingerprints below
python tools/build_fold_streams.py --release /path/to/abc2027_uwb_train_v1 --out DATA/folds
```

Keep the defaults (`--seed 2027`, `--train-stitchings 2`, the fold file as shipped). Every
stitching prints a fingerprint, a hash of its plan (stream id, clip and cut of every segment), and
`fold.json` stores it. They must equal:

| Fold | train 0 | train 1 |
|---|---|---|
| A | 5760f8093c6433c0 | 27ae219e907ab9ea |
| B | 9f52151329e5f589 | cf5d4c3382ab73ec |
| C | 3531fd8cad97663e | 645d06a6f38a98df |
| D | 80e902563ff366bc (official) | 15824eccf42780b4 |

| Fold | held-out 0 | held-out 1 |
|---|---|---|
| A | d8deead0cfe9437a | 39b17a2c3e522d7b |
| B | dfb0ff6bb8c0c582 | 10ba71a9ed553778 |
| C | 267822d1b7a72e49 | e34d880f0840e366 |
| D | e45ab3c4a7f6eb94 (official) | d595eaf5e54fe1fa |

What each check covers:

- **The fingerprints** cover the plan: which clip, which cut, which stream, in which order.
- **The build** then checks that the virtual rebuild of all 18 official streams is bit for bit
  identical, and that every fall label is an exact copy of its source.

Together these mean the signal and the labels are the same as everyone else's. The fingerprints
were made with numpy 2.5.3; another numpy version is fine as long as they match.

**2. Read the folds with your own pipeline.** Each `DATA/folds/<fold>` is laid out like the
official package (`streams/train`, `streams/val`, `streams_events.csv`,
`streams_composition.csv`), so code that reads the release can read a fold. Check two things:

- **Virtual files.** The stream files are HDF5 virtual datasets; h5py ≥ 2.9 with HDF5 ≥ 1.10 reads
  them like ordinary files. A reader that can't (another HDF5 library, or code that copies the
  files elsewhere) needs `--storage full`, about 19 GB per fold. Fingerprints and labels are the
  same either way.
- **Stream counts.** A fold has 26 training streams (`train_00…train_25`) and 10 held-out streams
  (`val_00…val_09`), not 13 and 5. Code that assumes the official counts, or reads the official
  package's CSVs or streams directly, must be pointed at the fold folder.

**3. Protocol rules, all required for comparable numbers:**

1. **Train on the fold's training streams only.** Everything fitted, not just the network, must be
   fitted on the fold's `streams/train` only: normalizer statistics, feature statistics, location
   or scene priors, post-processing parameters, thresholds. Nothing may be fitted on its
   `streams/val`, on the official package's streams, or on other folds.
2. **No selection on the held-out streams.** Use a fixed epoch budget and a final or weight-averaged
   model; no best checkpoint. Tune the decoder or post-processing nested across folds (section 12).
3. **Predict every held-out stream.** All 10 `val_XX` streams of each fold, one CSV per fold in the
   submission format (`stream_id`, `t_start`, `t_end`, `score`).
4. **Score with the shared scorer:**
   `python tools/score_folds.py --release /path/to/abc2027_uwb_train_v1 --fold A=DATA/folds/A:predA.csv … --fold D=DATA/folds/D:predD.csv`.
   Report the pooled score, the per-fold scores and the worst fold.
5. **Run several seeds.** Report at least 2 seeds per fold. Training is not reproducible per seed,
   and seed noise is 0.02–0.03.

Followed this way, two people's pooled scores measure the same thing on the same data. Differences
smaller than about 0.03 are within noise (section 11).

## 16. Files

| File | What |
|---|---|
| `configs/folds/subject_folds.yaml` | the 4 folds |
| `tools/build_fold_streams.py` | builds the fold streams (checks: official plan, virtual bit-for-bit, every label) |
| `tools/score_folds.py` | pooled, per-fold and per-stitching official scores |
| `tools/cv_kit/cv_folds.py` | loader for other pipelines (streams, per-timestep labels, events, submission CSVs) and the same scorer (`python cv_folds.py score A=… D=…`) |
| `metadata/fold_clips.csv` | every source clip: kind, subject, official split, the fold that holds it out, labels, the duration the plan uses |
| `metadata/dry_run.txt` | the official-plan check and every fold's planned stitchings |
| `configs/folds/stream_checksums.csv` | SHA-256 of every stream's content and label file of the reference build (compared after every default build) |

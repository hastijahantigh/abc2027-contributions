# falljepa: preprocessing and model, compared with the Stage05 collaboration package

This note explains how `falljepa` turns the official UWB-FallTrack streams into training data and
what the model does with them. It then compares both with the collaboration package
`ABC2027_collaboration_20261008` (Stages 01–05: dataset audit, Stage05 preprocessing and the
PITS/PoolFormer review).

All values below come from `configs/uwb/U0.yaml` and the code in `src/falljepa/`. Where the README
disagrees with the config, the config wins. For the package, the sources are its Persian reports
under `workspace/projects/abc_challenge/docs/` (`01_official_baselines.md`,
`04_deep_dataset_audit.md`, `05_preprocessing_blueprint.md`,
`05_preprocessing_implementation.md`, `05_poolformer_code_review.md`) and
`configs/stage05_preprocessing.json`.

**Status of both sides.** Neither side has a model trained on the real data yet. The package
ships a verified preprocessing pipeline, audits and an untrained tokenizer prototype; it says so
itself ("No trained competitive detector is provided"). falljepa ships a full train / decode /
score pipeline that has been verified only on a synthetic release. So this is a comparison of
designs, not of scores. The only real-data scores so far are the official baselines that the
package reproduced: Energy 0.315 and Logistic 0.258 mean event F1 on val.

---

## 1. The data facts both designs start from

| Fact | Value | Source |
|---|---|---|
| Streams | 13 train (6.18 h), 5 val (2.05 h); `cir` is float32 `(3, T, 368)` at 50 Hz | official package |
| Radar order | `29171, 29444, 30722` | both check it |
| Falls | 359 train, 119 val; median 1.66 s / 2.0 s; range 0.48–5 s | |
| Flagged falls | 173 of the 359 training falls (48 %); ignore zones (±2 s) when scored with `--ignore-flagged` | |
| Scored falls | 186 train, 67 val | |
| Ranking | mean event F1 at tIoU 0.1 / 0.3 / 0.5; ties broken by median \|start error\| at 0.3, then false alarms/h | `official/evaluate.py` |
| Stream construction | streams are joined source recordings (crossfade 0.2 s), with a join roughly every 16 s; ADL recordings are reused in several streams | Stage02/04 |
| Offline is allowed | no causal / online rule in the package README; the official Energy baseline itself uses a whole-stream percentile | `current_rules_2027.md` |

Several Stage04 findings shape the falljepa design directly:

- **Window labels cannot localize.** If you take the official 4 s windows, label them with
  `tIoU ≥ 0.1`, feed the *true* labels to the official decoder and score the result, you get val
  F1@0.1 = 1.0, F1@0.3 = 0.18 and **F1@0.5 = 0.0** (mean 0.39). The median output interval is
  8 s. Classifier quality cannot fix this; only the target and the decoder can.
- **Output length.** A fixed 2 s interval can reach tIoU ≥ 0.5 on 98 % of falls; a fixed 4 s one
  on about half. So the output intervals must vary in length.
- **Coupled holds.** In about 24 % of timesteps, all three radars repeat their previous profile
  exactly.
- **Radar dominance follows the recording group.** In Dataset 1, 29444 carries about 99 % of the
  motion energy; in Dataset 2, 30722 does.
- **The signal is broadband and low-rank PCA is a trap.** 10–15 % of the power is above 10 Hz.
  PCA at 95 % explained variance keeps only 26 % of the within-window dynamics for radar 29171.

---

## 2. falljepa preprocessing

Code: `src/falljepa/data/uwb.py`, `data/labels.py`, `data/folds.py`.

### 2.1 What is kept unchanged

The signal stays on the 50 Hz grid, with no filtering, resampling, PCA, clipping or
duplicate-frame removal. All 368 values per radar are kept and all three radars stay in their
official order. Any stream that is not finite or has the wrong shape or radar order raises an
error. On these points falljepa matches the package's blueprint exactly.

### 2.2 Normalization

`fit_normalizer` computes:

1. **Mean and std per (radar, bin)** in float64, over **every timestep of the training streams**
   (about 1.1 M). The std floor is 1e-6.
2. **The std of the temporal difference** of the z-scored signal, per (radar, bin), computed on
   the continuous stream.

The result is cached in `runs/_cache/normalizer_train.json`. In cross-validation it is refitted
for each fold, on the training-stream timesteps outside the held-out fold
(`normalizer_train_cv{k}.json`). Val and test are only transformed, never fitted.

### 2.3 Input channels

The model input is a 7-channel image with rows for range bins and columns for time:

```
x : [7, 368, L]
    0..2  z-scored CIR of the three radars
    3..5  temporal difference of those, divided by its training std
          (computed on the continuous stream, so a window's first column has its real predecessor)
    6     valid: 1 on real timesteps, 0 on padding past a stream end
```

### 2.4 Windows

| | Value |
|---|---|
| Window | 25.6 s = 1280 timesteps |
| Margin | `(frame_width − frame_stride) / 2` = 14 timesteps of real signal on each side, so a window is read as 1308 timesteps and its edge frames see data, not padding |
| Training | windows start at random positions; streams are chosen in proportion to their length. One window per 12.8 s of stream per epoch (about 1,740 windows, about 218 steps at batch 8) |
| Val / test | a fixed grid with a 6.4 s hop, plus a last window that ends at the stream end; every timestep is covered by about 4 windows |
| Padding | only past the true stream ends: zeros with `valid = 0` |

Windows deliberately **cross the joins between source recordings**. A window of 25.6 s holds a
whole fall recording (cropped to at most about 23 s) together with the joins on both sides, and
the test streams contain joins too. The model has to learn that a join is not a fall.

### 2.5 Labels: four states per timestep

`stream_targets` gives every 20 ms timestep one state and one loss weight:

| State | Timesteps |
|---|---|
| `none` | no fall |
| `begin` | the first 0.3 s of a fall, capped at a third of the fall |
| `middle` | the rest of the fall |
| `end` | the last 0.3 s of a fall, capped at a third of the fall |

A fall covers `[round(t_start·50), round(t_end·50))`. Flagged falls (`ignore`, `quality=low`,
`needs_review`) keep their states with loss weight **0.5**. The scorer neither rewards nor
penalizes predictions on them, and they double the number of falls seen in training.
`flagged_margin_s` (0 by default) can extend the lower weight to neighbouring timesteps.
Nothing is masked around fall boundaries. The boundaries are learned as the `begin` and `end`
states instead.

### 2.6 Augmentation (training only)

These act on the signal channels and leave `valid` alone:

- **Per-radar gain:** `exp(N(0, 0.1))`.
- **Additive noise:** `N(0, 0.05)`, applied only where `valid = 1`.
- **Radar drop:** with probability 0.1, one whole radar (both its z-score and difference channels)
  is zeroed.

### 2.7 Cross-validation

`data/folds.py` uses the package's own Stage04 fold assignment, copied to
`configs/folds/stage04_source_folds.csv`. It has 6 folds, each one presumed fall subject plus one
presumed ADL subject.

Windows cross recordings, so a fold is not a set of windows. Instead, each timestep belongs to
every recording that covers it, using `streams_composition.csv`; at a 0.2 s crossfade it belongs
to both sides. For held-out fold *k*:

- the **fit mask** lets only non-*k* timesteps feed the normalizer, the state losses and the
  JEPA/SIGReg losses (a JEPA frame pair counts only when both frames lie entirely outside *k*);
- the **score mask** lets only fold *k*'s timesteps be scored, with the official evaluator.
  False alarms per hour are computed over fold *k*'s duration.

The input is not cut, so fold *k*'s raw signal still appears as unlabeled context next to
training timesteps. This is the one known source of optimism. The protocol is:

1. Compare configurations by their 6-fold score at the last epoch.
2. Tune the decoder on the pooled out-of-fold probabilities (`falljepa.tune_decoder`).
3. Train once on all of train.
4. Score val once.

---

## 3. The falljepa model

Code: `src/falljepa/model/`. This is gir's frame JEPA (HTR-JEPA), read along time instead of
along a text line. It has about 2.35 M parameters.

### 3.1 Tensor flow

```
x [B, 7, 368, 1308], valid [B, 1308]
│
├─ frames: unfold the time axis, width 32 (0.64 s), stride 4 (0.08 s)  → 320 frames per window
│          each timestep is covered by 32/4 = 8 frames
│
├─ Ev  FrameEncoder (isolated: a frame never sees values outside itself)
│        patches of 23 bins × 8 timesteps × 7 channels → 16 × 4 = 64 tokens of 1288 values
│        linear embed → d = 128, + 2-D position (row, column) + CLS
│        4 transformer layers, CLS pooling → PreProjector MLP → z_i ∈ R^128     ─► SIGReg
│
├─ + frame position embedding
├─ Cv  2 causal transformer layers over frames                → h_i         (padding frames masked)
├─ Bv  2 bidirectional layers                                 → r_i
│
├─ state head  r_i → logits for each of the frame's 32 timesteps × 4 states   ─► CE (per frame)
│   frames → timesteps: Hann-weighted mean of the 8 covering frames' logits   ─► CE (per timestep)
│   (also the weighted std of P(fall) across those frames: "disagreement")
│
└─ Hv  2 causal predictor layers on h_≤i → ẑ_{i+8}, the latent of the first frame that shares
       no timestep with frame i (target detached)                            ─► MSE
```

### 3.2 Losses

| Loss | Weight | What it trains |
|---|---|---|
| `State_Frame` | 1.0 | weighted CE for each frame on each of its 32 timesteps; class weights none 1, begin 4, middle 2, end 4 |
| `State_Step` | 0.5 | the same CE on the Hann-averaged per-timestep logits, which are what gets decoded |
| `MSE_Latent` | 0.25 | JEPA next-frame latent prediction, horizon 8 frames, target detached |
| `SIGReg` | 0.1 | Epps–Pulley regularizer that keeps `z` close to an isotropic Gaussian (LeJEPA), against collapse |

The JEPA term is auxiliary. In gir it never beat weight 0: an attached target collapsed `z` (rank
17 of 128), and a detached one still cost 0.55 pt. The first ablation is therefore
`--model.loss_weights.MSE_Latent 0`, while watching `CopyRatio`, the predictor's error relative to
just copying the current latent.

### 3.3 From windows to events

1. **Windows → stream.** The per-timestep log-probabilities of the overlapping windows are
   averaged with a Hann weight over each timestep's position in its window (floor 1e-3, so the
   stream edges still get a value), then renormalized.
2. **Stream → states.** A Viterbi decoder runs over the grammar
   `none → begin → (middle) → end → none`. It allows `begin → end` for very short falls, and every
   state change costs `switch_penalty = 2` nats. Each decoded fall has exactly one start and one
   end, so a flicker in P(fall) cannot split one fall into two, which would add a false alarm.
3. **States → events.** Every non-`none` run of at least 0.4 s becomes
   `t_start, t_end, score = mean P(fall)` (optionally minus `spread_penalty` × disagreement), with
   `t_alarm = t_start`. A thresholding decoder is available as an alternative.

Every validation epoch decodes the whole val streams and scores them with the unchanged official
evaluator (`--ignore-flagged`). `val/event/F1_mean` selects the checkpoint, except in CV, where
the last epoch is used. The decoder settings (switch penalty, threshold, minimum duration, spread
penalty) are tuned out of fold, never on val.

### 3.4 Optimization

AdamW with learning rate 1.5e-4 and weight decay 0.05, cosine schedule with 5 % warmup, batch 8,
150 epochs (about 33k steps), bf16, gradient clipping at 1.0.

---

## 4. What the collaboration package built

### 4.1 Stage05 preprocessing (`abc-stage05-v1`)

| Item | Value |
|---|---|
| Native tensor | `(B, 3, T, 368)` at 50 Hz; no filter, resampling, PCA, clipping or frame removal. An exact legacy adapter to `(B, T, 1104)` is also provided |
| Window | 4 s (200 frames) with a 1 s hop and no extra context. 4 s is chosen for comparability with the baseline; the package says it is not proven optimal |
| Training windows | **only inside the interior of a single source recording**, with no crossfade and no other source in the receptive field: 15,202 windows (68 % of the possible ones) |
| Inference windows | label-free and composition-free, starting every 1 s to the end of the stream, with the tail padded: 29,639 windows covering every frame |
| Scaler | mean and std per (radar, component). It uses **one occurrence per source** (its longest single-source interior), so that reused ADL recordings are not counted several times: 391,085 frames from 467 sources. There are 7 versions: `all_train` plus `outer_0..5`; the API also supports nested inner exclusions |
| Labels | dense binary `fall_label` (flagged falls included) |
| Loss mask | excludes padding, flagged falls ±2 s, and **a 0.5 s band on each side of every onset and offset**. As a result, 2 of the 186 scored training falls (both exactly 1 s long) get no positive supervision, and flagged falls get none at all |
| Diagnostics, not used as input | `adjacent_profile_hold`, `extreme_z_mask` (\|z\| > 20), `finite_mask` |
| Folds | the same 6 Stage04 folds; heldout evaluation uses pure single-source windows |
| Tests | 39 synthetic unit tests plus 54 real windows read; numeric hashes match Stage04 for all 18 streams |

### 4.2 Model side

- **Supplied PITS/PoolFormer code (reviewed, not adopted as is).** Its classification head
  produces one logit per window, with no start/end, frame logits or masks. Its contrastive
  pretraining is not supervised and has a token-position shortcut. Setting `patch_len=1` with the
  default `dct_mean_rm` makes every token zero. Its PoolFormer mixes each of the 1104 scalar
  variables separately in time, so the radars and range bins are only combined at the linear
  head.
- **`radar_tokens.py` prototype (untrained).** One token per radar per 20 ms timestep: a shared
  `368 → D` projection, a radar identity embedding and a learned gate that fuses the radars into
  `(B, T, D)`. Invalid frames are masked before and after the projection. Its tests cover the
  real-batch interface only: random weights, zero optimizer steps. It has no temporal model, head
  or decoder.
- **Proposed next step:** a decoder-only experiment that keeps the Stage01 logistic classifier,
  features, windows and scaler fixed and changes only the window→interval decoder.

---

## 5. Side-by-side comparison

### 5.1 Preprocessing

| Topic | Stage05 package | falljepa | Comment |
|---|---|---|---|
| Rate, components, radars | 50 Hz, 368, 3 in fixed order | same | agree; both follow the PSD, PCA and hold findings |
| Validation of input | shape, radar order, time grid, finiteness, hashes; non-finite input stops the run | shape, radar order, finiteness | the package also hashes and checks the time grid |
| Window length | 4 s | 25.6 s | falljepa needs context to see the approach, the fall and the lying still that ends it, at the cost of crossing joins |
| Hop | 1 s, for training and inference | random starts for training; 6.4 s for inference (4× coverage) | |
| Training windows vs joins | must lie inside one source interior (15,202 windows; context makes this fall quickly, e.g. 8 s gives 41 %) | cross joins freely | the package is strictly source-safe; falljepa trains on what test streams look like (a join about every 16 s) |
| Scaler population | one occurrence per source (467 sources, 391 k frames) | every training timestep (about 1.1 M); reused ADL recordings count more than once | the package's choice is cleaner; the difference is probably small but untested |
| Scaler versions | all-train, 6 outer folds, nested inner | all-train, plus one per CV fold computed from the fit mask | falljepa has no nested inner level, because it tunes the decoder on pooled OOF probabilities |
| Extra channels | none; holds and extreme z are diagnostics only | temporal difference, plus a valid channel | the difference stresses motion over the static room profile, which the package flags as a possible shortcut |
| Padding | explicit `valid` / `padding` masks, not used as negatives | `valid` channel and mask; padded frames are never attended to and have weight 0 in the loss | agree |
| Target | binary fall per frame | 4 states (none / begin / middle / end) per timestep | the states give the decoder a grammar and train the boundaries directly |
| Boundaries | 0.5 s on each side **excluded** from the loss (label uncertainty) | the 0.3 s at each end **is** the `begin` / `end` class, with class weight 4 | opposite answers to the same uncertainty; see §6 |
| Flagged falls | no positive supervision; ±2 s ignored | falls at weight 0.5 | falljepa uses 173 more falls; the package plays it safe |
| Augmentation | none by default; gain scaling and radar dropout are listed as separate experiments | gain, noise, radar drop on by default | radar drop interacts with the recording-group dominance in §1 and needs a Dataset-split ablation |
| CV unit | source recording, as pure single-source windows | source recording, as timestep fit/score masks on whole streams | falljepa's held-out score includes joins and transitions; held-out raw signal is visible as unlabeled context |
| Storage | lazy HDF5 reads, no normalized cache | lazy HDF5 reads, each worker opens its own handles | agree |

### 5.2 Model and output

| Topic | Package (baseline, PITS review, prototype) | falljepa |
|---|---|---|
| Token | one token per radar per timestep: the whole 368-value profile → D | a 23-bin × 8-timestep × 7-channel patch; 64 patches → one vector per 0.64 s frame |
| Radar fusion | explicit: radar embedding and learned gate after the projection | implicit: the patch embedding mixes the radar channels linearly (each radar keeps its own weights) |
| Temporal model | none yet (PoolFormer reviewed but not adopted) | frame transformer → 2 causal + 2 bidirectional layers |
| Self-supervision | PITS reconstruction + contrastive (reviewed: has a position shortcut, unsupervised) | JEPA next-frame latent (detached) + SIGReg; auxiliary, and ablated against weight 0 |
| Output | baseline: one probability per 4 s window → merged intervals | per-timestep probabilities of 4 states |
| Decoder | official: merge adjacent positive windows (median 8 s intervals) | Viterbi under the fall grammar, minimum duration, mean P(fall) as score; tuned out of fold |
| Event length | multiples of the window and hop | any multiple of 20 ms |
| Uses future context | Energy baseline: yes (whole-stream percentile) | yes (bidirectional layers and centred windows); allowed, because scoring is offline |

### 5.3 How falljepa responds to each Stage04 finding

| Stage04 finding | falljepa response |
|---|---|
| Window target + official decoder → F1@0.5 = 0 even with perfect labels | per-timestep states and an interval decoder, not window labels |
| Fixed 4 s / 8 s outputs cannot reach tIoU 0.5 | runs of variable length at 20 ms resolution |
| Context costs source-safe training windows | context is free: windows cross joins, and the folds are masks on timesteps rather than window lists |
| Radar dominance follows the recording group | radars stay as separate channels; radar drop augmentation; cross-radar attention is a listed ablation |
| PCA at 95 % variance loses dynamics; > 10 Hz has power | no PCA, no resampling |
| Holds are coupled across radars (about 24 %) | the JEPA horizon is 8 frames, the first frame with no shared timesteps, so prediction cannot be won by copying |
| Static room context could be a shortcut | difference channel; per-window instance norm (RevIN) is listed as an ablation |

---

## 6. Where the two disagree, and how to settle it

1. **Training across joins vs single-source interiors.** The package's rule makes every training
   window belong to one source group, which removes any doubt in CV. falljepa's rule trains on the
   joins the test streams contain and on 25.6 s of context. The package's own table shows why
   strict source safety and long context do not mix: at 8 s with 2 s of context, only 29 % of
   windows survive. The cost of falljepa's choice is held-out signal used as unlabeled context in
   CV. *Test:* rerun one CV fold with the held-out timesteps zeroed (`valid = 0`) in the training
   windows and compare the fold score.
2. **Boundary bands: excluded vs targeted.** The package masks ±0.5 s of every boundary because the
   labels are uncertain there. falljepa makes the boundaries their own classes, because tIoU and the
   start-error tie-break score exactly those boundaries. Both are reasonable. *Test:* a
   `boundary_ignore_s` option in `labels.py` that sets the weight to 0 within ±x s of each
   boundary, compared in 6-fold CV with x ∈ {0, 0.25, 0.5}.
3. **Flagged falls.** Package: never positive. falljepa: positive at half weight. The scorer
   ignores them either way, so the question is only whether they help the model. *Test:*
   `flagged_weight` ∈ {0, 0.5, 1} in CV.
4. **Scaler population.** Package: one occurrence per source. falljepa: every timestep. *Test:*
   load the package's `normalizers/all_train.json` and `outer_k.json` as falljepa's `stats_path`.
   They only lack `diff_std`, which can be computed with the same masks. This also lets the two
   pipelines share inputs exactly.
5. **Default augmentation.** The package treats gain scaling and radar dropout as experiments, not
   defaults. *Test:* `augment: {}` against the default, reported per Dataset group.
6. **Tokenization.** The package's profile token (whole 368 profile per radar per timestep, fused
   explicitly) and falljepa's range × time patch are two different answers to "one token per
   timestamp". *Test:* replace `FrameEncoder`'s patch embedding with `radar_tokens.py`'s projection
   and gate (frame width 32, stride 4), keeping everything else fixed.

The package also suggested a decoder-only experiment with the Stage01 logistic model. It does not
depend on falljepa, and `falljepa.model.aggregate.decode_events` with the `threshold` decoder can
be reused for it directly.

---

## 7. Known inconsistencies in this repo's README

The README's narrative ("The idea, analyzed" and "The JEPA term") still describes an earlier
setting: 20 s windows with a 10 s hop, stride 8 (4 frames per timestep), JEPA weight 0.5 and
horizon 4. The config, and the README's own hyperparameter table, use 25.6 s / 6.4 s, stride 4
(8 frames per timestep), weight 0.25 and horizon 8. This document follows the config.

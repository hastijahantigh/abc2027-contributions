# FallJEPA: from raw UWB radar to fall events

This document follows the data end to end: what the raw data is, how it is turned into training
samples, what the model computes, how it is trained and validated, and how its output becomes the
fall events that are submitted. All numbers are those of the base experiment `configs/uwb/U0.yaml`.
Where the README or a docstring disagrees (some still say 20 s windows or a JEPA weight of 0.5),
the config is what actually runs.

Code references are given as `file:function` so each step can be checked in the source.

---

## 0. The whole pipeline on one page

```
 RAW (organizers)                     3 XeThru X4M200 radars, each: CIR frames of 368 range bins
      │  align to 50 Hz grid,         (fall recordings + daily-activity recordings, LIARA apartment)
      │  concatenate with 0.2 s crossfade
      ▼
 RELEASE (what we get)                streams/{train,val}/stream_XX.h5 : cir float32 (3, T, 368)
      │                               streams_events.csv              : fall start/end per stream
      │                               streams_composition.csv         : which recording covers which seconds
      ▼
 PREPROCESSING (ours, src/falljepa/data/)
      │  1. labels      every 20 ms timestep → state none/begin/middle/end + loss weight
      │  2. normalizer  per (radar, range bin) z-score, fitted on training streams only
      │  3. channels    3 z-scored radars + 3 temporal differences + 1 "valid" mask = 7 channels
      │  4. windows     25.6 s (1280 timesteps) + 14-timestep margin each side → [7, 368, 1308]
      │  5. augment     (training only) per-radar gain, Gaussian noise, drop a radar
      ▼
 MODEL FallJEPA (src/falljepa/model/)
      │  frame the window: 320 frames of 32 timesteps (0.64 s), stride 4 (0.08 s)
      │  Ev  frame encoder (ViT over 64 patches, each frame isolated)   → z_i  ∈ R^128
      │  Cv  2 causal transformer layers over the 320 frames            → h_i
      │  Bv  2 bidirectional transformer layers                         → r_i
      │  head: r_i → logits of 4 states for each of the 32 timesteps frame i covers
      │  Hv  2 causal layers: h_≤i → predicted z_{i+8}  (JEPA, target detached)
      │  frames → timesteps: Hann-weighted mean of the 8 frames covering each timestep
      ▼
 LOSS = 1.0·CE(per frame) + 0.5·CE(per timestep) + 0.25·MSE(latent prediction) + 0.1·SIGReg(z)
      │  AdamW 1.5e-4, cosine schedule, batch 8, 150 epochs, bf16
      ▼
 INFERENCE on a whole stream
      │  windows every 6.4 s → per-timestep log-probs → Hann-weighted merge of overlapping windows
      │  Viterbi decoding under the grammar none→begin→middle→end→none
      │  every non-none run ≥ 0.4 s = one fall; score = mean P(fall)
      ▼
 SUBMISSION CSV  stream_id, t_start, t_end, score, t_alarm  → official evaluator (event F1)
```

---

## 1. The task and how it is scored

The ABC2027 UWB challenge gives continuous radar streams in which people go about daily life and
sometimes fall. For each fall we must predict **when it starts and when it ends**.

Scoring is event level (`src/falljepa/official/evaluate.py`, the organizers' evaluator, unchanged):

- A prediction is a row `(stream_id, t_start, t_end, score)`.
- Predictions are matched one-to-one to ground-truth falls, greedily in order of decreasing `score`.
  A match needs a temporal IoU (overlap / union of the two intervals) of at least θ.
- **Ranking score = mean of the event F1 at θ = 0.1, 0.3 and 0.5.** Ties are broken by the median
  absolute start error at θ = 0.3, then by false alarms per hour at θ = 0.3.
- Falls that are *flagged* (`quality = low` or `needs_review`, under `--ignore-flagged`, the test
  rule) are **ignore zones**: they count neither as hits nor as misses, and an unmatched prediction
  overlapping one (widened by ±2 s) is dropped instead of being a false alarm.

Two consequences shape everything below. First, a good *boundary* matters: at θ = 0.5 the
predicted interval must overlap the true one tightly. Second, *fragmentation* is costly: if one
fall is predicted as two pieces, one piece is a false alarm and lowers F1.

---

## 2. The raw data

### 2.1 Sensors and source recordings (organizers' side)

- Recorded in the LIARA apartment (UQAC, about 40 m²) with **three Novelda XeThru X4M200 UWB
  radars**, serials 29171, 29444, 30722 (always in this order).
- Each radar outputs **CIR frames** (channel impulse response): one value per **range bin**, 368
  bins. A CIR frame is a snapshot of how much signal is reflected at each distance from the radar.
  Something moving changes the values at its distance from frame to frame; static furniture gives
  a constant profile.
- Source recordings are of two corpora with *different* people:
  - **fall recordings** (`Chute01`–`Chute04`, simulated on a mat at three fixed positions), in
    three "Datasets"; each contains one annotated fall;
  - **activity recordings** (ADL, 15 daily activities: walking, cooking, using a computer, ...)
    plus a few "other"/empty-room recordings.

### 2.2 How the organizers built the streams (`abc2027_uwb_train_v1/scripts/build_streams.py`)

1. **Alignment** (`src/abc2027/align.py`): the three radars have their own clocks. Each is
   resampled to a common **50 Hz grid** by nearest-frame lookup (no interpolation, no filtering).
   Result per recording: `cir` of shape `(3, T, 368)`, one timestep = **20 ms**.
2. **Planning** (`src/abc2027/streams.py`, seed 2027): streams of about 30 minutes are planned by
   mixing falls and activities at about 57 falls per hour.
   - a fall recording is cropped to a random **3–8 s before** and **5–10 s after** the fall;
   - activity recordings are cut into random **8–20 s chunks**;
   - each fall is used once; each activity recording is reused 3× in train (2× in val/test).
3. **Concatenation** with a **0.2 s linear crossfade** at each join (never inside a fall).
4. The fall's interval is mapped into stream time and written to `streams_events.csv`.

### 2.3 What we receive (the release)

| File | Content |
|---|---|
| `streams/train/stream_00..12.h5` | 13 training streams, **6.18 h** |
| `streams/val/stream_00..04.h5` | 5 validation streams, **2.05 h**, *different subjects* from train |
| each `.h5` | `cir` float32 `(3, T, 368)` = (radar, timestep, range bin); `t` in seconds; `radar_serials` |
| `streams_events.csv` | 478 falls: `stream_id, t_start, t_end, ignore, quality, needs_review, source_path, ...` |
| `streams_composition.csv` | every segment of every stream: `stream_id, kind (adl/fall/other), path, t_start, t_end` |

Test streams (5, ids `test_00..04`) are withheld. Stream ids are `<split>_<XX>`
(`src/falljepa/data/uwb.py:stream_files`).

Key facts that the hyperparameters are derived from:

- falls last **0.48–5 s**, median **1.66 s** (train) / 2.0 s (val);
- only about **3 % of timesteps are falls**;
- **173 of 359 training falls are flagged** (48 %); val has 119 falls of which 67 are scored;
- a join between recordings occurs about every 16 s, so joins are frequent and must not be
  mistaken for falls.

---

## 3. Preprocessing

All of it lives in `src/falljepa/data/` and happens on the fly inside the PyTorch `Dataset`; no
preprocessed copy of the data is written, except the normalizer statistics.

### 3.1 Labels: from event intervals to per-timestep states (`data/labels.py`)

Every timestep of a stream gets one of **four states**:

| id | state | meaning |
|---|---|---|
| 0 | `none` | no fall |
| 1 | `begin` | the first 0.3 s of a fall |
| 2 | `middle` | the rest of the fall |
| 3 | `end` | the last 0.3 s of a fall |

Procedure (`stream_targets`), for each fall `[t_start, t_end)`:

1. `a = round(t_start · 50)`, `b = round(t_end · 50)` (half-open timestep range).
2. Boundary band `n = max(min(15, (b − a) // 3), 1)`: 0.3 s = 15 timesteps, but never more than a
   third of the fall so short falls still have a middle.
3. `state[a:b] = middle`, then `state[a:a+n] = begin`, `state[b−n:b] = end`.

Why four states and not just fall / no-fall: a band at each boundary teaches the model *where*
falls start and end (what tIoU and the start-error tie-break measure), and it gives the decoder a
grammar (Section 6.3) that forbids splitting one fall into several.

**Loss weight per timestep.** All timesteps have weight 1, except those inside a **flagged** fall,
which get **0.5**. Flagged falls are ignored by the scorer, so they are kept as (down-weighted)
falls instead of thrown away: this nearly doubles the falls the model learns from, at no cost at
test time. (`flagged_margin_s = 0`, so the no-fall timesteps around them keep weight 1.)

### 3.2 Normalization (`data/uwb.py:fit_normalizer`)

Raw CIR values have very different scales per radar and per range bin. The normalizer computes, over
**every timestep of the training streams only** (accumulated in float64, in chunks of 20 000):

- `mean[r, b]`, `std[r, b]` for each radar `r` and range bin `b` (3 × 368 each; std floored at 1e-6);
- `diff_std[r, b]`: the std of the temporal difference of the z-scored signal (see 3.3).

It is fitted on first use and cached as JSON (`runs/_cache/normalizer_train.json`; one file per
fold in cross-validation, where the held-out fold's timesteps are excluded). Validation and test
data are normalized with the training statistics: no statistic ever comes from the data being
evaluated.

### 3.3 Input channels (`data/uwb.py:StreamStore.read`)

For a span of timesteps `[lo, hi)` of a stream the store builds **7 channels**:

| channels | what | formula |
|---|---|---|
| 0–2 | z-scored CIR of the 3 radars | `z = (x − mean) / std` |
| 3–5 | temporal difference of each radar | `d_t = (z_t − z_{t−1}) / diff_std` |
| 6 | `valid` | 1 on real timesteps, 0 on padding |

- The **difference channels** emphasize motion over the static room profile (the walls and
  furniture are constant and vanish in the difference). They are computed on the **continuous
  stream**: one extra timestep before `lo` is read so the first difference of a window uses its
  true predecessor (only the very first timestep of a stream gets `d = 0`).
- Positions outside the stream (before 0 or after `T`) are zero and have `valid = 0`, so the model
  can tell padding from a quiet signal.
- The tensor is transposed to **`[C, 368, L]`**: rows = range bins, columns = time. A window is
  therefore an *image* whose horizontal axis is time, like a line of handwriting in gir (the
  handwriting model this one was adapted from).
- Non-finite values raise an error. There is no other filtering, resampling or background removal.

### 3.4 Windows (`data/uwb.py:WindowDataset`)

The model never sees a whole stream at once; it sees **windows**.

- **Window length 25.6 s = 1280 timesteps.** It is long enough to hold an entire fall recording
  (≤ 23 s: approach, fall, lying still) *with the joins on both sides*, so the model learns that a
  join is not a fall and sees the context that tells a fall from sitting down.
- **Margin.** Each window is read with `m = (frame_width − frame_stride)/2 = (32 − 4)/2 = 14`
  extra timesteps of real signal on each side, so the first and last frames of a window see real
  data rather than padding. The sample is therefore **`x: [7, 368, 1308]`**.
- **Training windows: random starts.** Per item, a stream is drawn with probability proportional to
  its length and a start uniformly in `[0, T − 1280]`. The RNG is seeded by `(seed, epoch, index)`,
  so each epoch sees different, reproducible windows. The *number* of windows per epoch is
  `Σ_streams ceil(T / 640)` (one per 12.8 s of stream), ≈ 1745 windows ≈ **218 batches of 8**.
  Random starts let every fall appear at every position in the window; on a fixed grid each fall
  would only ever be at two positions.
- **Validation/test windows: fixed grid** `0, 320, 640, …` (hop **6.4 s**), plus a last window
  that ends exactly at the stream end. Every timestep is covered by about **4 windows**, which are
  later averaged (Section 6.1).

Each item is a dict:

| key | shape | meaning |
|---|---|---|
| `x` | `[7, 368, 1308]` | input channels |
| `valid` | `[1308]` | real timestep or padding |
| `fit` | `[1308]` | may this timestep train the model? (0 on padding and, in CV, on the held-out fold) |
| `state` | `[1308]` int | target state per timestep (margins keep their real labels) |
| `weight` | `[1308]` | loss weight per timestep (`label weight × fit`) |
| `stream`, `start` | scalars | where the window is, to put predictions back into the stream |

### 3.5 Data augmentation (training only, `WindowDataset._augment`)

Applied to the 6 signal channels (not to `valid`), identically to a radar's z-score and difference
channels:

- **gain**: each radar multiplied by `exp(N(0, 0.1))` (about ±10 %);
- **noise**: additive `N(0, 0.05)` on real timesteps;
- **radar drop**: with probability 0.1, one whole radar is set to zero (robustness to a weak or
  missing sensor).

### 3.6 Data loading

`UWBDataModule` (`data/uwb.py`) wires it together: training batch 8 (`drop_last`), eval batch 16,
8 workers, pinned memory. HDF5 files are opened lazily, once per worker. `frame_width`,
`frame_stride`, the number of input channels (7) and the number of frames per window (320) are
**linked from the data config into the model** by the CLI (`cli.py:add_arguments_to_parser`), so
they are never set twice.

---

## 4. The model: `FallJEPA` (`src/falljepa/model/fall_jepa.py`)

2.35 M parameters, `d_model = 128`, 4 attention heads everywhere. Shapes below are for one window
(batch dimension `B` omitted).

| part | parameters |
|---|---|
| frame encoder `Ev` | 959 k |
| pre-projector | 132 k |
| causal context `Cv` (2 layers) | 396 k |
| bidirectional `Bv` (2 layers) | 396 k |
| latent predictor `Hv` (2 layers) | 396 k |
| state head | 33 k |
| frame position embedding, horizon embedding | 41 k |

### 4.1 Framing: the window becomes a sequence of frames

`extract_frames` slides a frame of **w = 32 timesteps (0.64 s)** with stride **s = 4 (0.08 s)**
along the time axis of `x [7, 368, 1308]`:

```
number of frames F = (1308 − 32) / 4 + 1 = 320          frame i covers timesteps [4i, 4i + 32)
```

Every timestep is covered by `w / s = 8` frames. A frame is shorter than almost every fall, so
different frames see the onset, the descent and the landing.

A frame is **valid** if at least one of its timesteps is real; invalid frames (past the stream end)
are never attended to.

### 4.2 Frame encoder `Ev` (`model/encoder.py:FrameEncoder`): one frame → one vector

Each frame `[7, 368, 32]` is encoded **in isolation**: it cannot see anything outside its own
0.64 s. It is a small Vision Transformer:

1. **Patching**: the frame is cut into patches of **23 range bins × 8 timesteps** →
   `368/23 = 16` rows × `32/8 = 4` columns = **64 patches**, each flattened to `7·23·8 = 1288`
   values (all 7 channels together, so the three radars are mixed already in the patch).
2. **Patch embedding**: a linear layer 1288 → 128, plus a learned 2-D position embedding
   (row = range band, column = time within the frame). Positions are within the frame only, so the
   same motion gives the same vector wherever it occurs in the window.
3. A learned **[CLS] token** is prepended → 65 tokens.
4. **4 pre-LayerNorm transformer blocks** (self-attention + MLP with 4× hidden size, GELU,
   dropout 0.1).
5. The final LayerNorm'd [CLS] token is the frame's vector.
6. **Pre-projector** (`PreProjector`): Linear 128→512 (no bias) → BatchNorm → GELU → Linear
   512→128. Its output is **`z_i ∈ R^128`**, the latent that SIGReg regularizes and that the JEPA
   predicts.

Result: `z: [320, 128]`.

### 4.3 Context `Cv`: causal attention across frames

- Add a learned **frame position embedding** (one vector per frame index, 320 × 128).
- **2 causal transformer blocks**: frame `i` attends only to frames `≤ i`, and only to valid frames.
- LayerNorm → **`h: [320, 128]`**.

The causal part exists for the JEPA predictor (it must not see the future it predicts).

### 4.4 Bidirectional layers `Bv`

- **2 non-causal transformer blocks** on `h` (each frame sees the whole window), LayerNorm →
  **`r: [320, 128]`**.

Scoring is offline, so recognition may look at the future: what happens after a candidate fall
(lying still on the floor) is strong evidence.

### 4.5 State head: every frame labels each of its 32 timesteps

```
r_i → Linear(128,128) → LayerNorm → GELU → Linear(128, 32·4) → reshape [32, 4]
```

`frame_logits: [320, 32, 4]`: for each frame, logits of the 4 states for each of the 32 timesteps
it covers.

### 4.6 Frames → timesteps (`model/aggregate.py:frames_to_steps`)

Each timestep now has 8 opinions (from the 8 frames covering it). They are combined as a **weighted
mean of logits**, with a **Hann weight** over the position inside the frame,
`w(p) = 0.5 − 0.5·cos(2π (p + 0.5)/32)`: a frame is trusted most about its centre and least about
its own edges.

- Output covers `(320 − 1)·4 + 32 = 1308` timesteps; the 14-timestep margins are cut off →
  **`step_logits: [1280, 4]`**, one prediction per timestep of the window.
- Also computed: `step_spread [1280]`, the weighted std of `P(fall) = 1 − P(none)` across the 8
  frames (how much the frames disagree). It is saved and can lower an event's score
  (`spread_penalty`, 0 by default).

### 4.7 Latent predictor `Hv` (the JEPA part)

JEPA = Joint-Embedding Predictive Architecture: instead of predicting the future *signal*, predict
the future *latent*.

- Input `h + horizon_embed` (a learned vector for the horizon), **2 causal blocks**, LayerNorm.
- Output at frame `i` is the prediction **ẑ_{i+8}**, the latent of the frame 8 steps later, i.e.
  the first frame that shares **no** timestep with frame `i` (8·4 = 32 = w). With a horizon of 1,
  neighbouring frames share 75 % of their timesteps and the background is static, so simply copying
  `z_i` would already win.
- The target `z_{i+8}` is **detached** (no gradient flows into the target). In gir's ablations an
  attached target collapsed the latents.
- Pairs are used only if both frames are valid (and, in CV, wholly outside the held-out fold):
  312 pairs per full window.

This branch is **auxiliary**: nothing at inference reads `Hv`. It shapes the frame encoder to
produce latents that carry predictive information about motion.

### 4.8 What the model returns (`FallJEPA.forward`)

| key | shape (per window) | used by |
|---|---|---|
| `frame_logits` + `frame_state`, `frame_weight` | `[320·32, 4]`, `[320·32]` | State_Frame loss |
| `step_logits_flat` + `step_state`, `step_weight` | `[1280, 4]`, `[1280]` | State_Step loss |
| `z_v_pred`, `z_v_target` | `[312, 128]` | MSE_Latent loss |
| `z_v_proj` | `[320, 128]` (valid frames) | SIGReg loss, effective-rank metric |
| `z_v_copy` | `[312, 128]` | CopyRatio diagnostic |
| `step_logits`, `step_spread` | `[1280, 4]`, `[1280]` | inference |

---

## 5. Training

### 5.1 Losses (`src/falljepa/loss/losses.py`, weights in `U0.yaml`)

```
L = 1.0 · State_Frame + 0.5 · State_Step + 0.25 · MSE_Latent + 0.1 · SIGReg
```

**State_Frame** (supervised, per frame). Weighted cross-entropy between each frame's logits and the
true state of each of its 32 timesteps (`StateCELoss`):

```
loss = Σ weight_t · cw[y_t] · CE_t  /  Σ weight_t · cw[y_t]
```

with class weights `cw = [none 1, begin 4, middle 2, end 4]` against the 97 / 3 imbalance (the
boundaries are the rarest and set tIoU), and `weight_t` from Section 3.1 (0.5 on flagged falls, 0
on padding and, in CV, on the held-out fold). Margins keep their real labels, so every frame is
supervised on all 32 of its timesteps.

**State_Step** (supervised, per timestep). The same loss on the Hann-averaged `step_logits` of the
1280 window timesteps. This is what is actually decoded, so it is trained directly too.

**MSE_Latent** (self-supervised). Mean squared error between `ẑ_{i+8}` and the detached `z_{i+8}`.
Weight 0.25: it stays an auxiliary term until an ablation (`--model.loss_weights.MSE_Latent 0`)
shows it helps.

**SIGReg** (self-supervised, from LeJEPA). Pushes the distribution of the frame latents `z` towards
an isotropic standard Gaussian, which prevents collapse (all frames mapping to nearly the same
vector). Each step: draw 1024 random unit directions, project the latents on each, and compare the
empirical characteristic function of each 1-D projection with that of `N(0, 1)` at 17 points in
`[0, 3]` (Epps–Pulley statistic, Gaussian-weighted), scaled by the batch size 8.

**Logged diagnostics** (`module.py:_shared_step`): every loss term; `CopyRatio` = MSE(predictor) /
MSE(copy `z_i` as the prediction), below 1 means the predictor beats copying; `EffRank_z` (val),
the effective rank of the latents (128 = fully spread, small = collapse); gradient norm of each
model part.

### 5.2 Optimization (`src/falljepa/optimizer/`, `U0.yaml: trainer`)

| setting | value |
|---|---|
| optimizer | AdamW, β = (0.9, 0.95) |
| learning rate | 1.5e-4 peak (2e-4 at batch 16, scaled by √½) |
| weight decay | 0.05 on weight matrices of linear layers only; none on biases, norms, embeddings, tokens |
| schedule | linear warmup over the first 5 % of steps, then cosine decay to 1e-6, updated every step |
| batch | 8 windows (= 2560 frames) |
| epochs | 150 × ~218 steps ≈ 33 k steps |
| precision | bf16 mixed |
| gradient clipping | global norm 1.0 |
| dropout | 0.1 in every transformer block |
| seed | 0 |

### 5.3 One training step

1. The loader gives 8 random windows `x [8, 7, 368, 1308]` with labels (Section 3.4) and augments
   them (3.5).
2. The model frames each window (2560 frames), encodes the frames, runs context, bidirectional
   layers, state head and predictor (Section 4).
3. The four losses are computed and summed with their weights.
4. Backward, clip, AdamW step, LR scheduler step.
5. At the start of each epoch `set_epoch` changes the random window starts.

### 5.4 Validation during training (every epoch)

Validation does exactly what the test does, on the **whole val streams**:

1. Grid windows (hop 6.4 s), no augmentation, eval batch 16. The losses are logged too.
2. Each window's per-timestep log-probabilities are put back into its stream (Section 6.1).
3. Each stream is decoded into events (6.2–6.3).
4. The events are written as a submission CSV `predictions/val-epochNNN.csv` and scored with the
   **official evaluator under `--ignore-flagged`**; the full report goes to
   `predictions/val-epochNNN.report.json`.
5. Logged: `val/event/F1_mean` (the ranking score), F1 at θ = 0.1/0.3/0.5/0.7, precision, recall,
   false alarms per hour, median start error, mean tIoU.

The checkpoint with the **highest `val/event/F1_mean`** is kept as `best-epoch=NNN.ckpt`, plus
`last.ckpt`.

### 5.5 Run bookkeeping (`cli.py`, `callbacks.py`)

Each run is a folder `runs/<group>/<name>/<id>/` with the full resolved `config.yaml`,
`checkpoints/`, `predictions/`, `meta/` (git commit, diff, command, host, library versions) and
`csv/` metrics; W&B logs offline and is uploaded later (`tools/pull_sync.sh`).

```bash
uv run python run.py fit      -c configs/uwb/U0.yaml --data.release /path/to/abc2027_uwb_train_v1
uv run python run.py validate -c runs/UWB/Frame/U0/<id>/config.yaml --ckpt_path best
uv run python run.py test     -c runs/UWB/Frame/U0/<id>/config.yaml --ckpt_path best --data.test_splits '[test]'
```

---

## 6. Inference: from a stream to fall events

### 6.1 Windows → stream (`aggregate.py:StreamAccumulator`)

Every timestep is covered by about 4 windows. For each stream the per-timestep log-probabilities of
all windows are summed with a **Hann weight over the position inside the window** (1280 positions,
plus a floor of 0.001 so the first and last timesteps of the stream, seen only by a window edge,
still get a value), divided by the sum of weights, and renormalized with a log-softmax. This
ensembles the overlapping windows and trusts each window most where it has context on both sides.

Result per stream: `logp [T, 4]` and `spread [T]`.

### 6.2 Decoding (`aggregate.py:viterbi`)

A **Viterbi** search finds the most likely state sequence under the fall grammar:

```
none → none | begin
begin → begin | middle | end
middle → middle | end
end → end | none
```

Every change of state costs a **switch penalty of 2 nats**, which suppresses flicker. The path may
start only in `none` or `begin`. Because a fall must go `begin → … → end` before returning to
`none`, a dip in P(fall) inside a fall cannot split it into two events.

(An alternative `threshold` decoder, P(fall) > threshold with nearby runs merged, exists for
comparison.)

### 6.3 Runs → events (`aggregate.py:runs_to_events`)

- Every maximal run of non-`none` states is one fall `[a, b)`, in seconds `a/50, b/50`.
- Runs shorter than **0.4 s** are dropped (the shortest real fall is 0.48 s).
- `score` = mean `P(fall)` over the run (the evaluator uses it to order matching), `t_alarm` =
  `t_start`.

### 6.4 Output

A CSV with `stream_id, t_start, t_end, score, t_alarm` (plus `confidence`, `disagreement`), valid
as a submission as written. In validate/test mode the merged probabilities are also saved as
`*.probs.npz`, so the decoder can be re-tuned without re-running the model.

---

## 7. Model selection and honest evaluation

The val set has only 119 falls (67 scored) in 5 streams. Tuning the decoder, comparing configs
*and* picking the checkpoint on it would make the val score optimistic. So there is also a grouped
cross-validation on the training streams.

### 7.1 Folds (`data/folds.py`, `configs/folds/stage04_source_folds.csv`)

- 6 folds, each made of one presumed fall subject and one presumed activity subject (the
  collaborator's Stage04 assignment), keyed per **source recording**.
- Windows cross recordings, so a fold cannot be a set of windows. Instead `streams_composition.csv`
  says which recording covers each timestep (both, inside a crossfade), giving a per-timestep fold
  mask.
- For held-out fold *k*:
  - **fit mask**: timesteps not in fold *k*. Only they feed the normalizer, the state losses
    (`weight = 0` elsewhere) and the self-supervised losses (a frame counts only if wholly outside
    fold *k*);
  - **score mask**: fold *k*'s timesteps. Validation runs on the *training* streams, and only
    events whose midpoint is in fold *k* are scored (false alarms per hour over fold *k*'s
    duration).
- The input is not cut: fold *k*'s signal is still seen, unlabeled, as context next to training
  timesteps. That is the remaining optimism.
- Splitting by stream would be wrong: a subject's recordings, and the same activity recording
  (reused 3×), appear in several streams.

### 7.2 Protocol

```bash
RELEASE=/path/to/abc2027_uwb_train_v1 GPUS="0 1 2" tools/run_cv.sh configs/uwb/U0.yaml
```

1. Train the 6 folds (`U0.yaml` + `cv.yaml`; no checkpoint selection, the last epoch is used).
   Compare configurations by their 6-fold score.
2. Validate each fold's last checkpoint, writing out-of-fold probabilities, then tune the decoder
   on all of them pooled (`falljepa.tune_decoder`): Viterbi switch penalty {0.5…8} or threshold
   {0.3…0.9} × merge gap, × minimum duration {0.2…0.6 s} × spread penalty {0, 0.5, 1}, ranked by
   the official ranking score.
3. Train once on all of train with the chosen config and decoder.
4. Score the official val once. That number is the estimate of test performance.

---

## 8. Hyperparameters at a glance

| group | setting | value | reason |
|---|---|---|---|
| data | sampling rate | 50 Hz (20 ms) | as released |
| | window | 25.6 s = 1280 steps (+14 margin each side) | a whole fall recording with both joins |
| | train windows | random starts, 1 per 12.8 s of stream per epoch | falls at every position |
| | inference hop | 6.4 s | about 4 windows per timestep |
| | channels | 3 z-score + 3 difference + valid = 7 | motion over static clutter; explicit padding |
| | augmentation | gain σ 0.1, noise σ 0.05, radar drop 0.1 | |
| labels | begin/end band | 0.3 s (≤ a third of the fall) | |
| | flagged falls | falls at weight 0.5 | ignored by the scorer; 48 % of train falls |
| frames | width / stride | 32 / 4 steps (0.64 / 0.08 s) | 8 frames per timestep |
| | patch | 23 bins × 8 steps → 64 tokens | |
| model | d_model, heads | 128, 4 | |
| | layers | encoder 4, causal 2, bidirectional 2, predictor 2 | |
| | JEPA horizon | 8 frames, target detached | first non-overlapping frame |
| loss | weights | State_Frame 1, State_Step 0.5, MSE 0.25, SIGReg 0.1 | |
| | class weights | none 1, begin 4, middle 2, end 4 | |
| optim | AdamW | lr 1.5e-4, wd 0.05, β2 0.95, 5 % warmup, cosine to 1e-6 | |
| | batch, epochs | 8, 150 (~33 k steps), bf16, clip 1.0 | |
| decoder | Viterbi | switch penalty 2, min duration 0.4 s | starting point; tune out of fold |

---

## 9. What is deliberately *not* done

- No band-pass filtering, resampling, background subtraction or per-window normalization: the only
  transforms are the global z-score and the temporal difference.
- No statistic or threshold comes from val or test data.
- The raw source recordings are not used (forbidden by the challenge rules); only the released
  streams are.
- Known risk, from the release README: the radar context *before* a fall (the mat area) is already
  distinguishable from activity recordings. The model may partly learn "where" rather than "how
  someone moves". Per-window instance normalization is listed in the README as a future ablation
  against this.

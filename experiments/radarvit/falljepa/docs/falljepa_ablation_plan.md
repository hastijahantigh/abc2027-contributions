# FallJEPA ablation plan: train on all of train, score on official val

Every ablation is trained on the 13 training streams and scored on the 5 official val streams
(`configs/uwb/U0.yaml` + `configs/uwb/ablation.yaml`, not `cv.yaml`), for **100 epochs**
(about 22k steps; U0 used 150, but val F1 is flat from about epoch 60). Each run changes **one
thing** relative to a reference.

---

## 0. Protocol: how to tell a real gain from noise

Val has only 67 scored falls. One fall moves recall by about 1.5 points. From epoch 60 to 149 a
single run moves by ±0.04, and picking the best epoch on val adds about +0.08 (meeting notes, §4).
If each ablation is one run scored at its best val epoch, most differences will be noise. So:

| Rule | Why |
|---|---|
| **Primary metric: mean `val/event/F1_mean` over the last 30 epochs** (70–99), with the same fixed decoder for every run | No epoch is selected on val, and averaging 30 epochs removes most of the epoch-to-epoch swing |
| Also report: last epoch, best epoch (marked *selected on val*), F1@0.5, median start error, FA/h | The boundary ablations (A3, A4) act mainly on F1@0.5 and start error |
| **Secondary metric with less noise: timestep AP of P(fall)** on scored val timesteps (flagged falls masked out) | It uses about 10⁴ fall timesteps instead of 67 events, so it shows representation changes the event F1 cannot resolve |
| **Seeds:** 3 for the stage-1 runs (they set the reference), 2 for every other ablation; more if Δ is within 2 seed SDs | Seed variance is the noise floor. Without it, no difference can be called real |
| A change **wins** if Δ(primary) > 2 × seed SD and both seeds agree in sign | Simple and conservative. Pool the seeds of a run before comparing |
| Decoder: Viterbi, switch penalty 2, min 0.4 s (U0), **never tuned on val** during the ablations | Otherwise the decoder tuning is mixed into the model comparison. A3 and A4 need their own decoder; give it one fixed default too (see there) |
| Save `val-validate.probs.npz` of the last checkpoint (`last.ckpt`, the final epoch) for every run | Decoder variants (min score, NMS, regression refinement) can then be compared without training again |

**Runs up to tag `20261011-033741` (stage 1 and 2):** Lightning 2.6 wrote `last.ckpt` only together
with a new best checkpoint, so their `last.ckpt` is the best epoch on val. Their
`val-validate.probs.npz`, and the AP column computed from it, are therefore selected on val, and
resumes restarted from the best epoch, not the interrupted one. The `F1_*@30` columns come from the
per-epoch metrics and are not affected. Since then `last.ckpt` is the latest epoch
(`BestAndLastCheckpoint` in `src/falljepa/callbacks.py`).

**Caveat to write into the report:** if val ranks ~20 ablations, val is no longer an honest
estimate of test. The final configuration's val number will be optimistic. Score the final
combination (stage 4) once and report it as "val, after model selection on val".

---

## 1. Infrastructure (done)

Stage 1, then stage 2 on its winner, unattended:

```bash
RELEASE=/path/to/abc2027_uwb_train_v1 GPUS="0 1 2" nohup tools/run_stages.sh > runs/stages.log 2>&1 &
```

Single batches of runs, and the comparison:

```bash
RELEASE=/path/to/abc2027_uwb_train_v1 GPUS="0 1 2" SEEDS="0 1 2" tools/run_ablations.sh A0 A1 A1b
uv run python tools/compare_ablations.py --tag <TAG> --ref A0
```

1. **Overlay configs**: `configs/uwb/ablation.yaml` (100 epochs, group `UWB/Ablation`) and one file
   per ablation in `configs/uwb/ablations/` (A0, A1, A1b, A2, A2c, A3, A4a, A4b, A4-cs). A spec like `A1+A2` stacks
   overlays in order. **They are deep-merged by `tools/compose_config.py`** into
   `<run dir>/composed.yaml`, not chained with `-c`: LightningCLI's `-c` replaces the `losses` and
   `loss_weights` dicts whole, so a stage-2 overlay that changes one loss would bring back the JEPA
   losses A1 removed. In an overlay, `__delete__` removes a key (A1: `MSE_Latent: __delete__`).
   Overlays therefore only write what they change.
2. **Driver** `tools/run_ablations.sh`: queues every (spec, seed) over `GPUS`, one run at a time per
   GPU. Each run is `runs/UWB/Ablation/<spec>-s<seed>/<TAG>`: `fit` (with `--seed_everything` and
   `--data.seed` = seed), then `validate --ckpt_path last` (the final epoch; writes
   `val-validate.probs.npz`). Arguments after `--` go to every `fit`. Run again with the same `TAG`:
   finished runs are skipped, interrupted ones resume from `last.ckpt` (the epoch they stopped at).
3. **Pipeline** `tools/run_stages.sh`:
   - Stage 1: `STAGE1` (default `A0 A1 A1b`) × `STAGE1_SEEDS` (default `0 1 2`). If any run fails,
     it stops: no winner is chosen from incomplete runs.
   - Winner: the highest mean `F1_mean@30` over seeds (`compare_ablations.py --best`). It is
     written to `runs/UWB/Ablation/stages-<TAG>.best`, with the table in `stages-<TAG>.stage1.md`.
   - Stage 2: `<winner>+X` for X in `STAGE2` (default `A2 A2c A3 A4a A4b A4-cs`) ×
     `STAGE2_SEEDS` (default `0 1`). The table, with Δ to the winner, goes to `stages-<TAG>.stage2.md`.
   - Overlays are looked up when stage 2 starts. An X with no overlay is skipped, with the command to
     run it later on the same TAG. All six default X exist.
   - Both scripts are wrapped in `{ ... }`, so a `git pull` while they run does not change the
     running script. Runs that start later use the code in the checkout at that time. New
     features are therefore off by default.
4. **Comparison** `tools/compare_ablations.py`: the table of §0 per spec (mean ± SD over seeds),
   Δ to `--ref` and whether it passes the 2 SD rule. `--specs` restricts it to some specs, `--best`
   prints only the winner, and `--csv` writes the per-run rows. Runs with fewer validated epochs
   than their `trainer.max_epochs` are left out (their last 30 epochs are earlier ones: A1b-s1 of
   `20261011-033741` stopped at 83, so its `@30` covered epochs 53–82); `--incomplete` keeps them,
   marked `*`. The timestep AP reads
   `--events` (default `runs/_dataset/streams_events.csv`; the pipeline passes
   `$RELEASE/streams_events.csv`).
5. **Optional Hv**: `predictor_layers: 0` builds no predictor and emits no `z_v_*` keys (no
   `CopyRatio`). `pre_projector: false` makes the pre-projector an identity (A1b).
6. Tests: `test_no_predictor_means_no_jepa_branch`, `test_overlays_deep_merge_and_delete`, and
   `test_cli_ablation_overlays_and_comparison` (A1b+A2 composed, fit, validate, compare on the
   synthetic release).

The winner is the highest mean, even when it is within noise of the second. If A0/A1/A1b are within
2 SD of each other, the simpler model (A1b, then A1) is the better reference. Choose it by hand:
`TAG=<TAG> SEEDS="0 1" tools/run_ablations.sh A1b+A2 ...`.

---

## 2. The ablations

### Stage 1: reference and protocol (3 seeds each)

| id | change | config / code | question |
|---|---|---|---|
| **A0** | none: U0 as it is (JEPA 0.25, Hv, SIGReg 0.1) | — | the baseline |
| **A1** | **no JEPA loss, no Hv** (your 1st) | `predictor_layers: 0`, `pred_horizons: []`, drop `MSE_Latent`; SIGReg stays | Does the JEPA branch help at all? In CV, no-JEPA won on val in 5 of 6 folds. Also logs the time and memory saved |
| A1b | A1 **and no SIGReg / pre-projector** | `SIGReg: 0` (and an identity pre-projector) | Without A1b, "no JEPA" is ambiguous: SIGReg also comes from LeJEPA. A1b is a purely supervised model |

The **reference R** for stages 2–3 is the winner of A0/A1/A1b, probably A1. All later rows change
exactly one thing relative to R.

### Stage 2: your ablations (2 seeds each)

**A2: no causal layers, 4 bidirectional** (your 2nd)
- `context_layers: 0`, `bidir_layers: 4` (`A2.yaml`), so the parameter count is unchanged. Config only:
  with empty `context`, `h = LN(z + pos)`.
- If R still has Hv, Hv then predicts from frames that have no context. Hv is causal itself, so
  nothing leaks, but the JEPA term changes too. That is a second reason to run A2 on the no-JEPA R.
- Optional control A2c, `context 0, bidir 2`: separates "bidirectional" from "fewer layers".

**A3: binary fall / no-fall instead of 4 states** (your 3rd), `A3.yaml`
- Labels: `data.num_states: 2` → `labels.stream_targets` gives NONE / FALL (= 1) over `[a, b)`, no
  bands. Model `num_states: 2` (the overlay sets both; they must match). `class_weights: [1, 3]`,
  about the mean weight of begin/middle/end in U0, so the total weight on falls is unchanged.
- Decoder: `aggregate.viterbi` takes 2 or 4 states. With 2, the grammar is none ↔ fall with the same
  switch penalty (a 2-state HMM smoother). Penalty 2, min 0.4 s, the same as R.
- What to look at: the 4 states mainly prevent fragmentation and set boundaries. Expect the
  difference in **FA/h, F1@0.5 and start error** more than in F1@0.1.

**A4: boundary regression head (ActionFormer)** (your 4th), `A4a.yaml`, `A4b.yaml`, `A4-cs.yaml`

*Targets* (`labels.stream_bounds`, from the stream-level events, so they are correct even when a
fall crosses a window edge): for a timestep `t` inside fall `[a, b)`, `d_s = (t − a)/50`,
`d_e = (b − t)/50` seconds. Outside falls the weight is 0. Batches carry `bounds` and `bound_weight`.
`data.regress_center` is ActionFormer's *center sampling*: only the central fraction of a fall gets a
target (1.0 by default, 0.5 in A4-cs).

*Head* (`model.regression: true`): `Linear → LN → GELU → Linear(d, w·2)` on `r_i`, then softplus. Each
frame predicts `(d_s, d_e)` for each of its 32 timesteps. They are averaged over frames with the
Hann weights (`aggregate.frames_mean`) → `step_bounds`. At inference they become **absolute**
`[t − d_s, t + d_e]` in stream seconds and are merged over windows like the log-probabilities
(`StreamAccumulator.result_bounds`). They are saved as `bounds/<stream>` (float32) in the probs file.

*Loss* `BoundaryDIoULoss` (term `Boundary`, weight 1.0): 1D DIoU between the predicted and true
intervals, on fall timesteps, × the label weight (flagged falls 0.5).

| id | head | decoding | what it answers |
|---|---|---|---|
| **A4a** combine | 4 states + regression | Viterbi runs as in R, then `decoder.refine`: start = P(fall)-weighted median of the run's regressed starts, end the same | Does regression fix the late ends and wide starts (F1 0.76 → 0.65 from θ 0.3 to 0.5)? |
| A4a-aux | same checkpoint as A4a | Viterbi only, regression ignored | Is any gain from the decoding, or from regression as an auxiliary loss shaping `r`? Decode A4a's saved probs without refine. No extra training. Not automated yet |
| **A4b** replace | binary (A3) + regression | `decoder.kind: proposals`: every 5th timestep (0.1 s) with P(fall) > 0.5 proposes its regressed interval, scored by the mean P(fall) over it. **Greedy hard NMS** at tIoU 0.3 and the same 0.5 on the score | Does the full ActionFormer formulation beat the 4-state grammar? |
| A4-cs | A4a with center sampling 0.5 | as A4a | the ActionFormer default |

A4b uses hard NMS, not the Soft-NMS of the first draft. ActionFormer keeps hundreds of rescored
proposals because mAP rewards that, but event F1 counts every surviving duplicate as a false alarm.
Its settings are fixed in `A4b.yaml` and are not tuned on val.

### Stage 3: other ablations likely to help (2 seeds each, ordered by expected gain)

These follow from the error analysis (meeting notes §4–5): overfitting, misplaced detections,
late ends, Chute04, uncalibrated scores.

| id | change | why / evidence | cost |
|---|---|---|---|
| **E1** | **EMA of the weights** (decay 0.999, evaluate and checkpoint the EMA weights) | train loss → 0.011 while val loss rises after epoch 4. ±0.04 between epochs is exactly what weight averaging removes. Also makes "last epoch" a stable choice | callback, ~40 lines (`torch.optim.swa_utils.AveragedModel`) |
| **E2** | **Range-shift augmentation**: roll each window by ±k range bins (k ≤ 8, ≈ ±40 cm), the same shift for all three radars | Fall mats sit at 3 fixed positions. A range shift weakens the "where" shortcut (the release README: pre-fall context vs ADL AUC 1.000) and may help Chute04 (29 % recall) | in `WindowDataset._augment` |
| **E3** | **Time-stretch augmentation** (resample the window ×0.8–1.25, then labels the same way) + **time/range masking** (SpecAugment: 2 masks of ≤ 0.5 s, ≤ 2 bands) | Falls are 0.48–5 s long. Stretching widens the range of speeds the model sees. Masking is cheap regularization against overfitting | `_augment` + label resampling |
| **E4** | **Smoothing loss** (MS-TCN T-MSE: truncated MSE between neighbouring log-probs, τ = 4, weight 0.15) | 6 of 13 misses are misplaced detections, and fragmentation turns one fall into a miss + a false alarm | one loss class on `step_logits` |
| E5 | **Per-window instance normalization** (subtract each window's per-(radar, bin) median over time before the z-score) | removes the static room profile. A second shortcut control, compared with E2 | `StreamStore.read` |
| E6 | **Dropout 0.2** (from 0.1) + **stochastic depth 0.1** | overfitting. Simplest regularizer, a check for E1 | config + `Block` |
| E7 | **Boundary band** `boundary_s: 0.5` (from 0.3) | ends are often late, and wider begin/end bands give the boundary states more data. Config only. If A3 shows the 4 states matter, this is the next question | config |
| E8 | **Focal loss** (γ = 2) instead of class-weighted CE, or label smoothing 0.1 | calibration: at P(fall) > 0.9 only 68 % are falls. The event score is the mean P(fall), so better calibration orders hits above false alarms | loss variant |

Decoder-only checks, on the saved probs of R (no training, report them but do not select on them):
min event score (false alarms have median score 0.86, hits 0.97), `infer_hop_s: 3.2`
(8 windows per timestep).

Lower priority (bigger code changes, smaller or unclear expected gain): a second frame width
(64/8, multi-scale), cross-radar attention instead of mixing radars in the patch, EMA target encoder
for JEPA (only if A0 beats A1).

### Stage 4: combine and confirm

1. R + every stage-2/3 change that won, added one at a time in order of gain. Stop when an addition no longer helps (2 seeds).
2. The final combination: 3 seeds, primary metric, plus the 3-seed ensemble (average the
   probs, then decode). Report it as "val after selection on val".
3. Only after that: tune the decoder, on CV once the held-out-signal problem is fixed
   (meeting notes §6), otherwise do not tune it.

---

## 3. Run budget

| stage | runs (ablation × seeds) |
|---|---|
| 1 | A0, A1, A1b × 3 = 9 |
| 2 | A2, A3, A4a, A4b, A4-cs × 2 (+ A2c × 2) = 10–12 |
| 3 | E1–E8 × 2 = 16 |
| 4 | ~4–6 additions × 2 + final × 3 ≈ 12 |

About 50 runs of 100 epochs (each 2/3 of a U0 run), about as much as 33 U0 runs.

## 4. Implementation order

1. ~~Infrastructure + optional Hv (§1)~~ done: stage 1 can start.
2. While stage 1 runs: EMA callback, augmentations (E1–E3). These are small and self-contained.
3. ~~Binary labels + generalized Viterbi (A3), the regression head, absolute-position merging,
   refinement decoder (A4a), the proposals decoder + NMS (A4b)~~ done (tests: binary targets,
   2-state Viterbi, DIoU loss, refine, proposals, regression head, fit + validate of every stage-2 overlay).
4. Smoothing loss, instance norm, focal loss (E4, E5, E8).

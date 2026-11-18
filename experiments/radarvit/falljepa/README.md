# falljepa: gir's frame JEPA for ABC2027 UWB fall detection

Detect falls in continuous UWB radar streams (ABC2027, UWB-FallTrack) with the frame/JEPA model of
gir (HTR-JEPA), read along time instead of along a text line.

```
stream ──► 25.6 s windows (grid every 6.4 s at inference; random starts for training)
            └─► each window: 320 overlapping frames of 0.64 s, moving 0.08 s
                  z_i = Ev(frame_i)        isolated frame encoder (gir)           → SIGReg
                  h_i = Cv(z_≤i)           causal context
                  r_i = Bv(h)              2 bidirectional layers
                  frame_i → state of each of its 32 timesteps: none / begin / middle / end   → CE
                  ẑ_{i+8} = Hv(h_≤i)       latent of the next non-overlapping frame (detached)  → MSE
            frames → timesteps:  Hann-weighted mean of the 8 frames covering each timestep  → CE
windows → stream:  Hann-weighted mean of the overlapping windows' log-probabilities
stream → events:   Viterbi under none → begin → middle → end → none; score = mean P(fall)
```

## Usage

```bash
uv sync
uv run pytest                                            # CPU, synthetic release, ~10 s
uv run python tools/check_setup.py --release /path/to/abc2027_uwb_train_v1   # GPU, data, folds, one batch
uv run python run.py fit -c configs/uwb/U0.yaml --data.release /path/to/abc2027_uwb_train_v1
uv run python run.py fit -c configs/uwb/U0.yaml --model.loss_weights.MSE_Latent 0 --run.name U0-noJEPA

RELEASE=/path/to/abc2027_uwb_train_v1 GPUS="0 1 2" tools/run_cv.sh configs/uwb/U0.yaml   # 6-fold CV + decoder

uv run python tools/build_fold_streams.py --release /path/to/abc2027_uwb_train_v1 --check-official   # generator reproduces the official plan
uv run python tools/build_fold_streams.py --release /path/to/abc2027_uwb_train_v1 --out DATA/folds   # subject-held-out folds A-D
uv run python run.py fit -c configs/uwb/AP.yaml --data.release DATA/folds/A \
    --data.stats_path runs/_cache/fold_A/normalizer.json --data.sampler.groups DATA/folds/A/groups.csv
RELEASE=/path/to/abc2027_uwb_train_v1 GPUS="0 1 2" SEEDS="0 1 2" tools/run_ablations.sh A0 A1 A1b   # train → official val
RELEASE=/path/to/abc2027_uwb_train_v1 GPUS="0 1 2" nohup tools/run_stages.sh > runs/stages.log 2>&1 &   # stage 1, then stage 2 on its winner
uv run python tools/compare_ablations.py --tag <TAG> --ref A0                                      # docs/falljepa_ablation_plan.md

RUN=runs/UWB/Frame/U0/<id>
uv run python run.py fit      -c $RUN/config.yaml --ckpt_path last   # resume
uv run python run.py validate -c $RUN/config.yaml --ckpt_path best   # whole val streams, official scorer
uv run python run.py test     -c $RUN/config.yaml --ckpt_path best --data.test_splits '[test]'
```

From a machine that can reach W&B (the server cannot), fetch every run without its checkpoints, upload
the W&B runs, then analyse one:

```bash
uv run wandb login                                           # once
tools/pull_sync.sh user@server:/home/abc2027/mehdi           # runs/, logs, runs/_dataset/*.csv; then W&B upload
uv run python tools/analyze_run.py --run runs/UWB/Frame/U0/<id> --tune   # -> <run>/analysis/val-validate.json
```

`--data.release` is the extracted official package (`streams/{train,val}/stream_XX.h5`,
`streams_events.csv`). The normalizer is fitted on the training streams on first use and cached at
`data.stats_path`. Each validation epoch decodes every val stream and scores it with the official
evaluator (`--ignore-flagged`), so `val/event/F1_mean` is the ranking score and chooses the best
checkpoint (`checkpoints/best-epoch=NNN.ckpt`). `checkpoints/last.ckpt` is always the latest epoch,
so `--ckpt_path last` resumes where a run stopped and validates a checkpoint chosen without val.
`predictions/` holds a valid submission CSV per epoch and the evaluator's report.
Run folders, resume, W&B and metadata work as in gir.

## Layout

| Path | What |
|---|---|
| `src/falljepa/model/fall_jepa.py` | `FallJEPA`: framing, encoder, context, state head, latent predictor |
| `src/falljepa/model/aggregate.py` | frames→timesteps, windows→stream, Viterbi and event decoding |
| `src/falljepa/data/labels.py` | per-timestep states and loss weights from `streams_events.csv` |
| `src/falljepa/data/uwb.py` | normalizer, lazy HDF5 windows with margins, `UWBDataModule` |
| `src/falljepa/data/folds.py` | recording-grouped cross-validation with the friend's 6 folds |
| `src/falljepa/tune_decoder.py`, `tools/run_cv.sh` | decoder tuning on out-of-fold probabilities; the 6-fold driver |
| `configs/folds/subject_folds.yaml`, `tools/build_fold_streams.py` | 4 folds holding out 2 fall + 2 ADL subjects each (D = official split); builds each fold (2 training + 2 held-out stitchings: 26 + 10 streams, ~12.5 h + ~4.2 h) from `segments/` with the organizers' generator and the clip durations that reproduce the official streams bit for bit; streams are HDF5 virtual datasets over one `segments.h5` (4.9 GB in all, ~5 MB per stream file) |
| `tools/score_folds.py`, `docs/stream_cv.md` | pooled / per-fold / per-stitching official scores of the fold runs; the full CV guide (design, assumptions, build, train, decide) |
| `data/units.py`, `model/recording_head.py`, `configs/uwb/RS.yaml`, `tools/run_rs_cv.sh` | the recording head: per detected source recording, P(fall) and start/end distributions (MBR decoding), in place of Amir's post-processing; complex I/Q input (README v1.1); one run over the 4 CV folds; see `docs/falljepa_recording_head.md` §10 |
| `tools/pull_sync.sh`, `wandb_sync.sh`, `analyze_run.py` | fetch runs from the server and upload W&B (gir's scheme); error analysis of a run |
| `src/falljepa/module.py`, `cli.py`, `callbacks.py` | training, stream scoring, run folders |
| `src/falljepa/official/evaluate.py` | the official evaluator, unchanged |
| `configs/uwb/U0.yaml`, `cv.yaml` | the base experiment; its cross-validation overlay |
| `configs/uwb/ablation.yaml`, `ablations/*.yaml`, `tools/compose_config.py`, `run_ablations.sh`, `run_stages.sh`, `compare_ablations.py` | ablations on the official val (100 epochs, several seeds): deep-merged overlays, the run queue, stage 1 → stage 2 on its winner, the comparison; see `docs/falljepa_ablation_plan.md` |
| `tests/` | labels, coverage, normalization, frame isolation, causality, decoding, CLI fit + validate |

The package is `falljepa`, not `abc`: a package named `abc` shadows Python's own `abc` module.

### Copied from gir (commit `4ec55b2`; gir itself is unchanged)

| gir file | here | changes |
|---|---|---|
| `src/model/encoder.py` | `model/encoder.py` | `FrameEncoder` unchanged; `TextEncoder` removed |
| `src/model/layers.py` | `model/layers.py` | FlexAttention → `scaled_dot_product_attention` (fixed-length windows need no packing; runs on CPU) |
| `src/model/htr_jepa.py` | `model/fall_jepa.py` | `PreProjector` kept; model rewritten for time (no text side) |
| `src/loss/base.py`, `losses.py` | `loss/` | MSE, Cosine, SIGReg unchanged; CE/CTC → `StateCELoss` |
| `src/optimizer/*` | `optimizer/` | unchanged |
| `src/cli.py`, `callbacks.py`, `module.py`, `run.py` | same names | IAM/CER → UWB/official event F1; links frame/data settings into the model |

## The idea, analyzed

**Large windows.** A 20 s window with a 10 s hop covers each timestep twice, not three times: 40 s
sits on the edge of 20-40 and 40-60, so only 30-50 sees it with context. Windows are merged with a
Hann weight, which at 50 % overlap sums to a constant, so in practice the window a timestep is
central in decides. For a real ensemble use `--data.infer_hop_s 5` (4 windows per timestep).
Training uses random window starts by default (`train_crop: grid` gives the exact fixed grid):
on the grid every fall would only ever appear at two positions in a window. 20 s holds a whole
fall recording (10-20 s, with its margins) plus the joins around it. So windows cross joins
between source recordings, and the model learns that a join is not a fall; test streams have one
about every 16 s.

**Frames and states.** A frame (0.64 s) is shorter than a fall (median 1.66 s), so different frames
see the onset, the middle and the landing, as intended. Begin and end are the first and last
0.3 s of a fall (at most a third of it). These are bands, because an instant cannot be classified
per timestep. The states give the decoder a grammar: every decoded fall has exactly one start and
one end, and a flicker of P(fall) cannot split one fall into several. A split costs a false alarm,
and with it F1. Each timestep is predicted by 4 frames. Their logits are averaged with a Hann
weight over the position in the frame, because a frame knows least about its own edges. The loss
applies both to each frame (your idea) and to the average (what is decoded).

**The JEPA term, from gir's ablations.**
- In w1, MSE on an attached target collapsed the frame latents (effective rank 17 of 128) and cost
  1.56 pt CER.
- In w2, detaching the target restored the rank, but Hv still did not beat the model without it
  (+0.55 pt).

So the target is detached here, and its weight is 0.5. The horizon is 4 frames, the first frame
that shares no timestep with this one: at k = 1, neighbouring frames share 75 % of their
timesteps, about 24 % of timesteps repeat the previous one exactly, and the background is static,
so copying wins. The first ablation must be `MSE_Latent: 0`, gir's C1/S2 gate. Watch `CopyRatio`
(< 1 means the predictor beats copying) and `EffRank_z`. On the synthetic data both already show
the failure signs (CopyRatio ≈ 2.8, rank 7 of 32).

In gir, text was Hv's action. Here nothing plays that role, so the prediction is unconditional and
the future of a frame is genuinely uncertain at a fall onset. The closest analogue is conditioning
Hv on the label sequence during training: recognition never reads Hv, so this is safe. It is not
implemented yet.

## Recent time-series ideas

Already in: frames are patches (PatchTST, 2023), as range × time patches of 23 CIR bins ×
8 timesteps, with 64 tokens per frame (368 = 16 × 23). Also in: SIGReg (LeJEPA, 2025), grammar-constrained decoding (HMM/CRF decoding from action
segmentation), and overlap ensembling across frames and windows, which acts as test-time
augmentation.

Next, as one-change ablations, most promising first:
1. **Boundary regression** (ActionFormer 2022, TriDet 2023): every timestep also regresses its
   distance to the fall's start and end. This is the most direct gain for F1@0.5 and for the
   start-error tie-breaker.
2. **Multi-scale features** (ActionFormer's pyramid, TimeMixer 2024): falls last 0.5-5 s. Add a
   second frame width (64/16) or a pooled context pyramid.
3. **Smoothing loss** (MS-TCN 2019): a truncated MSE between neighbouring log-probabilities, so
   detections fragment less (fewer false alarms).
4. **Shortcut control**: per-window instance normalization (RevIN 2022) against the static room
   profile. The README measures pre-fall context against ADL at AUC 1.000. Compare it with the
   difference channel (on by default).
5. **Cross-radar fusion** (iTransformer / Crossformer 2023-24): encode each radar with shared
   weights, then attend across the three, instead of mixing them in the patch embedding.
6. **Masked-span JEPA** (V-JEPA 2, 2025) instead of only next-frame prediction, or an EMA target
   encoder instead of a detached target.

Not worth it here: state-space models such as Mamba (a window is only 125 frames, so attention is
cheap), and forecasting foundation models such as Chronos, TimesFM, MOMENT and Moirai (built for
forecasting few-channel series, not segmenting 1104-channel radar).

## Hyperparameters

Chosen from the data, not tuned: falls 0.48–5 s (median 1.66 s train, 2.0 s val); a fall recording
is cropped to 3–8 s before and 5–10 s after the fall (at most about 23 s); activity chunks are
8–20 s; 3 % of timesteps are falls.

| Setting | Value | Why |
|---|---|---|
| window | 25.6 s (1280 timesteps, 320 frames) | a whole fall recording with the joins on both sides: the approach, the fall and the lying still that ends it |
| inference hop | 6.4 s | 4 windows per timestep, a real ensemble (10 s gave 2, of which only the central one counts) |
| training windows | random starts, 1 per 12.8 s of stream an epoch | a fall at every position in the window |
| frame width | 32 timesteps (0.64 s) | shorter than all falls but the shortest, so onset, descent and landing fall in different frames |
| frame stride | 4 (0.08 s) | 8 frames per timestep, gir S0's ratio; gir w1's coarser stride (F1) cost 2.7 pt |
| patch | 23 bins × 8 timesteps | 64 tokens a frame |
| d_model, heads, layers | 128, 4; encoder 4, causal 2, bidirectional 2, predictor 2 | gir S0 size (2.35 M parameters); scoring is offline, and gir w2 gained 1.7 pt from bidirectional layers |
| latent horizon | 8 frames (= w / s) | the first frame sharing no timestep; k = 1 is won by copying |
| JEPA | target detached, weight 0.25 | gir w1/w2; auxiliary until it beats weight 0 |
| class weights | none 1, begin 4, middle 2, end 4 | falls are 3 % of timesteps; boundaries are rarest and set tIoU |
| begin/end band | 0.3 s, at most a third of the fall | |
| flagged falls | falls at weight 0.5 | ignored by the scorer, 48 % of training falls |
| optimizer | AdamW 1.5e-4, weight decay 0.05, cosine, 5 % warmup, batch 8, 150 epochs (~33k steps) | batch 16 did not fit next to another job on a 24 GB GPU |
| decoder | Viterbi, switch penalty 2, minimum 0.4 s | a starting point: tune it out of fold |

## Validation: the official val set, and grouped cross-validation

The official val streams have different subjects from train, as the test does, so they are the
honest held-out set for the final number. They hold only 119 falls (67 scored with
`--ignore-flagged`) in 5 streams. That is too few to also tune on without overfitting. Tuning the
decoder, comparing configurations and picking a checkpoint on val all make the val score
optimistic.

Grouped cross-validation on the training streams (`data/folds.py`, the friend's 6 folds: each is
one presumed fall subject and one presumed activity subject) supplies the missing tuning data.
Windows cross recordings, so a fold cannot be a set of windows. Instead every timestep belongs to
the recording(s) covering it. For held-out fold k:
- normalizer, state losses and self-supervised losses use only timesteps outside fold k;
- the score counts only fold k's timesteps, with the official evaluator.

The input is not cut, so fold k's signal is still seen as unlabeled context next to training
timesteps. That is the remaining optimism. Why the folds matter: activity recordings appear in 3
training streams, and a subject's recordings are spread over many streams. Splitting by stream
would put the same person, often the same recording, on both sides.

The protocol:
1. Compare configurations by their 6-fold score, using the last epoch: `cv.yaml` chooses no checkpoint.
2. Tune the decoder on the pooled out-of-fold probabilities (`falljepa.tune_decoder`).
3. Train once on all of train with that configuration and decoder.
4. Score val once.

## What this changes in preprocessing

| Friend's v1 | Here | Why |
|---|---|---|
| 4 s windows, each inside one source recording | 20 s windows across joins | the model needs long context, and test streams have joins |
| 6 recording-grouped folds over windows | the same 6 folds over timesteps, as fit/score masks | windows cross recordings |
| scaler: one occurrence per source | per (radar, bin), over every training timestep | simpler; activity recordings reused 3× weigh a little more |
| raw z-score only | + temporal difference (on the continuous stream) + valid channel | motion over static clutter; explicit padding |
| flagged falls: no positive supervision | flagged falls as falls at weight 0.5 | the scorer ignores them, and they are 48 % of training falls |
| ±0.5 s boundary band ignored | begin/end states | the boundaries are what tIoU and the start error score |
| 50 Hz, no filtering or resampling | unchanged | |
| train-only statistics, label-free inference | unchanged | |

## Status

- **Verified on a synthetic release only.** No real HDF5 streams are in this folder. The 16 tests
  pass: labels, window coverage, normalization statistics, frame isolation, causality, padding,
  one-batch overfit, the decoding grammar, window merging, fold masks, masked normalization and scoring, decoder tuning, and CLI fit plus validate (also in cross-validation) from the
  checkpoint.
- **Synthetic learning run.** A 25-epoch run on the synthetic falls reaches F1 1.0 under the
  official evaluator, with starts about 0.25 s early.
- **Not tuned on real data.** The hyperparameters are reasoned defaults. Tune the decoder with the
  cross-validation, not on val.

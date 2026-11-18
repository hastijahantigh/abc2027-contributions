# FallJEPA handover: run the model on ABC27 and swap its encoder for a ViT

From Mehdi, 2026-10-31. This zip holds the full code of our current join-free fall detector, how to
run it on ABC27 against the data I use, and where a ViT can replace the encoder.
`VIT_IDEAS.md` describes the ViT options. `falljepa/handover/vit_sketch.py` holds two working,
tested ViT modules you can start from.

```
abc2027_falljepa_vit_handover/
  HANDOVER.md                 this guide
  VIT_IDEAS.md                where and how a ViT fits, ranked, with the risks
  falljepa/                   the code (a snapshot of my working tree, branch stream-cv)
    src/falljepa/             the package (model, data, losses, training module)
    configs/uwb/JF5T.yaml     the baseline to start from ("JF6" in my reports)
    configs/uwb/JF7b.yaml     the newest variant (same trunk, more timing heads; a tie with JF6)
    tools/run_jf_cv.sh        trains + validates + scores the 4 CV folds for one seed
    tools/score_folds.py      the shared scorer (official evaluator, pooled / per fold)
    docs/                     design docs (read the four listed in section 9)
    handover/                 vit_sketch.py, test_vit_sketch.py, results/ (my JF6 CV scores)
    report/                   two HTML reports (JF, JF2) that explain the pipeline with figures
```

---

## 1. The task in one paragraph

ABC2027 UWB-FallTrack: 3 UWB radars give a continuous stream of complex channel impulse responses
(184 range bins per radar, 50 Hz). The task is to output every fall as (`stream_id`, `t_start`, `t_end`,
`score`). The official evaluator matches by temporal IoU and averages F1 over tIoU 0.1 / 0.3 / 0.5,
so the **timing** of the start and end matters as much as detection. We do not trust the official
val (it is small and over-used). We compare models on a **subject-held-out 4-fold CV**. Each fold has
26 training streams and 10 held-out streams, and fold D reuses the official split.
`docs/stream_cv.md` describes it.

Our models are **join-free**: they never use the points where the organisers stitched clips into
streams, nor source-recording information. That is a group decision, for fairness and for the paper.
Please keep it that way. Do not add features derived from the joins or the recordings.

## 2. Setup on ABC27

The server is the SSH host ABC27 (user `abc2027`, port 44322). It needs the university network or the VPN.

```bash
# from your laptop: copy the code to your own folder on the server (do not use mine: mehdi*, nsd1)
scp -P 44322 abc2027_falljepa_vit_handover.zip abc2027@<ABC27 address>:/home/abc2027/
ssh ABC27
cd /home/abc2027 && unzip abc2027_falljepa_vit_handover.zip
mv abc2027_falljepa_vit_handover/falljepa /home/abc2027/<yourname>_vit && cd /home/abc2027/<yourname>_vit
git init && git add -A && git commit -qm "FallJEPA snapshot from Mehdi 2026-10-31"   # so you can diff your changes

export PATH=/home/abc2027/.local/bin:$PATH      # uv is not on PATH over ssh
uv sync                                          # Python env from uv.lock (torch, lightning, h5py, ...)
uv run pytest -q                                 # CPU tests on a synthetic release, a few minutes
uv run pytest -q handover/test_vit_sketch.py     # the ViT sketch, seconds
```

**The data is already on the server.** Read it, do not copy or rebuild it:

| What | Path |
|---|---|
| CV folds A–D (26 train + 10 held-out streams each) | `/home/abc2027/dataset/abc2027_uwb_cv_folds/{A,B,C,D}` |
| Official package (its evaluator scores the folds) | `/home/abc2027/dataset/abc2027_uwb_train_v1` |

Both are world-readable and checksum-verified, so your numbers are comparable with mine. Caches
(normalizer statistics, physics features) are written under `runs/_cache/fold_<F>/` inside **your**
folder on first use. They are fitted on the fold's training streams only.

**The GPU is shared.** It is one RTX 4090 (24 GB), and another user often holds about 12 GB. JF6 peaks
at about 9.5 GB (nvidia-smi). Run **one** training at a time, check `nvidia-smi` before you launch,
and tell me when you start a queue so we do not both run at once and go out of memory. If a ViT
variant needs more memory, lower `data.batch_size` and raise `trainer.accumulate_grad_batches` so the
step stays at 8 chunks.

## 3. Run the baseline first (JF6)

Reproduce JF6 on one fold before changing anything. That proves the setup works.

```bash
cd /home/abc2027/<yourname>_vit && export PATH=/home/abc2027/.local/bin:$PATH
export FOLDS_DIR=/home/abc2027/dataset/abc2027_uwb_cv_folds OFFICIAL=/home/abc2027/dataset/abc2027_uwb_train_v1

# one fold (D, the official split), seed 0, about 1 h
FOLDS=D GPU=0 CONFIG=configs/uwb/JF5T.yaml NAME=JF5T GROUP=UWB/JF5T SEED=0 TAG=jf6-check \
  nohup tools/run_jf_cv.sh > runs/jf6_check.log 2>&1 &
tail -f runs/jf6_check.log
```

For one fold, `tools/run_jf_cv.sh`:

1. trains for 15 epochs (a fixed budget, SWA from epoch 2; nothing is selected on the held-out streams);
2. validates `checkpoints/swa.ckpt` on the 10 held-out streams;
3. scores them with the official evaluator.

The script handles three cases:

- **Resume:** run the same command with the same `TAG`. Finished folds are skipped, and an interrupted fold resumes from `last.ckpt`.
- **The full CV:** use `FOLDS="A B C D"`, one seed at a time. That is about 4–5 h per seed.
- **Extra overrides:** put them after `--`, e.g. `tools/run_jf_cv.sh -- --model.model.init_args.dropout 0.2`.

Outputs:

```
runs/UWB/JF5T-D/JF5T-s0/jf6-check/
  config.yaml                       the full resolved config (re-run / validate with -c this)
  checkpoints/{last,swa}.ckpt
  predictions/val-validate.csv      submission format: the full model
  predictions/val-validate.queries.csv / .dense.csv   stage 1 alone / the dense head alone (diagnostics)
  csv/fit-*/metrics.csv             training curves (W&B is offline on the server)
runs/_analysis/cv_JF5T_jf6-check.json   score: pooled, per_fold, per_stitching, fold_mean, worst_fold
```

**What you should see on fold D, seed 0: about .84** (my 2 seeds gave .844 and .878). If you get
something far below that, stop and tell me before training ViT variants.

## 4. The numbers to beat

Here are the JF6 results (`configs/uwb/JF5T.yaml`) on the 4-fold CV, without post-processing. The
files are in `handover/results/`.

| | A | B | C | D | fold mean | pooled |
|---|---|---|---|---|---|---|
| JF6 seed 0 | .902 | .916 | .930 | .844 | .898 | .897 |
| JF6 seed 1 | .900 | .897 | .925 | .878 | .900 | .900 |
| JF7b (2-seed mean) | .903 | .902 | .941 | .866 | .903 | — |

Rules for a fair comparison (from `docs/stream_cv.md` §17):

- **Run at least 2 seeds** per variant. Training is not reproducible even with the same seed (GPU
  nondeterminism), and seed noise is about ±.02–.03 per fold. A difference below about .02 in the
  fold mean is a tie.
- **Do not select anything on the held-out streams.** Use the fixed 15 epochs, the SWA checkpoint and
  no early stopping. Fit everything (normalizer, thresholds) on the fold's training streams only.
- **Report the fold mean, the pooled score, each fold and the worst fold.** Fold D is the hardest and
  usually decides.
- **Do not change the data or the folds for one model only.** If you think more data helps, propose
  it as an ablation for all models.

## 5. The model in one page

Code: `src/falljepa/model/fall_set.py` (FallSetJEPA), which subclasses `fall_jepa.py` (FallJEPA, the trunk).

```
stream ──► windows of 32 s (1600 steps of 20 ms) + 14 steps margin each side; every 6 s at inference
           training item = a "chunk" of 4 consecutive windows 6 s apart (50 s), 8 chunks per step

one window:  x [13, 184, 1628]   channels = I x3 radars, Q x3, I_c x3, Q_c x3 (clutter removed), valid
   │
   ├─ TRUNK (fall_jepa.FallJEPA.encode)  ◄── the part you replace
   │   stem   model/stem.RadarStem: complex conv (|.|, phase-invariant) → 3 real convs → range attention
   │          pooling; "early" fusion stacks the 3 radars as channels  → z [400, 64], token i = steps [4i, 4i+32)
   │   pre-projector (MLP + BN) → z                                     → SIGReg loss (keeps z spread out)
   │   local  model/stem.TCN: 4 dilated residual convs (1 2 4 8), 4.9 s receptive field, non-causal
   │   global 3 bidirectional transformer blocks with RoPE (no absolute positions) → r [400, 64]
   │
   ├─ dense head: per-step fall / no-fall from r (CE losses; also a Viterbi baseline output)
   ├─ STAGE 1 (set_heads.FallQueries): 3 DETR-style queries per window → P(fall), start and end
   │          distributions over the window's steps (set-matched BCE + soft CE + tIoU losses)
   └─ STAGE 2 (set_heads.GroupRefiner): proposals of overlapping windows are grouped on the stream;
              each group is refined from its members, the stream tokens of its region and "event
              physics" (before/after signal features) → 2 slots per group, P(fall), start, end

stream decode (module.py / postprocess.py, decoder.kind "sets"): threshold .5, NMS → the submission CSV
```

Where things live:

| File | What |
|---|---|
| `model/fall_jepa.py` | the trunk: `__init__` builds stem / TCN / blocks, `encode()` runs them, `forward()` the dense head |
| `model/stem.py` | RadarStem, ComplexConv2d, RangePool, RadarFusion, TCN |
| `model/layers.py` | Attention (with `key_valid`, RoPE), Block, FeedForward, `rotary` |
| `model/fall_set.py` | FallSetJEPA: cuts chunks into windows, stage 1, stage 2 training, inference over streams |
| `model/set_heads.py`, `model/sets.py` | FallQueries, GroupRefiner, BoundaryRefiner (JF7b), Hungarian matching, grouping |
| `data/uwb.py` | HDF5 stream reading, normalizer, channels, windows and chunks, augmentations (`UWBDataModule`) |
| `data/sampler.py` | balanced sampler (4 of 8 chunks hold a fall, 2 hold a hard negative) |
| `loss/losses.py` | every loss named in the config's `losses:` block |
| `module.py`, `cli.py`, `callbacks.py` | Lightning module, run folders and config CLI (jsonargparse), SWA (`WeightAveraging`) |
| `official/evaluate.py` | the official evaluator, unchanged |

Configs are jsonargparse YAML: `model.model.init_args` are the constructor arguments of `FallSetJEPA`.
Any of them can be overridden on the command line (`--model.model.init_args.d_model 96`).

## 6. Where the encoder sits, and the contract a replacement must keep

`FallJEPA.encode()` (in `fall_jepa.py`) is the only place the trunk runs:

```python
z = self.stem(x, Fn).reshape(B * Fn, -1)      # trunk "stem":  x [B, 13, 184, L] -> [B, Fn, d]
z = self.pre_proj(z).view(B, Fn, -1)
h = z if self.rope else z + self.frame_pos_embed[:Fn]
if self.tcn is not None:
    h = self.tcn(h, frame_valid)              # local "tcn":   [B, Fn, d], frame_valid [B, Fn] -> [B, Fn, d]
for blk in self.bidir: r = blk(r, key_valid=frame_valid)   # global: 3 RoPE blocks
```

A replacement for the **stem** (`forward(x, frames)`) or for the **local encoder** (`forward(h, valid)`)
with the same signature drops in without touching anything else. It must keep five things:

1. **Shapes.** The output is `[B, Fn, d_model]`, with Fn = 400 tokens for a 32 s window (one token per
   4 steps = 80 ms). Everything downstream indexes tokens as `4 i` steps.
2. **Alignment.** Token i must describe steps `[4 i, 4 i + 32)`, the same footprint as the old frame i.
   `tests/test_jf4.py::test_stem_token_i_sees_exactly_frame_i` checks this with gradients.
   `handover/test_vit_sketch.py` does the same for the ViT sketch.
3. **Valid mask.** Tokens past the stream end (`frame_valid` False) must not leak into valid ones.
   Mask them as attention keys and zero them at the output, as the TCN does.
4. **No absolute positions in the trunk.** The stream stage averages the tokens of the same instant
   seen from overlapping windows. A learned absolute position embedding makes those tokens differ by
   window position. Use RoPE or relative positions inside the trunk. (FallQueries adds its own
   positions to its keys.)
5. **Radar dropout.** `data.augment.radar_drop_p` zeroes a whole radar. A fusion over radars should mask
   a radar whose signal is all zero (`RadarFusion` and the sketch both do).

Two practical notes:

- **BatchNorm.** The SWA callback recomputes BatchNorm statistics (`bn_batches: 50`), so BN layers
  are fine. LayerNorm on raw energies amplifies noise at still range positions (`docs/falljepa_JF4_design.md` §3.1).
- **JEPA predictor.** JF6 has none (`predictor_layers: 0`). A non-causal local encoder would leak the
  future into it. Keep `predictor_layers: 0` unless your ViT is causal.

## 7. Adding a ViT, step by step

The sketch is in `handover/vit_sketch.py`:

- `RadarViT` replaces `RadarStem`: a complex tubelet patch embedding aligned to frames, then
  attention over radar × range patches within each frame.
- `LocalViT` replaces `TCN`: RoPE transformer blocks with attention limited to ±30 tokens (4.9 s, the
  TCN's reach).

1. **Copy the module.** `cp handover/vit_sketch.py src/falljepa/model/vit.py`, then change its two
   imports to relative ones (`from .layers import ...`, `from .stem import ...`).
2. **Build it in `FallJEPA.__init__`** (`fall_jepa.py`, around line 128):
   - Allow the new names: `assert trunk in ("frames", "stem", "vit") and local in ("none", "tcn", "vit")`.
   - Trunk: in the `else:` branch, build `self.stem = RadarViT(...)` when `trunk == "vit"`. Keep the
     attribute name `stem`, so `encode()` needs no change.
   - Local: `self.tcn = TCN(...) if local == "tcn" else LocalViT(...) if local == "vit" else None`.
3. **Add your new arguments** (e.g. `vit_depth`, `vit_patch_bins`, `local_depth`, `local_radius`) to
   `FallJEPA.__init__` **and** to `FallSetJEPA.__init__` (`fall_set.py`, the "JF4 trunk" block), and
   pass them through its `super().__init__(...)` call. jsonargparse reads the config keys from these
   signatures.
4. **Write the config.** `cp configs/uwb/JF5T.yaml configs/uwb/JFV.yaml`. Change `run.group` and
   `run.name`, and set `trunk: vit` and/or `local: vit` plus your arguments. Change one thing at a
   time: first the local encoder only (stem kept), then the stem only (TCN kept), then both.
5. **Test it on CPU.** Copy `tests/test_jf4.py::test_cli_jf4_fits_and_validates` into a
   `tests/test_vit.py` with `["JFV"]` as the parameter. It fits 2 tiny epochs and validates on the
   synthetic release. Also keep `handover/test_vit_sketch.py` passing.
6. **Check GPU memory.** Before a long run on ABC27, run 1 epoch on one fold and read nvidia-smi:
   `tools/run_jf_cv.sh -- --trainer.max_epochs 1` with a throwaway TAG.
7. **Run fold D with seeds 0 and 1** and compare with JF6 D (.844 / .878). Then run all 4 folds if it
   holds up:
   `FOLDS="A B C D" CONFIG=configs/uwb/JFV.yaml NAME=JFV GROUP=UWB/JFV SEED=0 TAG=<date>-jfv-s0 tools/run_jf_cv.sh`

Transformers usually want a warmup. JF6 uses a constant AdamW 1e-3
(`falljepa.optimizer.AdamWConstant`); `falljepa.optimizer.AdamWWithCosineScheduler` adds a linear warmup and a cosine decay
(`init_args: {learning_rate: 1.0e-3, weight_decay: 1.0e-4, min_lr: 1.0e-5, warmup_ratio: 0.05}`). If a
ViT trains unstably, try a lower learning rate or this schedule before you judge the architecture.
Note that the schedule changes the baseline too: for a fair comparison, also run JF6 with it, or keep
the constant rate.

## 8. Practical pitfalls

- `uv` is at `/home/abc2027/.local/bin/uv`. Export the PATH in every ssh session and in your queue scripts.
- Use `nohup ... &`, or a queue script like `tools/run_jf7_queue.sh`, so a dropped ssh does not kill training.
- W&B is offline on the server (`run.wandb: offline`). Read `csv/fit-*/metrics.csv`, or sync later from a machine with internet.
- Never point `--run.root` at my folders. Each person's normalizer and physics caches live in their own `runs/_cache`.
- `num_workers: 10`. The server has 12 CPUs, so leave them alone while training runs.
- If you change the input channels (e.g. add contrast maps), `in_channels` and the `data` options must
  agree. See `docs/falljepa_JF4_design.md` §4.1–4.2.

## 9. What to read, in this order

1. `docs/falljepa_join_free.md` §0–6: the join-free model (stage 1, chunks, stage 2, decoding).
2. `docs/falljepa_JF4_design.md` §1–3 and §11: why the stem and TCN look the way they do, measured on
   the radar data, and the fold D results that led to JF6. **This is the most important one for a
   trunk change.** It also explains why JF2's per-frame transformer (a small ViT on 23 × 8 patches,
   `trunk: frames` in `model/encoder.py`) lost to the stem.
3. `docs/stream_cv.md` §9–11 and §17: training a fold, scoring, deciding between models.
4. `docs/falljepa_JF7_design.md`: the current timing work (only needed if you touch the heads).
5. `report/falljepa_JF_report.html`, `report/falljepa_JF2_report.html`: figures and error analysis.

Questions: ask me. If something in this guide does not match what you see on the server, trust the
server and tell me.

# Ideas: putting a ViT into FallSetJEPA

Companion to `HANDOVER.md` (§6 explains the contract a replacement must keep, §7 the wiring steps).
The ideas are ranked by how likely they are to help, judged from what we measured on this data.
Ideas 1 and 2 have working, tested code in `falljepa/handover/vit_sketch.py`.

## 0. What we already know (read before choosing)

- **A small ViT has been tried once: JF2's trunk.** `trunk: frames` in `model/encoder.py` cut each
  window into 400 overlapping 0.64 s frames. Each frame was split into 23-bin × 8-step patches and
  encoded with a 2-layer transformer and a CLS token. The convolutional stem (JF4/JF5) replaced it
  and gained on fold D (.811 → .847–.864). Almost all of the gain came from fewer false alarms. The
  reasons behind the stem's design (`docs/falljepa_JF4_design.md` §1–3) suggest why:
  1. **Phase.** Each radar has an arbitrary global phase, so a linear patch embedding of raw I/Q has to
     learn phase invariance from 12 h of data. The stem's complex conv followed by |·| has it built in.
  2. **Fine range-time structure.** A fall is about 1 m of range migration over about 1 s, at
     0.3–0.65 bins per frame. 23 × 8 patches flattened into one linear map blur it. Small 5 × 9 kernels keep it.
  3. **Few data.** About 12 h of training streams per fold.
- **So a ViT should keep the stem's inductive biases**: a phase-invariant front end, frame-aligned
  tokens and range attention with a range embedding. The global transformer already exists: 3 RoPE
  blocks over the 400 tokens. "Adding a ViT" is most promising as a *local* encoder or a *hybrid*
  stem, and least promising as a raw-patch ViT that replaces everything.
- **Where the errors are now** (JF6/JF7b error analysis):
  - Timing: about +.035 of the score is lost to starts and ends, and long falls start about 0.5 s late.
  - Misses: about +.04. They are mostly weak-motion falls of subjects P1_8/P1_9, Chute04.
  - False alarms: about +.02, from Autre / Faire_menage ADLs.

  A trunk change is most likely to help the misses and the false alarms. Timing is mostly a head problem.

---

## 1. LocalViT: windowed transformer instead of the TCN  ★ start here

**What.** Replace `model/stem.TCN` (4 dilated convs, ±30 tokens) by 2–4 pre-LN transformer blocks
with RoPE whose attention is restricted to a band of ±r tokens. `vit_sketch.LocalViT(depth,
radius)` does this. The stem stays, and so do the 3 global blocks.

**Why it may help.** Attention can pick *which* moments in the 4.9 s neighbourhood matter (the
moment of impact, the still period after it). Convolutions weight neighbours by fixed offsets. With
`radius: 30` the reach equals the TCN's, so the comparison is clean.

**Settings to try.** These are runs, not a grid search; 2 seeds each on fold D first:

| run | change from JF6 |
|---|---|
| V1a | `local: vit`, depth 2, radius 30 (same reach as TCN, fewer params: 66 k vs 100 k) |
| V1b | depth 4, radius 15 (stacked reach ±60) |
| V1c | depth 2, radius 60 (±4.8 s per block) |

**Cost.** Band attention is computed as full 400 × 400 attention with a mask, which is cheap at d = 64.
Memory should stay close to JF6's.

**Optional variant** (not in the sketch). Combine the two: TCN first, then local attention (`local: tcn+vit`). This tests
whether attention adds anything on top of the convolution rather than replacing it.

## 2. RadarViT: a frame-aligned ViT stem with a phase-invariant patch embedding  ★ second

**What.** Replace `RadarStem` with a ViT over range patches. `vit_sketch.RadarViT` does this:

- **Patch embedding.** Per radar, a *complex* tubelet embedding (`ComplexConv2d`, kernel 23 bins ×
  32 steps, stride 23 × 4). Patch column i then covers exactly steps [4i, 4i + 32), frame i. That
  gives 8 range patches per radar per frame.
- **Energies.** |·| → log1p, which is invariant to each radar's phase, as in the stem's L0.
  Then a linear map to d = 64 and range + radar embeddings.
- **Encoder.** Transformer blocks over the 24 radar × range tokens **of the same frame**: spatial
  attention only, so frames stay independent and time is mixed later by the local and global
  encoders. A radar dropped by `radar_drop_p` is masked.
- **Pooling.** A learned query attends to the 24 tokens → one token per frame.

Tests (`handover/test_vit_sketch.py`) check:

- token i depends only on steps [4i, 4i + 32);
- the output is invariant to each radar's phase;
- the module runs forward and backward inside `FallSetJEPA.encode`.

**Why it may help.** This is "late" radar fusion by attention. The network can see *which* radar
and *which* range moved, and the range embedding keeps "where".

**Risk, and what to try if it loses to the stem.** A 23 × 32 patch with one linear map is coarse.
That was reason 2 in §0. Fixes, in order:

1. **Smaller patches.** `patch_bins: 8` gives 23 range tokens per radar, 69 per frame. Attention per
   frame is still cheap.
2. **Hybrid ("conv stem + ViT").** Keep the stem's L0–L1 (complex conv, 5 × 9 kernels), then patchify
   its feature map instead of the raw I/Q. This is the standard fix for ViTs on small data (Xiao et al.
   2021, "Early convolutions help transformers see better").
3. **Factorised space-time attention.** In the style of TimeSformer / ViViT: in each block, attention
   over range × radar within a frame, then over time within a band. This replaces stem + local
   encoder together. It is more expressive, and it is the largest change. Try it only if 2a or 2b is
   at least on par.

**Cost.** RadarViT has about 197 k params (depth 2, d 64) vs 78 k for the stem. Attention runs over
24 tokens × 400 frames × 32 windows per step, so measure memory with a 1-epoch run first.

## 3. Masked pre-training of the ViT on the fold's training streams

**What.** ViTs gain the most from self-supervised pre-training. The repo already has the pieces of a
JEPA: SIGReg on z, a latent predictor, `pred_horizons`. Two options:

- **I-JEPA / V-JEPA style.** Mask blocks of frames (and/or range patches) and predict the latents of
  the masked tokens from the visible ones with a small predictor. An EMA or detached target encoder
  gives the targets (`detach_target: true` exists). Then fine-tune with the current losses.
- **MAE style.** Mask 50–75 % of patches and reconstruct the log-energy of the masked patches.

**Protocol.** Pre-train **only on the fold's own training streams**, once per fold. Their unlabeled
time also counts: about 12 h per fold, mostly ADL and background. Never pre-train on held-out
streams or other folds' held-out subjects. That would leak subjects.

**Expected value.** This is uncertain: it is little data for pre-training. But it addresses the main
weakness of ViTs here. Try it after 1 and 2 have shown which ViT architecture is on par.

## 4. ImageNet / AudioSet pre-trained ViT (AST-style) — low priority

Treating each radar's range × time map as an image and fine-tuning a pre-trained ViT (e.g. DeiT-tiny, AST)
is the classic route. I do not recommend it as a first step:

- The inputs are complex I/Q, not RGB.
- 184 bins × 1600 steps per window would need patches that break the frame alignment.
- A DeiT-tiny is 5 M params with d = 192, against our 64. The heads would need a projection.
- Memory on the shared GPU gets tight: 32 windows per step.

If you try it, feed log-energy maps (|z| and |c| per radar as channels) and project the patch tokens
to d = 64 with frame-aligned temporal pooling (§6 of the handover).

## 5. Smaller ideas worth an ablation

- **Range-only attention in the stem.** Replace `RangePool` (softmax pooling) with 1–2 transformer
  blocks over the 23 range positions before pooling. This is the cheapest "ViT" change. It does not
  touch the convs.
- **Attention over radars before fusion.** Use `radar_fusion: attention` (it exists already)
  combined with idea 2's per-radar tokens.
- **Token count.** The tokens are 80 ms apart. A ViT with temporal stride 8 (160 ms, 200 tokens) would
  halve compute, but **breaks the 4i alignment** that the heads assume. Only do this with an upsampling
  layer back to 400 tokens.
- **Drop path / stochastic depth** (0.1) and a lower learning rate (5e-4) are the usual regularisers
  for small-data ViTs. Try them if a ViT overfits: high train F1, held-out F1 dropping across epochs in `metrics.csv`.

## 6. Order of work I would follow

1. Reproduce JF6 on fold D (handover §3), 1 seed.
2. V1a (LocalViT, stem kept) on fold D, 2 seeds. Compare with .844 / .878.
3. RadarViT with the TCN kept, fold D, 2 seeds. If it is clearly worse, try 2.1 (smaller patches),
   then 2.2 (hybrid).
4. The best of 2–3, then both together, on all 4 folds × 2 seeds. Compare fold mean and worst fold with
   JF6 (.898 / .900).
5. Only then: pre-training (idea 3).

Keep a short log of each run: config, TAG, fold scores, peak GPU memory. Send me the
`runs/_analysis/cv_*.json` files, and I can put them in the same tables and error analysis as my runs.

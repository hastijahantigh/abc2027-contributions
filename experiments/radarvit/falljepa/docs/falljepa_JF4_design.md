# JF4: a stage-1 trunk shaped by the radar data (design, 2026-10-29)

Branch `stream-cv`. Base: **JF2** (`configs/uwb/JF2.yaml`, CV .861 pooled on both seeds), not JF3.
JF4 changes only **stage 1**: the trunk (embedding, local encoder, global encoder) and its inputs.
Stage 2 (group refiner, event physics), decoding, sampler, folds and epochs stay as in JF2.
Nothing is implemented yet.

The data measurements are in `runs/_analysis/jf4_data/`: the scripts `events.py`, `contrast.py` and
`migration.py`, and their tables. They cover all 18 published streams: 474 falls and 1772 ADL
spans of 2 s. The scripts read only the raw streams, `streams_events.csv` and
`streams_composition.csv`; the composition file is used for analysis only, never as model input.

## 0. What stage 1 gets wrong in JF2

From the JF2 error analysis (`report/falljepa_JF2_errors.html`, both seeds):

| Error | JF2 | What it says about stage 1 |
|---|---|---|
| Timing errors | 73 misses + 77 false alarms | Stage 1 is already +1.5 s late on these falls. It fires on the strongest motion burst. The label is a state step: the scene changes from the standing profile to the floor profile. |
| Undetected, weak falls | 50 misses that JF2 never fires on. Weakest-motion quartile: 29 % missed vs 3 % in the strongest. Chute04 recall .71. | The trunk does not see weak motion. The dense head is silent too (median P .07), so this is a front-end (embedding) problem. |
| ADL false alarms | 32 (0.97 / h), 21 of them on Faire_menage and Autre | Real motion bursts with no lasting change of the scene. |
| Unseen sessions | fixed by JF2's clutter channels and background swap | Keep the invariance; do not reintroduce the session background. |

JF2 also showed where the gains come from. The trunk moved stage 1 with the fixed merge from .684 to
.832, and the dense head from .632 to .791. The stream stage added only +.03. Improving the trunk
is therefore the larger lever.

## 1. What the data says

| # | Measurement | Value | Consequence for the design |
|---|---|---|---|
| D1 | Motion energy of a fall on its strongest radar vs the 2nd | **39×** median (Chute01–03: 65–80×, Chute04: 16×). Strongest vs weakest: 280×. ADL: 2.3× and 4.7×. | A fall is seen by **one** radar; all three take that role (110 / 201 / 163 falls). Early fusion mixes two quiet radars into every patch. |
| D2 | Share of the motion on the strongest radar | .98 (Chute04 .93) | Same as the teammate's profile-step share (.99 for falls vs .50 at joins, `report/falljepa_vs_triradar.html`). |
| D3 | Same-bin correlation across radars, `\|clutter-removed\|` at the motion bins | .06 (falls), .01 (ADL) | Bin k of radar A and bin k of radar B are different places. |
| D4 | Fall motion extent in range | contiguous run p50 **8 bins** (p90 14); half the energy in 6 bins; peak bin p50 13 (near the radar) | ≈ 40 cm at 5.14 cm per bin. A 23-bin patch edge cuts the extent in about **31 %** of falls. |
| D5 | Motion during a fall | the motion centroid moves p50 37 bins; speeds 0.8–2 m/s (Chute04 1.4) ≈ 0.3–0.65 bins per frame | Falls are diagonal streaks in range × time: 2-D kernels see them. |
| D6 | Phase | Per-frame phase step at the motion bins: p50 0.9 rad, p90 1.42 rad (uniform noise would give 1.57). The static phase drifts 0.07–0.67 rad (SD over 10 s blocks). | With λ ≈ 3.4–4.1 cm, the phase aliases above about 0.5 m/s, so the fast part of a fall is aliased at 50 Hz. Absolute phase carries no information. A model on raw I/Q must learn rotation invariance from about 26 training streams. |
| D7 | Doppler band energy of the clutter-removed signal (0–.5 / .5–1 / 1–2 / 2–4 / 4–8 / 8–25 Hz) | falls .26 .11 .11 .14 .15 .22; ADL .20 .13 .11 .14 .16 .26 | The spectrum alone does not separate falls from ADL. The range × time pattern and the state change have to. |
| D8 | Fall timing vs the label centre | motion peak −0.12 s (IQR −0.56 … 0.25), inside the label 76 % | Motion locates most falls. |
| D9 | Before/after contrast of the complex profile (1 s windows, 0.3 s gap) | **with the joins masked:** peak −0.01 s from the centre (IQR −0.44 … 0.63), inside the label 73 %. **Unmasked:** peak 1.2 s *before* the label start. | The contrast is the labeller's quantity and is centred on the label. Unmasked, joins dominate it; the nearest join to a fall start is p10 0.9 s, p50 3.0 s. |
| D10 | Lowest raw-energy radar | radar 29171 (index 0) for 83 % of falls and 99.7 % of ADL | "The radar with the lowest energy" is almost always the same radar. |

## 2. The four proposed ideas

### 2.1 CNN embedding at the beginning: **yes. This is the core of JF4.**

Today each 0.64 s frame is cut into 23-bin × 8-step patches of all 19 channels. Each patch is
mapped by one `Linear(3496 → 64)` (224 k parameters), and a 2-layer transformer runs per frame. The
data shows four problems with this:

- **No range equivariance (D4, D5).** The person is 8 bins wide and moves about 37 bins. The linear
  map must learn the same motion at each of the 23 offsets inside a patch, and 31 % of falls are cut
  by a patch edge.
- **Linear before any energy.** Weak falls (Chute04 0.4×, Dataset 3 0.2× the motion) need a
  matched-filter energy, |Σ w·x|, i.e. coherent integration over range and time and then a modulus.
  The linear patch map can only form that energy after squeezing everything into 64 dimensions.
- **Phase-sensitive (D6).** The absolute phase of each radar is arbitrary, and the linear map has to
  learn invariance to it.
- **Redundant.** Each 8-step patch column is embedded by 4 overlapping frames.

**How:** a convolutional stem run once over the whole window, per radar with shared weights
(section 3.1). Its first layer is a complex convolution followed by a modulus. That makes it
invariant to each radar's phase by construction and turns it into a learned bank of
range–Doppler matched filters. A per-radar stem is required here: a complex convolution that mixes
radars before the modulus would not be phase-invariant, because each radar has its own oscillator.

### 2.2 Difference in embedding space, e(t) − e(t−1) concatenated to e(t): **not at lag 1. Yes as a long-lag before/after contrast, computed on the signal.**

- **Lag 1 adds nothing.** Tokens are 0.08 s apart and each sees 0.64 s, so consecutive tokens share
  87 % of their input. e(t) − e(t−1) is a small, noisy derivative. A kernel-3 convolution over the
  tokens (the first TCN layer, 2.3) can form it exactly, together with any other neighbour
  difference. A fixed concatenation is a strict special case.
- **What the label needs is a long-lag difference.** The label marks where the static scene
  steps from "standing" to "on the floor", on the radar where that change is largest. The matching
  quantity is the before/after contrast at the scale of 0.5–2 s. D9 shows it is centred on the label
  and complementary to the motion peak (D8). Stage 1 has no such input today. JF2's event physics
  computes it only in stage 2, around proposals that are already late, and the refiner follows them.
- **Where to compute it.** In the **signal**, not in embedding space. On the complex profile,
  `mean(z[t+g … t+g+w]) − mean(z[t−g−w … t−g])` cancels the session background exactly, as long as
  the background is constant. A difference of embeddings does not, because the encoder is nonlinear.
  JF4 therefore gives the stem three contrast maps per radar (section 4.1). The TCN then learns any
  further differences in embedding space.
- **The joins.** A join is also a step of the scene (D9, unmasked), so a contrast input makes joins
  salient. JF4 pairs the contrast with a **step-swap augmentation**: decoy background steps at random
  times, away from falls (4.3). A step then no longer tells the model where a fall is. The per-radar
  fusion also lets the model see that a join changes all three radars, while a fall changes one (D2).

### 2.3 TCN instead of the frame transformer, no pooling: **yes, as the local encoder after the stem.**

- **The frame transformer cannot see the fall as a unit.** It sees one 0.64 s frame at a time. A fall
  takes 1–3 s (p50 1.75 s, p90 3 s), and the labeller reads 1–1.5 s of scene before and after it, so
  the event lives at about 5 s. Today only the global transformer sees that scale, through absolute
  positions and softmax attention, which are poor at precise local order (onset → descent → impact →
  stillness).
- **The TCN covers that scale and keeps the resolution.** A non-causal dilated TCN over the 0.08 s
  tokens has a receptive field of 61 tokens (4.9 s, centred). That covers the fall and its
  before/after context. With no pooling, the 0.08 s grid stays, and the pointer needs it for 20 ms
  spans.
- **Separation of roles.** With a convolutional stem the frame transformer has no job left:
  - the **stem** is the 0.66 s micro-motion encoder;
  - the **TCN** is the 5 s event encoder;
  - the **transformer** is the 32 s context encoder.
- **Consequence for JEPA.** A centred TCN sees the future, so the JEPA predictor (causal context →
  z(i+8)) would leak. JF4-ST drops the predictor and `MSE_Latent`, as in the A1 ablation (no-JEPA won
  5 of 6 folds in the AP era). SIGReg stays on the stem tokens z. JF4-S keeps JEPA, because there z
  has the frame's footprint.

### 2.4 Each radar minus the radar with the lowest energy: **no. The measurement rules it out.**

- **It lowers the signal.** Motion SNR (fall span vs the stillest 1 s before it) drops from **440 to
  245** (medians). It is lower in **99 %** of falls and 98 % of ADL spans.
- **The bins do not match (D3).** Bins of different radars are different places, and their phases
  come from independent oscillators. The subtraction adds the reference radar's noise and motion to
  the other two.
- **The reference is often the fall's own radar.** In **23 %** of falls (110 of 474), the
  lowest-energy radar is the radar that sees the fall. The subtraction then copies the fall into the
  other two radars and erases the one-radar signature (D1), which separates falls from ADL and joins.
- **It is a fixed radar (D10).** In practice it is "minus radar 29171", not an adaptive reference.

What the idea is after (a common reference, "which radar differs") is better served by each radar's
own clutter removal (JF2 has it), the contrast maps (2.2), and attention across radar tokens (3.2),
which can learn "one radar changed, the others did not".

## 3. The JF4 trunk

```
x [B, 22, 184, 1628]  (per radar: z, clutter c, 3 contrast maps; + valid)
  │  per radar (shared weights; radars folded into the batch)
  ├─ L0  complex conv  (z, c) → 16 complex, k 5 bins × 9 steps, stride (2, 2), no bias → |·| → log1p → BN
  ├─ L1  conv 16 (+3 contrast, avg-pooled 2×2) → 32, k 3 × 5, stride (1, 2), BN, GELU      [92 bins, 0.08 s]
  ├─ L2  conv 32 → 48, k 3 × 3, stride (2, 1), BN, GELU                                    [46 bins]
  ├─ L3  conv 48 → 64, k 3 × 3, stride (2, 1), BN, GELU                                    [23 bins ≈ 41 cm each]
  ├─ crop to 400 tokens: token i centred on step 4 i + 16 (= frame i), receptive field 21 bins × 33 steps
  ├─ range attention pool: 23 positions + range and radar embeddings → 4 heads → [64]  per radar and token
  │
  ├─ radar fusion, per token: a learned query attends to the 3 radar tokens (+ radar embeddings);
  │                           a radar whose signal is all zero (radar dropout) is masked     → [64]
  ├─ + physics frame embedding (as JF2) → pre-projector → z            (SIGReg; JEPA in JF4-S only)
  │
  ├─ local:  TCN, 4 residual blocks, dilations 1 2 4 8, k 3, non-causal → receptive field 61 tokens = 4.9 s
  ├─ global: 3 bidirectional transformer blocks with RoPE (no absolute positions) → r [400, 64]
  │
  └─ r → dense head (unchanged) · FallQueries (keys get the learned position embedding) · stage 2 (unchanged)
```

### 3.1 Stem (per radar, shared weights)

- **Input per radar.** Raw z and clutter-removed c as two complex channels (4 real), plus the
  three contrast maps (from 4.1). The amplitude channels |z| and |c| are dropped: the stem computes
  its own moduli.
- **L0: complex convolution, no bias.**
  - Written as a real `Conv2d` with the tied block weights `[[Wr, −Wi], [Wi, Wr]]`. A bias would break
    rotation equivariance.
  - Then |y| = sqrt(re² + im² + 1e−6), then log1p, then BatchNorm.
  - This is exactly invariant to a global phase rotation of each radar: z and c rotate together,
    since c = z − movmean(z).
  - Kernel 5 bins × 9 steps (26 cm × 0.18 s). It can resolve range migration of 0.3–0.65 bins per
    frame (D5) and Doppler up to Nyquist.
  - Stride 2 in range and in time.
- **L1–L3.** Real convolutions with BatchNorm and GELU. The contrast maps enter at L1, average-pooled
  2 × 2 to the L0 grid. Strides bring the grid to 23 range positions and one token per 4 steps.
- **Receptive field.** 21 bins (1.1 m, about 2× the fall's extent) × 33 steps (0.66 s, the old
  frame). Token i is centred on frame i's centre, so the dense head, the JEPA horizon and every
  existing `4 i` index stay valid.
- **Size and cost.** About 55 k parameters, against 224 k for today's patch embedding. About 30 G
  multiply-accumulates per micro-batch of 8 windows (the same as today). Activations are about
  1.4 GB, against about 2.2 GB for today's unfolded patches plus the frame transformer.
- **BatchNorm, not LayerNorm.** A per-position LayerNorm would blow up the noise at still positions.
  The SWA callback already recomputes every BatchNorm (`bn_batches: 50`).

### 3.2 Range pooling and radar fusion

- **Range attention pooling.**
  - LayerNorm(u_p + range_embed[p] + radar_embed[r]) over the 23 positions p.
  - 4 heads, each a softmax over p, from `Linear(64 → 4)`; values `Linear(64 → 64)`, split 4 × 16;
    output `Linear(64 → 64)`.
  - The range embedding keeps "where" (the mats are at fixed ranges). The softmax can act as a soft
    max for small, concentrated motion.
- **Radar fusion.**
  - One pre-LN cross-attention layer: a learned fusion query attends to the 3 radar tokens, each
    with its radar embedding. 4 heads (the existing `layers.Attention` with `key_valid`), then an FFN
    (64 → 128 → 64).
  - This is late fusion: the model sees *which* radar moved or changed (D1, D2) instead of a mix.
  - A radar whose input is all zero, i.e. dropped by `radar_drop_p`, is masked.
  - The attention weights are logged per token. They should concentrate on one radar during falls
    (diagnostic).
- **Physics frame embedding.** JF2's 12 physics features, averaged per patch column, are projected
  and added to the fused token before the pre-projector, where JF2 added them to each frame's tokens.

### 3.3 Local encoder: TCN

- **Block(δ).** `x + Drop(GELU(LN(Conv1d(GELU(LN(Conv1d(x, k3, dil δ))), k3, dil δ))))`, d = 64,
  dropout .1. LN is over channels, here it sits on contextual tokens rather than raw energy.
- **Dilations** 1, 2, 4, 8: receptive field 1 + 4 · 15 = 61 tokens (4.9 s). The first layer's
  [−1, 0, +1] taps include the user's lag-1 difference.
- **Padding.** Inputs are multiplied by `frame_valid` before every convolution, so padding past the
  stream end stays zero.
- **Size.** About 98 k parameters.

### 3.4 Global encoder

- **3 bidirectional blocks** (`layers.Block`), replacing JF2's 2 causal + 2 bidirectional. Each
  block uses **RoPE** on q and k (head dimension 16, base 10000, positions = token index).
- **No absolute position embedding in the trunk.** The same instant then gives the same token in
  each of the 4–5 windows that see it. The stream stage averages exactly these tokens
  (`TokenAccumulator`, `_chunk_tokens`), and with JF2's absolute positions it averages
  position-dependent vectors.
- **FallQueries still needs positions** so its 3 queries can specialise, as in DETR. Its
  cross-attention memory becomes `tokens + frame_pos_embed` (the existing 400 × 64 parameter).
  The trunk tokens r passed to the pointer and to stage 2 stay position-free.

### 3.5 Unchanged

- **Model:** d_model 64, 4 heads, FallQueries (3 queries, 2 layers, pointer with the 12 physics step
  features), the dense head and its losses, stage 2, the decoder.
- **Data:** windows of 32 s every 6 s, chunks of 4 windows.
- **Training:** sampler, 15 epochs, SWA from epoch 2, AdamW 1e-3, 7 GB cap.
- **Trunk size:** about 0.30 M parameters, against about 0.53 M in JF2.

## 4. Data and augmentation

### 4.1 Input channels (`StreamStore.read`)

- **Layout.** `[I ×3, Q ×3, I_c ×3, Q_c ×3, D1 ×3, D2 ×3, D3 ×3, valid]` = 22 channels.
  - The raw I/Q stay first, so stage 2's event physics (`x[:, :6]`) is unchanged.
  - New data option `amplitude: false` drops |z| and |c|.
  - New option `contrast: [[0.3, 0.5], [0.3, 1.0], [0.5, 2.0]]` gives (gap, width) in seconds.
- **The contrast maps.** `D_(g,w)(t, bin) = | mean z[t+g, t+g+w) − mean z[t−g−w, t−g) |` per radar
  and bin.
  - They are computed with a cumulative sum on the normalised complex z. The read already has 4 s of
    context on each side for the clutter mean; the largest scale needs 2.5 s.
  - At the stream ends, the mean is taken over the steps that exist.
- **Invariances.** A constant background cancels, so `bg_swap` leaves them unchanged, as it leaves
  the clutter channels.
- **Raw dropout.** `raw_channels: 6`: the raw I/Q are dropped with p .3, as in JF2. The contrast maps
  stay.
- **Background swap.** `_swap_background` skips its amplitude recompute when `amplitude: false`.

### 4.2 Channel count link

`cli._in_channels` gets the new flags: 3 × (2 + 2 [+ 3]) + 1 = 13 (JF4-S, JF4-ST) or 22 (JF4).

### 4.3 Step swap: decoy joins (`augment.step_swap_p: 0.5`, JF4 only)

- **What it does.** On a training chunk with p .5, the raw z (normalised, with its context) gets a
  background step at a random time t*:
  `z[:, t ≥ t*] += ramp · (P_other − P_own)`.
  - `P_other` is a random training stream's 2 s mean profile, as in `bg_swap`.
  - `P_own` is the mean of the 2 s before t*.
  - The ramp goes from 0 to 1 over 10 steps, like the generator's 0.2 s crossfade.
- **Where.** It is applied before the clutter and contrast channels are derived, so they see the
  step the way they see a real join.
- **When t* is allowed.** t* is uniform over the chunk, excluding [a − 3 s, b + 1 s] around every
  labelled fall near the chunk. A decoy never sits on a fall's own step.
- **Effect.** Background steps appear in ADL and at random distances from falls. "A step, then a
  fall 1–3 s later" (the join prior) stops paying off, and "a step without motion" is a learnable
  negative.
- **Labels** are unchanged.

## 5. Variants and configs

Four variants, each adding one idea to the one before. Fusion is now its own step (added
2026-10-29 at the user's request), so the CNN stem and late fusion are judged separately.

| Variant (config) | Adds to the one before | Trunk | Inputs | Losses |
|---|---|---|---|---|
| **S-early** (`JF4_Se.yaml`) | the convolutional stem (idea 1), early fusion kept | per-radar L0 (complex conv + modulus, shared weights) → the 3 radars' moduli concatenated at the same bin (48 channels) → shared L1–L3 → range attention pool → JF2's frame position embedding, 2 causal + 2 bidirectional blocks, JEPA | 13 channels | JF2's |
| **S-late** (`JF4_S.yaml`) | late fusion | per-radar L0–L3 → range pool per radar → attention across the 3 radar tokens (3.2) → as S-early | 13 channels | JF2's |
| **ST** (`JF4_ST.yaml`) | the TCN (idea 3) and RoPE | the better S → TCN → 3 RoPE bidirectional blocks; no causal context, no JEPA predictor | 13 channels | JF2's without `MSE_Latent` |
| **STC** (`JF4.yaml`) | the contrast maps (idea 2, long lag) and the step swap | as ST | 22 channels + step swap | as ST |

**S-early in detail.**
- L0 must stay per radar: a complex convolution across radars would not be phase-invariant.
- After the modulus, the radars meet the way they do in JF2: as channels at the same bin index.
  L1 takes 3 × 16 = 48 channels, about 23 k parameters instead of 7.7 k. L2–L3 are unchanged.
- The range pool has a range embedding and no radar embedding. Radar dropout is just zeros, as in
  JF2.
- This variant answers "does a convolutional, phase-invariant embedding beat the linear patch?"
  without changing how the radars are combined.

**ST and STC** use S-late unless S-early beats it on the rule in section 6.

New model arguments (`FallSetJEPA` → `FallJEPA`):
- `trunk: frames | stem` (`frames` is JF2's path, unchanged)
- `stem_channels: [16, 32, 48, 64]`
- `range_heads: 4`
- `radar_fusion: early | attention`
- `local: none | tcn`, `tcn_dilations: [1, 2, 4, 8]`
- `global_layers`, `rope: true`
- `contrast_channels` (linked from data)

JF2's `frame_patch_size`, `frame_encoder_layers` and `frame_pooling` are ignored when `trunk: stem`.

## 6. Runs and what decides

### 6.1 Fast screen: fold D only (the user's plan, 2026-10-29)

Every variant trains on **fold D only, seeds 0 and 1**:
`FOLDS=D SEED=<s> CONFIG=configs/uwb/JF4_<v>.yaml NAME=JF4<v> GROUP=UWB/JF4<v> tools/run_jf_cv.sh`.

- **Why fold D.** It is JF2's weakest fold (.811 / .810; RS .852, Amir .922). Its held-out streams
  hold the official val subjects: falls F8, F9 and ADL A0, A11. It has 134 scored falls, so one fall
  is worth about .007.
- **Cost.** About 1.5 h per run:
  - in sequence: 4 variants × 2 seeds ≈ **12 h**;
  - with both seeds in parallel: about **6 h**. JF3 peaked at 4.5 GB, so two runs take about 9 GB
    beside the other user's 12 GB. This only works if the 7 GB cap is per run.
- **Order.** Each stage starts when the one before is read:
  1. CPU tests + GPU smoke (section 7).
  2. S-early and S-late, both seeds.
  3. ST on the better S, both seeds.
  4. STC, both seeds.
- **The baseline already exists.** JF2 fold D, seeds 0 and 1:
  - TP 111 / 113, FP 23 / 23, FN 23 / 21.
  - Its `.queries` / `.dense` fold D scores are fetched from ABC27 for the ladder.

**What fold D can and cannot show.**
- **Noise.** One fold is noisier than the pooled CV. JF3's two seeds scored .808 and .843 on fold D,
  with the same code. A 2-seed mean resolves about ±.03 (≈ 4 falls). Read error counts, not only
  the score.
- **Which variants it tests well.** Fold D's errors are mostly timing: 37 of 44 misses are
  mislocalised, and most false alarms are the same detections. So fold D tests ST and STC well.
- **Which it tests poorly.** The weak, never-fired falls that S-early / S-late target sit mostly in
  folds A and C (P3_3, P2_2). On fold D, the S variants are judged on "no loss", plus their
  `.queries` / `.dense` scores and the recall of fold D's weakest falls (Chute04).
- **Selection bias.** Choosing on fold D makes it a selection set. The official val numbers in the
  JF2 report come from the same subjects. Confirm the winner on folds A–C (2 seeds, about 9 h)
  before calling JF4 better than JF2.

| Stage | Question | Read first (fold D, per seed) |
|---|---|---|
| S-early | does a convolutional, phase-invariant embedding beat the linear patch? | score; `.queries` / `.dense`; TP / FP / FN; Chute04 recall; stage-1 proposal recall at P > .05 |
| S-late | does seeing which radar moved help? | as S-early; FP split into ADL vs fall-recording; fusion attention concentrated on one radar during falls |
| ST | does the 5 s context place falls better? | timing errors: misses with tIoU < .3 + off-label false alarms (JF2 s0 17 + 3 misses, s1 8 + 9); centre offset of stage-1 proposals on JF2's fold D timing errors; median tIoU of hits |
| STC | does stage 1 find the state step once it can see it? | timing errors; false alarms within 3 s after a join; recall by distance to the nearest join (analysis only) |

**Rule.**
- **Keep** a variant, and build the next stage on it, if its 2-seed fold D mean is ≥ the previous
  best − .01 and its "read first" errors move the right way.
- **S-late is chosen over S-early** unless S-early is better by ≥ .02 (2-seed mean). Late fusion has
  the physical argument (D1–D3); early fusion has the history.
- **A fold D win** is a 2-seed mean ≥ .835 (JF2 + .025), with fewer timing errors on both seeds.
- **If both S variants lose by more than .03,** stop and diagnose before ST. Run JF2 +
  phase-rotation augmentation (each radar's I/Q rotated by φ ~ U[0, 2π)) to separate the invariance
  from the convolutional stem.
- **Report** per seed: the ladder (`.dense` / `.queries` / main) and the fold D rows of the error
  tables, re-run with `runs/_analysis/jf2_err/jf2_err_srv.py` and `build_report.py`.

### 6.2 Confirmation: folds A–C

The best variant, and the best S if ST or STC win, runs on folds A–C with seeds 0 and 1. With the
fold D runs, that gives the full pooled CV against JF2's .861. About 9 h per variant (two seeds in
parallel: 4.5 h).

## 7. Checks before the runs

**CPU (`tests/test_jf4.py`):**
- **Alignment.** An impulse at step 4 i + 16 peaks at stem token i. With `trunk: stem`, `Fn` is 400
  for 32 s windows.
- **Phase invariance.** Rotating radar r's (z, c) by any φ leaves the stem output unchanged
  (to 1e-5). Rotating radar r alone changes nothing in the other radars' tokens.
- **Radar mask.** A zeroed radar gets zero fusion weight. The model output is finite with 1 or 2
  radars dropped.
- **Contrast.** Adding a constant complex profile to z leaves D unchanged. A synthetic profile step
  peaks at the step for every scale. Stream-end handling.
- **Step swap.** t* never falls inside [a − 3 s, b + 1 s]. The clutter and contrast channels show
  the step. Labels are unchanged.
- **TCN.** The receptive field is 61 tokens (gradient test). Invalid frames stay zero.
- **End to end.** S-early / S-late / ST / STC on the synthetic release (as the JF3 tests). Stage 2 still reads
  raw I/Q from channels 0–5.

**GPU smoke:** peak memory under 7 GB with `sampler.split: 2`, and step time against JF2 (about 8
batches/s). If memory is over, use `torch.utils.checkpoint` on L0–L1 first, then L0 stride 2 × 4.

## 8. Risks, and what was considered and left out

**Risks.**
- **Late fusion reverses the AP-era choice** to keep early radar fusion. The reason to revisit it
  is D1/D2: a fall lives on one radar, and bins of different radars are different places (D3).
  S-early tests the early-fusion stem directly, so the choice is made on data.
- **ST changes three things** (TCN, RoPE, no JEPA). They are coupled: a centred TCN breaks the
  JEPA setup. If ST is ambiguous, the clean split is ST + JEPA on a causal side branch
  (z → causal blocks → predictor, outside the main path).
- **BatchNorm with mixed sessions.** The batch statistics come from 24 radar-windows per
  micro-batch. SWA recomputes them, but watch the raw-weights vs SWA gap per fold. x1 showed SWA can
  hurt.
- **The contrast maps can still help the model find joins.** Step 3's join diagnostics decide
  whether the step swap is enough.
- **No evidence yet that any of this beats JF2.** JF3 shows a sound diagnosis can still fail.

**Considered, not proposed:**
- **Radar minus the lowest-energy radar:** measured to hurt (2.4).
- **A fixed lag-1 embedding difference:** subsumed by the TCN (2.2).
- **A temporal pyramid (ActionFormer FPN):** falls are 12–37 tokens, and full attention over 400
  tokens is cheap.
- **A transformer over range × time tokens after the stem:** 184 tokens per frame, about 5× the cost.
- **Dropping the raw channels entirely:** a later ablation. The raw profile may carry the "lying on
  the mat" state.
- **A larger d_model:** the model already memorises training at d = 64.

## 9. Implementation map

| File | Change |
|---|---|
| `src/falljepa/model/stem.py` (new) | `ComplexConv2d`, `RadarStem`, `RangePool`, `RadarFusion`, `TCN`, rotary helper |
| `src/falljepa/model/layers.py` | `Attention(rope=...)`, `Block(rope=...)` |
| `src/falljepa/model/fall_jepa.py` | `trunk`, `local`, `global_layers`, `rope`; `encode()` branches on `trunk` |
| `src/falljepa/model/set_heads.py` | `FallQueries(..., key_pos)`: position embedding added to the cross-attention memory |
| `src/falljepa/model/fall_set.py` | pass-through arguments; `raw_channels: 6` with the new layout |
| `src/falljepa/data/uwb.py` | `amplitude`, `contrast` in `StreamStore.read`; `step` decoy in `read`; `_swap_background` without amplitude; `step_swap_p` in the training dataset |
| `src/falljepa/cli.py` | `_in_channels` from the new flags; link `contrast` to the model |
| `configs/uwb/JF4_Se.yaml`, `JF4_S.yaml`, `JF4_ST.yaml`, `JF4.yaml` | as section 5 |
| `tests/test_jf4.py` | section 7 |

## 10. Implementation and the fold D queue (2026-10-29)

- **Code** (uncommitted, branch `stream-cv`):
  - `model/stem.py`; `layers.rotary` and `Attention`/`Block(rope)`.
  - `FallJEPA(trunk, radar_fusion, stem_channels, range_heads, contrast_channels, local, tcn_dilations, rope)`, passed through `FallSetJEPA`.
  - `FallQueries(key_pos)`.
  - Data: `UWBDataModule(amplitude, contrast)`, `contrast_maps`, `StreamStore.read(step=...)`, `WindowDataset._step` (`augment.step_swap_p`, `step_swap_s`), `in_channels`.
  - Configs: `configs/uwb/JF4_Se.yaml`, `JF4_S.yaml`, `JF4_ST.yaml`, `JF4.yaml`. Each is JF2's config with the same batch: 4 chunks, split 2, accumulate 4.
- **Tests:** `tests/test_jf4.py` has 14 tests: the section 7 checks and the four configs end to end. The full suite passes (105).
- **Contrast maps.** They are window sums of an edge-padded cumulative sum, computed after removing each bin's mean. The window sums are float32, with an error of about 3e-4 relative. Cost: 27 ms per 50 s chunk.
  - The first version (float64 fancy indexing, 60 ms) left the GPU at 23 % with 10 loader workers at 100 %.
- **Batch layout (2026-10-29, user request: use about 12 GB, train as fast as possible).** The optimizer step
  is still JF2's 8 chunks (JF2 accumulated 4 GPU batches of 2), now in one GPU batch:
  - `batch_size: 8`, `sampler.split: 1`, `accumulate_grad_batches: 1`;
  - `fall_slots: 4`, `hard_slots: 2`, so each step has the same composition;
  - `eval_batch_size: 32`;
  - 677 optimizer steps per epoch, as JF2.
- **Speed and memory** (fold D, chunks per second from optimizer step 30 to 90; `runs/bench_jf4.tsv` and
  `runs/bench/` on ABC27):

  | Variant | 8 × 1 | 4 × 2 | 2 × 4 (JF2's layout) | Peak memory at 8 × 1 |
  |---|---|---|---|---|
  | S-early | 30.0 | | | 9.6 GB |
  | S-late | 28.2 | 24.0 | 20.0 | 11.9 GB |
  | ST | 26.7 | | | 11.9 GB |
  | JF4 (contrast in the loader) | 14.6 | 15.0 | 14.1 | 12.8 GB |

  - JF4 with 11 loader workers instead of 10: 13.3 chunks/s.
  - JF4 was loader-bound: the contrast maps cost the CPU workers half their throughput.
- **Contrast in the model.** `data.contrast_in_model: true` (in `JF4.yaml`) moves the contrast maps to
  the GPU.
  - The loader sends the raw I/Q with the contrast's reach (2.5 s) on each side as `ctx`.
  - `FallJEPA.with_contrast` → `stem.contrast_from_context` computes the same channels and inserts them
    where the loader would have put them.
  - The step swap and the radar dropout reach `ctx` too.
  - A test checks that both paths give the same input (2e-3).
  - Loader cost per chunk (laptop): 34 ms, against 63 ms with the contrast in the loader and 37 ms for ST.
  - Not yet smoke-tested on the GPU: the queue reaches JF4 last.
- **Queue.** `/home/abc2027/mehdi_jf4/runs/jf4_queue.sh` (log `runs/jf4_queue.log`) was started 2026-10-29 ~10:00 UTC. It runs one at a time, in this order:
  - S-early s0, s1;
  - S-late s0, s1;
  - ST s0, s1;
  - JF4 s0, s1.
- **Tags** are `20261029-<jf4se|jf4s|jf4st|jf4stc>-s<seed>`. Scores land in `runs/_analysis/cv_<JF4Se|JF4S|JF4ST|JF4>[-queries|-dense]_<tag>.json`.
- **Fusion for ST and JF4.** Both use late fusion. If S-early beats S-late by ≥ .02, re-run them with `radar_fusion: early`.
- **Restart.** The first queue was stopped and its runs deleted. It was restarted 2026-10-29 at 10:46
  UTC with the 8 × 1 layout.
- **Expected time:** about 1 h per run, about 8 h in total.

## 11. Fold D results and JF5 (2026-10-30)

Main score (mean F1 at tIoU .1/.3/.5, SWA weights), fold D, seeds 0 and 1. Pooling the two seeds gives
the same numbers to within .001 (`runs/_analysis/pooled_seeds_*.json`).

| Variant | s0 | s1 | Mean | Queries | Dense | FP / FN (2 seeds pooled) |
|---|---|---|---|---|---|---|
| JF2 | .811 | .810 | .810 | .777 | .764 | 46 / 44 |
| **Se** | .851 | .877 | **.864** | **.849** | .804 | 19 / 35 |
| S | .844 | .850 | .847 | .831 | .796 | 23 / 42 |
| ST | .839 | .850 | .844 | .845 | **.850** | 20 / 45 |
| JF4 (STC) | .866 | .841 | .854 | .839 | .834 | 15 / 41 |

What the runs say:

- **The stem is the gain:** +.034 to +.054 over JF2 in every variant, almost all of it from false alarms (46 → 15–23).
  Nothing added after the stem beats the noise: the spread among Se, S, ST and JF4 (.020) is the size of
  the seed SD (.004–.018 here, .02–.03 historically).
- **Late fusion:** −.017 vs early, just under the queue's .02 re-run rule.
- **TCN + RoPE (no JEPA):** main score unchanged (−.003), dense +.054. The fall queries and stage 2 already
  give the timing the TCN adds to the dense head.
- **Contrast maps:** +.009 main, −.010 queries. In the dense head they wipe out P1_8 Chute04: median
  in-label P is .62/.31 in Se and .00/.00 in JF4.
- **Misses are one subject, not the architecture:** 16 of the 134 scored falls are missed by ≥ 6 of the
  8 runs. They are all Dataset 1, sessions P1_8 and P1_9 (12 Chute04, 4 Chute01), and 11 of them get no
  prediction at all. Each run misses 14–16 of them plus only 2–8 others. No trunk change touched them.
- **Ensembles (dense head, `tools/ensemble_dense.py`):** pairing Se with S gains no more than pairing two
  seeds of one model (+.027 vs +.032 over the members' mean). So the fusions are not complementary.
- **Training:** the raw-weight event F1 at epoch 5 ≈ epoch 15 in every run, and SWA adds .02–.04. More epochs would not help.

**JF5 = Se, unchanged** (`configs/uwb/JF5.yaml`), on folds A–D × seeds 0, 1 (`tools/run_jf5_queue.sh`,
tags `20261030-jf5-s{0,1}`, about 1 h per run, 8 h in all). Reasons:

1. It has the best main and queries scores, and it is the simplest and fastest variant.
2. Each addition was measured on fold D and gave no main gain. An untested combination (Se + TCN) would
   make the 4-fold result uninterpretable if it underperformed.
3. It keeps JEPA and the dense head's partial signal on Chute04, which JF4's contrast maps lose.

The comparison is JF2's 4-fold CV (.861 pooled, both seeds), same folds and protocol. The P1 Chute04 misses
need their own analysis (signal level vs what the model sees), not another trunk change.

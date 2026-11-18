# Learning Amir's post-processing inside the model: the recording head

Branch `stream-cv`, 2026-10-24. Sections 0–9 are the design. Section 10 says what was built: one
configuration, `configs/uwb/RS.yaml`, trained once per fold of the subject-held-out CV
(`docs/stream_cv.md`) with `tools/run_rs_cv.sh`. Where section 10 differs from the design, section 10
holds. Nothing has been trained yet.

## 0. Summary

- **What the post-processing (PP) really does.** It converts per-timestep P(fall) into a different
  question: *which source recording holds a fall?* Then it lets the organizers' labeller place the
  fall inside that recording.
  - Its largest step (+.10) runs the labelling rule on the right recording.
  - The gate (+.03 to +.06) is a test on the whole recording.
  - Merge and NMS exist only because the detector outputs runs of timesteps, not recordings.
  - Null test (2026-10-18): with no model at all, one blip per detected recording already scores
    .52. Real model + PP scores .944. The PP's own ceiling with oracle locations is .953.
- **The labels are a function of the recording.** The ground truth is
  `propose_fall(source recording)`, and a stream is recordings joined by 0.2 s crossfades. So the
  natural output unit of the model is **the recording, not the timestep**.
- **Proposal.** Keep the FallJEPA trunk, with 32 s windows instead of 16 s, and add a **recording
  head**. It is a two-stage detector whose region of interest (RoI) is a source recording:
  1. **Unit proposals.** A unit is a recording between two detected joins.
     - v1: Amir's join detector, unchanged.
     - v2: a learned join heatmap, supervised for free by `streams_composition.csv`.
  2. **Recording-physics layer.** It has no parameters. It computes the labeller's own quantities
     relative to each unit:
     - state curves against the unit's first-1 s and last-1.5 s profiles;
     - motion relative to the unit's median;
     - the rule's transition score;
     - the profile change per radar.
  3. **Recording decoder.** Two attention blocks masked to the unit, plus a `[REC]` query. Per
     unit it predicts:
     - **P(the unit holds a fall)**;
     - **a start distribution and an end distribution** over its timesteps, a span head like
       extractive question answering.
- **Decoding.** Emit one event per unit with P > .5. Choose its span by minimum Bayes risk (MBR)
  for the challenge metric, the mean of F1 at tIoU .1/.3/.5. There is no frame threshold, no
  merge, no NMS, no gate and no external rule.
- **More data for localization.** The rule is defined on *every* recording. So its interval on
  ADL, Autre and Empty units is a free auxiliary target. That is over 10× as many localization
  examples as the 184–195 clean training falls of a fold (rule distillation). They are fewer
  distinct recordings, since ADL files are reused.
- **What to expect.** On official val the PP is within .01 of its ceiling, so a large jump there
  is not the goal. The gains this design targets are:
  - no constant tuned on val;
  - training on the unit the metric counts;
  - the falls the rule cannot place (6.5 % of training falls stay below tIoU .5);
  - the Chute04 recall gap.

  The CV decides whether it beats the PP (§7).

---

## 1. Amir's post-processing, step by step

Sources: `src/falljepa/postprocess.py`, `tools/amir_postprocess.py`,
`teammate_pipeline_20261013/ABC2027_best_pipeline.md`, `report/falljepa_amir_postprocess.html`,
and the official `src/abc2027/annotate.py`. Gains are A1 (3 seeds) on official val.

| # | Step | What it computes | What it actually encodes | Gain | Constants |
|---|---|---|---|---|---|
| 1 | Detect | runs of P ≥ .3, ≥ 0.2 s, gaps ≤ 0.2 s merged; score = max P | a candidate fall exists near here | −.034 vs Viterbi alone (more recall, more FPs) | θ, min run, gap |
| 2 | Joins | J(t) = min over radars of log(step / local median step); peaks J > 1.5, 1 per s | where source recordings start and end; a splice moves all 3 radars, a fall mostly one | (enables 3) | step 10, gap 6, baseline 501, J > 1.5, radius 50 |
| 3 | Span | nearest joins ≥ 1 s from the detection centre; whole span if 2–20 s, else −4/+5 s inside it | *the analysis unit is the recording* | | guard 50, 100–1000, 200/250 |
| 4 | Refine | official `propose_fall` internals on the span (raw-frame motion) | the label is a function of the recording's first 1 s, last 1.5 s, median motion, p30 of motion, … | **+.101** (+.033 on a −4/+5 window) | the labeller's ~10 |
| 5 | Raw-frame motion | motion between raw frames, not 50 Hz grid frames | 24 % of grid frames are sample-and-hold repeats | +.019 | — |
| 6 | Merge + NMS 5.9 s | one event per 5.9 s, highest score | one fall per recording | needed by 4 | 5.9 s |
| 7 | Gate | drop unless max over radars of ‖median(last 1.5 s) − median(first 1 s)‖² ≥ 2.12e-5 | a fall leaves a new static profile on one radar; activity doesn't | **+.031** (+.058 ensemble) | 2.12e-5 |
| 8 | One per recording | best event between two joins | structure of the stream generator | +.000 | — |

Three things follow from this.

1. **Steps 3–8 are recording-level reasoning that a 16 s window model cannot do.** The rule needs
   the *first* and *last* seconds of the recording. A window that does not hold the whole
   recording, or that crosses a join, sees the wrong profiles. That is why the −4/+5 s window only
   gave +.033 and the right recording gave +.101.
2. **The network's real job under the PP is recording classification.** Evidence:
   - noise or time-shifted outputs fall to .12–.34;
   - a constant blip per recording gives .52;
   - oracle locations give .953;
   - A4 regression-head boundaries lost to the rule: .757 against .847 for the same detections.

   A head that regresses boundaries from local window features cannot see the recording's
   reference profiles.
3. **Everything with a tuned constant was tuned on official val or the official training
   subjects.** That includes θ, the gate threshold, NMS 5.9 s and the join threshold. The new CV
   holds out subjects, so a learned version is the fair comparison there.

## 2. Data facts that shape the heads

Measured on the official train and val composition (24 Autre/Empty clips, 1,844 ADL pieces,
478 fall clips):

| Fact | Value | Consequence |
|---|---|---|
| Fall clip length | 7.3–18.6 s (median 12.4) | A whole fall recording plus 1 s margins fits in **≤ 22 s**. Use 32 s windows. |
| ADL piece length | 0.5–20 s (median 12); 88 pieces < 2 s | Units can be tiny. The physics layer needs the official `_profiles` fallback (len/5 frames). |
| Autre/Empty clip length | 77–120 s, used whole | Long units must be chunked (§3.2). These are also the hardest negatives. |
| Fall start inside its clip | 0–7.9 s (median 3.0); 5 % within 0.1 s of the clip start | A span may start at the unit edge, so the span domain includes a 0.5 s margin. |
| Fall clips not starting at the source recording's start | 14 % | The label's "standing" profile then came from outside the stream. Only a learned head can absorb this offset. |
| Fall duration | 0.48–5.0 s (median 1.74) | Span length prior 0.3–6 s. |
| Falls whose neighbour clip is also a fall | 164 links over 478 falls (about 34 %) | A missed join can put 2 falls in one unit (§5.3). |
| Joins | ~130 clips per 30-minute stream; heuristic detector P/R .94/.94 (train), .96/.90 (val) | Unit proposals are good but not perfect. Train on the same proposals as at test. |

## 3. Architecture

```
z-CIR + valid, physics (12 ch)
  └─► FallJEPA trunk (unchanged: frame encoder → causal context → bidirectional), 32 s window
        tokens r_i every 0.08 s (frame width 32, stride 4) ──┬─► dense head: P(fall) per 50 Hz step (kept, auxiliary)
                                                             ├─► join head: join heatmap per token (v2)
                                                             │
units u = [join_k + 6, join_k+1 − 5) inside the window ──────┤
  (v1: Amir's join peaks, cached;  v2: join-head peaks)      │
                                                             ▼
                       recording-physics layer (no parameters, per unit)
                        state_r(t), motion_r(t)/median_u, rule score_r(t), d(t, unit start), d(t, unit end)
                        unit scalars: log profile change per radar (sorted), top-radar share, …
                                                             │
                       recording decoder: 2 blocks, attention masked to u ± 1 s, [REC] query
                                                             │
                      ┌──────────────────────────────────────┼──────────────────────────────┐
                 P_fall(u)                       start logits ℓs(t), t ∈ u ± 0.5 s      end logits ℓe(t)
             (replaces θ, merge,                 (softmax over the unit: one span per unit by construction;
              NMS, gate, 1/recording)             replaces the external rule)
```

### 3.1 Trunk

- **Same as the AP/rdrop+dswa model.** d 64, frame encoder 2 layers, context 2 causal + 2
  bidirectional, physics features, radar dropout.
- **One change: the window is 32 s** (1,600 timesteps, about 400 frame tokens).
  - The longest unit (22 s) plus 1 s of context on each side plus the inference hop (8 s) must fit
    in one window: 22 + 2 + 8 = 32.
  - Attention over 400 tokens at d 64 is cheap.
  - `frame_pos_embed` grows to the new `window_frames`.

### 3.2 Units: the regions of interest

- **Definition.** A unit is the stretch between two consecutive join peaks, without the crossfade:
  `[peak_k + 6, peak_{k+1} − 5)`. These are Amir's offsets; the 1 s guard is not needed because a
  unit is not centred on a detection.
- **Long units.** A unit longer than 22 s is cut into 20 s chunks at a 10 s hop. Each chunk is a
  unit with a `chunk` flag. This covers Autre/Empty recordings, and two recordings merged by a
  missed join. It replaces Amir's "−4/+5 s if the span is not 2–20 s".
- **Where unit proposals come from.**
  - **v1:** Amir's `join_peaks(join_score(cir))`, already cached per stream by `StreamJoins`.
    Training uses the *same* detector on the training streams, so the units the model learns on
    have the same errors as at test.
  - **v2:** the learned join head (§3.5). During training, units come from ground-truth joins
    with jitter ±3 steps, 5 % dropped joins and 2 % spurious joins at fall onsets. These rates
    mimic the detector's errors. At inference, units come from the join head.
- **Which units a window supervises.** Only units fully inside the window, at least 1 s from its
  edges. That is about 2–3 per 32 s window, so about 80 per batch of 32.

### 3.3 Recording-physics layer (no parameters)

This layer is the labeller written as tensors. It computes on the **unnormalized** CIR, because
the official distances weight the range bins that way. It can be cached per stream like
`data/physics.py`'s features (v1), or computed in `forward` from the unit mask (v2).

For each unit u, radar r and timestep t:

| Channel | Formula (official function) | Why |
|---|---|---|
| `state_r(t)` | d(x_t, P_start) / (d(x_t, P_start) + d(x_t, P_end)), with P_start = median of u's first 1 s and P_end = median of u's last 1.5 s (`_profiles`, `_state`), smoothed 0.4 s | the labeller's state curve |
| `motion_r(t)` | log1p(raw-frame motion / median over u), smoothed 0.4 s | the labeller normalizes by the recording median |
| `score_r(t)` | rise(t) · log1p(motion_sum(t) / median), with rise = state(t + 1.5 s) − state(t) (`_transition_bounds`) | the rule's objective curve; its argmax is where the rule puts the fall |
| `pos(t)` | distance to u's start and to u's end, in s, clipped at 20 s, sinusoidal | the profiles live at the unit edges |

Unit scalars, for the `[REC]` query:

- `log ‖P_end − P_start‖²` per radar, **sorted descending**. The top one is the gate statistic.
  Sorting makes the head invariant to which radar it is, as the anchor rule is.
- The share of the top radar in the summed change: .99 for falls, .50 for joins.
- log of max motion / unit median motion.
- Mean motion over the last 1.5 s / max motion: the stillness that Amir's declared gate used.
- Unit duration is **left out** by default. A fall clip is never longer than 18.6 s, but that is
  the generator, not the physics. Test it only as an ablation.

**Soft anchor radar.** The decoder mixes the per-radar channels with weights softmax(β · log
change_r), with β learned and initialized at 5. This is the official "anchor = argmax change" made
differentiable, so a fall seen on two radars can use both.

### 3.4 Recording decoder: the output modules

Per unit u, the input tokens are the trunk tokens r_i on u ± 1 s, so the decoder sees both joins.
Each token gets `Linear(physics channels)` added.

1. **Two transformer blocks** (d 64, 4 heads), with attention masked to u ± 1 s, plus a learned
   `[REC]` token that attends to all of them.
2. **Fall head:**
   `P_fall(u) = σ(MLP([REC] ⊕ Linear(unit scalars)))`.
   - This head replaces θ, merge, NMS, the gate and one-per-recording all at once.
   - The gate becomes a learned function of its own statistic plus the network's evidence, not a
     hard 2.12e-5 cut.
3. **Span head.** For each token, `Linear(d, 2w)` gives start and end logits for its w = 32
   timesteps. They are averaged over overlapping frames with the trunk's Hann weights
   (`frames_to_steps`), which gives 50 Hz logits ℓs(t), ℓe(t).
   - The **softmax runs over the unit**, on u ± 0.5 s: q_s = softmax(ℓs), q_e = softmax(ℓe).
   - So there is exactly one span per unit, as in SQuAD's start/end pointers.
   - This replaces the external rule. The rule's own objective is an input (`score_r`), so the
     head starts near the rule and learns its corrections, such as the 14 % of clips cut after
     the recording start and the falls where the rule picks another transition.
4. **Dense head** (the existing `State_Frame`/`State_Step` P(fall)). It is kept as an auxiliary
   loss, to keep the trunk well conditioned, and as a diagnostic. It is not used for decoding.

### 3.5 Join head (v2)

- **Output.** One logit per trunk token (0.08 s).
- **Target.** A Gaussian heatmap, σ = 1 token, around every composition join of the training
  streams. The joins are exact and free: the clip boundaries in `streams_composition.csv`.
- **Loss.** CenterNet focal loss.
- **Input.** Amir's J(t) is added as a 13th physics channel, so the head starts from the heuristic.
- **Decoding.** Local maxima with p > .5, at least 1 s apart. This is CenterNet's max-pool peak
  picking, a parameter-free selection like an RPN's.
- **Why bother.** The heuristic misses 6–10 % of joins. A missed join merges two recordings and
  gives both of them the wrong profiles. The join head can learn the cases the min-over-radars
  rule gets wrong, such as quiet radars and crossfades between near-identical rooms.

## 4. Training

### 4.1 Targets per unit

Computed once per stream and cached, like the physics features.

- `y_fall(u) = 1` if a labelled fall **starts** in u, holds ≥ 40 % of its duration in u, and ends
  ≤ 2 s after u (§10.2).
  - Weight 0 if that fall is flagged (it is an ignore zone under `--ignore-flagged`).
  - Weight 0 if a fall starts in u but u cannot own it.
  - A unit that holds only a fall's tail is a no-fall unit, weight .5.
- `span(u)`: the label's start and end timesteps, for fall units.
- `rule(u)`: `postprocess.official_bounds(cir[u])` with raw motion, for **every** unit long
  enough (≥ 2 s), fall or not. This is the rule-distillation target.

### 4.2 Losses

| Term | Formula | Weight |
|---|---|---|
| `Rec` | BCE(P_fall(u), y_fall) over supervised units | 1.0 |
| `Span` | soft CE of q_s and q_e against Gaussians N(s\*, σ) and N(e\*, σ), σ = 0.1 s, fall units | 1.0 |
| `Rule` | the same soft CE against `rule(u)`, **non-fall** units only | 0.2 |
| `State_Frame`, `State_Step` | as in AP | 0.5, 0.25 |
| `Join` (v2) | focal loss on the join heatmap | 0.5 |

Two more terms to try as ablations, not in v1:

- an expected-tIoU loss on (E_q[s], E_q[e]);
- σ = 0.2 s, since the README says labels are good to about ±0.5 s.

Why the `Rule` term helps:

- A fold has about 190 clean training falls, but about 2,600 ADL, Autre and Empty units of ≥ 2 s
  in its 2 stitchings (about 1,380 ADL pieces per stitching, as in the official train).
- The rule's interval on those units is exactly the function the span head must learn: "where
  would the labeller put a fall in this recording?"
- On fall units the target is the label itself, so the rule never overrides the truth.
- The weight is low because the decoder never uses the span of a non-fall unit.

### 4.3 Sampling

- **The sampler (`BalancedBatchSampler`) is reused.** It already picks a source recording per slot
  (16 fall, 16 no-fall, 4 of them Autre).
- **New:** the 32 s window is centred on that recording's unit, with a random shift that keeps the
  unit ≥ 1 s from both edges. For an Autre/Empty recording, a random 20 s chunk is used.
- **Units per batch.** The other units fully inside the window are supervised too, giving about
  80 units per batch, with fall:non-fall about 1:3.
- **Recipe.** Unchanged: radar dropout .15, AdamW 1e-3 constant, dense SWA.
- **Epoch budget.** Fix it once for all folds, as in `docs/stream_cv.md` §9: 15 epochs at 2
  stitchings, dense SWA from epoch 2.
- **Remix.** Leave it off in v1. Remixed windows have no cached heuristic joins; with v2's
  ground-truth units they work unchanged.

## 5. Decoding

### 5.1 The whole chain

1. **Trunk.** Run it on 32 s windows at an 8 s hop, about 225 windows per 30-minute stream. The
   current setup runs about 900 windows of 16 s at a 2 s hop.
   - Dense outputs and the join heatmap are merged with triangular weights, as now.
2. **Units.** Take the join peaks (v1 cached, v2 the join head) and form units, chunking any unit
   longer than 22 s.
3. **Recording decoder.** Run it on each unit, in the window where the unit is most central. With
   a hop of 8 s, a unit of ≤ 22 s is always ≥ 1 s from that window's edges.
4. **Events.** Every unit with P_fall > τ gives exactly one event, with score P_fall.
   - Use τ = .5 by default, untuned. Tune it nested across folds only at the end
     (`docs/stream_cv.md` §12).
   - The scorer has no score threshold, so τ is the only decision knob.
5. **Chunks.** Only inside a chunked long unit, two events that overlap keep the higher P_fall.

### 5.2 MBR span selection

The metric rewards a span by whether its tIoU clears .1, .3 and .5. So instead of argmax start
and argmax end, choose the span that maximizes the expected metric:

```
(ŝ, ê) = argmax over candidate (s, e)  Σ_{s', e'} q_s(s') q_e(e') · Σ_{θ ∈ {.1,.3,.5}} 1[tIoU((s,e),(s',e')) ≥ θ]
```

- Candidates and hypotheses are the top 16 starts × top 16 ends with 0.3 s ≤ e − s ≤ 6 s. That is
  about 256 × 256 tIoU evaluations per unit, which takes microseconds.
- MBR prefers a span that is safe under uncertainty over the single most likely pair.
- It is a pure decoding change. Its saved q_s and q_e can be compared against argmax without
  retraining.

### 5.3 Two falls in one unit

- This happens only when a join between two fall clips is missed: about 34 % of falls have a fall
  neighbour, times the 6–10 % miss rate, so about 2–3 % of falls.
- v1 accepts the loss.
- If the diagnostics show it matters, add a second span slot per unit with a set loss (Hungarian
  matching, as in DETR). Slot 2 is almost always empty. The better fix is v2's join recall.

### 5.4 What is learned, what is structural, what stays fixed

| PP piece | Constants | In the recording-head model |
|---|---|---|
| θ .3, min run .2 s, merge gaps .2 s | 3 | **gone**: P_fall per unit |
| Join score and peaks | 6 | v1 kept as the unit proposer; **v2 learned** (peak picking stays a parameter-free selection) |
| Span selection: guard, 2–20 s, −4/+5 s | 5 | **structural**: units, plus chunking of long units |
| Official rule: profiles, state, spike suppression, smoothing, anchor, transition bounds | ~10 | **learned** span head; the rule's quantities are inputs (parameter-free layer), not decisions |
| Raw-frame motion | — | input feature (already in `physics.py`) |
| Merge + NMS 5.9 s | 1 | **gone**: one span per unit by construction |
| Gate 2.12e-5 | 1 | **learned** inside P_fall, from the same statistic |
| One per recording | — | **structural** |
| What cannot be learned | | hard selection of join peaks and units (as an RPN's proposals), chunking of long units, τ |

## 6. Alternatives considered and why not

| Alternative | Why not |
|---|---|
| Dense per-timestep heads with in-model NMS (ActionFormer/TriDet heatmap + boundary distributions) | Each timestep still sees only its window. The boundaries depend on the recording's first and last seconds. A4 (ActionFormer-style regression) already lost to the rule, .757 against .847. |
| DETR-style queries per window, Hungarian matching | Needs many falls to learn where the queries go. Here the stream already hands over the anchors (recordings), so learning them is wasted data. Kept only as the 2-slot fallback of §5.3. |
| Soft units: attention biased by −α · (join mass between i and j), no hard units | Elegant and fully differentiable, but the labeller needs the *edge* profiles, which soft attention would have to discover from about 190 falls. Worth trying after v2, and it is the fallback if test streams turn out not to be concatenations. |
| Train the recording head on frozen, cached trunk features (fast) | The trunk memorizes its training falls (BCE about 1e-3), so in-sample features are overconfident. Train end to end. Cached features are fine only for decoding ablations on held-out streams. |
| Give the rule's interval as a one-hot input ("rule hint") | Nearly copies the PP. Keep it as an ablation (S2-hint) to measure how far the learned head is from the rule. |

## 7. Plan on the new CV

Protocol, from `docs/stream_cv.md` §9–11:

- 4 folds, a fixed budget, dense SWA, no selection on the held-out streams.
- Pooled score over all held-out streams; per fold; worst fold.
- A variant wins if:
  - its pooled score is ≥ +.02 higher (2 seeds);
  - it is better on ≥ 3 of 4 folds;
  - its worst fold is not worse.

| Stage | Run | What it isolates | Runs |
|---|---|---|---|
| R0 | rdrop+dswa (AP) + Amir's PP; also decoded without gate and without refine (decode only) | the reference, and the PP's gain on unseen subjects | 4 folds × 2 seeds |
| S1 | **RS-hybrid**: the recording-head model, decoded with P_fall as detector but the rule's bounds | learned recording classification against threshold + NMS + gate | decode only, from S2's runs |
| S2 | **RS v1**: P_fall + span head, heuristic units, `Rule` term on | the whole proposal | 4 × 2 |
| S2 ablations | −`score_r` inputs (pure learned); −`Rule` term; σ .2; MBR against argmax; rule hint | where the span quality comes from | decode-only variants are free; trained ones 4 × 1 |
| S3 | **RS v2**: learned join head, ground-truth units in training | learned joins (also report join P/R per fold against the heuristic) | 4 × 2 |

Diagnostics to log per fold:

- **Unit AUROC** of P_fall, by unit kind: fall against Autre/Empty and against ADL. Compare it
  with the gate statistic alone: .916 and .987 on training.
- **Span tIoU** against the rule's tIoU on the same clean held-out falls, paired: how often each
  wins, and the fraction ≥ .5.
- **Calibration** of P_fall, which decides whether τ = .5 is safe.
- **Error budget** in Amir's §6.4 classes: ok / loc_05 / loc_03 / lost / undetected; false alarms
  by unit kind.

Cost:

- **Training.** The trunk does about 2× the work per window. With 15 epochs × about 690 steps,
  one fold run should take about 1.5 h. This is an estimate, not measured.
- **Inference.** About 4× fewer windows than now, plus one small decoder pass per unit.

## 8. What to implement

| File | What |
|---|---|
| `src/falljepa/data/units.py` | units from join peaks, chunking, unit targets (`y_fall`, span, rule bounds, weights), unit-centred window sampling |
| `src/falljepa/data/unit_physics.py` | per-stream unit-relative channels and unit scalars on the unnormalized CIR, cached under `stats_path` like `physics.py` |
| `src/falljepa/model/recording_head.py` | `RecordingDecoder`: masked blocks, `[REC]`, soft anchor, fall and span heads |
| `src/falljepa/model/fall_jepa.py` | calls the decoder on `batch["units"]`; optional join head |
| `src/falljepa/loss/losses.py` | `UnitBCELoss`, `SpanSoftCELoss` (also used for `Rule`), `JoinFocalLoss` |
| `src/falljepa/model/aggregate.py`, `module.py` | decoder `kind: units` (MBR, τ); unit outputs saved in `probs.npz` for decode-only variants |
| `configs/uwb/ablations/RS.yaml`, `RS-pure.yaml`, `RS-join.yaml` | overlays on AP + rdrop + dswa |
| `tests/` | units and chunking on the synthetic release; masked span softmax; MBR on toy distributions; the physics layer equals `annotate._state`/`_profiles` on one unit |

Batches carry the units as padded tensors:

- `unit_bounds [B, U, 2]`, `unit_mask [B, U]`, `unit_scalars [B, U, S]`;
- targets `unit_fall`, `unit_weight`, `span [B, U, 2]`, `rule_span [B, U, 2]`.

The decoder runs on `[B·U, ≤ 300 tokens, 64]`.

## 9. Risks

- **Too few falls for the span head.** About 190 clean falls per fold may not teach it the rule's
  precision. The rule's objective as input and the `Rule` term on about 2,600 negative units address
  this. If S2's span tIoU stays below the rule's, keep S1 (learned P_fall, rule bounds), which
  still removes θ, NMS and the gate.
- **Units depend on how the streams are built,** exactly like the PP. The README says the test
  streams are built the same way. If they were not, units degrade to 20 s chunks, and the model
  still runs as a chunked window detector.
- **Autre is still scarce.** Each fold trains on about 18 Autre/Empty recordings. Chunks multiply the
  units, not the people.
- **Noise.** GPU runs are not reproducible per seed, and one fold has about 60 clean falls (±.06).
  Effects below about .02 pooled need more seeds than this plan has.

## 10. What was built: the one run

The user asked for a single run, not staged variants. RS is therefore the full proposal with v1
units (Amir's join detector), plus README v1.1's complex input. It also decodes its own dense head
with Amir's chain, so that one training run gives both numbers.

### 10.1 Complex input (README v1.1)

README v1.1 says the 368 values of a frame are **184 complex range bins stored as
`[I_0 … I_183, Q_0 … Q_183]`**. The amplitude of bin k is `sqrt(x[k]² + x[k+184]²)`.

What that means for the current model:
- The AP model cuts the 368 rows into 16 patches of 23. **The I and Q of one bin always land in
  different patches**: rows 0–7 hold I, rows 8–15 hold Q.
- A patch embedding is linear in its patch. So it can never combine I and Q of the same bin, and it
  cannot form phase or Doppler. Those need I and Q of the same bin over time.
- `range_shift` augmentation moved I bins into Q bins at the 184 boundary.
- Per-value z-scoring gives I and Q of one bin different scales, which distorts phase.

RS (`data.complex_input: true`):
- **Input:** `[10, 184, L]`: I of the 3 radars, Q of the 3 radars, the amplitude of the 3 radars,
  and valid.
- **Patches:** 23 bins × 8 timesteps, so 8 rows. A patch now holds I and Q of the same bins over
  0.16 s, and its linear embedding can learn complex filters (a slow-time DFT is linear in I/Q).
- **Normalization:** one std per complex bin, `sqrt((std_I² + std_Q²) / 2)`, for both I and Q.
  The mean is still per value.

Nothing else needs to change for the complex layout:
- the physics features, the join detector and the labeller sum squared differences over the 368
  values, which is exactly Σ_k |Δx_k|² over the complex bins;
- the persistence-gate statistic is the same for the same reason.

### 10.2 A fall belongs to the unit it starts in

On the official train and val streams, Amir's join detector puts a peak **inside 16 of the 253
clean falls (6.3 %)**:
- always 36–178 timesteps after the fall's start, at the landing;
- the landing changes all three radars at once, like a join.

With the design's "≥ 90 % of the fall in one unit" rule, those falls would have no unit. RS uses
instead:
- **Owner.** A fall belongs to the unit it starts in.
- **Span.** Its span may end up to 2 s after the unit.
- **Tail.** The unit holding the tail learns "no fall here" (weight .5).
- **Result.** **250 of 253** clean falls get a scored owner unit.

Consequences:
- The span domain is the unit from 0.5 s before to 2 s after.
- The decoder sees the unit ± 2 s.
- A window decodes only units ≥ 2 s from its inner edges.
- So the inference hop is 6 s (22 + 2 × 2 + 6 = 32).

### 10.3 Settings

Data:
- 32 s windows, training hop 2 s, inference hop 6 s.
- Frames 32/4; physics features; interior normalizer.
- Balanced sampler with 16 fall slots, now each holding a whole, scored fall unit, and 4 Autre
  slots.
- Radar dropout .15.

Training:
- 15 epochs, dense SWA every 50 steps from epoch 3, AdamW 1e-3 constant.
- Validation every 5 epochs, for progress only. `checkpoints/swa.ckpt` is the model.

Losses:

| Term | Weight |
|---|---|
| `State_Frame`, `State_Step` | .5, .25 |
| `MSE_Latent`, `SIGReg` | .25, .1 |
| `Unit` | 1 |
| `Span` | 1 |
| `Rule` | .2 |

Decoding:
- One event per unit with P > .5, its span by MBR.
- The run also writes `<name>.amir.csv`: Amir's chain (θ .3, rule, NMS, gate) on the dense head of
  the same model, logged as `val/amir/*`.

### 10.4 Checked before the run

These checks ran locally, on CPU.

- **Tests.** All 65 pass (`tests/test_recording_head.py` adds 11). They cover:
  - the units and the ownership rule;
  - the labeller channels, which equal the official `_state`/`_profiles` and `official_bounds`;
  - the complex layout;
  - the decoder's rows and domains;
  - the losses training the head;
  - MBR;
  - window coverage of every unit;
  - an end-to-end `fit` + `validate` of RS.yaml on the synthetic release.
- **Real val streams:**
  - the unit caches for 5 streams build in 17 s;
  - 32 training windows load in 0.3 s, holding 20 scored fall units;
  - all 600 units are decoded by some inference window;
  - the span targets lie inside their domains.
- **Model size:** 524k parameters.
- **Memory:** a batch-32 training step peaks at about 8.2 GB on CPU, against 7.2 GB for AP.

### 10.5 Run it

```bash
FOLDS_DIR=/home/abc2027/dataset/abc2027_uwb_cv_folds OFFICIAL=/home/abc2027/dataset/abc2027_uwb_train_v1 \
  GPU=0 nohup tools/run_rs_cv.sh > runs/rs_cv.log 2>&1 &
```

The runner:
- trains folds A–D one after another;
- validates each fold's SWA checkpoint;
- writes `runs/_analysis/cv_RS_<TAG>.json` (the recording head) and `cv_RS-amir_<TAG>.json` (Amir's
  chain on the same models), each pooled, per fold and per stitching.

Rerun with the same `TAG=` to resume. If the GPU is short of memory, append
`-- --data.batch_size 16 --data.sampler.fall_slots 8 --data.sampler.hard_slots 2`.

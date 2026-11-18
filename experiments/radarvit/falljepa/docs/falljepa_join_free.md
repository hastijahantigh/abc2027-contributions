# A join-free fall detector: window proposals and a learned stream stage

Branch `stream-cv`, 2026-10-26. Config `configs/uwb/JF.yaml`, model `falljepa.model.FallSetJEPA`, runner
`tools/run_jf_cv.sh`. Nothing has been trained yet.

## 0. Why

RS (`docs/falljepa_recording_head.md`) scores .786 on the subject-held-out CV, but it reasons over
*source recordings*. Their boundaries are found by Amir's join detector, which detects the 0.2 s
crossfades the stream generator inserts. That is a property of how the challenge built its streams,
not of falls. For the paper and for a fair entry we want a model that never uses joins, recordings
or the labeller's recording-relative quantities, and that still decides once per fall over the
whole stream.

**Join-free means:**
- no join detector, units, unit channels or unit scalars;
- no Amir span, labelling rule or persistence gate at decoding;
- no `annotate.py` at all.

Training may still use the training labels and the composition file for sampling and normalization,
as every model does.

## 1. The model in one picture

```
stream ──► 32 s windows every 6 s ──► trunk (unchanged: frame encoder, causal, bidirectional) ──► tokens r (0.08 s)
                                            │
                    stage 1, per window:    ├─► dense head  P(fall) per 20 ms step       (auxiliary; also the
                    3 fall queries          │                                              join-free baseline decoder)
                    (DETR decoder)          └─► 3 × (P(fall), start distribution, end distribution, hidden state)
                                                                    │
                                     proposals with P > .05, from every window that saw them
                                                                    │
                    stream stage:           group proposals whose spans overlap (tIoU ≥ .2)
                                            region = the group ± 3 s
                                            ├─ stream tokens: r of every window, averaged per 0.08 s (centre-weighted)
                                            ├─ event physics: the scene before vs after the group's span (no parameters)
                                            └─ proposal tokens: each member's hidden state, P, span, window position
                                                                    │
                                            group refiner: 2 slots + 2 transformer blocks
                                                                    │
                                            per slot: P(fall), start / end distributions over the region
                                                                    │
                    decoding:               slots with P > .5, span by MBR, NMS at tIoU .5 (duplicates only)
```

## 2. Stage 1: fall queries in each window

- **Why queries.** A 32 s window often holds two falls, and "one fall per window" would drop the
  second. Three learned queries (DETR) cross-attend to the window's 400 tokens; each says "a fall
  here, from … to …" or "nothing".
- **Pointer spans.** A query's start (end) logit at step t is `q · (k_frame(t) + k_offset(t mod 4) + k_step(t))`:
  - `k_frame` comes from the trunk token that owns t (0.08 s);
  - `k_offset` is one of 4 learned offsets inside that token;
  - `k_step` is a projection of the 12 physics features at that 20 ms step.

  This gives 20 ms resolution without a dense decoder, and a full distribution, which the stream
  stage and MBR need.
- **Targets: set matching.**
  - Each window's falls are matched to its queries with the lowest-cost assignment. The cost is
    −log P(fall) − ½ (log P_start(a) + log P_end(b)), brute-forced since there are ≤ 3 queries.
  - A fall wholly inside the window is a target.
  - A fall cut by the window edge, or a flagged fall, may absorb a query but gives it no
    objectness loss. A flagged fall still trains the span at weight .5.
  - Unmatched queries learn "no fall".

## 3. Training on chunks, so the stream stage trains on real proposals

A training item is a **chunk**: 4 consecutive inference windows (32 s, 6 s apart), so 50 s of stream.
The model cuts it into its windows and runs stage 1 on each of them, exactly as at inference. The
stream stage then sees the proposals of 4 overlapping windows, as it does on a whole stream. This is
the Faster R-CNN recipe: the second stage trains on the current proposals of the first.

- **Batch.** 8 chunks = 32 windows per optimizer step, the same trunk cost as RS. They come as 2
  accumulated batches of 4 chunks to stay under 7 GB of GPU memory (§8). Chunks lie on an 8 s grid,
  so an epoch has as many optimizer steps as RS.
- **Sampler.** Amir's balanced sampler over chunks, per batch of 4: 2 fall chunks, 1 Autre chunk, 1 other.
  That is 2 Autre chunks per optimizer step, twice RS's share of hard negatives.
- **Groups per chunk:**
  - the model's own proposals (P > .05), grouped as at inference, at most 6;
  - each scored fall that no group covers, as a group around its true span jittered by 0.3 s;
  - one random no-fall region.

  The extra groups keep stage 2 learning while stage 1 is still poor.
- **Robustness.** A member proposal is dropped with probability .25 (at least one stays), and the
  group's reference span is jittered by 0.2 s, so stage 2 does not learn to trust perfect training
  proposals.
- **Gradients.** Stage 2 back-propagates into the queries and the trunk, as in a two-stage detector.

## 4. Stage 2: the group refiner

For one group, the region is the union of its members' spans ± 3 s (at most 20 s).

1. **Stream tokens.** At every 0.08 s of the region, the trunk tokens of all windows that cover it
   are averaged. Each window is weighted by how central the token is in it (triangular). This is
   the stream-level view; with 6 s hops, 4–5 windows contribute per token.
2. **Event physics** (no parameters, `set_heads.event_physics`). Profiles are taken just before and
   just after the group's reference span [a, b], the p-weighted median of its members:
   P_before = mean over [a − 1.2 s, a − 0.2 s], and P_after = mean over [b + 0.2 s, b + 1.7 s].
   Per radar and step it computes:
   - **state**: d(x, P_before) / (d(x, P_before) + d(x, P_after)), smoothed 0.4 s;
   - **motion** relative to the region's mean;
   - **rise**: state(t + 1.5 s) − state(t);
   - the position relative to a and to b.

   Per group it also gives log ‖P_after − P_before‖² per radar, which is the gate's statistic but
   around the event instead of the recording, and the stillness after the event.

   These are the labeller's quantities without the recording: "the person was standing, moved,
   and now the picture is different and still". A soft anchor, softmax(β · log change) with β
   learned, mixes the radars as RS did.
3. **Proposal tokens.** Each member's query hidden state is joined with its logit, its span relative
   to the region, its length and its distance to its window's edge. The refiner can then weigh
   "4 windows agree" against "one window, at its edge".
4. **Refiner.** It has 2 slots, so a group may hold 0, 1 or 2 falls:
   - each slot starts from a learned vector plus an MLP of the group scalars;
   - 2 transformer blocks run over [slots, proposals, region tokens];
   - each slot has an objectness head and a pointer start/end head over the region's steps.
5. **Targets.** Falls wholly inside the region are matched to the slots like stage 1. Partly-inside
   and flagged falls are ignore targets.

## 5. Decoding

- **Main (`<name>.csv`).** Every slot with P > .5 gives one event. Its span is chosen by MBR for the
  mean F1 at tIoU .1/.3/.5. Two groups can reach the same fall through their 3 s margins, so
  events overlapping a higher-scored one at tIoU ≥ .5 are dropped. This only removes duplicates.
- **Ladder, written by the same run:**
  - `<name>.queries.csv`: stage 1 with a non-learned merge. In each group, the most central
    member's P and MBR span are used.
  - `<name>.dense.csv`: the dense head with the old Viterbi decoder.

  Together with the main file this gives three join-free numbers from one training, which shows
  what each stage adds.

## 6. Settings (`configs/uwb/JF.yaml`)

- **Trunk and recipe:** the RS trunk and recipe. Complex input, physics, interior normalizer, 32 s
  windows, radar dropout .15, 15 epochs, dense SWA from epoch 3, AdamW 1e-3. Nothing is selected on
  the held-out streams.
- **Heads:**
  - stage 1: 3 queries, 2 decoder layers;
  - stage 2: 2 slots, 2 blocks, at most 12 members and 6 groups per chunk.
- **Losses:**

  | Term | Weight |
  |---|---|
  | State_Frame, State_Step (dense) | .5, .25 |
  | MSE_Latent, SIGReg | .25, .1 |
  | Query_Obj, Query_Span | 1, 1 |
  | Group_Obj, Group_Span | 1, 1 |

- **Inference:** windows every 6 s, proposals at P > .05.

## 7. What to compare

On the 4 folds, against RS .786 (join-based) and Amir's chain .777 (join-based):

| Variant | What it is |
|---|---|
| `dense` | the join-free baseline: dense head and Viterbi |
| `queries` | stage 1 with a fixed merge |
| main | the full model |

A main score close to RS is a good join-free result. The physics-free ablation is
`--model.model.init_args.event_physics false`. It shows how much the before/after features give
without any recording information.

## 8. Checked before the run (CPU, 2026-10-26)

- **Tests.** All 74 pass; `tests/test_join_free.py` adds 9. They cover:
  - set matching with scored, flagged and cut falls;
  - grouping and regions;
  - event physics finding a new scene on one radar;
  - the heads learning a span;
  - the refiner on padded groups;
  - the non-learned merge across windows;
  - the token accumulator;
  - chunks in the data and both stages training end to end;
  - a full `fit` + `validate` of JF.yaml on the synthetic release, writing all three CSVs.
- **Real sizes.** On 2 chunks of 50 s, the forward pass gives 8 windows, 24 queries and 10 groups.
- **Model size:** 663k parameters (RS: 524k).
- **GPU memory on ABC27** (2026-10-27, fold A, 150–300 training batches and a full validation decode,
  nvidia-smi sampled every 0.5 s):

  | Setting | Peak |
  |---|---|
  | 8 chunks per batch, eval batch 32 | 11.6 GB (training; validation 7.9 GB) |
  | the same with `expandable_segments` | 11.6 GB |
  | 4 chunks × 2 accumulated, eval batch 16, `expandable_segments` | **6.4 GB**, flat |

  The trunk on 32 windows is most of it. The stream stage's physics now runs 16 groups at a time.
- **MBR tie-break.** `aggregate.mbr_span` now takes `tie_iou`. With a peaked distribution every span
  within tIoU .5 ties, so JF adds 1e-3 × the expected tIoU to break ties. RS's default (0) is unchanged.

## 9. First ablations if JF underperforms

- **Batch diversity.** A step of 4-window chunks holds only 8 independent places, against RS's 32
  windows. Try `--data.chunk_windows 2 --data.batch_size 8 --data.sampler.fall_slots 4 --data.sampler.hard_slots 1`
  (16 windows per batch, the same memory).
- **Physics off:** `--model.model.init_args.event_physics false`.
- **Detection threshold.** Read the slots in `val-validate.probs.npz` (`slots/<sid>`: group, slot,
  logit, start, end) to see whether .5 is the right cut. Decide that across folds, never on one.

## 10. Expected weak spots

- **Recording-relative labels.** About 14 % of fall clips do not start at their recording's start,
  so their "standing" profile comes from outside the stream. Some tIoU .5 accuracy is lost for good.
- **Missed falls stay missed.** A fall that no window proposes has no group. Stage-1 recall at
  P > .05 is the ceiling. The run logs it as `<stage>/sets/proposal_recall`.

## 11. JF2: the changes from the JF error analysis

JF (tag 20261027-jf1) scored .752 pooled on the CV, against RS .788 and Amir's chain .778 on the same
streams. The error analysis (`report/falljepa_JF_errors.html`) found three causes:
- **unseen recording sessions**: every session has its own static background, and held-out sessions
  far from all training sessions lose most falls;
- **falls 1–2.5 s after a join**: JF finds .50–.76 of them, against .68–.93 for RS;
- **misplaced falls**: long falls are cut to the impact, and the wrong motion burst is chosen.

JF2 (`configs/uwb/JF2.yaml`) changes all of these in one run. Every change is an option, and `JF.yaml`
is unchanged. JF2 stays join-free.

| Change | Where | Targets |
|---|---|---|
| 9 clutter-removed channels: signal minus its 8 s centred moving mean (`data.clutter_s`) | `StreamStore.read` | unseen sessions |
| raw channels zeroed for the trunk with p .3 (`raw_drop_p`) | `FallSetJEPA.forward` | unseen sessions |
| static scene swapped with another session's 2 s mean profile, p .5 (`augment.bg_swap_p`) | `WindowDataset._swap_background` | unseen sessions |
| event physics anchors: the stillest 0.5 s in [a − 4, a − 0.2) s and 1 s in [b + 0.2, b + 4) s (`stable_anchors`) | `set_heads.event_physics` | late starts, short leads |
| soft span targets (σ 0.25 s) + `SpanIoULoss` (1 − tIoU of the expected span), both stages | losses | late starts |
| proposals within 3 s of a group's extent join it (`group_gap_s`); regions ± 5 s, ≤ 24 s | `sets.group_proposals` | wrong burst, duplicates |
| an event whose centre is within 3 s of a stronger one is dropped (`decoder.centre_s`) | `sets.nms` | duplicates (+0.007 on JF's saved slots) |

- **Data.** JF's protocol is unchanged: the same folds (2 training stitchings, 26 streams), sampler,
  epochs and held-out streams. More training stitchings (`--train-stitchings 8` with the balanced
  sampler's `steps_scale: 0.25` to keep the epoch) were left out so the comparison with JF, RS and
  Amir's chain stays like for like; if tried, it should be an ablation for every model.
- **Input.** The model sees 19 channels: raw I, Q and amplitude, the same three clutter-removed, and
  valid. The event physics always reads the raw I and Q, never the dropped ones.
- **Run.**

  ```bash
  FOLDS_DIR=/home/abc2027/dataset/abc2027_uwb_cv_folds OFFICIAL=/home/abc2027/dataset/abc2027_uwb_train_v1 \
    CONFIG=configs/uwb/JF2.yaml NAME=JF2 GROUP=UWB/JF2 GPU=0 tools/run_jf_cv.sh
  ```

- **What to read.** Read the per-group numbers of the error analysis, not only the pooled score:
  - recall of the low-similarity sessions;
  - recall at lead ≤ 2.5 s;
  - the start error of falls over 2 s;
  - false alarms inside fall recordings.

  All changes are in one run, so a gain cannot be credited to one of them; ablate afterwards.

## 12. JF3: placing the fall where the label puts it (design, 2026-10-28)

The JF2 error analysis (`report/falljepa_JF2_errors.html`) covers 4 folds and 2 seeds. Its largest
error is timing. In 73 of 123 misses, JF2 fired on the right fall recording but at the wrong time:
47 detections overlap the fall with tIoU < .3, and 26 fire 1.5–5 s after it. The same detections
are 77 of the 109 false alarms. Placing them correctly would take F1@.3 from .885 to .959.

The errors are mostly late, on Chute04 and on fold D (P1_8 / P1_9). On fold D, Amir's chain hits
77 % of them, because the join anchors the start.

### What the diagnosis shows

Tables are in `runs/_analysis/jf2_err/`: `burst.csv`, `props_near.csv` and `rule_test.csv`, with
the scripts that wrote them.

- **Decoding is not the cause.** No timing error had a slot with tIoU ≥ .3 that NMS or the 3 s
  centre suppression removed.
  - In 30 of the 73, a stage-1 proposal had the right span, but the refiner moved it.
  - In the other 43, no proposal and no slot had it.
- **JF2 places the fall on the motion burst. The label is a state step.**
  - For hits, the label matches the main burst: onset −0.06 s, end +0.12 s (medians).
  - For timing errors, the motion is longer or split: 6 bursts above 20 % of the peak, against 3
    for hits. The label covers only the first part.
  - JF2 predicts at the burst's peak and centre of mass (+0.13 s from the peak): +1.59 s from the
    label centre.
  - The labeller (`abc2027/annotate.py`) places the fall where the scene steps from the standing
    profile to the floor profile. It takes the 1.5 s window with the largest
    rise × log1p(motion), cut at 20–80 % of the rise and widened over motion ≥ 15 %. That is not
    the strongest burst.
- **Stage 1 is already late on these falls.** The P-weighted centre of the nearby proposals is
  +1.52 s from the label (hits: +0.02 s). Only 8 % have most of their proposal mass on the right
  span. The refiner follows them.
- **The refiner never trains on a late reference span.**
  - Its groups' reference spans are jittered by N(0, 0.2 s) (`span_jitter_s`), and its extra groups
    sit on the true span ±0.3 s.
  - On training falls, stage 1 is accurate, so the event-physics anchors are nearly always right.
  - At inference on these falls, the reference is 1–3 s late.
- **The anchors already hold the answer.** The offline test (`jf2_rule_srv.py`) rebuilt JF2's
  groups from the saved proposals. It then ran the labeller's transition rule on the radar that
  changes most, with JF2's own stable anchors (the stillest 0.5 s before and 1 s after the
  reference span):

  | profiles for the rule | timing errors recovered | hits kept |
  |---|---|---|
  | V0: JF2's stable anchors around the reference span | **55 / 73** (fold D 84 %) | 764 / 889 |
  | V1: stillest stretches before / after the group's extent | 46 / 73 | 699 / 889 |
  | V2: first 1 s / last 1.5 s of the region (the labeller's own choice) | 32 / 73 | 562 / 889 |

  V0's median offset on the timing errors is 0.00 s; JF2's is +1.59 s. As a hard replacement, V0
  would break 125 hits. A post-hoc selector between the two spans also fails: a gradient-boosted
  selector on summary features recovers +5 net across the folds. Amir's join-free F was unstable
  for the same reason (fold C .909 → .699). **The rule and the model are right on different falls,
  and only the full signal tells them apart. So the choice belongs inside the refiner.**

### The changes

All of them are options; `JF2.yaml` stays as it is. The model stays join-free: the rule runs on the
group's region and anchors, never on a recording.

| # | Change | Where | Why |
|---|---|---|---|
| T1 | **Late reference in training.** With p .5 (`ref_shift_p`), move a group's reference span by δ, \|δ\| ~ U(0.8, 3.0) s (`ref_shift_s`), late : early = 2 : 1. This also applies to the extra groups around uncovered falls. Targets stay the true labels. The region still holds the fall, since the margin is 5 s. | `FallSetJEPA._stage2`, `_extra_groups` | The refiner and the physics anchors then see the late references that inference gives them. |
| T2 | **Transition candidate.** `event_physics` also runs the labeller's `_transition_bounds` on the anchor radar's state and motion (ported, CPU, per group; a parity test against `annotate.py`). It returns (a) two channels: the rule score rise × log1p(motion), and an indicator of the rule span; (b) one extra member token: a learned embedding plus member features (the rule span in the region, its length, the height of the rise, the motion score; the rule score in the logit slot). | `set_heads.event_physics`, `GroupRefiner.forward`, `sets.member_features` | The refiner's pointer can copy the transition span when the signal supports it (V0: 55/73) and keep the burst span otherwise. |
| T3 | **Cascade, ablation only.** A second refiner pass with the first pass's span as the reference: the physics is recomputed and the weights are shared, on detached first-pass spans. | `FallSetJEPA.refine`, `_stage2` | Corrects anchors that start late without relying on T1 alone. |

Not changed, with the reason:
- NMS and centre suppression: 0 of the 73 timing errors come from decoding.
- The score threshold: false alarms score as high as hits.
- Post-hoc rule replacement or selection: −125 hits, or +5 net.
- The span-loss weights: hits are unbiased, with a median centre offset of +0.02 s.

### Runs and what decides

- **JF3** = JF2 + T1 + T2 (`configs/uwb/JF3.yaml`), seeds 0 and 1, on the same protocol:
  - the CV folds as built (2 training stitchings);
  - the same sampler, epochs and memory settings (7 GB cap);
  - `tools/run_jf_cv.sh`.

  That is about 5 h per seed. If JF3 wins, ablate with **JF3-T1** (T1 alone, 2 seeds), then
  **JF3+T3**.
- **Checks before the run (CPU):**
  - the ported rule matches `annotate._transition_bounds` on synthetic state/motion curves and on 20
    training recordings;
  - with T1, the shifted reference stays inside the region;
  - one training step and one inference pass stay under 7 GB.
- **Scoring:** `jf2_err_srv.py` and `build_report.py` with the new tags (same matching), and the
  pooled score from `score_folds.py`.

| Metric | JF2 (s0 + s1) | Target for JF3 |
|---|---|---|
| timing errors: misses + false alarms | 73 + 77 | ≤ 45 + 45 |
| hits lost (JF2 hits that JF3 misses) | — | ≤ 15 |
| fold D score (2-seed mean) | .811 | ≥ .85 (RS .852) |
| Chute04 recall | .71 | ≥ .80 |
| pooled score (2-seed mean) | .861 | ≥ .88 |
| median tIoU of hits | .77 | not lower |

**Risks.**
- T1 could make the refiner trust its members less, which would cost hit localisation. Watch the
  median tIoU of hits.
- The rule needs a visible rise, so it does not help the weak, quiet falls (section 3 of the report).
  Those stay a separate problem.
- An ADL burst also gets a rule candidate. The candidate only changes the span, not the
  objectness, but check the ADL false-alarm rate (0.97 / h).

### Implementation (2026-10-28)

- **T1:** `FallSetJEPA(ref_shift_p, ref_shift_s, ref_shift_late)`, in `_stage2`.
  - It applies to model groups and to the groups around uncovered falls. `_extra_groups` now returns
    how many of its groups are fall groups, so the no-fall groups are never shifted.
- **T2:** `FallSetJEPA(transition)` → `GroupRefiner(transition)` → `event_physics(transition=True)`.
  - It returns 13 channels plus `[G, 6]` rule features. The rule member token is valid when the
    region holds a transition.
  - `set_heads.transition_bounds` is the labeller's `_transition_bounds`, step for step. A test
    checks it against `annotate.py` on 40 random curves.
- **Config:** `configs/uwb/JF3.yaml` is JF2 plus these 4 options.
- **Tests:** `tests/test_join_free.py`, the last 5 tests, including JF3 end to end on the synthetic
  release.
- **Limit found by the tests:** when a long still stretch lies between the scene change and the
  later burst, the stable before-profile lands on the floor, and the rule follows the late
  reference. In the JF2 timing errors, the motion runs on from the fall, which is the case the rule
  handles (55 of 73 offline).

## 13. JF4: a data-shaped stage-1 trunk (design, 2026-10-29)

JF4 starts from JF2, not JF3, and changes only stage 1. The full design, the data measurements
behind it and the run ladder are in `docs/falljepa_JF4_design.md`. The changes:

- **Stem.** A per-radar convolutional stem whose first layer is a complex convolution plus a
  modulus (phase-invariant), replacing the linear patch embedding and the frame transformer.
- **Fusion.** Range attention pooling, then attention across the three radar tokens (late fusion).
- **Local encoder.** A dilated TCN with a 4.9 s receptive field and no pooling.
- **Global encoder.** RoPE bidirectional blocks.
- **Contrast inputs.** Multi-scale before/after contrast maps of the complex profile, with a
  step-swap (decoy join) augmentation.

Measured on the 18 published streams, subtracting the lowest-energy radar lowers the fall motion SNR
in 99 % of falls, so it is not used.

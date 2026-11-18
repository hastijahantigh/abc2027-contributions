# JF7: start and end timing (2026-10-30)

JF7a and JF7b change only the span objective and the stream stage of JF6 (= JF5T, `configs/uwb/JF5T.yaml`).
The trunk, the data and the training schedule are unchanged. No labeller rule is used.

## 1. Why: the timing errors of JF6

From `runs/_analysis/err_jf56/` (JF6, 2 seeds × 4 folds, 1012 scored falls):

- **Fixing timing is worth +.034 to +.042** (oracle, ensemble and single seeds). Missed falls are worth +.038 to +.046, false alarms +.015 to +.018.
- **Of the 102 falls detected at tIoU < .5:**
  - in 76 the start is the worse boundary, 68 of those more than 0.5 s late;
  - 64 predicted falls are shorter than the label.
- **Long falls are the problem.** Falls of 2.5–3 s have median tIoU .60, and 22 % are below .5. Their median start is 0.84 s late. Falls up to 2 s have median tIoU .83–.86.
- **Most of these falls never get a good span.** In 74 of the 102, no stage-1 proposal reaches tIoU .5 either. In the other 28, a proposal did, and stage 2 output a worse span. Choosing the best proposal for every fall would take recall at tIoU .5 from .813 to .867.
- **Training and decoding disagree.**
  - Training: a cross-entropy on the start and end marginals, plus the tIoU of the expected positions.
  - Decoding (`mbr_span`): the span that maximises the expected challenge score.
  - Ranking: P(fall), which ignores timing.
- **Stage 2 trains on small errors only.** Its reference spans are jittered by N(0, 0.2–0.3 s), much smaller than the 1–2.5 s errors it meets at inference.

## 2. JF7a (`configs/uwb/JF7a.yaml`)

**1. Pair-tIoU span loss** (`loss.SpanPairIoULoss`, `Query_PairIoU` and `Group_PairIoU`, weight 2 each).
- It is 1 − the expected gain of a span drawn from the start and end distributions, over every (start, end) pair at token resolution (4 steps).
- The gain of a pair is .5 × its tIoU plus .5 × the mean of sigmoid((tIoU − θ) / .05) over θ = .1, .3 and .5. This is the smooth version of what `mbr_span` maximises.
- Pairs whose end is not after the start, at token level, count 0.
- It replaces `SpanIoULoss`. The soft cross-entropies stay at weight .5 (was 1) to keep a dense signal early in training.

**2. Slot quality head** (`GroupRefiner(quality=True)`, loss `Group_Quality`: BCE against a soft target).
- Each slot predicts the tIoU of its MBR span with the fall it is matched to. Otherwise the target is its best tIoU with any fall in the region, otherwise 0. Slots matched to a cut fall get weight 0.
- At inference the event score is P × q^α, with α = `decoder.quality_alpha`, default 1. The P > .5 threshold is unchanged, so this only changes which overlapping slot NMS keeps, and the order the evaluator matches in.

**3. Member boundary channels** (`GroupRefiner(member_channels=True)`). The region gets two more step channels: each member's start and end as a Gaussian of 0.2 s, weighted by the member's P (max over members). The stage-2 pointer can then copy a member's boundary and adjust it, which targets the 28 cases above.

## 3. JF7b (`configs/uwb/JF7b.yaml`): JF7a plus a boundary refinement stage

**The module** (`set_heads.BoundaryRefiner`).
- Each start and each end is placed again from a ±3 s crop at 50 Hz around its current position.
- Inputs per step: the region's step channels (event physics and member bumps), the refined stream token that owns the step, the slot's hidden state, the boundary kind and the offset from the centre.
- Five residual dilated convolutions (dilations 1–16, about ±1.2 s of reach per position) give logits over the crop.

**Training** (`Boundary_CE`: soft cross-entropy, σ = 0.1 s).
- Crops are centred on the matched falls' starts and ends, moved by an error drawn from `fall_set.BOUNDARY_ERRORS`:
  - starts: 45 % N(0, 0.3 s), 35 % late by U(0.5, 2.5 s), 20 % early by U(0.5, 2 s);
  - ends: 50 % N(0, 0.3 s), 30 % late by U(0.5, 2 s), 20 % early by U(0.5, 2 s).
- So the stage learns to undo the error sizes JF6 actually makes.

**Inference** (`FallSetJEPA.refine_bounds`). For every slot with P > .05, the start and the end move to the argmax of their crops. These spans are written both to the slots and to the events.

**Left out for now:**
- the dense head's P(fall) as input to the stage-1 pointer;
- oversampling long falls;
- cascade matching at increasing IoU.

These are the next step if JF7a/JF7b move the start errors on long falls.

## 4. Protocol

- 4 folds × seeds 0 and 1 for each variant, on ABC27: `tools/run_jf7_queue.sh`, tags `20261030-jf7a-s{0,1}` and `20261030-jf7b-s{0,1}`, JF7a first.
- Compare with JF6 pooled .899 (fold mean), and with PP .918.
- Watch:
  - F1 at tIoU .5;
  - the late-start rate on falls longer than 2.5 s (`runs/_analysis/err_jf56/analyze.py`);
  - recall at .5 against the best-proposal ceiling (.867).
- Seed SD is about .015, so differences under .02 are ties.

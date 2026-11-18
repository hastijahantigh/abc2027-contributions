# FallJEPA: notes for the group meeting (2026-10-11)

## 1. Task

- ABC2027 UWB-FallTrack. Three UWB radars, 368 range bins each, 50 Hz.
- Find each fall in a long stream. Give its start time and end time.
- The ranking score is the mean event F1 at tIoU 0.1, 0.3 and 0.5.
- Two of the three thresholds need good start and end times. Detection alone is not sufficient.
- Validation has only 67 scored falls. One fall changes recall by approximately 1.5 points.

## 2. Model (short)

- We use the frame JEPA from gir (HTR-JEPA). We read it along time, not along a text line.
- Input: 25.6 s windows. Each window has 320 frames of 0.64 s.
- Output: one of four states for each 20 ms step: none, begin, middle, end.
- A Viterbi decoder changes the states into events (start, end, score).
- The model does not give times directly. The decoder reads them from the states.
- An auxiliary JEPA loss predicts the latent of a frame 8 frames ahead.

## 3. Results

| Model on official val | Mean F1 | FA/h |
|---|---|---|
| Energy baseline (official) | 0.315 | 49 |
| Logistic baseline (official) | 0.258 | — |
| Full run, last epoch | 0.654 | — |
| Full run, epoch 78 (selected on val) | 0.737 | 10.2 |
| 6-fold ensemble, no JEPA (nothing selected on val) | **0.757** | 7.3 |

- The model is approximately 2.4 times better than the best baseline.
- The largest gain is at strict overlap: F1 0.65 at θ 0.5. The baselines get 0.25 and 0.01.
- At θ 0.3 the model finds 54 of 67 falls. Median start error is 0.35 s.
- Joins between recordings do not cause problems: 1 false alarm in 21 (see section 5).

## 4. Be careful with the 0.737

- We selected epoch 78 from 150 epochs on the same val streams. This adds approximately 0.08.
- From epoch 60 to 149, the val score is 0.65 ± 0.04.
- The model overfits. Train loss goes to 0.011. Val loss is lowest at epoch 4 and then increases.
- The honest single-model number is approximately 0.65.
- The ensemble number (0.757) is more honest. We did not select its members on val.

## 5. Errors

- **Misplaced detections.** 6 of the 13 misses have a detection near them (less than 4 s). Each one counts as a miss and a false alarm.
- **Chute04.** Recall is 29% (4 of 14). Other fall types: 86–100%.
- **Labels of exactly 2.00 s.** The model finds 55% of these falls, and 86% of the others. Possibly these labels do not align with the motion. We must examine the signal.
- **Boundaries.** Starts are centred but wide. Ends are frequently late. F1 decreases from 0.76 (θ 0.3) to 0.65 (θ 0.5).
- **Score.** False alarms have lower scores than hits (median 0.86 against 0.97). The official scorer does not remove low scores. The decoder must do this.
- **Joins between ADL data and fall data: no problem.** The streams join ADL clips and fall clips with a 0.2 s crossfade. The jump in the signal at a join does not cause false alarms. On val, we examined each join with no fall within 1.5 s, in a ±1 s window:

  | Join | Joins examined | Max P(fall) > 0.5 | False alarms |
  |---|---|---|---|
  | ADL → fall clip | 85 | 1 | 0 |
  | Fall clip → ADL | 98 | 0 | 0 |
  | ADL → ADL | 355 | 2 | 1 |

  The median of the max P(fall) at a join is 0.000. At an ADL→fall join, the model does not start the fall too early. Only 1 of the 21 false alarms is at a join, and that join is ADL → ADL.
  One limit: 12 scored falls start less than 2 s after an ADL→fall join. The model finds 6 of them. But it does not give a false detection at the join. It gives no response. 4 of the 6 misses have labels of exactly 2.00 s (see above). Thus the cause is probably the labels, not the join.

## 6. Cross-validation: a problem

- We ran 6-fold CV by subject. We ran it with JEPA and without JEPA.
- On its held-out fold, each model is best at approximately epoch 8. At the last epoch, it is almost zero (F1 0.11 / 0.06).
- The same checkpoints get 0.39–0.74 on official val.
- Thus the problem is not general overfitting. Possibly the CV protocol causes it.
- Our hypothesis: the held-out signal is in the training windows as unlabeled context. The model sees it for 100 epochs. Val streams are never seen.
- Next test: train one fold with the held-out signal set to zero in the input.
- Until we fix this, do not use the CV curves to set the epoch count or tune the decoder.

## 7. JEPA or no JEPA

- On the held-out folds, the two are almost the same.
- On val, no-JEPA wins in 5 of 6 folds. The ensemble is better by +0.085.
- Decision: we set the JEPA loss weight to 0 as the default.

## 8. Next steps

1. Fix the CV (hide the held-out signal). Do this on one fold first.
2. Tune the decoder on out-of-fold predictions. Add a minimum score. On val, a tuned decoder decreases false alarms by 50% (10.2 → 4.9/h).
3. Decrease overfitting: weight averaging (EMA or SWA), more dropout, more augmentation.
4. Add a boundary regression head (as in ActionFormer and TriDet).
5. Examine Chute04 and the 2.00 s labels on the signal.
6. Calibrate the probabilities. At P(fall) > 0.9, only 68% of steps are falls.
7. Submission plan: the 6-fold no-JEPA ensemble. Score val one time only.

## 9. Questions for the group

- Is a CV that uses held-out signal as unlabeled context a known problem? Do you know a better protocol?
- Fall clips and ADL clips come from different places. The static radar scene can show where a fall clip is. Do we ask the organizers before we use this?
- Must each change that we keep get a full 6-fold CV? A single val run cannot show a change smaller than approximately 0.1 F1.

## Reports to share

| File | Content | For whom |
|---|---|---|
| `report/falljepa_U0_report.html` | Main result: how the model works, scores, errors, plan | All. Show this first. |
| `report/falljepa_cv_report.html` | CV problem, JEPA vs no JEPA, ensemble | All. Show sections 2–4. |
| `docs/falljepa_fall_timing.html` | How the model gets start and end times (interactive demo) | If people ask how times are made |
| `report/abc2027_uwb_report.html` | Dataset, scoring rules and baselines | Persons new to the challenge |
| `docs/falljepa_vs_stage05.md` | Comparison with the Stage05 collaboration package | The persons who made the package |

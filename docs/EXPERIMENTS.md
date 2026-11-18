# Experiment notes

## Anchor-TCN development

The raw/EMA comparison source is retained in `experiments/anchor_tcn/anchor_tcn_compare.py`. The full-window experiment kept 800-frame (16-second) input windows and visited every indexed training window once per epoch. Hops 292, 80 and 40 gave validation mean F1 0.7733, 0.8060 and 0.7240. Changing hop also changed the number of optimization updates; this does not isolate overlap alone.

The hierarchical Transformer used local groups of 10 frames and 80 global tokens, with 770,179 parameters against the TCN's 774,339. Two complete saved runs are included: five epochs scored 0.7034, ten epochs scored 0.6917. The short PDF reports the latter. Normalization sampling also differed from the TCN comparison.

Extending TCN training to 30 epochs gave scores 0.8060, 0.7796, 0.7556, 0.7650, 0.7590 and 0.7566 at epochs 5, 10, 15, 20, 25 and 30. Epoch 5 was best among these evaluated checkpoints.

## Signal postprocessing

The selected sequence was confidence filtering, raw-CIR boundary refinement, a profile-change check, and overlap-based duplicate removal. Confidence was 0.51, initial/final NMS tIoU was 0.1, and the profile-change threshold was 2.12e-5. Mean F1 increased from 0.8060 to 0.9005 after boundary refinement, 0.9073 after the profile check, and 0.9141 after final NMS. At tIoU 0.3 the final result had 61 true positives, 4 false positives and 6 false negatives.

The included audit and prediction CSVs were extracted from the historical postprocessing result archive. The separate frozen-application rerun is described in the short report, but its intact original archive is unavailable. These remain development-validation results. Signal processing uses future context and does not establish real-time alarm latency.

## Anchor-TCN subject-held-out CV

Four shared subject-held-out folds were used, with seeds 2027 and 2028. Each job trained for five fixed full epochs on 26 training streams and evaluated val_00 through val_09. For each scored fold, decoder/postprocessing settings were selected using the other three folds. Checkpoint policy was final; no per-held-out-fold checkpoint selection was used.

| Processing method | Mean pooled F1 across seeds |
|---|---:|
| Fixed original decoder | 0.604078 |
| Tuned original decoder | 0.627906 |
| Tuned anchor intervals + refinement + gate + NMS | 0.718813 |
| Probability-run candidates + tuned signal processing | 0.736772 |
| Method selected using the other three folds | 0.732715 |

The selected procedure scored 0.716259 for seed 2027 and 0.749171 for seed 2028; the mean was 0.732715 and sample standard deviation 0.023272. The probability-run route was the best observed individual method overall, while 0.732715 measures the procedure that chooses a method using the other folds. Those are different reported quantities; the selected method can vary by target fold. All-fold deployment tuning is recorded separately and is not independent CV evidence.

This CV study provides an important generalization check on the earlier validation result. The two scores should not be described as a controlled 0.9141-to-0.7327 performance drop: data construction, held-out subjects, and selection protocol differ.

## FallJEPA/JF6 experiments with RadarViT

The supplied RadarViT encoder replaced RadarStem within JF6 (called JF5T in this checkout). The local TCN and subsequent temporal attention and event heads remained. Training used 15 epochs, seed 0, and four held-out folds, with four microbatches of two chunks per optimizer update and final averaged weights.

| Model | A | B | C | D | Fold mean |
|---|---:|---:|---:|---:|---:|
| JF6 | 0.9024 | 0.9159 | 0.9305 | 0.8443 | 0.8983 |
| RadarViT | 0.8877 | 0.9054 | 0.9167 | 0.8575 | 0.8918 |

The original `tools/ensemble_seeds.py` fused same-fold event CSVs at tIoU 0.3, with confidence-weighted mean boundaries. Keeping either model's detection gave fold mean 0.8993 (pooled 0.8981), 473 TP, 51 FP and 33 FN. Requiring both gave fold mean 0.9046 (pooled 0.9033), 441 TP, 11 FP and 65 FN. Counts use tIoU 0.3. The agreement rule was exploratory on the same held-out folds. A fused-score cutoff of 0.5 gave the same saved metrics in this run.

The four-seed training add-on packages the experimental JF6/RadarViT configuration for a prepared training release. Its four seeds refer to independent training runs on that release. The four-fold evaluation above used seed 0.

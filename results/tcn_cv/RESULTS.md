# TCN subject-held-out CV results

Four folds, seeds [2027, 2028], 5 full epochs each. Checkpoint policy: final.
Each model trained on its fold's 26 training streams and predicted val_00 through val_09.
The network, normalization and window settings were fixed before CV. For each scored fold, decoder and postprocessing selection used only the other three folds.

| Seed | Pooled mean F1 | A | B | C | D |
|---|---:|---:|---:|---:|---:|
| 2027 | 0.716259 | 0.624049 | 0.752778 | 0.632997 | 0.814554 |
| 2028 | 0.749171 | 0.682171 | 0.805031 | 0.619273 | 0.861423 |

Mean pooled F1 across seeds: **0.732715**; sample SD: **0.023272**.

| Processing stage | Mean pooled F1 across seeds | SD |
|---|---:|---:|
| tcn_fixed | 0.604078 | 0.010041 |
| tcn_tuned | 0.627906 | 0.000643 |
| tcn_fixed_refined | 0.664881 | 0.009349 |
| tcn_fixed_refined_gate | 0.666932 | 0.009898 |
| tcn_tuned_refined | 0.703623 | 0.002148 |
| tcn_tuned_refined_gate | 0.713553 | 0.008434 |
| tcn_tuned_refined_gate_nms | 0.718813 | 0.009126 |
| signal_runs_tuned | 0.736772 | 0.021325 |
| selected | 0.732715 | 0.023272 |

The ranking mean uses event F1 at tIoU 0.1, 0.3 and 0.5; 0.7 is diagnostic.
metrics_all.csv contains every saved official scalar metric at pooled, fold, stitching and stream level: precision, recall, F1, counts, false alarms/hour and boundary errors where available.
Full official JSON reports, all selection settings, sweeps, training losses and window audits are included in the ZIP.
The selected method can differ by target fold. The choice uses the other three folds, not the target result.
All-fold deployment tuning is saved separately and is not an independent CV score.
The earlier 0.9141 was obtained on one repeatedly tuned validation split. It is not directly comparable to the pooled subject-held-out CV score.

Total recorded training time across 8 runs: 77.57 minutes.
Total recorded inference time: 0.76 minutes.
These totals exclude data verification, preprocessing, postprocessing, tuning and report writing.

Recovery changed only report assembly after the duplicate seed keyword error. No models, predictions, thresholds or official scores were changed.
This is a CV evaluation; no hidden-test score or measured alarm latency is available.

# Frozen TCN: signal postprocessing

Same saved TCN predictions; no model training or checkpoint modification.

| Method | F1 .1 | F1 .3 | F1 .5 | Mean F1 | Precision .3 | Recall .3 | TP/FP/FN .3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tcn_fixed | 0.8455 | 0.8293 | 0.7154 | 0.7967 | 0.9107 | 0.7612 | 51/5/16 |
| tcn_tuned | 0.8657 | 0.8358 | 0.7164 | 0.8060 | 0.8358 | 0.8358 | 56/11/11 |
| tcn_fixed_refined | 0.8780 | 0.8780 | 0.8455 | 0.8672 | 0.9643 | 0.8060 | 54/2/13 |
| tcn_fixed_refined_gate | 0.8780 | 0.8780 | 0.8455 | 0.8672 | 0.9643 | 0.8060 | 54/2/13 |
| tcn_tuned_refined | 0.9104 | 0.9104 | 0.8806 | 0.9005 | 0.9104 | 0.9104 | 61/6/6 |
| tcn_tuned_refined_gate | 0.9173 | 0.9173 | 0.8872 | 0.9073 | 0.9242 | 0.9104 | 61/5/6 |
| tcn_tuned_refined_gate_nms | 0.9242 | 0.9242 | 0.8939 | 0.9141 | 0.9385 | 0.9104 | 61/4/6 |
| signal_runs_tuned | 0.9147 | 0.9147 | 0.8837 | 0.9044 | 0.9516 | 0.8806 | 59/3/8 |
| reference_center_5p9 | 0.9037 | 0.9037 | 0.8741 | 0.8938 | 0.8971 | 0.9104 | 61/7/6 |

Selected: **tcn_tuned_refined_gate_nms**; difference from tuned original TCN: **+0.1082** mean F1.

The 5.9-second center-suppression row is diagnostic and is excluded from automatic selection.
Refined anchor rows begin with the original selected detections and confidence scores; boundaries, profile gating and, in the final-nms row, duplicate suppression change.
The signal-runs row also changes candidate generation and merging. It does not use learned distance heads.
All reported values are validation development results. Tuning on these same streams makes them optimistic estimates of unseen performance.
The official JSON reports include precision, recall, F1, counts, boundary errors and false alarms/hour.
No measured alarm time is supplied. Any evaluator latency field is not measured runtime or a real-time guarantee.
Raw-CIR signal processing uses future context and reconstructed stream joins; performance on differently constructed streams is unmeasured.

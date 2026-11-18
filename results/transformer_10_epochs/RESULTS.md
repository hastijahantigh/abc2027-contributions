# ABC2027 hierarchical Transformer

Status: COMPLETE (1/1 models).

Each 16-second input has 80 local groups of 0.2 seconds. A local Transformer creates one token per group; a global Transformer processes all 80 tokens.
Frame-level heads combine local frame features with global context to predict confidence and start/end distances.
The input mode and all model settings are saved in run_config.json.
Every trainable indexed window is visited exactly once in each full epoch, including partial final minibatches.
Full validation streams are inferred with a 292-frame retained-core hop.

## Main challenge objective

Ranking mean = (event F1 at tIoU 0.1 + 0.3 + 0.5) / 3. No model or threshold is selected using tIoU 0.7.

## Fixed confidence 0.8 and NMS 0.1

| Hop | Windows/epoch | F1 @ .1 | F1 @ .3 | F1 @ .5 | Mean F1 | Precision @ .3 | Recall @ .3 | TP / FP / FN @ .3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 13888 | 0.727 | 0.694 | 0.529 | **0.650** | 0.778 | 0.627 | 42 / 12 / 25 |

## Validation-selected confidence and NMS

| Hop | Windows/epoch | F1 @ .1 | F1 @ .3 | F1 @ .5 | Mean F1 | Precision @ .3 | Recall @ .3 | TP / FP / FN @ .3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 13888 | 0.782 | 0.752 | 0.541 | **0.692** | 0.758 | 0.746 | 50 / 16 / 17 |

The sweep checks confidence 0.005 and 0.01 through 0.99 in steps of 0.01 with NMS 0.1, 0.3, 0.5, 0.7.
The complete official reports additionally include F1 at tIoU 0.7, mean tIoU, boundary errors, false alarms/hour, latency, and tolerance fractions.
These are validation development results, not hidden-test estimates.

## What was tested

Inputs contain 800 frames (16 seconds). Each 10-frame group (0.2 seconds) yields one local token.
The default 80-frame training hop gives 90% input overlap.
Only windows whose entire retained core is masked are excluded; their counts are recorded in the window plans.
CSV manifests store window coordinates and label counts; CIR is loaded on demand without saving duplicate windows.
Sampling is exhaustive and shuffled without replacement. There is no additional 50% event oversampling.

## Interpretation limits

This run keeps the earlier TCN's 16-second crop, 292-frame retained core, and 80-frame training hop, while changing the backbone and train-only normalization sampler.
There is one seed. Repeat runs are needed before claiming a reliable improvement.
The global Transformer uses bidirectional context. No real-time alarm latency claim is made.

For audit, inspect window_manifest.csv, window_plan.json, epoch_coverage.json and training_summary.json inside each run.
Return RESULTS_TO_SHARE.zip after completion.

## Observed mean F1

Best development mean: hop_080 = 0.6917.

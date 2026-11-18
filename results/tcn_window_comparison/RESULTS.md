# Full-window training experiment

Status: COMPLETE (3/3 models).

EMA and the original 774,339-parameter TCN are retained in every run.
Every trainable indexed window is visited exactly once in each full epoch, including partial final minibatches.
All runs use standard inference with the same 292-frame retained-core hop.

## Main challenge objective

Ranking mean = (event F1 at tIoU 0.1 + 0.3 + 0.5) / 3. No model or threshold is selected using tIoU 0.7.

## Same fixed confidence 0.8 and NMS 0.1

| Training hop | Windows/epoch | Total crop visits | Training batches | F1 @ .1 | F1 @ .3 | F1 @ .5 | Mean F1 | FP @ .3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 292 frames | 3808 | 19040 | 4760 | 0.000 | 0.000 | 0.000 | **0.000** | 0 |
| 80 frames | 13888 | 69440 | 17360 | 0.846 | 0.829 | 0.715 | **0.797** | 5 |
| 40 frames | 27763 | 138815 | 34705 | 0.810 | 0.744 | 0.545 | **0.700** | 9 |

## Same validation confidence/NMS sweep for every model

| Training hop | Windows/epoch | Total crop visits | Training batches | F1 @ .1 | F1 @ .3 | F1 @ .5 | Mean F1 | FP @ .3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 292 frames | 3808 | 19040 | 4760 | 0.848 | 0.800 | 0.672 | **0.773** | 8 |
| 80 frames | 13888 | 69440 | 17360 | 0.866 | 0.836 | 0.716 | **0.806** | 11 |
| 40 frames | 27763 | 138815 | 34705 | 0.844 | 0.766 | 0.562 | **0.724** | 12 |

The sweep checks confidence 0.005 and 0.01 through 0.99 in steps of 0.01 with NMS 0.1, 0.3, 0.5, 0.7.
These are validation-selected development results, not hidden-test estimates.

## What was tested

Inputs remain 800 frames (16 seconds). Hop 292 is the context-preserving reference: its retained cores tile time, while input windows already overlap 63.5%.
Hop 80 gives 90% input overlap. Hop 40 gives 95% input overlap.
The denser grids create larger indexed datasets. They are not randomly capped at 20,000 crop draws.
Only windows whose entire retained core is masked are excluded; their counts are recorded in the window plans.
CSV manifests store window coordinates and label counts; CIR is loaded on demand without saving duplicate windows.
Sampling is exhaustive and shuffled without replacement. There is no additional 50% event oversampling.

## Interpretation limits

The number of full epochs is fixed. Denser overlap therefore increases BOTH temporal training positions and optimization iterations.
This tests the supervisor's more-samples proposal; it does not isolate overlap from extra compute.
All grid models share this full-coverage training policy. Compare them to one another; the historical random-sampler score is not a matched baseline for this experiment.
The class proportions are measured in every manifest; full-stream training has more background exposure than the previous event-biased sampler.
There is one seed. Small differences need repeat runs before a general improvement is claimed.
The original GroupNorm and bidirectional context remain. No real-time latency claim is made.

For audit, inspect window_manifest.csv, window_plan.json, epoch_coverage.json and training_summary.json inside each run.
Return RESULTS_TO_SHARE.zip after completion.

## Observed mean F1

Best development mean: hop_080 = 0.8060.
Difference from hop 292 reference: +0.0326.

# Original AnchorTCN: longer training

Status: COMPLETE (6/6 checkpoints evaluated).

Original 774,339-parameter model; 800-frame inputs; 292-frame retained core/inference hop.
One shuffled complete pass per epoch, with original normalization, loss, optimizer and constant learning rate.
Intermediate validation does not change training or stop it early. All requested epochs run.

## Original decoder, fixed confidence 0.8 / NMS 0.1

| Checkpoint | Epochs | Crop visits | Batches | F1 .1 | F1 .3 | F1 .5 | Mean F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hop_080/epoch_005 | 5 | 69440 | 17360 | 0.8455 | 0.8293 | 0.7154 | 0.7967 |
| hop_080/epoch_010 | 10 | 138880 | 34720 | 0.8217 | 0.8217 | 0.6357 | 0.7597 |
| hop_080/epoch_015 | 15 | 208320 | 52080 | 0.7965 | 0.7788 | 0.6372 | 0.7375 |
| hop_080/epoch_020 | 20 | 277760 | 69440 | 0.8000 | 0.7826 | 0.6435 | 0.7420 |
| hop_080/epoch_025 | 25 | 347200 | 86800 | 0.8033 | 0.7705 | 0.6393 | 0.7377 |
| hop_080/epoch_030 | 30 | 416640 | 104160 | 0.7611 | 0.7434 | 0.5841 | 0.6962 |

## Original decoder, validation-selected confidence / NMS

| Checkpoint | Epochs | Crop visits | Batches | F1 .1 | F1 .3 | F1 .5 | Mean F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hop_080/epoch_005 | 5 | 69440 | 17360 | 0.8657 | 0.8358 | 0.7164 | 0.8060 |
| hop_080/epoch_010 | 10 | 138880 | 34720 | 0.8387 | 0.8387 | 0.6613 | 0.7796 |
| hop_080/epoch_015 | 15 | 208320 | 52080 | 0.8167 | 0.8000 | 0.6500 | 0.7556 |
| hop_080/epoch_020 | 20 | 277760 | 69440 | 0.8361 | 0.8033 | 0.6557 | 0.7650 |
| hop_080/epoch_025 | 25 | 347200 | 86800 | 0.8462 | 0.8000 | 0.6308 | 0.7590 |
| hop_080/epoch_030 | 30 | 416640 | 104160 | 0.8254 | 0.8095 | 0.6349 | 0.7566 |

Selection uses official mean event F1 at tIoU 0.1, 0.3, 0.5, then official tie-breaks.
The reports include precision, recall, TP/FP/FN and timing errors at each tIoU.
These are development scores on the same validation set used for selection, not hidden-test estimates.
best_checkpoint.json records the best original-decoder checkpoint so far. Checkpoint files are never replaced.
The separate signal postprocessor is not run here. Its earlier 0.9141 score is not a direct comparator for these decoder scores.
Every epoch directory contains last.pt and inference/ compatible with the existing tcn_signal_postprocess.py.
For a postprocessed comparison, apply the same frozen postprocessing configuration to each checkpoint.

Return RESULTS_TO_SHARE.zip. Checkpoints and frame arrays remain on the server and are excluded from that small archive.

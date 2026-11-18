
# ABC2027 radar fall detection experiments

**Experimental contributions by Hasti Jahantigh**

This repository documents the experiments I was assigned in the ABC2027 team project on continuous fall detection from three UWB radars. I ran the experiments, analyzed the results, and reported my findings to the team. My work covered Anchor-TCN, a hierarchical Transformer, signal-based postprocessing, subject-held-out cross-validation, and an experimental FallJEPA/JF6 track with RadarViT and prediction fusion.

The associated team paper is **FallPGR: Proposal Grouping and Refinement for Temporal Fall Localization in Continuous Multi-Radar UWB Streams**, which I co-authored. The paper describes the final team system and its ensemble. The FallJEPA/JF6 checkout here belongs to my assigned RadarViT experiments, which kept JF6’s local TCN. The final FallPGR system uses its own encoder variants and ensemble.

## Results at a glance

All F1 means use event F1 at temporal IoU 0.1, 0.3 and 0.5. Evaluation protocols differ between sections.

| Experiment | Evaluation | Mean F1 | Evidence |
|---|---|---:|---|
| Anchor-TCN, 90% input overlap | Original validation split, tuned decoder | 0.8060 | [Window comparison](results/tcn_window_comparison/RESULTS.md) |
| Hierarchical Transformer, 5 epochs | Original validation split | 0.7034 | [5-epoch results](results/transformer_5_epochs/RESULTS.md) |
| Hierarchical Transformer, 10 epochs | Original validation split | 0.6917 | [10-epoch results](results/transformer_10_epochs/RESULTS.md) |
| Anchor-TCN + signal postprocessing | Original validation split, development tuning | 0.9141 | [Postprocessing](results/tcn_postprocessing/RESULTS.md) |
| Anchor-TCN CV, method selection using other folds | Four folds; mean pooled score across 2 seeds | 0.7327 | [CV results](results/tcn_cv/RESULTS.md) |
| JF6 baseline | Four-fold mean, seed 0 | 0.8983 | [Fusion results](results/radarvit_jf6/ensemble_JF6_RadarViT_s0.json) |
| JF6 with RadarViT encoder | Four-fold mean, seed 0 | 0.8918 | [RadarViT results](results/radarvit_jf6/cv_JF6Radar_hasti-radar-v1-s0-micro4.json) |
| JF6 + RadarViT, either model | Four-fold mean, seed 0 | 0.8993 | [Fusion results](results/radarvit_jf6/ensemble_JF6_RadarViT_s0.json) |
| JF6 + RadarViT, agreement rule | Four-fold mean, exploratory | 0.9046 | [Fusion results](results/radarvit_jf6/ensemble_JF6_RadarViT_s0.json) |

The 0.9141 score was obtained on a repeatedly used validation split. It is not directly comparable with the subject-held-out CV scores and is not hidden-test performance. The complete CV method comparison and the distinction between selected and best observed methods are explained in [the experiment notes](docs/EXPERIMENTS.md).

## My contribution

I compared input preprocessing and training-window sampling, tested a hierarchical Transformer, adapted signal-based boundary refinement to Anchor-TCN outputs, and evaluated Anchor-TCN under the shared CV protocol. In the FallJEPA/JF6 study, I integrated and tested the supplied RadarViT encoder, evaluated its prediction fusion with JF6, and reported the results and limitations.

The FallJEPA/JF6 study uses an existing framework and an encoder supplied in the handover. Vision-TriRadar informed the signal postprocessing, and the challenge organizers supplied the annotation and evaluation helpers. The included code retains its original attribution. See [contribution and attribution](docs/CONTRIBUTION.md).

## Repository contents

- [Anchor-TCN and Transformer scripts](experiments/anchor_tcn/)
- [RadarViT experiment checkout](experiments/radarvit/falljepa/)
- [Experiment notes and CV interpretation](docs/EXPERIMENTS.md)
- [Original short reports](reports/)
- [Execution prerequisites and reproduction limits](docs/REPRODUCTION.md)
- [File provenance](provenance/files.json)

Saved result files are included for inspection. Dataset files, trained weights and the original RadarViT/JF6 prediction CSVs are not included. The repository scope is my assigned experiments and their available evidence.

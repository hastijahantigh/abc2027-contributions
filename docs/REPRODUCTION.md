# Using the source and saved results

## Inspect results without a server

Start with the root README, `docs/EXPERIMENTS.md`, and the result folders. CSV and JSON evidence can be inspected independently of training. Original reports and scripts preserve historical paths; those are provenance, not paths guaranteed to exist on a new machine.

## Anchor-TCN and Transformer

The standalone source snapshots are under `experiments/anchor_tcn/`. Training requires the original ABC2027 release and a suitable Python environment with GPU-enabled PyTorch, NumPy, SciPy and h5py. The release supplies `src/abc2027/evaluate.py` and `annotate.py`. Dataset paths must be configured for the local environment. Inspect each script's `--help` before running; this package preparation did not install dependencies or rerun model training.

`tcn_shared_cv_final.py` bundles its model, postprocessing and CV helper code. It expects the already prepared shared folds and the official release. Its default experiment is four folds, two seeds and five fixed epochs. `tcn_cv_finish_results.py` is the historical report-assembly recovery utility and is not required for a new training run unless that recovery is needed.

## RadarViT

The original experiment checkout is under `experiments/radarvit/falljepa/`. Use its `pyproject.toml`, `uv.lock` and `RADAR_RUN.md`. The historical fusion used the existing JF6 baseline predictions from a separate team checkout, not a newly trained baseline under the Hasti tag. Its script is `tools/ensemble_seeds.py`; supply your same-fold CSV paths through `--set`, together with `--fold-names A,B,C,D --iou 0.3 --thr 0.5`.

The original eight RadarViT/JF6 prediction CSVs are unavailable in this package, so the saved fusion scores cannot be independently regenerated from this package alone. The result JSONs and fusion implementation are included.

The `JF6_RadarViT_final/` directory contains the original four-seed training add-on for the experimental JF6/RadarViT configuration. Its README explains the required prepared final-release YAML. This checkout already contains RadarViT, so do not apply its patch twice.

## Evidence coverage

Included: experimental source snapshots; original Transformer and thirty-epoch TCN result contents; extracted TCN window and postprocessing evidence; TCN CV summaries, protocol, official reports and selections; RadarViT/fusion result JSONs; short reports.

Not included: datasets, caches, trained model weights, original RadarViT/JF6 CSV predictions, the intact original overlap/postprocessing archives, or the intact frozen-application result archive. The existing extracted evidence is preserved under its own name. Complete TCN CV window manifests and parameter-sweep tables were omitted from this compact repository; the locally retained original archive is identified by SHA-256 in `provenance/package_checks.json`.

Preparation checks validate file integrity, Python syntax, report-to-summary consistency and Markdown links. They do not constitute a fresh reproduction of training, inference or full evaluation.

The local CV launcher declares the same bundled payload hash as the recorded run, but its whole-file SHA-256 differs. The component source snapshots saved by the historical run are preserved under `provenance/tcn_cv_bundled_code/`.

# Hasti's JF6 range-attention experiment

Upload the single `hasti_range_v1.zip` archive. It contains this complete project, the original lockfile, the unmodified JF5T baseline config, and the new JF6Range config. The shared fold streams and official evaluator are read in place. Each model trains from scratch on its fold's training streams. The baseline summary JSONs in the handover cannot be used as ensemble members. The ensemble needs the two models' actual held-out event CSVs.

## Install in a new directory

From your computer, use your configured ABC27 SSH alias or substitute your actual login and port. The commands below assume the handover's account and Hasti directory.

```sh
ssh ABC27 "mkdir -p /home/abc2027/hasti"
scp hasti_range_v1.zip ABC27:/home/abc2027/hasti/
ssh ABC27
```

On ABC27, run the following in Bash. The guard prevents a second extraction over a running experiment.

```bash
cd /home/abc2027/hasti
if [ ! -e hasti_range_v1 ]; then unzip -q hasti_range_v1.zip; fi
cd hasti_range_v1
export PATH="/home/abc2027/.local/bin:$PATH"
uv sync --frozen
uv run --frozen pytest -q tests/test_range_attention.py 'tests/test_jf4.py::test_cli_jf4_fits_and_validates[JF6Range]'
```

The last test fits a small CPU model on the synthetic release, validates its SWA checkpoint and checks its event CSV. For the complete upstream suite use `uv run --frozen pytest -q`.

## Exercise the real training path

Use the shared GPU when available. The launcher defaults to GPU 0 and the two data paths below. Override `GPU`, `FOLDS_DIR`, or `OFFICIAL` in the shell if the server layout differs.

```bash
export FOLDS_DIR=/home/abc2027/dataset/abc2027_uwb_cv_folds
export OFFICIAL=/home/abc2027/dataset/abc2027_uwb_train_v1
ls "$FOLDS_DIR/D/groups.csv" "$OFFICIAL/src/abc2027/evaluate.py"
nvidia-smi
bash tools/hasti_range.sh smoke
```

The smoke run fits two real training batches at the configured batch size, including backpropagation. It uses its own run directory and does not score validation or produce a comparison SWA checkpoint. Successful completion verifies the training batch fits at that moment. Full validation and BN refresh still run during the longer job.

## Compare models on fold D

Run from the project root. This single background job trains the baseline, then the range variant, then scores their event ensemble. A failure stops the sequence. The default is fold D and seed 0.

```bash
mkdir -p runs
nohup bash -c 'set -e; for action in baseline range ensemble; do bash tools/hasti_range.sh "$action"; done' > runs/hasti-D-s0.log 2>&1 &
tail -f runs/hasti-D-s0.log
```

With the default settings, find the results in:

- `runs/_analysis/cv_JF5T_hasti-v1-s0.json` for JF6.
- `runs/_analysis/cv_JF6Range_hasti-v1-s0.json` for range attention.
- `runs/_analysis/ensemble_hasti-v1-s0-D.json` for the ensemble.
- `runs/UWB/JF6Range-D/JF6Range-s0/hasti-v1-s0/predictions/val-validate.csv` for the variant's event intervals.

Each training job uses the inherited 15 epochs, fold-specific training statistics, final SWA checkpoint and original decoder. The ensemble matches events at tIoU 0.3. Use the **votes>=1** row as the predeclared comparison, allowing a fall found by either detector to count. The script also prints agreement and score-threshold rows. Keep those diagnostic so that held-out labels do not select the fusion rule.

## All four folds and two seeds

After the first job finishes, run this from the same folder:

```bash
nohup bash -c 'set -e; export FOLDS="A B C D"; for SEED in 0 1; do export SEED; for action in baseline range ensemble; do bash tools/hasti_range.sh "$action"; done; done' > runs/hasti-4fold.log 2>&1 &
tail -f runs/hasti-4fold.log
```

The same tag lets the upstream runner reuse completed folds and resume interrupted fits with a `last.ckpt`. Run only one queue at a time. Four-fold scoring refreshes the baseline and range summary JSONs with the same names. Their per-fold entries include D. The ensemble summaries have distinct fold-list filenames. If a teammate shares verified baseline event CSVs from the same fold, seed, protocol and scorer, you may substitute those paths for the baseline in the included ensemble tool and skip baseline retraining.

## What changed

`configs/uwb/JF6Range.yaml` differs from `JF5T.yaml` in experiment metadata and the single model switch `range_attention: true`. Within `RangePool`, one existing transformer Block attends to the 23 fused range positions of each frame independently, after adding the range embedding once and before softmax pooling. It has width 64, four heads, MLP ratio 2 and dropout 0.1. The same TCN, three temporal transformer blocks and downstream heads consume its output. Baseline configs default to the original pooling path. Neither model needs DINOv2 weights.

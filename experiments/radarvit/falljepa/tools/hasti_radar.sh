#!/usr/bin/env bash
# Run from Hasti's unpacked project. Examples in HASTI_RUN.md.
set -euo pipefail
cd "$(dirname "$0")/.."
# The shared server can have several editable installs of the falljepa package.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/home/abc2027/.local/bin:$PATH"
export FOLDS_DIR=${FOLDS_DIR:-/home/abc2027/dataset/abc2027_uwb_cv_folds}
export OFFICIAL=${OFFICIAL:-/home/abc2027/dataset/abc2027_uwb_train_v1}
export GPU=${GPU:-0} SEED=${SEED:-0} FOLDS=${FOLDS:-D}
export RUNS="$PWD/runs"
mkdir -p "$RUNS"
split=${MICRO_SPLIT:-4}
case "$split" in 1|2|4|8) ;; *) echo "MICRO_SPLIT must be 1, 2, 4 or 8" >&2; exit 2 ;; esac
tag="hasti-radar-v1-s$SEED"
if [[ $split != 1 ]]; then tag="$tag-micro$split"; fi
batch_args=(--data.sampler.split "$split" --trainer.accumulate_grad_batches "$split"
            --data.eval_batch_size "$((32 / split))")

case "${1:-}" in
  smoke)
    CUDA_VISIBLE_DEVICES="$GPU" uv run --frozen python run.py fit -c configs/uwb/JF6Radar.yaml \
      --data.release "$FOLDS_DIR/D" --data.sampler.groups "$FOLDS_DIR/D/groups.csv" \
      --data.stats_path "$RUNS/_cache/fold_D/normalizer.json" \
      "${batch_args[@]}" --trainer.accelerator gpu --trainer.max_epochs 1 --trainer.limit_train_batches "$((2 * split))" \
      --trainer.limit_val_batches 0 --run.wandb disabled --run.root "$RUNS" \
      --run.group hasti-smoke --run.name JF6Radar --run.id "smoke-$(date +%Y%m%d-%H%M%S)"
    ;;
  baseline|radar)
    if [[ $1 == baseline ]]; then name=JF5T; else name=JF6Radar; fi
    CONFIG="configs/uwb/$name.yaml" NAME="$name" GROUP="UWB/$name" TAG="$tag" \
      bash tools/run_jf_cv.sh -- --trainer.accelerator gpu "${batch_args[@]}"
    ;;
  ensemble)
    read -ra folds <<< "$FOLDS"
    fold_list=$(IFS=,; echo "${folds[*]}")
    base="$RUNS/UWB/JF5T-{fold}/JF5T-s$SEED/$tag/predictions/val-validate.csv"
    range="$RUNS/UWB/JF6Radar-{fold}/JF6Radar-s$SEED/$tag/predictions/val-validate.csv"
    uv run --frozen python tools/ensemble_seeds.py --release "$OFFICIAL" --folds "$FOLDS_DIR" \
      --fold-names "$fold_list" --iou 0.3 --thr 0.5 \
      --set "JF6+radar=$base,$range" \
      --out "$RUNS/_analysis/ensemble_$tag-$fold_list.json"
    ;;
  *) echo "Usage: bash tools/hasti_radar.sh {smoke|baseline|radar|ensemble}" >&2; exit 2 ;;
esac

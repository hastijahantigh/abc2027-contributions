#!/usr/bin/env bash
# Six-fold recording-grouped cross-validation of one config, then the decoder tuned out of fold.
#
#   tools/run_cv.sh configs/uwb/U0.yaml [extra run.py arguments...]
#   RELEASE=/path/to/abc2027_uwb_train_v1 GPUS="0 1" tools/run_cv.sh configs/uwb/U0.yaml
#   NAME=U0-noJEPA tools/run_cv.sh configs/uwb/U0.yaml --model.loss_weights.MSE_Latent 0
#
# Folds run in parallel, one per GPU in GPUS (default "0"); FOLDS picks folds (default "0 1 2 3 4 5"). Each fold fits, then validates its last
# checkpoint, which writes predictions/val-validate.probs.npz; tune_decoder reads all six.
set -euo pipefail
CONFIG=$1; shift
NAME=${NAME:-$(basename "$CONFIG" .yaml)}
RELEASE=${RELEASE:-DATA/abc2027_uwb_train_v1}
GPUS=(${GPUS:-0})
FOLDS=(${FOLDS:-0 1 2 3 4 5})
TAG=$(date +%Y%m%d-%H%M%S)

run_fold() {
  local fold=$1 gpu=$2; shift 2       # the rest are run.py arguments
  local name="$NAME-cv$fold"
  CUDA_VISIBLE_DEVICES=$gpu uv run python run.py fit -c "$CONFIG" -c configs/uwb/cv.yaml \
    --data.release "$RELEASE" --data.cv_fold "$fold" --run.name "$name" --run.id "$TAG" "$@"
  CUDA_VISIBLE_DEVICES=$gpu uv run python run.py validate -c "runs/UWB/CV/$name/$TAG/config.yaml" --ckpt_path best
}

# One fold per GPU at a time; a failed fold stops the run before the decoder is tuned
failed=0
for ((i = 0; i < ${#FOLDS[@]}; i += ${#GPUS[@]})); do
  pids=()
  for ((j = 0; j < ${#GPUS[@]} && i + j < ${#FOLDS[@]}; j++)); do
    fold=${FOLDS[$((i + j))]}
    echo "[run_cv] $NAME fold $fold on GPU ${GPUS[$j]} (runs/UWB/CV/$NAME-cv$fold/$TAG)"
    run_fold "$fold" "${GPUS[$j]}" "$@" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
  done
done
if ((failed > 0)); then
  echo "[run_cv] $failed fold(s) failed; see the output above" >&2
  exit 1
fi

uv run python -m falljepa.tune_decoder --events "$RELEASE/streams_events.csv" \
  --probs runs/UWB/CV/"$NAME"-cv*/"$TAG"/predictions/val-validate.probs.npz \
  --out "runs/UWB/CV/$NAME-decoder-$TAG.json"
echo "[run_cv] decoder: runs/UWB/CV/$NAME-decoder-$TAG.json"

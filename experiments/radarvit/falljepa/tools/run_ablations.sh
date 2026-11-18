#!/usr/bin/env bash
# Train ablations on all of train, score them on the official val streams, several seeds each.
#
#   tools/run_ablations.sh A0 A1 A1b [-- extra run.py arguments...]
#   RELEASE=/path/to/abc2027_uwb_train_v1 GPUS="0 1 2" SEEDS="0 1 2" tools/run_ablations.sh A0 A1 A1b
#   SEEDS="0 1" tools/run_ablations.sh A1+A2 A1+A2c       # "+" stacks overlays: A1, then A2
#   BASE=configs/uwb/AP.yaml GROUP=UWB/AP CKPT=swa tools/run_ablations.sh JEPA noJEPA
#
# BASE: the base config(s) the overlays go on (default: U0.yaml and ablation.yaml); GROUP: the run
# group (default UWB/Ablation); CKPT: the checkpoint validated after fit, "last" (default) or "swa"
# (checkpoints/swa.ckpt, written by the WeightAveraging callback).
#
# Each spec names overlays in configs/uwb/ablations/. U0.yaml, ablation.yaml and the overlays are
# deep-merged by tools/compose_config.py into <run dir>/composed.yaml, which is what fit reads (so
# overlays stack: one that changes a loss keeps the losses an earlier one removed). Every (spec, seed)
# is one run, $RUNS/UWB/Ablation/<spec>-s<seed>/<TAG> (RUNS defaults to runs): it fits, then
# validates its last checkpoint, which writes predictions/val-validate.probs.npz. The runs are queued
# over GPUS, one at a time per GPU. Compare them with tools/compare_ablations.py.
#
# The whole script is one { ... } block, so bash parses it before running: editing this file (a git
# pull) while it runs does not change a running queue.
{
set -euo pipefail
SPECS=()
while (($#)) && [[ $1 != -- ]]; do SPECS+=("$1"); shift; done
[[ ${1:-} == -- ]] && shift
((${#SPECS[@]})) || { echo "usage: $0 SPEC... [-- run.py arguments]" >&2; exit 2; }
RELEASE=${RELEASE:-DATA/abc2027_uwb_train_v1}
GPUS=(${GPUS:-0})
SEEDS=(${SEEDS:-0 1})
TAG=${TAG:-$(date +%Y%m%d-%H%M%S)}
RUNS=${RUNS:-runs}
BASE=(${BASE:-configs/uwb/U0.yaml configs/uwb/ablation.yaml})
GROUP=${GROUP:-UWB/Ablation}
CKPT=${CKPT:-last}

overlays() {  # A1+A2 -> configs/uwb/ablations/A1.yaml configs/uwb/ablations/A2.yaml, one per line
  local part
  IFS=+ read -ra parts <<< "$1"
  for part in "${parts[@]}"; do
    [[ -f configs/uwb/ablations/$part.yaml ]] || { echo "no configs/uwb/ablations/$part.yaml" >&2; return 1; }
    echo "configs/uwb/ablations/$part.yaml"
  done
}

# Check every spec before anything starts
for spec in "${SPECS[@]}"; do overlays "$spec" > /dev/null; done

JOBS=()
for spec in "${SPECS[@]}"; do
  for seed in "${SEEDS[@]}"; do JOBS+=("$spec $seed"); done
done

run_job() {
  local spec=$1 seed=$2 gpu=$3; shift 3   # the rest are run.py arguments
  local name="$spec-s$seed"
  local dir="$RUNS/$GROUP/$name/$TAG"
  local ckpt=$CKPT
  [[ $CKPT == swa ]] && ckpt="$dir/checkpoints/swa.ckpt"
  local -a files=()
  local line
  while IFS= read -r line; do files+=("$line"); done < <(overlays "$spec")
  # Run again with the same TAG: finished runs are skipped, interrupted ones resume from last.ckpt
  if [[ -f $dir/predictions/val-validate.probs.npz ]]; then
    echo "[run_ablations] $name: already done ($dir)"
    return 0
  fi
  if [[ -f $dir/checkpoints/last.ckpt ]]; then
    echo "[run_ablations] $name on GPU $gpu: resuming from last.ckpt ($dir)"
    CUDA_VISIBLE_DEVICES=$gpu uv run python run.py fit -c "$dir/config.yaml" --ckpt_path last &&
    CUDA_VISIBLE_DEVICES=$gpu uv run python run.py validate -c "$dir/config.yaml" --ckpt_path "$ckpt"
    return
  fi
  echo "[run_ablations] $name on GPU $gpu ($dir)"
  uv run python tools/compose_config.py "$dir/composed.yaml" "${BASE[@]}" "${files[@]}" &&
  CUDA_VISIBLE_DEVICES=$gpu uv run python run.py fit -c "$dir/composed.yaml" \
    --data.release "$RELEASE" --seed_everything "$seed" --data.seed "$seed" \
    --run.root "$RUNS" --run.group "$GROUP" --run.name "$name" --run.id "$TAG" "$@" &&
  CUDA_VISIBLE_DEVICES=$gpu uv run python run.py validate -c "$dir/config.yaml" --ckpt_path "$ckpt"
}

# GPU j runs jobs j, j + n, j + 2n, ... in order; a failed job does not stop the others on its GPU
worker() {
  local j=$1; shift
  local failed=0 i
  for ((i = j; i < ${#JOBS[@]}; i += ${#GPUS[@]})); do
    run_job ${JOBS[$i]} "${GPUS[$j]}" "$@" || { echo "[run_ablations] FAILED: ${JOBS[$i]}" >&2; failed=$((failed + 1)); }
  done
  return $failed
}

echo "[run_ablations] ${#JOBS[@]} runs on GPUs ${GPUS[*]}, tag $TAG"
pids=()
for ((j = 0; j < ${#GPUS[@]}; j++)); do
  worker "$j" "$@" &
  pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || { echo "[run_ablations] some runs failed; see the output above" >&2; exit 1; }
echo "[run_ablations] done: uv run python tools/compare_ablations.py --root $RUNS/$GROUP --tag $TAG"
exit 0
}

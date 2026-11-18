#!/usr/bin/env bash
# The recording-head model (configs/uwb/RS.yaml) on the subject-held-out CV: every fold trains once
# (fixed budget, dense SWA, nothing selected on its held-out streams), validates checkpoints/swa.ckpt,
# and the folds are scored pooled, per fold and per stitching (tools/score_folds.py), twice: the
# recording head's own events, and Amir's chain on the same model's dense head.
#
#   FOLDS_DIR=/home/abc2027/dataset/abc2027_uwb_cv_folds OFFICIAL=/home/abc2027/dataset/abc2027_uwb_train_v1 \
#     GPU=0 nohup tools/run_rs_cv.sh > runs/rs_cv.log 2>&1 &
#
# FOLDS (default "A B C D"), SEED (0), TAG (a timestamp; pass the same TAG again to resume: finished
# folds are skipped, an interrupted one resumes from last.ckpt), RUNS (runs). Extra arguments go to
# run.py fit, e.g. -- --data.batch_size 16 --data.sampler.fall_slots 8 --data.sampler.hard_slots 2
{
set -euo pipefail
[[ ${1:-} == -- ]] && shift
FOLDS_DIR=${FOLDS_DIR:?set FOLDS_DIR to the built folds (tools/build_fold_streams.py --out)}
OFFICIAL=${OFFICIAL:?set OFFICIAL to the official package (its src/abc2027/annotate.py and evaluate.py)}
FOLDS=(${FOLDS:-A B C D})
GPU=${GPU:-0}
SEED=${SEED:-0}
TAG=${TAG:-$(date +%Y%m%d-%H%M%S)}
RUNS=${RUNS:-runs}
GROUP=UWB/RS
NAME=RS-s$SEED
echo "[run_rs_cv] tag $TAG, folds ${FOLDS[*]}, GPU $GPU"

SCORE_RS=() SCORE_AMIR=()
for F in "${FOLDS[@]}"; do
  dir="$RUNS/$GROUP-$F/$NAME/$TAG"
  if [[ ! -f $dir/predictions/val-validate.csv ]]; then
    if [[ -f $dir/checkpoints/last.ckpt ]]; then
      echo "[run_rs_cv] fold $F: resuming ($dir)"
      CUDA_VISIBLE_DEVICES=$GPU uv run python run.py fit -c "$dir/config.yaml" --ckpt_path last
    else
      echo "[run_rs_cv] fold $F: training ($dir)"
      CUDA_VISIBLE_DEVICES=$GPU uv run python run.py fit -c configs/uwb/RS.yaml \
        --data.release "$FOLDS_DIR/$F" --data.annotate_release "$OFFICIAL" \
        --data.stats_path "$RUNS/_cache/fold_$F/normalizer.json" --data.sampler.groups "$FOLDS_DIR/$F/groups.csv" \
        --seed_everything "$SEED" --data.seed "$SEED" \
        --run.root "$RUNS" --run.group "$GROUP-$F" --run.name "$NAME" --run.id "$TAG" "$@"
    fi
    CUDA_VISIBLE_DEVICES=$GPU uv run python run.py validate -c "$dir/config.yaml" --ckpt_path "$dir/checkpoints/swa.ckpt"
  fi
  SCORE_RS+=(--fold "$F=$FOLDS_DIR/$F:$dir/predictions/val-validate.csv")
  SCORE_AMIR+=(--fold "$F=$FOLDS_DIR/$F:$dir/predictions/val-validate.amir.csv")
done

mkdir -p "$RUNS/_analysis"
echo "[run_rs_cv] recording head"
uv run python tools/score_folds.py --release "$OFFICIAL" --out "$RUNS/_analysis/cv_RS_$TAG.json" "${SCORE_RS[@]}"
echo "[run_rs_cv] Amir's chain on the dense head of the same runs"
uv run python tools/score_folds.py --release "$OFFICIAL" --out "$RUNS/_analysis/cv_RS-amir_$TAG.json" "${SCORE_AMIR[@]}"
}

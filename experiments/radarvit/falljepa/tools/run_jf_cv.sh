#!/usr/bin/env bash
# The join-free model (configs/uwb/JF.yaml, docs/falljepa_join_free.md) on the subject-held-out CV:
# every fold trains once (fixed budget, dense SWA, nothing selected on its held-out streams), validates
# checkpoints/swa.ckpt, and the folds are scored pooled, per fold and per stitching (tools/score_folds.py)
# three times: the full model, stage 1 with the non-learned merge (.queries) and the dense head (.dense).
#
#   FOLDS_DIR=/home/abc2027/dataset/abc2027_uwb_cv_folds OFFICIAL=/home/abc2027/dataset/abc2027_uwb_train_v1 \
#     GPU=0 nohup tools/run_jf_cv.sh > runs/jf_cv.log 2>&1 &
#
# FOLDS (default "A B C D"), SEED (0), TAG (a timestamp; pass the same TAG again to resume: finished
# folds are skipped, an interrupted one resumes from last.ckpt), RUNS (runs). Extra arguments go to
# run.py fit, e.g. -- --model.model.init_args.event_physics false
# JF2: CONFIG=configs/uwb/JF2.yaml NAME=JF2 GROUP=UWB/JF2. CACHE suffixes the normalizer / physics cache
# folder (fold_$F$CACHE), for folds built differently (e.g. more training stitchings)
{
set -euo pipefail
[[ ${1:-} == -- ]] && shift
FOLDS_DIR=${FOLDS_DIR:?set FOLDS_DIR to the built folds (tools/build_fold_streams.py --out)}
OFFICIAL=${OFFICIAL:?set OFFICIAL to the official package (its src/abc2027/evaluate.py scores the folds)}
FOLDS=(${FOLDS:-A B C D})
GPU=${GPU:-0}
SEED=${SEED:-0}
TAG=${TAG:-$(date +%Y%m%d-%H%M%S)}
RUNS=${RUNS:-runs}
CONFIG=${CONFIG:-configs/uwb/JF.yaml}
GROUP=${GROUP:-UWB/JF}
CACHE=${CACHE:-}
# Fewer reserved-but-unused blocks when the stream stage's group count changes every batch (6.4 GB peak)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
NAME=${NAME:-JF}-s$SEED
echo "[run_jf_cv] $CONFIG, tag $TAG, folds ${FOLDS[*]}, GPU $GPU"

SCORE=() SCORE_Q=() SCORE_D=()
for F in "${FOLDS[@]}"; do
  dir="$RUNS/$GROUP-$F/$NAME/$TAG"
  if [[ ! -f $dir/predictions/val-validate.csv ]]; then
    if [[ -f $dir/checkpoints/last.ckpt ]]; then
      echo "[run_jf_cv] fold $F: resuming ($dir)"
      CUDA_VISIBLE_DEVICES=$GPU uv run python run.py fit -c "$dir/config.yaml" --ckpt_path last
    else
      echo "[run_jf_cv] fold $F: training ($dir)"
      CUDA_VISIBLE_DEVICES=$GPU uv run python run.py fit -c "$CONFIG" \
        --data.release "$FOLDS_DIR/$F" \
        --data.stats_path "$RUNS/_cache/fold_$F$CACHE/normalizer.json" --data.sampler.groups "$FOLDS_DIR/$F/groups.csv" \
        --seed_everything "$SEED" --data.seed "$SEED" \
        --run.root "$RUNS" --run.group "$GROUP-$F" --run.name "$NAME" --run.id "$TAG" "$@"
    fi
    CUDA_VISIBLE_DEVICES=$GPU uv run python run.py validate -c "$dir/config.yaml" --ckpt_path "$dir/checkpoints/swa.ckpt"
  fi
  SCORE+=(--fold "$F=$FOLDS_DIR/$F:$dir/predictions/val-validate.csv")
  SCORE_Q+=(--fold "$F=$FOLDS_DIR/$F:$dir/predictions/val-validate.queries.csv")
  SCORE_D+=(--fold "$F=$FOLDS_DIR/$F:$dir/predictions/val-validate.dense.csv")
done

mkdir -p "$RUNS/_analysis"
echo "[run_jf_cv] full model (stage 1 + stream stage)"
uv run python tools/score_folds.py --release "$OFFICIAL" --out "$RUNS/_analysis/cv_${NAME%-s*}_$TAG.json" "${SCORE[@]}"
echo "[run_jf_cv] stage 1, non-learned merge"
uv run python tools/score_folds.py --release "$OFFICIAL" --out "$RUNS/_analysis/cv_${NAME%-s*}-queries_$TAG.json" "${SCORE_Q[@]}"
echo "[run_jf_cv] dense head, Viterbi (join-free baseline)"
uv run python tools/score_folds.py --release "$OFFICIAL" --out "$RUNS/_analysis/cv_${NAME%-s*}-dense_$TAG.json" "${SCORE_D[@]}"
}

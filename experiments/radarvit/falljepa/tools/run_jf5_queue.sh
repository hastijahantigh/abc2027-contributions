#!/usr/bin/env bash
# JF5 (configs/uwb/JF5.yaml) on the 4 CV folds, seeds 0 and 1, one run at a time (tools/run_jf_cv.sh).
# Seed 0 runs all folds first, so a 1-seed CV score exists after about half the queue.
# Logs: runs/jf5_s<seed>.log; scores: runs/_analysis/cv_JF5*_20261030-jf5-s<seed>.json.
cd "$(dirname "$0")/.."
export PATH=/home/abc2027/.local/bin:$PATH
export FOLDS_DIR=/home/abc2027/dataset/abc2027_uwb_cv_folds OFFICIAL=/home/abc2027/dataset/abc2027_uwb_train_v1
export FOLDS="A B C D" GPU=0
for SEED in 0 1; do
  TAG=20261030-jf5-s$SEED
  CONFIG=configs/uwb/JF5.yaml NAME=JF5 GROUP=UWB/JF5 SEED=$SEED TAG=$TAG tools/run_jf_cv.sh > runs/jf5_s$SEED.log 2>&1
  echo "[jf5_queue] seed $SEED tag $TAG exit $? $(date -u)"
done

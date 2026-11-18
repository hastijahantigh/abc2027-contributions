#!/usr/bin/env bash
# JF7a then JF7b (docs/falljepa_JF7_design.md: JF6 + pair-tIoU span loss, slot quality head, member channels;
# JF7b adds the boundary refinement stage) on the 4 CV folds, seeds 0 and 1, one run at a time
# (tools/run_jf_cv.sh). Logs: runs/jf7{a,b}_s<seed>.log; scores: runs/_analysis/cv_JF7{a,b}*_20261030-jf7{a,b}-s<seed>.json.
cd "$(dirname "$0")/.."
export PATH=/home/abc2027/.local/bin:$PATH
export FOLDS_DIR=/home/abc2027/dataset/abc2027_uwb_cv_folds OFFICIAL=/home/abc2027/dataset/abc2027_uwb_train_v1
export FOLDS="A B C D" GPU=0
for V in a b; do
  for SEED in 0 1; do
    TAG=20261030-jf7$V-s$SEED
    CONFIG=configs/uwb/JF7$V.yaml NAME=JF7$V GROUP=UWB/JF7$V SEED=$SEED TAG=$TAG tools/run_jf_cv.sh > runs/jf7${V}_s$SEED.log 2>&1
    echo "[jf7_queue] JF7$V seed $SEED tag $TAG exit $? $(date -u)"
  done
done

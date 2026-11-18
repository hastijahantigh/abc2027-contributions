#!/usr/bin/env bash
# JF5-T (configs/uwb/JF5T.yaml: JF5 + TCN/RoPE, no JEPA predictor) on the 4 CV folds, seeds 0 and 1,
# one run at a time (tools/run_jf_cv.sh). With WAIT_PID it starts once that process (the JF5 queue) exits.
# Logs: runs/jf5t_s<seed>.log; scores: runs/_analysis/cv_JF5T*_20261030-jf5t-s<seed>.json.
cd "$(dirname "$0")/.."
export PATH=/home/abc2027/.local/bin:$PATH
export FOLDS_DIR=/home/abc2027/dataset/abc2027_uwb_cv_folds OFFICIAL=/home/abc2027/dataset/abc2027_uwb_train_v1
export FOLDS="A B C D" GPU=0
if [[ -n ${WAIT_PID:-} ]]; then
  echo "[jf5t_queue] waiting for PID $WAIT_PID $(date -u)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
for SEED in 0 1; do
  TAG=20261030-jf5t-s$SEED
  CONFIG=configs/uwb/JF5T.yaml NAME=JF5T GROUP=UWB/JF5T SEED=$SEED TAG=$TAG tools/run_jf_cv.sh > runs/jf5t_s$SEED.log 2>&1
  echo "[jf5t_queue] seed $SEED tag $TAG exit $? $(date -u)"
done

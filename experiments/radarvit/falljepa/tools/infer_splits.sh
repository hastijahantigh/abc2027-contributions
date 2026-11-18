#!/usr/bin/env bash
# Run a trained run's checkpoint(s) on whole streams of any split, the way validation does (grid
# windows, merged probabilities, Viterbi, official scorer with --ignore-flagged), so the training
# streams can be analysed like val. Run it on the server, from the repo root:
#
#   RUN=runs/UWB/Frame/U0/20261009-134854-7b9c tools/infer_splits.sh
#   RUN=... SPLITS="train val" CKPTS="best last" GPU=0 tools/infer_splits.sh
#
# Writes, for each split and checkpoint, <run>/predictions/<split>-<ckpt>.{csv,report.json,probs.npz}
# (run.py test names them test-test.*; they are renamed here). Then fetch them with tools/pull_sync.sh
# and run tools/analyze_falls.py (or tools/analyze_run.py --stage train-best).
set -euo pipefail
: "${RUN:?set RUN to the run folder, e.g. runs/UWB/Frame/U0/<id>}"
SPLITS=(${SPLITS:-train val})
CKPTS=(${CKPTS:-best last})
GPU=${GPU:-0}
PRED="$RUN/predictions"

for ckpt in "${CKPTS[@]}"; do
  for split in "${SPLITS[@]}"; do
    echo "[infer_splits] $RUN: $split with $ckpt"
    rm -f "$PRED"/test-test.csv "$PRED"/test-test.report.json "$PRED"/test-test.probs.npz
    CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled uv run python run.py test -c "$RUN/config.yaml" \
      --ckpt_path "$ckpt" --data.test_splits "[$split]" --data.cv_fold null --data.cv_assignment null
    for ext in csv report.json probs.npz; do
      mv "$PRED/test-test.$ext" "$PRED/$split-$ckpt.$ext"
    done
    python3 -c "import json,sys; r=json.load(open(sys.argv[1])); b=r['by_theta']['0.3']; \
print(f\"[infer_splits] {sys.argv[2]}: F1 mean {r['ranking_score']['f1_mean']:.3f}, tp {b['tp']} fp {b['fp']} fn {b['fn']}\")" \
      "$PRED/$split-$ckpt.report.json" "$split-$ckpt"
  done
done

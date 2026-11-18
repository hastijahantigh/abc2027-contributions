#!/usr/bin/env bash
# Stage 1 of the ablation plan, then stage 2 on its winner, unattended (docs/falljepa_ablation_plan.md):
#
#   RELEASE=/path/to/abc2027_uwb_train_v1 GPUS="0 1 2" nohup tools/run_stages.sh > runs/stages.log 2>&1 &
#
# 1. Stage 1: every spec of STAGE1 (default "A0 A1 A1b") × STAGE1_SEEDS (default "0 1 2"), through
#    tools/run_ablations.sh. If any run fails, the script stops: no winner is chosen from incomplete runs.
# 2. The winner: the stage-1 spec with the highest mean F1_mean over the last 30 epochs, mean over
#    seeds (tools/compare_ablations.py --best). It goes to $RUNS/UWB/Ablation/stages-<TAG>.best, the
#    stage-1 table to stages-<TAG>.stage1.md.
# 3. Stage 2: <winner>+X for every X of STAGE2 (default "A2 A2c A3 A4a A4b A4-cs") × STAGE2_SEEDS
#    (default "0 1"). An X whose overlay does not exist yet (when stage 2 starts) is skipped with a
#    message; it can be run later with the same TAG. The table of every run, Δ to the winner, goes to
#    stages-<TAG>.stage2.md.
#
# All runs share one TAG, so the winner's stage-1 seeds are stage 2's reference. Run again with the
# same TAG to continue after a crash: finished runs are skipped and interrupted ones resume from
# last.ckpt (tools/run_ablations.sh). Arguments after `--` go to every fit.
{
set -euo pipefail
[[ ${1:-} == -- ]] && shift
export RELEASE=${RELEASE:-DATA/abc2027_uwb_train_v1}
export GPUS=${GPUS:-0}
export RUNS=${RUNS:-runs}
export TAG=${TAG:-$(date +%Y%m%d-%H%M%S)}
STAGE1=(${STAGE1:-A0 A1 A1b})
STAGE2=(${STAGE2:-A2 A2c A3 A4a A4b A4-cs})
STAGE1_SEEDS=${STAGE1_SEEDS:-0 1 2}
STAGE2_SEEDS=${STAGE2_SEEDS:-0 1}
OUT="$RUNS/UWB/Ablation/stages-$TAG"
mkdir -p "$RUNS/UWB/Ablation"
compare() { uv run python tools/compare_ablations.py --root "$RUNS/UWB/Ablation" --tag "$TAG" --events "$RELEASE/streams_events.csv" "$@"; }

echo "[run_stages] $(date '+%F %T') stage 1: ${STAGE1[*]} × seeds $STAGE1_SEEDS, tag $TAG"
if ! SEEDS=$STAGE1_SEEDS tools/run_ablations.sh "${STAGE1[@]}" -- "$@"; then
  echo "[run_stages] stage 1 had failed runs: no winner chosen, stage 2 not started." >&2
  echo "[run_stages] fix them, then run again with TAG=$TAG to continue." >&2
  exit 1
fi

best=$(compare --specs "${STAGE1[@]}" --best)
echo "$best" > "$OUT.best"
compare --specs "${STAGE1[@]}" --ref "$best" | tee "$OUT.stage1.md"
echo "[run_stages] $(date '+%F %T') stage 1 winner: $best"

specs=()
for x in "${STAGE2[@]}"; do
  missing=""
  IFS=+ read -ra parts <<< "$x"
  for part in "${parts[@]}"; do
    [[ -f configs/uwb/ablations/$part.yaml ]] || missing+=" configs/uwb/ablations/$part.yaml"
  done
  if [[ -n $missing ]]; then
    echo "[run_stages] skipped $x: no$missing. Later: TAG=$TAG SEEDS=\"$STAGE2_SEEDS\" tools/run_ablations.sh $best+$x" >&2
  else
    specs+=("$best+$x")
  fi
done
((${#specs[@]})) || { echo "[run_stages] no stage-2 overlay exists; stopping." >&2; exit 1; }

echo "[run_stages] $(date '+%F %T') stage 2: ${specs[*]} × seeds $STAGE2_SEEDS"
status=0
SEEDS=$STAGE2_SEEDS tools/run_ablations.sh "${specs[@]}" -- "$@" || status=$?
compare --ref "$best" | tee "$OUT.stage2.md"
echo "[run_stages] $(date '+%F %T') done (stage 2 exit $status): $OUT.stage1.md, $OUT.stage2.md"
exit $status
}

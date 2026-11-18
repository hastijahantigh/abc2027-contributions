#!/bin/bash
# For a training server that cannot reach W&B: copies everything the analysis of its runs needs into
# the local runs/ with the same layout, then uploads the W&B runs with wandb_sync.sh. Run it from
# the repo root on a machine that can reach W&B (after `uv run wandb login` once); run it again to
# fetch and upload what was logged since.
#
#   tools/pull_sync.sh user@server:/home/abc2027/mehdi [--port N] [--dataset /path/on/server] [--no-sync] [local-runs-root]
#
# Copied: every run folder (config.yaml, meta/, csv/ metrics, predictions/ with the submission CSVs,
# the official reports and the per-timestep probabilities, wandb/), the logs under runs/
# (*.out), the normalizer caches, and the dataset's streams_events.csv and
# streams_composition.csv into runs/_dataset/. Never copied: checkpoints (*.ckpt) and the streams.
# --dataset defaults to /home/abc2027/dataset/abc2027_uwb_train_v1. --port is the server's SSH port
# (or set it once in ~/.ssh/config).

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 user@server:/path/to/project [--port N] [--dataset /path/on/server] [--no-sync] [local-runs-root]" >&2
    exit 1
fi

SOURCE="${1%/}"; shift
HOST=""                                  # a local path (no "host:") also works, for testing
if [[ "$SOURCE" == *:* ]]; then HOST="${SOURCE%%:*}:"; fi
DATASET=/home/abc2027/dataset/abc2027_uwb_train_v1
SYNC=1
LOCAL=runs
SSH=(-e ssh)
while [ $# -gt 0 ]; do
    case "$1" in
        --dataset) DATASET="${2%/}"; shift ;;
        --port) SSH=(-e "ssh -p $2"); shift ;;
        --no-sync) SYNC=0 ;;
        *) LOCAL="$1" ;;
    esac
    shift
done

mkdir -p "$LOCAL/_dataset"
rsync -az --prune-empty-dirs "${SSH[@]}" \
    --exclude='checkpoints/' --exclude='*.ckpt' \
    "$SOURCE/runs/" "$LOCAL/"
rsync -az "${SSH[@]}" "$HOST$DATASET/streams_events.csv" "$HOST$DATASET/streams_composition.csv" "$LOCAL/_dataset/"
echo "pulled into $LOCAL/ (checkpoints left on the server)"

if [ "$SYNC" = 1 ]; then
    "$(dirname "$0")/wandb_sync.sh" "$LOCAL"
fi

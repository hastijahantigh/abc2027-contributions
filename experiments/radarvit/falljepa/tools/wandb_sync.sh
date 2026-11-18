#!/bin/bash
# Uploads the offline W&B runs under runs/ that are not synced yet (already synced ones are
# skipped). A resumed or evaluated run has one offline folder per invocation, all with the
# same run id; wandb sync uploads them in order into that one run.
#
# By default only completed runs are uploaded; a run still being written is left for later
# (a run counts as completed once its logs/debug-internal.log says "stream: all finished").
#
#   tools/wandb_sync.sh [--live] [--no-skip-synced] [runs-root]
#
#   --live            also upload the runs that are still running (keeps streaming them)
#   --no-skip-synced  upload everything again, even runs that are already synced

ROOT=runs
ARGS=()
LIVE=0
for arg in "$@"; do
    case "$arg" in
        --live) LIVE=1; ARGS+=(--live) ;;
        --no-skip-synced) ARGS+=("$arg") ;;
        *) ROOT="$arg" ;;
    esac
done

while IFS= read -r -d '' dir; do
    if [ "$LIVE" = 1 ] || grep -q 'stream: all finished' "$dir/logs/debug-internal.log" 2>/dev/null; then
        printf '%s\0' "$dir"
    else
        echo "skip (still running): $dir" >&2
    fi
done < <(find "$ROOT" -type d -name 'offline-run-*' -print0 | sort -z) \
    | xargs -0 -r uv run wandb beta sync ${ARGS[@]+"${ARGS[@]}"}

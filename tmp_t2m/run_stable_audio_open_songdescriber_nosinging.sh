#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/jiahelei/github/audio_eval"
SOURCE_DIR="$REPO_ROOT/tmp/stable_audio_open/ttm_songdescriber_nosinging"
MANIFEST="$SOURCE_DIR/ttm_with_reference.jsonl"
PYTHON_BIN="/home/jiahelei/miniconda3/envs/eval/bin/python"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$REPO_ROOT/tmp_t2m/stable_audio_open_songdescriber_nosinging/$RUN_ID"
GENERATED_CACHE="$RUN_DIR/generated_cache"
REFERENCE_CACHE="$RUN_DIR/reference_cache"
RESULTS_DIR="$RUN_DIR/results"
LOG_DIR="$RUN_DIR/logs"
LOG_FILE="$LOG_DIR/eval.log"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest not found: $MANIFEST" >&2
    exit 1
fi

mkdir -p "$GENERATED_CACHE" "$REFERENCE_CACHE" "$RESULTS_DIR" "$LOG_DIR"
export MPLCONFIGDIR="$RUN_DIR/matplotlib"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "run_id=$RUN_ID"
echo "started_at=$(date --iso-8601=seconds)"
echo "repo_root=$REPO_ROOT"
echo "manifest=$MANIFEST"
echo "generated_cache=$GENERATED_CACHE"
echo "reference_cache=$REFERENCE_CACHE"
echo "results_dir=$RESULTS_DIR"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"

cd "$REPO_ROOT"

CMD=(
    "$PYTHON_BIN" -m audio_eval ttm
    "$MANIFEST"
    --metrics fd,kl,inception_score,clap,audiobox,utmos
    --metric-options openl3,passt,panns,,,
    --generated-cache "$GENERATED_CACHE"
    --reference-cache "$REFERENCE_CACHE"
    --results-dir "$RESULTS_DIR"
    --name stable_audio_open_songdescriber_nosinging_current
)

printf 'command='
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "finished_at=$(date --iso-8601=seconds)"
echo "result=$RESULTS_DIR/stable_audio_open_songdescriber_nosinging_current.json"

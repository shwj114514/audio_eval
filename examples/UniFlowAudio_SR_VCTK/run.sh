#!/usr/bin/env bash
# Derived from: the original UniFlowAudio_SR_VCTK/run.sh
# Change: pair LSD with its metric-owned ssr_eval option
# Unchanged: manifest, result directory, and result name
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"

cd "$REPO_ROOT"


CMD=(
  python -m audio_eval sr \
    "$EXAMPLE_DIR/sr.jsonl" \
    --metrics lsd \
    --metric-options ssr_eval \
    --results-dir "$EXAMPLE_DIR/results" \
    --name uniflow_audio_sr_vctk_single
)
printf '%q ' "${CMD[@]}"
"${CMD[@]}" "$@"

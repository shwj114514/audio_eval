#!/usr/bin/env bash

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"

cd "$REPO_ROOT"


CMD=(
  python -m audio_eval ttm \
    "$EXAMPLE_DIR/ttm.jsonl" \
    --metrics fd,kl,inception_score,clap,audiobox,utmos \
    --metric-options openl3_music,passt,panns,,, \
    --reference songdescriber_nosinging \
    --results-dir "$EXAMPLE_DIR/results" \
    --name stable_audio_open_songdescriber
)
printf '%q ' "${CMD[@]}"
"${CMD[@]}" "$@"

CMD=(
  python -m audio_eval ttm \
    "$EXAMPLE_DIR/ttm_with_reference.jsonl" \
    --metrics fd,kl,inception_score,clap,audiobox,utmos \
    --metric-options openl3_music,passt,panns,,, \
    --generated-cache "$EXAMPLE_DIR/generated_cache" \
    --reference-cache "$EXAMPLE_DIR/reference_cache" \
    --results-dir "$EXAMPLE_DIR/results" \
    --name stable_audio_open_songdescriber_noCache
)
printf '%q ' "${CMD[@]}"
"${CMD[@]}" "$@"

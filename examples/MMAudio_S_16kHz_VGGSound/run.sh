#!/usr/bin/env bash

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"

cd "$REPO_ROOT"


CMD=(
  python -m audio_eval v2a \
    "$EXAMPLE_DIR/v2a.jsonl" \
    --metrics fd,fd,fd,kl,kl,inception_score,imagebind,desync \
    --metric-options passt,panns,vggish,passt_ref_to_gen,panns_ref_to_gen,panns,,sliding_2seg \
    --generated-cache "$EXAMPLE_DIR/generated_cache" \
    --results-dir "$EXAMPLE_DIR/results" \
    --name mmaudio_s_16khz_vggsound_single
)
printf '%s\n' "DeSync protocol: sliding_2seg (4.8s window, 0.64s hop)"
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}" "$@"

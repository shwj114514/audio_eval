#!/usr/bin/env bash
# Derived from: the original StableAudioOpen_AudioCaps/run.sh
# Change: use OpenL3 FD and PaSST KL with the bundled AudioCaps references
# Unchanged: manifest, metrics, result directory, and result name
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"

cd "$REPO_ROOT"


CMD=(
  python -m audio_eval tta \
    "$EXAMPLE_DIR/tta.jsonl" \
    --metrics fd,kl,inception_score,clap,audiobox,utmos \
    --metric-options openl3,passt,panns,,, \
    --reference audiocaps \
    --results-dir "$EXAMPLE_DIR/results" \
    --name stable_audio_open_audiocaps
)
printf '%q ' "${CMD[@]}"
"${CMD[@]}" "$@"

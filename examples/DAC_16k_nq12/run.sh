#!/usr/bin/env bash

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"

cd "$REPO_ROOT"


CMD=(
  python -m audio_eval recon \
    "$EXAMPLE_DIR/recon.jsonl" \
    --metrics pesq,stoi,mel_stft,lsd,speaker_sim \
    --metric-options ,,,, \
    --results-dir "$EXAMPLE_DIR/results" \
    --name dac_16k_nq12
)
printf '%q ' "${CMD[@]}"
"${CMD[@]}" "$@"

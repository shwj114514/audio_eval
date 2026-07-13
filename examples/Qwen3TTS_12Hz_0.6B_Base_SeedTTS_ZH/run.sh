#!/usr/bin/env bash
# Derived from: the original Qwen3TTS SeedTTS-ZH run.sh
# Change: move the SeedTTS-ZH protocol from benchmark to per-metric options
# Unchanged: manifest, WER/SPK-SIM metrics, result directory, and result name
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"

cd "$REPO_ROOT"


CMD=(
  python -m audio_eval tts \
    "$EXAMPLE_DIR/tts.jsonl" \
    --metrics wer,speaker_sim \
    --metric-options seedtts_zh, \
    --results-dir "$EXAMPLE_DIR/results" \
    --name qwen3tts_12hz_0.6b_base_seedtts_zh
)
printf '%q ' "${CMD[@]}"
"${CMD[@]}" "$@"

#!/usr/bin/env bash
# Runs every checked-in example with the environment declared in pyproject.toml.
set -euo pipefail

# codec reconstruction
bash examples/DAC_16k_nq12/run.sh
# V2A
bash examples/MMAudio_S_16kHz_VGGSound/run.sh
# TTS
bash examples/Qwen3TTS_12Hz_0.6B_Base_SeedTTS_ZH/run.sh
# TTM
bash examples/StableAudioOpen_SongDescriber/run.sh
# TTA
bash examples/StableAudioOpen_AudioCaps/run.sh
# SR
bash examples/UniFlowAudio_SR_VCTK/run.sh

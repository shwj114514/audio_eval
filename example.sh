#!/usr/bin/env bash
# Runs every checked-in example with the environment declared in pyproject.toml.
set -euo pipefail

# codec reconstruction
export CUDA_VISIBLE_DEVICES="0"
bash examples/DAC_16k_nq12/run.sh
# V2A
export CUDA_VISIBLE_DEVICES="1"
bash examples/MMAudio_S_16kHz_VGGSound/run.sh
# TTS
export CUDA_VISIBLE_DEVICES="2"
bash examples/Qwen3TTS_12Hz_0.6B_Base_SeedTTS_ZH/run.sh
# TTM
export CUDA_VISIBLE_DEVICES="3"
bash examples/StableAudioOpen_SongDescriber/run.sh
# TTA
export CUDA_VISIBLE_DEVICES="4"
bash examples/StableAudioOpen_AudioCaps/run.sh
# SR
export CUDA_VISIBLE_DEVICES="5"
bash examples/UniFlowAudio_SR_VCTK/run.sh

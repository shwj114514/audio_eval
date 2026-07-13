# Qwen3-TTS 12Hz 0.6B Base on Seed-TTS ZH

This example contains one zero-shot voice-cloning output from
`Qwen/Qwen3-TTS-12Hz-0.6B-Base` and the corresponding Seed-TTS prompt audio.
The JSONL uses only `gen_path`, `ref_path`, and `prompt`.

Run the official Seed-TTS ZH metric protocol with:

```bash
bash run.sh
```

The evaluator writes `task: tts` and `benchmark: seedtts_zh` separately. WER
uses Paraformer-zh, and speaker similarity uses WavLM-Large + ECAPA-TDNN.

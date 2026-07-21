# MMAudio-S-16kHz on VGGSound

This example contains one official precomputed output from
`MMAudio-S-16kHz` on the VGGSound test set. `v2a.jsonl` uses muxed reference
videos for both reference audio and video features. The alternative
`v2a_with_silence_video.jsonl` shows how to provide a separate silent source
video through `video_path`; its `ref_path` still supplies the GT audio.

Run the single-sample metrics with:

```bash
bash run.sh
```

The command evaluates paired PANNs/PaSST KL, ImageBind, and Synchformer
DeSync. FD and Inception Score are distribution-level metrics and are kept in
the separate 15,220-sample full result under
`results/MMAudio_S_16kHz_VGGSound/`.

DeSync uses the official `hkchengrex/av-benchmark` implementation. Install it
in the evaluation environment; the official Synchformer checkpoint is
downloaded to `~/.cache/audio_eval/` on first use unless
`AUDIO_EVAL_SYNCHFORMER_CHECKPOINT` is set.

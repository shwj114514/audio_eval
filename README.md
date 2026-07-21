# audio-eval

`audio-eval` evaluates already generated audio. It does not know which model
generated the audio or how inference was performed. Task evaluators receive a
JSONL manifest; metric functions can also be called directly with paths, NumPy
arrays, PyTorch tensors, or `(audio, sample_rate)` tuples.


## Installation

Python 3.10 is recommended. The upstream OpenL3 package still uses the removed
`imp` module and therefore cannot be installed on Python 3.12.

```bash
conda create -n eval python=3.10 pip
conda activate eval
# Install the complete environment used by the examples.
pip install -e '.[paired,distribution,passt,speech,video]'
```

For a smaller installation, install the base package first and add only the
extra for the task you need, such as `pip install -e '.[paired]'` for PESQ,
STOI, and reconstruction metrics or `pip install -e '.[distribution]'` for
FD, KL, and Inception Score.

Heavy metric dependencies are optional, so installing PESQ does not require
NeMo or CLAP. Model checkpoints are downloaded by their upstream packages when
needed. PaSST is a separate extra because upstream `hear21passt` pins
`timm==0.4.12`; use a dedicated environment if another project requires a
newer timm.

## JSONL input

The canonical input for every task evaluator is a JSONL file: one JSON object
per line, not one JSON array. Only four fields are accepted:

| Field | Type | Meaning |
|---|---|---|
| `gen_path` | string, required | Generated or reconstructed audio |
| `ref_path` | string or null, optional | Task-dependent reference audio or media |
| `video_path` | string or null, optional | V2A source video; ignored by other tasks |
| `prompt` | string or null, optional | Text prompt or target transcript |

Paths may be absolute or relative to the JSONL file. Every referenced file
must exist. There is no `id` field: the stem of `gen_path` is the internal key,
so generated filename stems must be unique. Unknown fields are rejected.

`--metrics` and `--metric-options` are comma-separated lists of equal length.
Each option belongs to the metric at the same position. Use an empty item when
a metric has no option, for example `--metrics pesq,stoi --metric-options ,`.

The meaning of `ref_path` depends on the task: TTA/TTM use reference audio for
distribution metrics, TTS uses speaker reference audio, reconstruction/SR use
paired reference audio, and V2A uses its audio for reference-audio metrics.

### TTA and TTM

Text-to-audio and text-to-music use `gen_path` plus the text prompt:

```jsonl
{"gen_path":"generated/0001.wav","prompt":"A dog barking in a large room"}
{"gen_path":"generated/0002.wav","prompt":"Heavy rain and distant thunder"}
```

`prompt` is required when computing CLAP. Inception Score, AudioBox and UTMOS
only need `gen_path`.

FD and KL additionally need a real-data distribution. It can be supplied
in any one of these ways:

1. Add `ref_path` to every record.
2. Pass a real-audio directory or precomputed reference cache with `--reference`.
3. Pass a bundled reference name with `--reference`: `audiocaps`, `musiccaps`,
   `musiccaps_nosinging`, `songdescriber`, or `songdescriber_nosinging`.

`--metric-options` selects the metric algorithm, not the dataset. Use `openl3`
for the official FD setup and `passt` for the official KL setup. An explicit
`--reference` takes precedence over JSONL `ref_path`; if `--reference` is not
given, FD and KL use all JSONL `ref_path` entries.

When `ref_path` is used with FD or KL, either every record must provide it
or none may provide it. Generated/reference stems do not have to match in this
case because each JSONL record defines the association.

```jsonl
{"gen_path":"generated/0001.wav","ref_path":"reference/a.wav","prompt":"A dog barking"}
{"gen_path":"generated/0002.wav","ref_path":"reference/b.wav","prompt":"Heavy rain"}
```

```bash
audio-eval tta manifests/audiocaps.jsonl \
  --metrics fd,kl,clap \
  --metric-options openl3,passt, \
  --reference audiocaps

audio-eval ttm manifests/songdescriber.jsonl \
  --metrics fd,kl,clap \
  --metric-options openl3,passt, \
  --reference songdescriber
```

For JSONL manifests with `ref_path`, reusable feature caches can be supplied
separately. The evaluator creates missing directories/files and reuses complete
caches on later runs:

```bash
audio-eval ttm manifests/songdescriber.jsonl \
  --metrics fd,kl,inception_score,clap,audiobox,utmos \
  --metric-options openl3,passt,panns,,, \
  --generated-cache manifests/songdescriber_generated_cache \
  --reference-cache manifests/songdescriber_reference_cache
```

`--cache-dir` remains supported as the legacy shared cache root. OpenL3,
PANNs, PaSST, and VGGish features are stored in their corresponding `.npz`
feature caches.

### TTS

For TTS, `prompt` is the target transcript and `ref_path` is the reference
speaker audio:

```jsonl
{"gen_path":"generated/0001.wav","ref_path":"reference/speaker_a.wav","prompt":"The quick brown fox jumps over the lazy dog."}
{"gen_path":"generated/0002.wav","ref_path":"reference/speaker_b.wav","prompt":"Audio evaluation should be reproducible."}
```

- WER requires `prompt` for every record.
- Speaker similarity, PESQ, STOI, Mel-STFT and LSD require `ref_path` for every
  record.
- UTMOS only requires `gen_path`.
- TTS generated/reference stems do not have to match; records are paired by
  their JSONL line.

```bash
audio-eval tts manifests/seedtts_en.jsonl \
  --metrics wer,speaker_sim,utmos \
  --metric-options seedtts_en,,
```

`examples/Qwen3TTS_12Hz_0.6B_Base_SeedTTS_ZH` contains a one-sample Qwen3-TTS
example using this schema.

### Reconstruction and enhancement

Reconstruction uses paired generated and original audio. Both paths are
required, and their filename stems must match exactly:

```jsonl
{"gen_path":"generated/1089-134686-0000.wav","ref_path":"reference/1089-134686-0000.wav"}
{"gen_path":"generated/1089-134686-0001.wav","ref_path":"reference/1089-134686-0001.flac"}
```

The extensions may differ. `prompt` is not used.

```bash
audio-eval recon manifests/reconstruction.jsonl \
  --metrics pesq,stoi,mel_stft,lsd \
  --metric-options ,,,
```

### Audio super-resolution

Super-resolution uses the same paired schema as reconstruction:

```jsonl
{"gen_path":"generated/p361_002.wav","ref_path":"reference/p361_002.wav"}
```

`gen_path` is the model's super-resolved output and `ref_path` is the original
high-resolution target. The low-resolution model input is not used by the
metric and does not belong in the JSONL. Generated/reference stems must match.

```bash
audio-eval sr manifests/sr.jsonl \
  --metrics lsd \
  --metric-options ssr_eval
```

The metric-owned `ssr_eval` option selects the `ssr_eval==0.0.7` LSD settings
at 24 kHz. `examples/UniFlowAudio_SR_VCTK` contains a complete example.

### Video-to-audio

The V2A manifest pairs each generated audio file with its source video:

```jsonl
{"gen_path":"generated/---g-f_I2yQ_000001.flac","ref_path":"reference/---g-f_I2yQ_000001.mp4"}
{"gen_path":"generated/--U7joUcTCo_000000.flac","ref_path":"reference/--U7joUcTCo_000000.mp4"}
```

For this backward-compatible form, `ref_path` supplies the source video for
ImageBind and DeSync, and its audio track supplies the default real-audio
distribution for FD and KL. If the source video has no audio track, provide
the GT audio through `ref_path` and the video through `video_path`, as shown in
[`examples/MMAudio_S_16kHz_VGGSound/v2a_with_silence_video.jsonl`](examples/MMAudio_S_16kHz_VGGSound/v2a_with_silence_video.jsonl).
When `video_path` is omitted, V2A falls back to `ref_path` for video metrics.

`generated_cache/` and `reference_cache/` are created beside the JSONL
automatically when the cache arguments are omitted. If either directory is
missing or incomplete, the required features are extracted and saved there;
complete caches are reused on later runs.

```bash
audio-eval v2a manifests/vggsound.jsonl \
  --metrics fd,fd,fd,kl,kl,inception_score,imagebind,desync \
  --metric-options passt,panns,vggish,passt_ref_to_gen,panns_ref_to_gen,panns,,
```

`--generated-cache` and `--reference-cache` may still specify reusable cache
directories explicitly. `--reference audiocaps` or another supported audio
path/cache/preset overrides the FD/KL reference distribution only; ImageBind
and DeSync continue to use the paired source videos from `video_path`, its
`ref_path` fallback, or the video features in `--reference-cache`.

These options run PaSST/PANNs/VGGish FD, PaSST/PANNs KL, PANNs Inception
Score, ImageBind and Synchformer DeSync.

### Results

Results are written to `results/` by default. The output filename is the
manifest stem unless `--name` is provided. For example,
`manifests/seedtts_en.jsonl` produces `results/seedtts_en.json`. Result metadata
keeps the task name (`tta`, `ttm`, `tts`, `reconstruction`, `sr` or `v2a`).
Each result contains the aggregate values under `metrics`; metrics that support
per-item reporting also include `details` and `unpaired` fields.

## Python API

Metric functions can be used independently:

```python
from audio_eval.metrics.stoi import compute_stoi

result = compute_stoi(
    generated_tensor,
    reference_tensor,
    generated_sample_rate=24000,
    reference_sample_rate=16000,
)
```

Folder-level runners handle result writing:

```python
from audio_eval import evaluate_paired

result = evaluate_paired(
    "outputs/recon",
    "references/test",
    metrics=["pesq", "stoi", "mel_stft"],
    metric_options=["", "", ""],
)
```

The paired runner loads each generated/reference waveform once and passes the
in-memory arrays to every requested metric.

Task evaluators consume the same JSONL schema:

```python
from audio_eval import eval_recon, eval_tta, eval_ttm, eval_tts, eval_v2a

result = eval_tts(
    "manifests/seedtts_en.jsonl",
    metrics=["wer", "speaker_sim"],
    metric_options=["seedtts_en", ""],
)

result = eval_tta(
    "manifests/audiocaps.jsonl",
    metrics=["fd", "kl", "clap"],
    metric_options=["openl3", "passt", ""],
    reference="audiocaps",
)
```

The `seedtts_en` and `seedtts_zh` WER options are defined in `metrics/wer.py`.
Seed-TTS EN uses Whisper-large-v3, Seed-TTS ZH uses Paraformer-zh, and both use
WavLM speaker similarity. To match the released Seed-TTS scripts, EN uses
`scipy.signal.resample`, while ZH keeps the source sample rate for FunASR and
requires `zhconv`.
Set `AUDIO_EVAL_WAVLM_CHECKPOINT` to the official Seed-TTS/UniSpeech
`wavlm_large_finetune.pth` before running `speaker_sim` from the CLI.

## Cache behavior

Expensive feature extraction is part of metric computation; users do not call
a separate public extraction step.

- `audio_eval/features/openl3.py`, `panns.py`, `passt.py`, and `vggish.py` are
  the only implementations that run those feature models. Metric modules never
  contain a second model-forward path.
- PANNs caches 527-dimensional classifier probabilities and 2048-dimensional
  pre-classification embeddings in `panns.npz`. PaSST caches 527-dimensional
  classifier logits and 768-dimensional pre-classification embeddings in
  `passt.npz`. VGGish caches its 128-dimensional embeddings in `vggish.npz`.
  OpenL3 caches its window embeddings in `openl3.npz`.
- Feature caches preserve window-level outputs and their source-audio keys.
  KL and Inception Score consume classification outputs; FD consumes only
  pre-classification embeddings. Metric-specific aggregation happens after
  loading the shared cache.
- Passing or omitting an explicit cache directory does not select a different
  feature extractor. Existing legacy `.pth` PANNs/PaSST/VGGish bundles are not
  silently mixed with the current protocol.
- `compute_clap` caches audio embeddings.
- `compute_fd` and `compute_kl` accept a reference-audio path, a precomputed
  reference cache path, or a bundled reference name as their second argument.
- OpenL3 generated and reference embeddings are cached and reused by
  `compute_fd`. When explicit cache directories are supplied, the `.npz` files
  are written directly in those directories.
- `compute_fd`, `compute_kl`, and `compute_inception_score` accept
  `version="passt"` for PaSST caches; FD also accepts `version="openl3"` and
  `version="vggish"`.
- V2A automatically extracts paired ImageBind and Synchformer video features
  from `ref_path` when the reference cache is missing. Direct metric calls can
  also consume precomputed `av-benchmark` feature caches. DeSync evaluates the
  first and last 4.8-second windows.

Bundled FD references contain the official Stability AI OpenL3 statistics.
They use 44.1 kHz, stereo OpenL3 embeddings, a 0.5-second hop, and batch size
4; AudioCaps uses the `env` model while the music datasets use the `music`
model. Bundled KL references contain the official PaSST per-item probability
vectors. Generated PaSST probabilities are computed from 32 kHz mono audio
using 10-second windows with a 5-second hop and are cached automatically.

The default cache is `~/.cache/audio_eval`. Set `AUDIO_EVAL_CACHE` or pass
`cache_dir=` to change it. Cache keys include the input paths or array content,
file sizes, modification times, sample rate, and metric configuration.

## Metrics

Implemented metric modules in this first version are:

| Module | Public function | Backend/protocol |
|---|---|---|
| `fd.py` | `compute_fd` | OpenL3, PANNs, PaSST, or VGGish |
| `kl.py` | `compute_kl` | PANNs or PaSST paired logits |
| `inception_score.py` | `compute_inception_score` | PANNs or PaSST logits |
| `imagebind.py` | `compute_imagebind` | video/audio cosine similarity |
| `desync.py` | `compute_desync` | Synchformer first/last 4.8s |
| `clap.py` | `compute_clap` | LAION-CLAP |
| `audiobox.py` | `compute_audiobox` | AudioBox Aesthetics |
| `wer.py` | `compute_wer` | NeMo, Whisper, Paraformer, or HuBERT-Large |
| `speaker_sim.py` | `compute_speaker_sim` | WavLM-Large + ECAPA-TDNN |
| `utmos.py` | `compute_utmos` | UTMOS22 strong |
| `pesq.py` | `compute_pesq` | PESQ NB/WB |
| `stoi.py` | `compute_stoi` | STOI/ESTOI |
| `mel_stft.py` | `compute_mel_stft_loss` | multi-scale spectral losses |
| `lsd.py` | `compute_lsd` | standard or `ssr_eval` log-spectral distance |

Common metric options are:

| Metric | Options |
|---|---|
| FD | `openl3`, `panns`, `passt`, `vggish` |
| KL | `panns`, `passt`, `panns_ref_to_gen`, `passt_ref_to_gen` |
| Inception Score | `panns`, `passt` |
| WER | `seedtts_en`, `seedtts_zh`, `seedtts_zh_hard`, `librispeech_test_clean` |
| LSD | `standard`, `ssr_eval` |

An empty option is used for metrics whose backend is fixed, such as CLAP,
AudioBox, UTMOS, PESQ, and STOI.

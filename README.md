# audio-eval

`audio-eval` is a model-agnostic, composable toolkit for evaluating generated
and reconstructed audio. It provides task-level evaluators for text-to-audio,
text-to-music, text-to-speech, reconstruction, audio super-resolution, and
video-to-audio, alongside standalone metric APIs. Task evaluators use JSONL
manifests; metric functions accept file paths, NumPy arrays, PyTorch tensors,
or `(audio, sample_rate)` tuples.

## Evaluation Results

All results below were computed using `audio-eval`. Best scores are shown in
**bold**. ↑ means higher is better; ↓ means lower is better. `gen→ref`
denotes `KL(generated || reference)`.

### Video-to-Audio 🎬 → 🔊

All models were evaluated separately on Movie Gen Audio Bench and CineBench
(internal eval set).

Models evaluated: [Sonilo Sound Effects 1.0](https://sonilo.com/) and
[Mirelo v1.6](https://mirelo.ai/).

#### Movie Gen Audio Bench

<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th>Variant</th>
      <th>Better</th>
      <th>Sonilo Sound Effects 1.0</th>
      <th>Mirelo v1.6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">DeSync</td>
      <td>Sliding, 2 segments</td>
      <td align="center">↓</td>
      <td align="right"><strong>0.6521</strong></td>
      <td align="right">0.6694</td>
    </tr>
    <tr>
      <td>First–last</td>
      <td align="center">↓</td>
      <td align="right"><strong>0.5918</strong></td>
      <td align="right">0.6844</td>
    </tr>
    <tr>
      <td>IB</td>
      <td></td>
      <td align="center">↑</td>
      <td align="right"><strong>0.3178</strong></td>
      <td align="right">0.2945</td>
    </tr>
    <tr>
      <td rowspan="2">KL</td>
      <td>PANNs, gen→ref</td>
      <td align="center">↓</td>
      <td align="right"><strong>1.4203</strong></td>
      <td align="right">1.4802</td>
    </tr>
    <tr>
      <td>PaSST, gen→ref</td>
      <td align="center">↓</td>
      <td align="right"><strong>1.7706</strong></td>
      <td align="right">1.9005</td>
    </tr>
    <tr>
      <td rowspan="2">IS</td>
      <td>PANNs</td>
      <td align="center">↑</td>
      <td align="right">4.8333</td>
      <td align="right"><strong>5.0461</strong></td>
    </tr>
    <tr>
      <td>PaSST</td>
      <td align="center">↑</td>
      <td align="right"><strong>7.6874</strong></td>
      <td align="right">7.6660</td>
    </tr>
    <tr>
      <td rowspan="4">Audiobox</td>
      <td>CE</td>
      <td align="center">↑</td>
      <td align="right"><strong>3.7972</strong></td>
      <td align="right">3.4531</td>
    </tr>
    <tr>
      <td>CU</td>
      <td align="center">↑</td>
      <td align="right"><strong>6.1085</strong></td>
      <td align="right">5.5121</td>
    </tr>
    <tr>
      <td>PC</td>
      <td align="center">↑</td>
      <td align="right"><strong>2.8665</strong></td>
      <td align="right">2.8174</td>
    </tr>
    <tr>
      <td>PQ</td>
      <td align="center">↑</td>
      <td align="right"><strong>6.5849</strong></td>
      <td align="right">6.0517</td>
    </tr>
    <tr>
      <td rowspan="4">FD</td>
      <td>VGGish</td>
      <td align="center">↓</td>
      <td align="right"><strong>3.0817</strong></td>
      <td align="right">4.6969</td>
    </tr>
    <tr>
      <td>OpenL3</td>
      <td align="center">↓</td>
      <td align="right"><strong>82.9343</strong></td>
      <td align="right">122.3979</td>
    </tr>
    <tr>
      <td>PANNs</td>
      <td align="center">↓</td>
      <td align="right">18.0420</td>
      <td align="right"><strong>17.0103</strong></td>
    </tr>
    <tr>
      <td>PaSST</td>
      <td align="center">↓</td>
      <td align="right">210.3894</td>
      <td align="right"><strong>195.0931</strong></td>
    </tr>
  </tbody>
</table>

#### CineBench (internal eval set)

<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th>Variant</th>
      <th>Better</th>
      <th>Sonilo Sound Effects 1.0</th>
      <th>Mirelo v1.6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">DeSync</td>
      <td>Sliding, 2 segments</td>
      <td align="center">↓</td>
      <td align="right"><strong>0.2990</strong></td>
      <td align="right">0.3416</td>
    </tr>
    <tr>
      <td>First–last</td>
      <td align="center">↓</td>
      <td align="right"><strong>0.3040</strong></td>
      <td align="right">0.5540</td>
    </tr>
    <tr>
      <td>IB</td>
      <td></td>
      <td align="center">↑</td>
      <td align="right"><strong>0.3233</strong></td>
      <td align="right">0.3119</td>
    </tr>
    <tr>
      <td rowspan="2">KL</td>
      <td>PANNs, gen→ref</td>
      <td align="center">↓</td>
      <td align="right"><strong>0.7626</strong></td>
      <td align="right">0.9747</td>
    </tr>
    <tr>
      <td>PaSST, gen→ref</td>
      <td align="center">↓</td>
      <td align="right"><strong>0.8125</strong></td>
      <td align="right">1.2079</td>
    </tr>
    <tr>
      <td rowspan="2">IS</td>
      <td>PANNs</td>
      <td align="center">↑</td>
      <td align="right"><strong>2.6009</strong></td>
      <td align="right">2.5854</td>
    </tr>
    <tr>
      <td>PaSST</td>
      <td align="center">↑</td>
      <td align="right"><strong>3.0006</strong></td>
      <td align="right">2.9387</td>
    </tr>
    <tr>
      <td rowspan="4">Audiobox</td>
      <td>CE</td>
      <td align="center">↑</td>
      <td align="right"><strong>3.8704</strong></td>
      <td align="right">3.6893</td>
    </tr>
    <tr>
      <td>CU</td>
      <td align="center">↑</td>
      <td align="right"><strong>6.6173</strong></td>
      <td align="right">6.5818</td>
    </tr>
    <tr>
      <td>PC</td>
      <td align="center">↑</td>
      <td align="right"><strong>4.3200</strong></td>
      <td align="right">3.6931</td>
    </tr>
    <tr>
      <td>PQ</td>
      <td align="center">↑</td>
      <td align="right"><strong>7.0897</strong></td>
      <td align="right">6.8120</td>
    </tr>
    <tr>
      <td rowspan="4">FD</td>
      <td>VGGish</td>
      <td align="center">↓</td>
      <td align="right"><strong>2.2366</strong></td>
      <td align="right">4.0993</td>
    </tr>
    <tr>
      <td>OpenL3</td>
      <td align="center">↓</td>
      <td align="right"><strong>130.3204</strong></td>
      <td align="right">141.8064</td>
    </tr>
    <tr>
      <td>PANNs</td>
      <td align="center">↓</td>
      <td align="right"><strong>26.6477</strong></td>
      <td align="right">36.4970</td>
    </tr>
    <tr>
      <td>PaSST</td>
      <td align="center">↓</td>
      <td align="right"><strong>396.0998</strong></td>
      <td align="right">441.4231</td>
    </tr>
  </tbody>
</table>

### Text-to-Audio 📃 → 🔊

All models were evaluated separately on the Clotho test set and the AudioCaps
test set.

Models evaluated: [Sonilo Sound Effects 1.0](https://sonilo.com/),
[ElevenLabs SFX v2](https://elevenlabs.io/sound-effects), and
[Mirelo v1.6](https://mirelo.ai/).

#### Clotho

<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th>Variant</th>
      <th>Better</th>
      <th>Sonilo Sound Effects 1.0</th>
      <th>ElevenLabs SFX v2</th>
      <th>Mirelo v1.6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CLAP</td>
      <td></td>
      <td align="center">↑</td>
      <td align="right"><strong>0.4377</strong></td>
      <td align="right">0.3967</td>
      <td align="right">0.3223</td>
    </tr>
    <tr>
      <td rowspan="2">KL</td>
      <td>PANNs, gen→ref</td>
      <td align="center">↓</td>
      <td align="right"><strong>1.4424</strong></td>
      <td align="right">1.4950</td>
      <td align="right">1.7523</td>
    </tr>
    <tr>
      <td>PaSST, gen→ref</td>
      <td align="center">↓</td>
      <td align="right"><strong>1.5098</strong></td>
      <td align="right">1.6230</td>
      <td align="right">1.7790</td>
    </tr>
    <tr>
      <td rowspan="2">IS</td>
      <td>PANNs</td>
      <td align="center">↑</td>
      <td align="right">6.9713</td>
      <td align="right"><strong>7.3307</strong></td>
      <td align="right">5.7209</td>
    </tr>
    <tr>
      <td>PaSST</td>
      <td align="center">↑</td>
      <td align="right"><strong>10.3071</strong></td>
      <td align="right">9.0706</td>
      <td align="right">7.1142</td>
    </tr>
    <tr>
      <td rowspan="4">Audiobox</td>
      <td>CE</td>
      <td align="center">↑</td>
      <td align="right"><strong>3.7513</strong></td>
      <td align="right">3.7512</td>
      <td align="right">3.5656</td>
    </tr>
    <tr>
      <td>CU</td>
      <td align="center">↑</td>
      <td align="right"><strong>6.0252</strong></td>
      <td align="right">6.0047</td>
      <td align="right">5.5728</td>
    </tr>
    <tr>
      <td>PC</td>
      <td align="center">↑</td>
      <td align="right"><strong>3.0847</strong></td>
      <td align="right">2.8204</td>
      <td align="right">2.9840</td>
    </tr>
    <tr>
      <td>PQ</td>
      <td align="center">↑</td>
      <td align="right"><strong>6.4833</strong></td>
      <td align="right">6.4831</td>
      <td align="right">6.0801</td>
    </tr>
    <tr>
      <td rowspan="3">FD</td>
      <td>VGGish</td>
      <td align="center">↓</td>
      <td align="right"><strong>2.2883</strong></td>
      <td align="right">2.5429</td>
      <td align="right">5.4576</td>
    </tr>
    <tr>
      <td>OpenL3</td>
      <td align="center">↓</td>
      <td align="right">79.4856</td>
      <td align="right"><strong>53.2710</strong></td>
      <td align="right">86.8386</td>
    </tr>
    <tr>
      <td>PaSST</td>
      <td align="center">↓</td>
      <td align="right">185.1988</td>
      <td align="right"><strong>117.8920</strong></td>
      <td align="right">150.5898</td>
    </tr>
  </tbody>
</table>

#### AudioCaps

<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th>Variant</th>
      <th>Better</th>
      <th>Sonilo Sound Effects 1.0</th>
      <th>ElevenLabs SFX v2</th>
      <th>Mirelo v1.6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CLAP</td>
      <td></td>
      <td align="center">↑</td>
      <td align="right"><strong>0.5365</strong></td>
      <td align="right">0.5315</td>
      <td align="right">0.2756</td>
    </tr>
    <tr>
      <td rowspan="2">KL</td>
      <td>PANNs, gen→ref</td>
      <td align="center">↓</td>
      <td align="right"><strong>0.8548</strong></td>
      <td align="right">1.0386</td>
      <td align="right">1.9853</td>
    </tr>
    <tr>
      <td>PaSST, gen→ref</td>
      <td align="center">↓</td>
      <td align="right"><strong>0.9030</strong></td>
      <td align="right">1.1035</td>
      <td align="right">2.3337</td>
    </tr>
    <tr>
      <td rowspan="2">IS</td>
      <td>PANNs</td>
      <td align="center">↑</td>
      <td align="right">8.1439</td>
      <td align="right"><strong>8.8073</strong></td>
      <td align="right">6.0735</td>
    </tr>
    <tr>
      <td>PaSST</td>
      <td align="center">↑</td>
      <td align="right"><strong>11.1876</strong></td>
      <td align="right">9.4909</td>
      <td align="right">9.4753</td>
    </tr>
    <tr>
      <td rowspan="4">Audiobox</td>
      <td>CE</td>
      <td align="center">↑</td>
      <td align="right"><strong>3.7955</strong></td>
      <td align="right"><strong>3.7955</strong></td>
      <td align="right">3.5592</td>
    </tr>
    <tr>
      <td>CU</td>
      <td align="center">↑</td>
      <td align="right">5.4048</td>
      <td align="right">5.4048</td>
      <td align="right"><strong>5.5914</strong></td>
    </tr>
    <tr>
      <td>PC</td>
      <td align="center">↑</td>
      <td align="right"><strong>3.3695</strong></td>
      <td align="right">3.2447</td>
      <td align="right">2.8646</td>
    </tr>
    <tr>
      <td>PQ</td>
      <td align="center">↑</td>
      <td align="right">6.0934</td>
      <td align="right">6.0408</td>
      <td align="right"><strong>6.1982</strong></td>
    </tr>
    <tr>
      <td rowspan="3">FD</td>
      <td>VGGish</td>
      <td align="center">↓</td>
      <td align="right"><strong>1.4787</strong></td>
      <td align="right">1.6304</td>
      <td align="right">3.2801</td>
    </tr>
    <tr>
      <td>OpenL3</td>
      <td align="center">↓</td>
      <td align="right">174.5691</td>
      <td align="right"><strong>147.5690</strong></td>
      <td align="right">243.9284</td>
    </tr>
    <tr>
      <td>PaSST</td>
      <td align="center">↓</td>
      <td align="right">159.8685</td>
      <td align="right"><strong>128.7889</strong></td>
      <td align="right">218.4857</td>
    </tr>
  </tbody>
</table>

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

`--metric-options` selects the metric algorithm, not the dataset. OpenL3 FD
requires an explicit model choice: use `openl3_env` for environmental sounds
or `openl3_music` for music. Use `passt` for the official KL setup. An
explicit `--reference` takes precedence over JSONL `ref_path`; if
`--reference` is not given, FD and KL use all JSONL `ref_path` entries.
The ambiguous legacy FD option `openl3` is not accepted.

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
  --metric-options openl3_env,passt, \
  --reference audiocaps

audio-eval ttm manifests/songdescriber.jsonl \
  --metrics fd,kl,clap \
  --metric-options openl3_music,passt, \
  --reference songdescriber
```

For JSONL manifests with `ref_path`, reusable feature caches can be supplied
separately. The evaluator creates missing directories/files and reuses complete
caches on later runs:

```bash
audio-eval ttm manifests/songdescriber.jsonl \
  --metrics fd,kl,inception_score,clap,audiobox,utmos \
  --metric-options openl3_music,passt,panns,,, \
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

The V2A manifest pairs each generated audio file with its source video. 
```jsonl
{"gen_path":"generated/---g-f_I2yQ_000001.flac","ref_path":"reference/---g-f_I2yQ_000001.mp4"}
{"gen_path":"generated/--U7joUcTCo_000000.flac","ref_path":"reference/--U7joUcTCo_000000.mp4"}
```

If the GT reference audio and source video are separate files, use `ref_path`
for the audio and `video_path` for the video:

```jsonl
{"gen_path":"generated/---g-f_I2yQ_000001.flac","ref_path":"reference_audio/---g-f_I2yQ_000001.flac","video_path":"silence_video/---g-f_I2yQ_000001.mp4"}
{"gen_path":"generated/--U7joUcTCo_000000.flac","ref_path":"reference_audio/--U7joUcTCo_000000.flac","video_path":"silence_video/--U7joUcTCo_000000.mp4"}
```

In the first form, `ref_path` supplies the source video for ImageBind and
DeSync, and its audio track supplies the default real-audio distribution for
FD and KL. In the second form, `video_path` supplies the video for ImageBind
and DeSync, while `ref_path` supplies the GT audio for FD and KL, as shown in
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
    metric_options=["openl3_env", "passt", ""],
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
  `compute_fd`. Cache metadata must match the requested `openl3_env` or
  `openl3_music` configuration; incompatible caches are rejected. When
  explicit cache directories are supplied, the `.npz` files are written
  directly in those directories.
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
| FD | `openl3_env`, `openl3_music`, `panns`, `passt`, `vggish` |
| KL | `panns`, `passt`, `panns_ref_to_gen`, `passt_ref_to_gen` |
| Inception Score | `panns`, `passt` |
| WER | `seedtts_en`, `seedtts_zh`, `seedtts_zh_hard`, `librispeech_test_clean` |
| LSD | `standard`, `ssr_eval` |

An empty option is used for metrics whose backend is fixed, such as CLAP,
AudioBox, UTMOS, PESQ, and STOI.

## Citation

If you use `audio-eval` in your research, please cite the software:

```bibtex
@software{lei2026audioeval,
  author  = {Lei, Jiahe and Kong, Qiuqiang},
  title   = {audio-eval: Composable Evaluation for Generated and Reconstructed Audio},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/shwj114514/audio_eval}
}
```

Please cite the specific version of `audio-eval` used in your experiments.

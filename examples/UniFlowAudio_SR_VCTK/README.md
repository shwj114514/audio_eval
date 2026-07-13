# UniFlow-Audio SR on VCTK

This example uses the `p361_002` super-resolution pair published on the
official UniFlow-Audio project page:

- `input/p361_002.wav`: the published low-resolution input.
- `generated/p361_002.wav`: the published UniFlow-Audio SR output.
- `reference/p361_002.wav`: `p361_002_mic1.flac` from the fixed NVSR/ssr_eval
  VCTK-test release, resampled to 24 kHz as specified by UniFlow-Audio.

Run the official SR metric protocol with:

```bash
bash run.sh
```

The script pairs `--metrics lsd` with `--metric-options ssr_eval`. The metric-owned
`ssr_eval` option uses `ssr_eval==0.0.7`-compatible LSD: 24 kHz, `n_fft=1114`,
`hop_length=240`, and generated audio padded or truncated to the reference
length. For this public sample, UniFlow-Audio obtains LSD 1.707196, compared
with 2.896504 for the low-resolution input.

This single-sample score must not be compared directly with the paper's
1.49/1.53/1.58 aggregate scores over ESC-50, VCTK-test, and MUSDB.

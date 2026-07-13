# PaSST reference probabilities

The `.pkl` files in this directory are copied from
[`Stability-AI/stable-audio-metrics`](https://github.com/Stability-AI/stable-audio-metrics/tree/main/load/passt_kld),
commit `fd55536cc812c460ecc421220864993c7f168184`.

They contain precomputed 527-class PaSST probability vectors for AudioCaps,
MusicCaps, and SongDescriber reference audio.

SHA-256 checksums:

- `audiocaps-test__collectmean__reference_probabilities.pkl`:
  `b0ae50ab70fe4f47094e11a52d814ab5490d8ce8f342f84c1f7e310c50902475`
- `musiccaps-public-nosinging__collectmean__reference_probabilities.pkl`:
  `4255452ac04d04efcade3acf34976e97f7459689c513efe70f8ab3bcf542fbd6`
- `musiccaps-public__collectmean__reference_probabilities.pkl`:
  `24df46704664daa57d34f9e792c4d575c750202ae762297ee4c1c31d60db9f4f`
- `song_describer-nosinging__collectmean__reference_probabilities.pkl`:
  `0a33b0d836904f06eb0304bce4e4271eac7c171baa202de45ae472d6dbbbc433`
- `song_describer__collectmean__reference_probabilities.pkl`:
  `3d4274393e5960520327e2147f8c65930887927d5431c8891195eb5be2c119ec`

The upstream repository is licensed under the MIT License, Copyright (c) 2024
Stability AI. See `LICENSE.md` in this directory for the complete license text.

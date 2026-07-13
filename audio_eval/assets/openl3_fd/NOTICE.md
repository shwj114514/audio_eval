# OpenL3 reference statistics

The `.npz` files in this directory are copied from
[`Stability-AI/stable-audio-metrics`](https://github.com/Stability-AI/stable-audio-metrics/tree/main/load/openl3_fd),
commit `fd55536cc812c460ecc421220864993c7f168184`.

They contain the precomputed `mu_ref` and `sigma_ref` arrays used by Stability
AI's OpenL3 Fréchet Distance examples for AudioCaps, MusicCaps, and
SongDescriber. They are reference distribution statistics, not per-audio
embeddings.

SHA-256 checksums:

- `audiocaps-test__channels2__44100__openl3env__openl3hopsize0.5__batch4.npz`:
  `3c420a01dd417cfb494202a3e280ebd38fe467a9eebe71a562db7ccba4d93707`
- `musiccaps-public-nosinging__channels2__44100__openl3music__openl3hopsize0.5__batch4.npz`:
  `698f27e70e0e7fd1f85efd919fb8e395e666ce1a409819ef4f80fa33d5212a49`
- `musiccaps-public__channels2__44100__openl3music__openl3hopsize0.5__batch4.npz`:
  `48234027e115f60afd51b392398e2e3b7a09967f2d1774349d14962af0736b1c`
- `song_describer-nosinging__channels2__44100__openl3music__openl3hopsize0.5__batch4.npz`:
  `481ea5eb4adf269954b198d04bad07230f7a9ca2a2afaf15e9daf0eb1331c362`
- `song_describer__channels2__44100__openl3music__openl3hopsize0.5__batch4.npz`:
  `94fde8992200ce1c4d5ede5b98573e800a1da8a6aa967eba13b43b2c2b5be5e6`

The upstream repository is licensed under the MIT License, Copyright (c) 2024
Stability AI. See `LICENSE.md` in this directory for the complete license text.

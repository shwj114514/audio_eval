"""OpenL3 embeddings shared by Fréchet-based metrics."""

from __future__ import annotations

from pathlib import Path
import typing as tp

import numpy as np

from audio_eval.audio import collection_fingerprint, collection_items, load_audio
from audio_eval.cache import feature_cache_file, resolve_feature_cache
from audio_eval.common import AudioCollection


OPENL3_FEATURE_FILENAME = "openl3.npz"


def load_openl3_features(path: str | Path) -> tp.Dict[str, tp.Any]:
    cache_path = Path(path).expanduser()
    if cache_path.is_dir():
        cache_path = cache_path / OPENL3_FEATURE_FILENAME
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as loaded:
        required = {"embeddings", "num_audio"}
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"OpenL3 feature cache {cache_path} is missing arrays: {missing}")
        num_audio = int(loaded["num_audio"])
        keys = (
            loaded["keys"].astype(str).tolist()
            if "keys" in loaded.files
            else [str(index) for index in range(num_audio)]
        )
        clip_keys = (
            loaded["clip_keys"].astype(str).tolist()
            if "clip_keys" in loaded.files
            else []
        )
        return {
            "keys": keys,
            "clip_keys": clip_keys,
            "embeddings": loaded["embeddings"],
            "cache_path": cache_path,
        }


def get_openl3_features(
    audio: AudioCollection,
    *,
    sample_rate: int | None = None,
    target_sample_rate: int = 44100,
    channels: int = 2,
    content_type: str = "music",
    hop_size: float = 0.5,
    batch_size: int = 4,
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    refresh_cache: bool = False,
) -> tp.Dict[str, tp.Any]:
    """Extract OpenL3 embeddings; OpenL3 internally uses one-second windows."""
    if channels not in {1, 2}:
        raise ValueError("OpenL3 channels must be 1 or 2")
    if content_type not in {"env", "music"}:
        raise ValueError("OpenL3 content_type must be env or music")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    existing = resolve_feature_cache(audio, OPENL3_FEATURE_FILENAME)
    if existing is not None and not refresh_cache:
        return load_openl3_features(existing)

    fingerprint = collection_fingerprint(audio, sample_rate=sample_rate)
    output_path = feature_cache_file(
        OPENL3_FEATURE_FILENAME,
        fingerprint,
        backend=(
            f"openl3_{channels}ch_{target_sample_rate}_{content_type}_"
            f"hop{hop_size}_batch{batch_size}"
        ),
        cache_dir=cache_dir,
        output_dir=output_dir,
    )
    if output_path.is_file() and not refresh_cache:
        return load_openl3_features(output_path)

    try:
        import tensorflow as tf
        #  CPU：TensorFlow + oneDNN，
        #  GPU：TensorFlow + CUDA + cuDNN 9.10.2 + XLA
        # tf.config.set_visible_devices([], "GPU")
        import openl3
        import soxr
    except ImportError as error:
        raise ImportError(
            "OpenL3 features require `pip install audio-eval[distribution]`, including TensorFlow"
        ) from error

    items = collection_items(audio)
    if not items:
        raise ValueError("OpenL3 received no audio")
    keys = [key for key, _ in items]
    model = openl3.models.load_audio_embedding_model(
        input_repr="mel256",
        content_type=content_type,
        embedding_size=512,
    )

    batches: tp.List[np.ndarray] = []
    clip_keys: tp.List[str] = []
    for start in range(0, len(items), batch_size):
        batch_items = items[start:start + batch_size]
        left_audio: tp.List[np.ndarray] = []
        right_audio: tp.List[np.ndarray] = []
        sample_rates: tp.List[int] = []
        for _, source in batch_items:
            waveform, source_sample_rate = load_audio(source, sample_rate=sample_rate, mono=False)
            if waveform.ndim == 2 and waveform.shape[0] <= 8 and waveform.shape[0] < waveform.shape[1]:
                waveform = waveform.T
            peak = float(np.max(np.abs(waveform)))
            if peak > 0:
                waveform = waveform * (10.0 ** (-1.0 / 20.0) / peak)
            if source_sample_rate != target_sample_rate:
                waveform = soxr.resample(waveform, source_sample_rate, target_sample_rate)

            if waveform.ndim == 1:
                left_audio.append(waveform)
                if channels == 2:
                    right_audio.append(waveform)
            elif waveform.ndim == 2:
                if channels == 1:
                    left_audio.append(waveform.mean(axis=1))
                else:
                    left_audio.append(waveform[:, 0])
                    right_audio.append(waveform[:, 1] if waveform.shape[1] > 1 else waveform[:, 0])
            else:
                raise ValueError(f"Expected 1D or 2D audio, got shape {waveform.shape}")
            sample_rates.append(target_sample_rate)

        left_embeddings, _ = openl3.get_audio_embedding(
            left_audio,
            sample_rates,
            model=model,
            verbose=False,
            hop_size=hop_size,
            batch_size=batch_size,
        )
        if channels == 1:
            batch_embeddings = [np.asarray(value) for value in left_embeddings]
        else:
            right_embeddings, _ = openl3.get_audio_embedding(
                right_audio,
                sample_rates,
                model=model,
                verbose=False,
                hop_size=hop_size,
                batch_size=batch_size,
            )
            batch_embeddings = [
                np.concatenate([left, right], axis=1)
                for left, right in zip(left_embeddings, right_embeddings, strict=True)
            ]
        for (key, _), embeddings in zip(batch_items, batch_embeddings, strict=True):
            if embeddings.ndim != 2 or embeddings.shape[1] != 512 * channels:
                raise ValueError(f"Unexpected OpenL3 embedding shape for {key}: {embeddings.shape}")
            batches.append(np.asarray(embeddings, dtype=np.float32))
            clip_keys.extend([key] * embeddings.shape[0])

    embedding_array = np.concatenate(batches, axis=0)
    np.savez_compressed(
        output_path,
        keys=np.asarray(keys),
        clip_keys=np.asarray(clip_keys),
        embeddings=embedding_array,
        num_audio=np.asarray(len(items)),
        target_sample_rate=np.asarray(target_sample_rate),
        channels=np.asarray(channels),
        content_type=np.asarray(content_type),
        hop_size=np.asarray(hop_size),
        batch_size=np.asarray(batch_size),
        embedding_layer=np.asarray("openl3_mel256_512_per_channel"),
        schema_version=np.asarray(2),
    )
    return {
        "keys": keys,
        "clip_keys": clip_keys,
        "embeddings": embedding_array,
        "cache_path": output_path,
    }

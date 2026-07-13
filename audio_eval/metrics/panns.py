"""Shared internal PANNs pass used by FD, KL, and Inception Score."""

from __future__ import annotations

import hashlib
from pathlib import Path
import typing as tp

import numpy as np

from audio_eval.audio import collection_fingerprint, collection_items, load_audio
from audio_eval.cache import cache_file
from audio_eval.common import AudioCollection

_MODELS: tp.Dict[tp.Tuple[str, tp.Optional[str]], tp.Any] = {}


def load_panns_features(path: tp.Union[str, Path]) -> tp.Dict[str, tp.Any]:
    """Load a precomputed PANNs embedding/probability cache."""
    cache_path = Path(path).expanduser()
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    loaded = np.load(cache_path, allow_pickle=False)
    required = {"keys", "probabilities", "embeddings"}
    missing = sorted(required - set(loaded.files))
    if missing:
        raise ValueError(f"PANNs feature cache {cache_path} is missing arrays: {missing}")
    return {
        "keys": loaded["keys"].astype(str).tolist(),
        "probabilities": loaded["probabilities"],
        "embeddings": loaded["embeddings"],
        "cache_path": cache_path,
    }


def _load_model(*, device: str, checkpoint_path: tp.Optional[tp.Union[str, Path]]) -> tp.Any:
    cache_key = (device, str(checkpoint_path) if checkpoint_path is not None else None)
    if cache_key not in _MODELS:
        try:
            from panns_inference import AudioTagging
        except ImportError as error:
            raise ImportError("PANNs metrics require `pip install audio-eval[distribution]`") from error
        _MODELS[cache_key] = AudioTagging(
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
            device=device,
        )
    return _MODELS[cache_key]


def get_panns_features(
    audio: AudioCollection,
    *,
    sample_rate: int | None = None,
    backend: str = "panns_cnn14",
    cache_dir: str | Path | None = None,
    batch_size: int = 8,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    refresh_cache: bool = False,
) -> tp.Dict[str, tp.Any]:
    """Extract embeddings and probabilities once and cache both together."""
    if backend != "panns_cnn14":
        raise ValueError(f"Unsupported PANNs backend: {backend!r}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    target_sample_rate = 32000
    base_fingerprint = collection_fingerprint(audio, sample_rate=sample_rate)
    fingerprint_digest = hashlib.sha256()
    fingerprint_digest.update(base_fingerprint.encode())
    fingerprint_digest.update(f"backend={backend};sr={target_sample_rate}".encode())
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        stat = checkpoint.stat()
        fingerprint_digest.update(f"{checkpoint.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    fingerprint = fingerprint_digest.hexdigest()
    output_path = cache_file(
        "features",
        fingerprint,
        backend=backend,
        suffix=".npz",
        cache_dir=cache_dir,
    )

    if output_path.exists() and not refresh_cache:
        return load_panns_features(output_path)

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    items = collection_items(audio)
    keys = [key for key, _ in items]
    model = _load_model(device=device, checkpoint_path=checkpoint_path)
    probabilities: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []

    for start in range(0, len(items), batch_size):
        batch_items = items[start : start + batch_size]
        waveforms = [
            load_audio(
                source,
                sample_rate=sample_rate,
                target_sample_rate=target_sample_rate,
            )[0]
            for _, source in batch_items
        ]
        max_length = max(len(waveform) for waveform in waveforms)
        batch = np.zeros((len(waveforms), max_length), dtype=np.float32)
        for index, waveform in enumerate(waveforms):
            batch[index, : len(waveform)] = waveform
        batch_probabilities, batch_embeddings = model.inference(batch)
        probabilities.append(np.asarray(batch_probabilities, dtype=np.float32))
        embeddings.append(np.asarray(batch_embeddings, dtype=np.float32))

    probability_array = np.concatenate(probabilities, axis=0)
    embedding_array = np.concatenate(embeddings, axis=0)
    np.savez_compressed(
        output_path,
        keys=np.asarray(keys),
        probabilities=probability_array,
        embeddings=embedding_array,
    )
    return {
        "keys": keys,
        "probabilities": probability_array,
        "embeddings": embedding_array,
        "cache_path": output_path,
    }

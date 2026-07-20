"""PANNs classifier outputs and pre-classification embeddings."""

from __future__ import annotations

import hashlib
from pathlib import Path
import typing as tp

import numpy as np

from audio_eval.audio import collection_fingerprint, collection_items, load_audio
from audio_eval.cache import feature_cache_file, resolve_feature_cache
from audio_eval.common import AudioCollection


PANN_FEATURE_FILENAME = "panns.npz"
_MODELS: tp.Dict[tp.Tuple[str, tp.Optional[str]], tp.Any] = {}


def load_panns_features(path: str | Path) -> tp.Dict[str, tp.Any]:
    cache_path = Path(path).expanduser()
    if cache_path.is_dir():
        cache_path = cache_path / PANN_FEATURE_FILENAME
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as loaded:
        required = {"keys", "clip_keys", "probabilities", "embeddings"}
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"PANNs feature cache {cache_path} is missing arrays: {missing}")
        return {
            "keys": loaded["keys"].astype(str).tolist(),
            "clip_keys": loaded["clip_keys"].astype(str).tolist(),
            "probabilities": loaded["probabilities"],
            "embeddings": loaded["embeddings"],
            "cache_path": cache_path,
        }


def _load_model(*, device: str, checkpoint_path: str | Path | None) -> tp.Any:
    cache_key = (device, str(checkpoint_path) if checkpoint_path is not None else None)
    if cache_key not in _MODELS:
        try:
            from panns_inference import AudioTagging
        except ImportError as error:
            raise ImportError("PANNs features require `pip install audio-eval[distribution]`") from error
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
    output_dir: str | Path | None = None,
    batch_size: int = 8,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    refresh_cache: bool = False,
) -> tp.Dict[str, tp.Any]:
    """Return 527-D classifier probabilities and 2048-D embeddings per 10-second window."""
    if backend != "panns_cnn14":
        raise ValueError(f"Unsupported PANNs backend: {backend!r}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    existing = resolve_feature_cache(audio, PANN_FEATURE_FILENAME)
    if existing is not None and not refresh_cache:
        return load_panns_features(existing)

    target_sample_rate = 32000
    base_fingerprint = collection_fingerprint(audio, sample_rate=sample_rate)
    fingerprint_digest = hashlib.sha256()
    fingerprint_digest.update(base_fingerprint.encode())
    fingerprint_digest.update(
        f"backend={backend};sr={target_sample_rate};window=10;hop=5;tail=0.15;peak_db=-1".encode()
    )
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        stat = checkpoint.stat()
        fingerprint_digest.update(f"{checkpoint.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    output_path = feature_cache_file(
        PANN_FEATURE_FILENAME,
        fingerprint_digest.hexdigest(),
        backend="panns_cnn14_10s_5s",
        cache_dir=cache_dir,
        output_dir=output_dir,
    )
    if output_path.is_file() and not refresh_cache:
        return load_panns_features(output_path)

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    try:
        import soxr
    except ImportError as error:
        raise ImportError("PANNs features require soxr") from error

    items = collection_items(audio)
    if not items:
        raise ValueError("PANNs received no audio")
    keys = [key for key, _ in items]
    model = _load_model(device=device, checkpoint_path=checkpoint_path)
    probabilities: tp.List[np.ndarray] = []
    embeddings: tp.List[np.ndarray] = []
    clip_keys: tp.List[str] = []
    pending_windows: tp.List[np.ndarray] = []
    window_size = 10 * target_sample_rate
    step_size = 5 * target_sample_rate

    def flush() -> None:
        if not pending_windows:
            return
        batch_probabilities, batch_embeddings = model.inference(np.stack(pending_windows))
        if batch_probabilities.ndim != 2 or batch_probabilities.shape[1] != 527:
            raise ValueError(f"Unexpected PANNs classifier output shape: {batch_probabilities.shape}")
        if batch_embeddings.ndim != 2 or batch_embeddings.shape[1] != 2048:
            raise ValueError(f"Unexpected PANNs embedding shape: {batch_embeddings.shape}")
        probabilities.append(np.asarray(batch_probabilities, dtype=np.float32))
        embeddings.append(np.asarray(batch_embeddings, dtype=np.float32))
        pending_windows.clear()

    for key, source in items:
        waveform, source_sample_rate = load_audio(source, sample_rate=sample_rate)
        peak = float(np.max(np.abs(waveform)))
        if peak > 0:
            waveform = waveform * (10.0 ** (-1.0 / 20.0) / peak)
        if source_sample_rate != target_sample_rate:
            waveform = soxr.resample(waveform, source_sample_rate, target_sample_rate)
        waveform = np.asarray(waveform, dtype=np.float32)

        num_windows = 0
        for start in range(0, max(step_size, len(waveform) - step_size), step_size):
            window = waveform[start:start + window_size]
            if len(window) < window_size:
                if len(window) <= int(window_size * 0.15):
                    continue
                padded = np.zeros(window_size, dtype=np.float32)
                padded[:len(window)] = window
                window = padded
            pending_windows.append(window)
            clip_keys.append(key)
            num_windows += 1
            if len(pending_windows) == batch_size:
                flush()
        if num_windows == 0:
            raise ValueError(f"PANNs found no valid 10-second window for {key}")
    flush()

    probability_array = np.concatenate(probabilities, axis=0)
    embedding_array = np.concatenate(embeddings, axis=0)
    np.savez_compressed(
        output_path,
        keys=np.asarray(keys),
        clip_keys=np.asarray(clip_keys),
        probabilities=probability_array,
        embeddings=embedding_array,
        sample_rate=np.asarray(target_sample_rate),
        window_seconds=np.asarray(10.0),
        hop_seconds=np.asarray(5.0),
        classifier_output=np.asarray("sigmoid_probabilities"),
        embedding_layer=np.asarray("fc1_2048_before_classifier"),
        schema_version=np.asarray(2),
    )
    return {
        "keys": keys,
        "clip_keys": clip_keys,
        "probabilities": probability_array,
        "embeddings": embedding_array,
        "cache_path": output_path,
    }

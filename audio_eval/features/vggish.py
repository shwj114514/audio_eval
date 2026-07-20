"""VGGish embeddings for Fréchet distance."""

from __future__ import annotations

import hashlib
from pathlib import Path
import typing as tp

import numpy as np
import torch

from audio_eval.audio import collection_fingerprint, collection_items, load_audio
from audio_eval.cache import feature_cache_file, resolve_feature_cache
from audio_eval.common import AudioCollection


VGGISH_FEATURE_FILENAME = "vggish.npz"
# Cache one loaded VGGish model per device to avoid repeated initialization.
_MODELS: tp.Dict[str, torch.nn.Module] = {}


def load_vggish_features(path: str | Path) -> tp.Dict[str, tp.Any]:
    cache_path = Path(path).expanduser()
    if cache_path.is_dir():
        cache_path = cache_path / VGGISH_FEATURE_FILENAME
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as loaded:
        required = {"keys", "clip_keys", "embeddings"}
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"VGGish feature cache {cache_path} is missing arrays: {missing}")
        return {
            "keys": loaded["keys"].astype(str).tolist(),
            "clip_keys": loaded["clip_keys"].astype(str).tolist(),
            "embeddings": loaded["embeddings"],
            "cache_path": cache_path,
        }


def _load_model(device: str) -> torch.nn.Module:
    if device not in _MODELS:
        try:
            from av_bench.vggish.vggish import VGGish
        except ImportError as error:
            raise ImportError("VGGish features require `pip install audio-eval[video]`") from error
        _MODELS[device] = VGGish(device=torch.device(device), postprocess=False).eval()
    return _MODELS[device]


def get_vggish_features(
    audio: AudioCollection,
    *,
    sample_rate: int | None = None,
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    refresh_cache: bool = False,
) -> tp.Dict[str, tp.Any]:
    """Return every 0.96-second 128-D VGGish embedding without a classifier head."""
    existing = resolve_feature_cache(audio, VGGISH_FEATURE_FILENAME)
    if existing is not None and not refresh_cache:
        return load_vggish_features(existing)

    target_sample_rate = 16000
    base_fingerprint = collection_fingerprint(audio, sample_rate=sample_rate)
    fingerprint_digest = hashlib.sha256()
    fingerprint_digest.update(base_fingerprint.encode())
    fingerprint_digest.update(
        b"backend=av_bench_vggish;sr=16000;mono=true;center=true;patch=0.96;hop=0.96;postprocess=false"
    )
    output_path = feature_cache_file(
        VGGISH_FEATURE_FILENAME,
        fingerprint_digest.hexdigest(),
        backend="vggish_0.96s",
        cache_dir=cache_dir,
        output_dir=output_dir,
    )
    if output_path.is_file() and not refresh_cache:
        return load_vggish_features(output_path)

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(target_device)
    items = collection_items(audio)
    if not items:
        raise ValueError("VGGish received no audio")

    keys = [key for key, _ in items]
    clip_keys: tp.List[str] = []
    embeddings: tp.List[np.ndarray] = []
    minimum_samples = int(round(0.96 * target_sample_rate))
    with torch.inference_mode():
        for key, source in items:
            waveform, _ = load_audio(
                source,
                sample_rate=sample_rate,
                target_sample_rate=target_sample_rate,
            )
            waveform = waveform - waveform.mean()
            if len(waveform) < minimum_samples:
                waveform = np.pad(waveform, (0, minimum_samples - len(waveform)))
            tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).unsqueeze(0)
            feature = model(tensor, sample_rate=target_sample_rate).squeeze(0).cpu().numpy()
            if feature.ndim != 2 or feature.shape[0] == 0 or feature.shape[1] != 128:
                raise ValueError(f"Unexpected VGGish embedding shape for {key}: {feature.shape}")
            embeddings.append(np.asarray(feature, dtype=np.float32))
            clip_keys.extend([key] * feature.shape[0])

    embedding_array = np.concatenate(embeddings, axis=0)
    np.savez_compressed(
        output_path,
        keys=np.asarray(keys),
        clip_keys=np.asarray(clip_keys),
        embeddings=embedding_array,
        sample_rate=np.asarray(target_sample_rate),
        window_seconds=np.asarray(0.96),
        hop_seconds=np.asarray(0.96),
        embedding_layer=np.asarray("vggish_128_before_postprocess"),
        schema_version=np.asarray(2),
    )
    return {
        "keys": keys,
        "clip_keys": clip_keys,
        "embeddings": embedding_array,
        "cache_path": output_path,
    }

"""PaSST classifier logits and pre-classification embeddings."""

from __future__ import annotations

import contextlib
from functools import partial
import hashlib
import os
from pathlib import Path
import typing as tp

import numpy as np
import torch

from audio_eval.audio import collection_fingerprint, collection_items, load_audio
from audio_eval.cache import feature_cache_file, resolve_feature_cache
from audio_eval.common import AudioCollection


PASST_FEATURE_FILENAME = "passt.npz"
_MODELS: tp.Dict[str, torch.nn.Module] = {}


def load_passt_features(path: str | Path) -> tp.Dict[str, tp.Any]:
    cache_path = Path(path).expanduser()
    if cache_path.is_dir():
        cache_path = cache_path / PASST_FEATURE_FILENAME
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as loaded:
        required = {"keys", "clip_keys", "logits", "embeddings"}
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"PaSST feature cache {cache_path} is missing arrays: {missing}")
        return {
            "keys": loaded["keys"].astype(str).tolist(),
            "clip_keys": loaded["clip_keys"].astype(str).tolist(),
            "logits": loaded["logits"],
            "embeddings": loaded["embeddings"],
            "cache_path": cache_path,
        }


def _load_model(device: str) -> torch.nn.Module:
    if device not in _MODELS:
        try:
            from hear21passt.base import get_basic_model
        except ImportError as error:
            raise ImportError("PaSST features require `pip install audio-eval[passt]`") from error
        with open(os.devnull, "w") as output, contextlib.redirect_stdout(output):
            model = get_basic_model(mode="all")
        _MODELS[device] = model.to(device).eval()
    return _MODELS[device]


def get_passt_features(
    audio: AudioCollection,
    *,
    sample_rate: int | None = None,
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    batch_size: int = 8,
    device: str | None = None,
    refresh_cache: bool = False,
) -> tp.Dict[str, tp.Any]:
    """Return 527-D logits and 768-D embeddings per 10-second window."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    existing = resolve_feature_cache(audio, PASST_FEATURE_FILENAME)
    if existing is not None and not refresh_cache:
        return load_passt_features(existing)

    target_sample_rate = 32000
    base_fingerprint = collection_fingerprint(audio, sample_rate=sample_rate)
    fingerprint_digest = hashlib.sha256()
    fingerprint_digest.update(base_fingerprint.encode())
    fingerprint_digest.update(
        b"backend=passt_s_swa_p16_128_ap476;sr=32000;window=10;hop=5;tail=0.15;peak_db=-1;mode=all"
    )
    output_path = feature_cache_file(
        PASST_FEATURE_FILENAME,
        fingerprint_digest.hexdigest(),
        backend="passt_10s_5s",
        cache_dir=cache_dir,
        output_dir=output_dir,
    )
    if output_path.is_file() and not refresh_cache:
        return load_passt_features(output_path)

    try:
        import soxr
    except ImportError as error:
        raise ImportError("PaSST features require soxr") from error
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(target_device)
    items = collection_items(audio)
    if not items:
        raise ValueError("PaSST received no audio")

    keys = [key for key, _ in items]
    clip_keys: tp.List[str] = []
    logits: tp.List[np.ndarray] = []
    embeddings: tp.List[np.ndarray] = []
    pending_windows: tp.List[np.ndarray] = []
    window_size = 10 * target_sample_rate
    step_size = 5 * target_sample_rate

    def flush() -> None:
        if not pending_windows:
            return
        batch = torch.from_numpy(np.stack(pending_windows)).to(target_device)
        old_stft = torch.stft
        try:
            torch.stft = partial(torch.stft, return_complex=False)
            with torch.inference_mode():
                output = model(batch).detach().cpu().numpy().astype(np.float32, copy=False)
        finally:
            torch.stft = old_stft
        if output.ndim != 2 or output.shape[1] != 1295:
            raise ValueError(f"Unexpected PaSST mode='all' output shape: {output.shape}")
        logits.append(output[:, :527])
        embeddings.append(output[:, 527:])
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
            raise ValueError(f"PaSST found no valid 10-second window for {key}")
    flush()

    logits_array = np.concatenate(logits, axis=0)
    embedding_array = np.concatenate(embeddings, axis=0)
    np.savez_compressed(
        output_path,
        keys=np.asarray(keys),
        clip_keys=np.asarray(clip_keys),
        logits=logits_array,
        embeddings=embedding_array,
        sample_rate=np.asarray(target_sample_rate),
        window_seconds=np.asarray(10.0),
        hop_seconds=np.asarray(5.0),
        classifier_output=np.asarray("audioset_logits_527"),
        embedding_layer=np.asarray("passt_features_768_before_classifier"),
        schema_version=np.asarray(2),
    )
    return {
        "keys": keys,
        "clip_keys": clip_keys,
        "logits": logits_array,
        "embeddings": embedding_array,
        "cache_path": output_path,
    }

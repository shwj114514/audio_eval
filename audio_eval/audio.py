"""Audio loading shared by all metrics.

The public APIs accept paths, NumPy arrays, PyTorch tensors, or ``(audio,
sample_rate)`` tuples.  No ``AudioSample`` wrapper class is required.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from math import gcd
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}


def _is_tensor(value: Any) -> bool:
    return value.__class__.__module__.split(".")[0] == "torch" and hasattr(value, "detach")


def _is_array(value: Any) -> bool:
    return isinstance(value, np.ndarray) or _is_tensor(value)


def _to_numpy(value: Any) -> np.ndarray:
    if _is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim != 2:
        raise ValueError(f"Expected 1D or 2D audio, got shape {audio.shape}")
    if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
        return audio.mean(axis=0)
    return audio.mean(axis=1)


def load_audio(
    source: Any,
    *,
    sample_rate: int | None = None,
    target_sample_rate: int | None = None,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load or normalize one audio input.

    ``sample_rate`` is required for array/tensor inputs unless the input is an
    ``(audio, sample_rate)`` tuple. Paths carry their sample rate in the file.
    """
    if (
        isinstance(source, tuple)
        and len(source) == 2
        and _is_array(source[0])
        and isinstance(source[1], (int, np.integer))
    ):
        source, tuple_sample_rate = source
        if sample_rate is not None and sample_rate != int(tuple_sample_rate):
            raise ValueError("sample_rate conflicts with the value in (audio, sample_rate)")
        sample_rate = int(tuple_sample_rate)

    if isinstance(source, (str, os.PathLike, Path)):
        audio, source_sample_rate = sf.read(str(source), dtype="float32", always_2d=False)
        if sample_rate is not None and sample_rate != int(source_sample_rate):
            raise ValueError("Do not pass sample_rate for a file path with a different file sample rate")
        sample_rate = int(source_sample_rate)
    elif _is_array(source):
        if sample_rate is None:
            raise ValueError("sample_rate is required for NumPy array or PyTorch tensor input")
        audio = _to_numpy(source).astype(np.float32, copy=False)
    else:
        raise TypeError(f"Unsupported audio input type: {type(source)!r}")

    audio = np.asarray(audio, dtype=np.float32)
    if mono:
        audio = _to_mono(audio)
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinite values")
    if audio.size == 0:
        raise ValueError("Audio is empty")

    if target_sample_rate is not None and sample_rate != target_sample_rate:
        common = gcd(int(sample_rate), int(target_sample_rate))
        audio = resample_poly(audio, target_sample_rate // common, sample_rate // common, axis=-1)
        sample_rate = int(target_sample_rate)

    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)


def list_audio_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    files = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No supported audio files found in {directory}")
    return files


def collection_items(collection: Any) -> list[tuple[str, Any]]:
    """Return deterministic ``(key, source)`` items for a collection."""
    if isinstance(collection, (str, os.PathLike, Path)):
        path = Path(collection)
        if path.is_dir():
            return [
                (item.relative_to(path).with_suffix("").as_posix(), item)
                for item in list_audio_files(path)
            ]
        if path.is_file():
            return [(path.stem, path)]
        raise FileNotFoundError(path)

    if isinstance(collection, Mapping):
        return [(str(key), collection[key]) for key in sorted(collection, key=str)]

    if _is_array(collection) or (
        isinstance(collection, tuple)
        and len(collection) == 2
        and _is_array(collection[0])
    ):
        return [("0", collection)]

    if isinstance(collection, Sequence) and not isinstance(collection, (str, bytes)):
        return [(str(index), item) for index, item in enumerate(collection)]

    raise TypeError(f"Unsupported audio collection type: {type(collection)!r}")


def collection_fingerprint(collection: Any, *, sample_rate: int | None = None) -> str:
    """Fingerprint paths by manifest and arrays by content for cache invalidation."""
    digest = hashlib.sha256()
    digest.update(f"sample_rate={sample_rate}\n".encode())
    for key, source in collection_items(collection):
        digest.update(key.encode())
        if isinstance(source, (str, os.PathLike, Path)):
            path = Path(source)
            stat = path.stat()
            digest.update(str(path.resolve()).encode())
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        else:
            source_sample_rate = sample_rate
            if isinstance(source, tuple) and len(source) == 2 and _is_array(source[0]):
                source, source_sample_rate = source
            array = np.ascontiguousarray(_to_numpy(source))
            digest.update(f"{array.dtype}:{array.shape}:{source_sample_rate}".encode())
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


@contextmanager
def materialize_audio_collection(
    collection: Any,
    *,
    sample_rate: int | None = None,
) -> Iterator[Path]:
    """Expose any collection as a flat temporary directory for path-only backends."""
    with tempfile.TemporaryDirectory(prefix="audio_eval_") as temp_dir:
        root = Path(temp_dir)
        for index, (key, source) in enumerate(collection_items(collection)):
            safe_key = hashlib.sha1(key.encode()).hexdigest()[:12]
            if isinstance(source, (str, os.PathLike, Path)):
                source_path = Path(source).resolve()
                target = root / f"{index:08d}_{safe_key}{source_path.suffix.lower()}"
                target.symlink_to(source_path)
            else:
                audio, source_rate = load_audio(source, sample_rate=sample_rate, mono=False)
                target = root / f"{index:08d}_{safe_key}.wav"
                sf.write(target, audio, source_rate)
        yield root


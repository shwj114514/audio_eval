"""Fréchet distance with selectable OpenL3, PANNs, PaSST, or VGGish features."""

from __future__ import annotations

from pathlib import Path
import typing as tp
import numpy as np
from scipy import linalg

from audio_eval.audio import collection_fingerprint, collection_items, load_audio
from audio_eval.cache import cache_file, cat_features, ensure_audio_feature_cache, is_feature_map, load_feature_map, pair_feature_maps, stack_features
from audio_eval.common import MetricInput
from audio_eval.metrics.panns import get_panns_features, load_panns_features

_OPENL3_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "openl3_fd"

FD_OPTIONS: tp.Dict[str, tp.Dict[str, object]] = {
    "openl3": {"version": "openl3"},
    "panns": {"version": "panns"},
    "passt": {"version": "passt"},
    "vggish": {"version": "vggish"},
}

FD_REFERENCES: tp.Dict[str, tp.Dict[str, tp.Dict[str, object]]] = {
    "openl3": {
        "audiocaps": {"path": _OPENL3_ASSETS / "audiocaps-test__channels2__44100__openl3env__openl3hopsize0.5__batch4.npz", "content_type": "env"},
        "musiccaps": {"path": _OPENL3_ASSETS / "musiccaps-public__channels2__44100__openl3music__openl3hopsize0.5__batch4.npz", "content_type": "music"},
        "musiccaps_nosinging": {"path": _OPENL3_ASSETS / "musiccaps-public-nosinging__channels2__44100__openl3music__openl3hopsize0.5__batch4.npz", "content_type": "music"},
        "songdescriber": {"path": _OPENL3_ASSETS / "song_describer__channels2__44100__openl3music__openl3hopsize0.5__batch4.npz", "content_type": "music"},
        "songdescriber_nosinging": {"path": _OPENL3_ASSETS / "song_describer-nosinging__channels2__44100__openl3music__openl3hopsize0.5__batch4.npz", "content_type": "music"},
    }
}


def get_fd_options(option: str) -> tp.Dict[str, object]:
    if not option:
        return {}
    try:
        return dict(FD_OPTIONS[option])
    except KeyError as error:
        available = ", ".join(sorted(FD_OPTIONS))
        raise ValueError(f"Unknown FD option {option!r}. Available: {available}") from error


def frechet_distance(
    generated_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> float:
    if generated_embeddings.ndim != 2 or reference_embeddings.ndim != 2:
        raise ValueError("Embeddings must have shape [num_samples, embedding_dim]")
    if generated_embeddings.shape[0] < 2 or reference_embeddings.shape[0] < 2:
        raise ValueError("Fréchet distance requires at least two samples in each set")
    if generated_embeddings.shape[1] != reference_embeddings.shape[1]:
        raise ValueError("Generated and reference embedding dimensions differ")

    generated_mean = generated_embeddings.mean(axis=0)
    reference_mean = reference_embeddings.mean(axis=0)
    generated_covariance = np.cov(generated_embeddings, rowvar=False)
    reference_covariance = np.cov(reference_embeddings, rowvar=False)
    mean_difference = generated_mean - reference_mean
    covariance_mean, _ = linalg.sqrtm(
        generated_covariance.dot(reference_covariance).astype(np.complex128),
        disp=False,
    )
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(generated_covariance.shape[0]) * epsilon
        covariance_mean = linalg.sqrtm(
            (generated_covariance + offset).dot(reference_covariance + offset).astype(np.complex128)
        )
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(np.diag(covariance_mean).imag, 0, atol=1e-3):
            raise ValueError("Fréchet covariance square root has a large imaginary component")
        covariance_mean = covariance_mean.real
    score = (
        mean_difference.dot(mean_difference)
        + np.trace(generated_covariance)
        + np.trace(reference_covariance)
        - 2.0 * np.trace(covariance_mean)
    )
    return max(float(score), 0.0)


def _frechet_distance_from_statistics(
    generated_mean: np.ndarray,
    generated_covariance: np.ndarray,
    reference_mean: np.ndarray,
    reference_covariance: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> float:
    generated_mean = np.atleast_1d(generated_mean)
    reference_mean = np.atleast_1d(reference_mean)
    generated_covariance = np.atleast_2d(generated_covariance)
    reference_covariance = np.atleast_2d(reference_covariance)
    if generated_mean.shape != reference_mean.shape:
        raise ValueError("Generated and reference mean dimensions differ")
    if generated_covariance.shape != reference_covariance.shape:
        raise ValueError("Generated and reference covariance dimensions differ")

    mean_difference = generated_mean - reference_mean
    covariance_mean, _ = linalg.sqrtm(
        generated_covariance.dot(reference_covariance),
        disp=False,
    )
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(generated_covariance.shape[0]) * epsilon
        covariance_mean = linalg.sqrtm(
            (generated_covariance + offset).dot(reference_covariance + offset)
        )
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(np.diag(covariance_mean).imag, 0, atol=1e-3):
            raise ValueError("Fréchet covariance square root has a large imaginary component")
        covariance_mean = covariance_mean.real
    return max(float(
        mean_difference.dot(mean_difference)
        + np.trace(generated_covariance)
        + np.trace(reference_covariance)
        - 2.0 * np.trace(covariance_mean)
    ), 0.0)


def _openl3_embeddings(
    audio: MetricInput,
    *,
    sample_rate: int | None,
    target_sample_rate: int,
    channels: int,
    content_type: str,
    hop_size: float,
    batch_size: int,
    cache_dir: str | Path | None,
    refresh_cache: bool,
) -> tp.Tuple[np.ndarray, Path, int]:
    if channels not in {1, 2}:
        raise ValueError("OpenL3 channels must be 1 or 2")
    if content_type not in {"env", "music"}:
        raise ValueError("OpenL3 content_type must be env or music")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    fingerprint = collection_fingerprint(audio, sample_rate=sample_rate)
    if cache_dir is None:
        output_path = cache_file(
            "features",
            fingerprint,
            backend=f"openl3_{channels}ch_{target_sample_rate}_{content_type}_hop{hop_size}_batch{batch_size}",
            cache_dir=cache_dir,
        )
    else:
        cache_root = Path(cache_dir).expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        output_path = cache_root / (
            f"openl3_{channels}ch_{target_sample_rate}_{content_type}_"
            f"hop{hop_size}_batch{batch_size}_{fingerprint}.npz"
        )
    if output_path.is_file() and not refresh_cache:
        with np.load(output_path, allow_pickle=False) as loaded:
            return loaded["embeddings"], output_path, int(loaded["num_audio"])

    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        import openl3
        import soxr
    except ImportError as error:
        raise ImportError(
            "OpenL3 FD requires `pip install audio-eval[distribution]`, including TensorFlow"
        ) from error

    items = collection_items(audio)
    if not items:
        raise ValueError("OpenL3 FD received no audio")
    model = openl3.models.load_audio_embedding_model(
        input_repr="mel256",
        content_type=content_type,
        embedding_size=512,
    )
    batches: tp.List[np.ndarray] = []
    for start in range(0, len(items), batch_size):
        left_audio: tp.List[np.ndarray] = []
        right_audio: tp.List[np.ndarray] = []
        sample_rates: tp.List[int] = []
        for _, source in items[start : start + batch_size]:
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
            batches.append(np.concatenate(left_embeddings, axis=0))
        else:
            right_embeddings, _ = openl3.get_audio_embedding(
                right_audio,
                sample_rates,
                model=model,
                verbose=False,
                hop_size=hop_size,
                batch_size=batch_size,
            )
            batches.append(
                np.concatenate(
                    [
                        np.concatenate([left, right], axis=1)
                        for left, right in zip(left_embeddings, right_embeddings, strict=True)
                    ],
                    axis=0,
                )
            )

    embeddings = np.concatenate(batches, axis=0)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        num_audio=np.asarray(len(items)),
        target_sample_rate=np.asarray(target_sample_rate),
        channels=np.asarray(channels),
        content_type=np.asarray(content_type),
        hop_size=np.asarray(hop_size),
        batch_size=np.asarray(batch_size),
    )
    return embeddings, output_path, len(items)


def compute_fd(
    generated: MetricInput,
    reference: MetricInput | None = None,
    *,
    generated_sample_rate: int | None = None,
    reference_sample_rate: int | None = None,
    backend: str = "panns_cnn14",
    version: str = "panns",
    cache_dir: str | Path | None = None,
    generated_cache_dir: str | Path | None = None,
    reference_cache_dir: str | Path | None = None,
    batch_size: int = 8,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    refresh_cache: bool = False,
    target_sample_rate: int = 44100,
    channels: int = 2,
    content_type: str = "music",
    hop_size: float = 0.5,
) -> dict:
    if version not in {"openl3", "panns", "passt", "vggish"}:
        raise ValueError("version must be openl3, panns, passt, or vggish")

    reference_cache: Path | None = None
    if isinstance(reference, (str, Path)):
        reference_path = Path(reference).expanduser()
        if reference_path.exists():
            if reference_path.is_file() and reference_path.suffix.lower() in {".npz", ".pkl", ".pth"}:
                reference_cache = reference_path
                if version == "openl3" and reference_path.suffix.lower() == ".npz":
                    known_reference = next(
                        (
                            options
                            for options in FD_REFERENCES["openl3"].values()
                            if tp.cast(Path, options["path"]).name == reference_path.name
                        ),
                        None,
                    )
                    with np.load(reference_path, allow_pickle=False) as loaded:
                        if {"target_sample_rate", "channels", "content_type", "hop_size", "batch_size"} <= set(loaded.files):
                            target_sample_rate = int(loaded["target_sample_rate"])
                            channels = int(loaded["channels"])
                            content_type = str(loaded["content_type"])
                            hop_size = float(loaded["hop_size"])
                            batch_size = int(loaded["batch_size"])
                        elif known_reference is not None:
                            target_sample_rate = 44100
                            channels = 2
                            content_type = tp.cast(str, known_reference["content_type"])
                            hop_size = 0.5
                            batch_size = 4
                reference = None
            else:
                reference = reference_path
        elif str(reference) in FD_REFERENCES.get(version, {}):
            reference_options = FD_REFERENCES[version][str(reference)]
            reference_cache = tp.cast(Path, reference_options["path"])
            target_sample_rate = 44100
            channels = 2
            content_type = tp.cast(str, reference_options["content_type"])
            hop_size = 0.5
            batch_size = 4
            reference = None
        else:
            available = ", ".join(sorted(FD_REFERENCES.get(version, {})))
            raise FileNotFoundError(
                f"Unknown FD reference {reference!r} for {version}. Available bundled references: {available}"
            )

    if version == "openl3":
        generated_embedding_cache_dir = generated_cache_dir or cache_dir
        reference_embedding_cache_dir = reference_cache_dir or cache_dir
        generated_embeddings, generated_cache, num_generated = _openl3_embeddings(
            generated,
            sample_rate=generated_sample_rate,
            target_sample_rate=target_sample_rate,
            channels=channels,
            content_type=content_type,
            hop_size=hop_size,
            batch_size=batch_size,
            cache_dir=generated_embedding_cache_dir,
            refresh_cache=refresh_cache,
        )
        if generated_embeddings.ndim != 2 or generated_embeddings.shape[0] < 2:
            raise ValueError("OpenL3 FD requires at least two generated embedding windows")
        generated_mean = generated_embeddings.mean(axis=0)
        generated_covariance = np.cov(generated_embeddings, rowvar=False)

        num_reference: int | None = None
        if reference_cache is not None:
            with np.load(reference_cache, allow_pickle=False) as loaded:
                if {"mu_ref", "sigma_ref"} <= set(loaded.files):
                    reference_mean = loaded["mu_ref"]
                    reference_covariance = loaded["sigma_ref"]
                elif "embeddings" in loaded.files:
                    reference_embeddings = loaded["embeddings"]
                    if reference_embeddings.ndim != 2 or reference_embeddings.shape[0] < 2:
                        raise ValueError("OpenL3 reference cache needs at least two embeddings")
                    reference_mean = reference_embeddings.mean(axis=0)
                    reference_covariance = np.cov(reference_embeddings, rowvar=False)
                    num_reference = int(loaded["num_audio"]) if "num_audio" in loaded.files else None
                else:
                    raise ValueError(
                        "OpenL3 reference NPZ must contain mu_ref and sigma_ref or embeddings"
                    )
        else:
            if reference is None:
                raise ValueError("OpenL3 FD requires reference audio or a reference cache")
            reference_embeddings, reference_cache, num_reference = _openl3_embeddings(
                reference,
                sample_rate=reference_sample_rate,
                target_sample_rate=target_sample_rate,
                channels=channels,
                content_type=content_type,
                hop_size=hop_size,
                batch_size=batch_size,
                cache_dir=reference_embedding_cache_dir,
                refresh_cache=refresh_cache,
            )
            if reference_embeddings.ndim != 2 or reference_embeddings.shape[0] < 2:
                raise ValueError("OpenL3 FD requires at least two reference embedding windows")
            reference_mean = reference_embeddings.mean(axis=0)
            reference_covariance = np.cov(reference_embeddings, rowvar=False)

        score = _frechet_distance_from_statistics(
            generated_mean,
            generated_covariance,
            reference_mean,
            reference_covariance,
        )
        return {
            "fd": score,
            "version": version,
            "backend": "openl3_mel256",
            "content_type": content_type,
            "sample_rate": target_sample_rate,
            "channels": channels,
            "hop_size": hop_size,
            "num_generated": num_generated,
            "num_reference": num_reference,
            "num_embeddings_generated": generated_embeddings.shape[0],
            "generated_cache": str(generated_cache),
            "reference_cache": str(reference_cache),
        }

    filename = {
        "panns": "pann_features.pth",
        "passt": "passt_features_embed.pth",
        "vggish": "vggish_features.pth",
    }[version]
    layer = "2048" if version == "panns" else None
    generated_is_av_cache = generated_sample_rate is None and is_feature_map(
        generated, filename=filename, layer=layer
    )
    if version in {"passt", "vggish"} and not generated_is_av_cache:
        generated = ensure_audio_feature_cache(
            generated,
            sample_rate=generated_sample_rate,
            cache_dir=cache_dir,
            batch_size=batch_size,
            device=device,
            refresh_cache=refresh_cache,
        )
        if reference_cache is None:
            if reference is None:
                raise ValueError("FD requires reference audio or a reference cache")
            reference = ensure_audio_feature_cache(
                reference,
                sample_rate=reference_sample_rate,
                cache_dir=cache_dir,
                batch_size=batch_size,
                device=device,
                refresh_cache=refresh_cache,
            )
        generated_is_av_cache = True
    if version != "panns" or generated_is_av_cache:
        if reference is None and reference_cache is None:
            raise ValueError("FD requires reference features")
        generated_map = load_feature_map(generated, filename=filename, layer=layer)
        reference_map = load_feature_map(
            reference_cache if reference_cache is not None else reference,
            filename=filename,
            layer=layer,
        )
        if version == "passt":
            _, reference_tensor, generated_tensor, unpaired = pair_feature_maps(
                reference_map, generated_map
            )
        else:
            unpaired = []
            generated_tensor = (
                cat_features(generated_map) if version == "vggish" else stack_features(generated_map)
            )
            reference_tensor = (
                cat_features(reference_map) if version == "vggish" else stack_features(reference_map)
            )
        score = frechet_distance(generated_tensor.numpy(), reference_tensor.numpy())
        return {
            "fd": score,
            "version": version,
            "num_generated": len(generated_map),
            "num_reference": len(reference_map),
            "num_embeddings_generated": generated_tensor.shape[0],
            "num_embeddings_reference": reference_tensor.shape[0],
            "unpaired": unpaired,
        }

    generated_features = get_panns_features(
        generated,
        sample_rate=generated_sample_rate,
        backend=backend,
        cache_dir=cache_dir,
        batch_size=batch_size,
        device=device,
        checkpoint_path=checkpoint_path,
        refresh_cache=refresh_cache,
    )
    if reference_cache is not None:
        reference_data = load_panns_features(reference_cache)
    else:
        if reference is None:
            raise ValueError("FD requires reference audio or a reference cache")
        reference_data = get_panns_features(
            reference,
            sample_rate=reference_sample_rate,
            backend=backend,
            cache_dir=cache_dir,
            batch_size=batch_size,
            device=device,
            checkpoint_path=checkpoint_path,
            refresh_cache=refresh_cache,
        )
    score = frechet_distance(generated_features["embeddings"], reference_data["embeddings"])
    return {
        "fd": score,
        "version": version,
        "backend": backend,
        "num_generated": len(generated_features["keys"]),
        "num_reference": len(reference_data["keys"]),
        "generated_cache": str(generated_features["cache_path"]),
        "reference_cache": str(reference_data["cache_path"]),
    }

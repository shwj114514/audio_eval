"""
Fréchet distance with selectable OpenL3, PANNs, PaSST, or VGGish features.
Reference:
https://github.com/Stability-AI/stable-audio-metrics/blob/main/src/openl3_fd.py

Stable Audio Open:
* Reported result in the paper: 96.51
* Reproduced result: 97.9706206573

"""

from __future__ import annotations

from pathlib import Path
import typing as tp
import numpy as np
from scipy import linalg

from audio_eval.common import MetricInput
from audio_eval.features.openl3 import get_openl3_features
from audio_eval.features.panns import get_panns_features
from audio_eval.features.passt import get_passt_features
from audio_eval.features.vggish import get_vggish_features

_OPENL3_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "openl3_fd"

FD_OPTIONS: tp.Dict[str, tp.Dict[str, object]] = {
    # upd0721  The ambiguous legacy FD option `openl3` is not accepted.
    "openl3_env": {"version": "openl3", "openl3_content_type": "env"},
    "openl3_music": {"version": "openl3", "openl3_content_type": "music"},
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


def _resolve_openl3_content_type(
    requested: str | None,
    reference_content_type: str,
    *,
    reference_label: str,
) -> str:
    if requested is not None and requested != reference_content_type:
        raise ValueError(
            f"Requested OpenL3 content_type={requested!r}, but reference "
            f"{reference_label} uses content_type={reference_content_type!r}"
        )
    return reference_content_type


def frechet_distance(
    generated_embeddings: np.ndarray, # [num_embeddings, embedding_dim]
    reference_embeddings: np.ndarray, # [num_embeddings, embedding_dim]
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
    """Compute FD from precomputed means and covariances, such as OpenL3 reference statistics."""
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
    openl3_channels: int = 2,
    # Explicit OpenL3 model choice; only used with version="openl3".
    openl3_content_type: str | None = None,
    openl3_hop_size: float = 0.5,
) -> tp.Dict[str, tp.Any]:
    """
        Generated audio → embedding distribution → mean and covariance
        Reference audio → embedding distribution → mean and covariance
        Two Gaussian distributions → Fréchet distance

    """

    if version not in {"openl3", "panns", "passt", "vggish"}:
        raise ValueError("version must be openl3, panns, passt, or vggish")
    if openl3_content_type not in {None, "env", "music"}:
        raise ValueError("openl3_content_type must be env or music")

    reference_cache: Path | None = None
    if isinstance(reference, (str, Path)):
        # reference = "/data/reference" ｜ "/data/reference.npz"
        reference_path = Path(reference).expanduser()
        if reference_path.exists():
            if reference_path.is_file() and reference_path.suffix.lower() in {".npz", ".pkl", ".pth"}:
                # Treat it as a precomputed reference feature cache. The reference audio does not need to be loaded again afterward.
                reference_cache = reference_path
                if version == "openl3" and reference_path.suffix.lower() == ".npz":
                    # In addition to possibly storing `mu_ref` and `sigma_ref`, the OpenL3 reference cache may also store the feature-extraction configuration
                    # like `target_sample_rate` `channels`
                    known_reference = next(
                        (
                            options
                            for options in FD_REFERENCES["openl3"].values()
                            if tp.cast(Path, options["path"]).name == reference_path.name
                        ),
                        None,
                    )
                    with np.load(reference_path, allow_pickle=False) as loaded:
                        # load from https://github.com/Stability-AI/stable-audio-metrics/tree/main/load/openl3_fd
                        # Adopt the reference extraction configuration after verifying
                        # that it agrees with the explicitly selected public option.
                        if {"target_sample_rate", "channels", "content_type", "hop_size", "batch_size"} <= set(loaded.files):
                            reference_content_type = str(loaded["content_type"])
                            openl3_content_type = _resolve_openl3_content_type(
                                openl3_content_type,
                                reference_content_type,
                                reference_label=str(reference_path),
                            )
                            target_sample_rate = int(loaded["target_sample_rate"])
                            openl3_channels = int(loaded["channels"])
                            openl3_hop_size = float(loaded["hop_size"])
                            batch_size = int(loaded["batch_size"])
                        elif known_reference is not None:
                            # Use the official default reference configuration if the cache file does not contain the configuration.
                            reference_content_type = tp.cast(str, known_reference["content_type"])
                            openl3_content_type = _resolve_openl3_content_type(
                                openl3_content_type,
                                reference_content_type,
                                reference_label=str(reference_path),
                            )
                            target_sample_rate = 44100
                            openl3_channels = 2
                            openl3_hop_size = 0.5
                            batch_size = 4
                        else:
                            raise ValueError(
                                f"OpenL3 reference cache {reference_path} is missing "
                                "extraction configuration metadata and is not a known "
                                "bundled reference"
                            )
                reference = None
            else:
                reference = reference_path
        elif str(reference) in FD_REFERENCES.get(version, {}):
            reference_options = FD_REFERENCES[version][str(reference)]
            reference_cache = tp.cast(Path, reference_options["path"])
            reference_content_type = tp.cast(str, reference_options["content_type"])
            openl3_content_type = _resolve_openl3_content_type(
                openl3_content_type,
                reference_content_type,
                reference_label=str(reference),
            )
            target_sample_rate = 44100
            openl3_channels = 2
            openl3_hop_size = 0.5
            batch_size = 4
            reference = None
        else:
            available = ", ".join(sorted(FD_REFERENCES.get(version, {})))
            raise FileNotFoundError(
                f"Unknown FD reference {reference!r} for {version}. Available bundled references: {available}"
            )

    if version == "openl3":
        if openl3_content_type is None:
            raise ValueError(
                "OpenL3 FD requires an explicit content type; use metric option "
                "openl3_music or openl3_env"
            )
        generated_features = get_openl3_features(
            generated,
            sample_rate=generated_sample_rate,
            target_sample_rate=target_sample_rate,
            channels=openl3_channels,
            content_type=openl3_content_type,
            hop_size=openl3_hop_size,
            batch_size=batch_size,
            cache_dir=cache_dir,
            output_dir=generated_cache_dir,
            refresh_cache=refresh_cache,
        )
        generated_embeddings = generated_features["embeddings"]
        generated_cache = generated_features["cache_path"]
        num_generated = len(generated_features["keys"])
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
            reference_features = get_openl3_features(
                reference,
                sample_rate=reference_sample_rate,
                target_sample_rate=target_sample_rate,
                channels=openl3_channels,
                content_type=openl3_content_type,
                hop_size=openl3_hop_size,
                batch_size=batch_size,
                cache_dir=cache_dir,
                output_dir=reference_cache_dir,
                refresh_cache=refresh_cache,
            )
            reference_embeddings = reference_features["embeddings"]
            reference_cache = reference_features["cache_path"]
            num_reference = len(reference_features["keys"])
            if reference_embeddings.ndim != 2 or reference_embeddings.shape[0] < 2:
                raise ValueError("OpenL3 FD requires at least two reference embedding windows")
            reference_mean = reference_embeddings.mean(axis=0)
            reference_covariance = np.cov(reference_embeddings, rowvar=False)

        # Official OpenL3 reference caches may contain only `mu_ref` and `sigma_ref`,
        # so compute FAD from distribution statistics; the underlying formula is unchanged.
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
            "content_type": openl3_content_type,
            "sample_rate": target_sample_rate,
            "channels": openl3_channels,
            "hop_size": openl3_hop_size,
            "num_generated": num_generated,
            "num_reference": num_reference,
            "num_embeddings_generated": generated_embeddings.shape[0],
            "generated_cache": str(generated_cache),
            "reference_cache": str(reference_cache),
        }

    reference_source: MetricInput | None = reference_cache if reference_cache is not None else reference
    if reference_source is None and reference_cache_dir is not None:
        reference_source = Path(reference_cache_dir).expanduser()
    if reference_source is None:
        raise ValueError("FD requires reference audio or a reference feature cache")

    if version == "panns":
        generated_features = get_panns_features(
            generated,
            sample_rate=generated_sample_rate,
            backend=backend,
            cache_dir=cache_dir,
            output_dir=generated_cache_dir,
            batch_size=batch_size,
            device=device,
            checkpoint_path=checkpoint_path,
            refresh_cache=refresh_cache,
        )
        reference_data = get_panns_features(
            reference_source,
            sample_rate=reference_sample_rate,
            backend=backend,
            cache_dir=cache_dir,
            output_dir=reference_cache_dir,
            batch_size=batch_size,
            device=device,
            checkpoint_path=checkpoint_path,
            refresh_cache=refresh_cache,
        )
        feature_backend = backend
    elif version == "passt":
        generated_features = get_passt_features(
            generated,
            sample_rate=generated_sample_rate,
            cache_dir=cache_dir,
            output_dir=generated_cache_dir,
            batch_size=batch_size,
            device=device,
            refresh_cache=refresh_cache,
        )
        reference_data = get_passt_features(
            reference_source,
            sample_rate=reference_sample_rate,
            cache_dir=cache_dir,
            output_dir=reference_cache_dir,
            batch_size=batch_size,
            device=device,
            refresh_cache=refresh_cache,
        )
        feature_backend = "passt_s_swa_p16_128_ap476"
    else:
        generated_features = get_vggish_features(
            generated,
            sample_rate=generated_sample_rate,
            cache_dir=cache_dir,
            output_dir=generated_cache_dir,
            device=device,
            refresh_cache=refresh_cache,
        )
        reference_data = get_vggish_features(
            reference_source,
            sample_rate=reference_sample_rate,
            cache_dir=cache_dir,
            output_dir=reference_cache_dir,
            device=device,
            refresh_cache=refresh_cache,
        )
        feature_backend = "vggish_128"

    score = frechet_distance(generated_features["embeddings"], reference_data["embeddings"])
    return {
        "fd": score,
        "version": version,
        "backend": feature_backend,
        "num_generated": len(generated_features["keys"]),
        "num_reference": len(reference_data["keys"]),
        "num_embeddings_generated": generated_features["embeddings"].shape[0],
        "num_embeddings_reference": reference_data["embeddings"].shape[0],
        "generated_cache": str(generated_features["cache_path"]),
        "reference_cache": str(reference_data["cache_path"]),
    }

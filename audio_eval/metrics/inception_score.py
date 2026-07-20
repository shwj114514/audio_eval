"""Inception Score using PANNs or PaSST classification-head outputs."""

from __future__ import annotations

from pathlib import Path
import typing as tp

import numpy as np

from audio_eval.common import MetricInput
from audio_eval.features.panns import get_panns_features
from audio_eval.features.passt import get_passt_features


INCEPTION_SCORE_OPTIONS: tp.Dict[str, tp.Dict[str, object]] = {
    "panns": {"version": "panns"},
    "passt": {"version": "passt"},
}


def get_inception_score_options(option: str) -> tp.Dict[str, object]:
    if not option:
        return {}
    try:
        return dict(INCEPTION_SCORE_OPTIONS[option])
    except KeyError as error:
        available = ", ".join(sorted(INCEPTION_SCORE_OPTIONS))
        raise ValueError(
            f"Unknown Inception Score option {option!r}. Available: {available}"
        ) from error


def _normalized(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.maximum(probabilities.astype(np.float64), 1e-12)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def compute_inception_score(
    generated: MetricInput,
    *,
    sample_rate: int | None = None,
    backend: str = "panns_cnn14",
    version: str = "panns",
    splits: int = 10,
    shuffle: bool = True,
    seed: int = 2020,
    cache_dir: str | Path | None = None,
    generated_cache_dir: str | Path | None = None,
    batch_size: int = 8,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    refresh_cache: bool = False,
) -> tp.Dict[str, tp.Any]:
    if version not in {"panns", "passt"}:
        raise ValueError("version must be panns or passt")
    if splits < 1:
        raise ValueError("splits must be positive")

    if version == "panns":
        features = get_panns_features(
            generated,
            sample_rate=sample_rate,
            backend=backend,
            cache_dir=cache_dir,
            output_dir=generated_cache_dir,
            batch_size=batch_size,
            device=device,
            checkpoint_path=checkpoint_path,
            refresh_cache=refresh_cache,
        )
        probabilities = _normalized(features["probabilities"])
        feature_backend = backend
    else:
        features = get_passt_features(
            generated,
            sample_rate=sample_rate,
            cache_dir=cache_dir,
            output_dir=generated_cache_dir,
            batch_size=batch_size,
            device=device,
            refresh_cache=refresh_cache,
        )
        probabilities = _softmax(features["logits"])
        feature_backend = "passt_s_swa_p16_128_ap476"

    if probabilities.shape[0] < splits:
        splits = 1
    if shuffle:
        permutation = np.random.RandomState(seed).permutation(probabilities.shape[0])
        probabilities = probabilities[permutation]

    split_scores: tp.List[float] = []
    for index in range(splits):
        start = index * probabilities.shape[0] // splits
        end = (index + 1) * probabilities.shape[0] // splits
        probability = probabilities[start:end]
        marginal = np.maximum(probability.mean(axis=0, keepdims=True), 1e-12)
        divergence = np.sum(
            probability * (np.log(np.maximum(probability, 1e-12)) - np.log(marginal)),
            axis=-1,
        )
        split_scores.append(float(np.exp(divergence.mean())))

    return {
        "inception_score": float(np.mean(split_scores)),
        "std": float(np.std(split_scores)),
        "version": version,
        "backend": feature_backend,
        "num_samples": len(features["keys"]),
        "num_windows": len(features["clip_keys"]),
        "splits": splits,
        "shuffle": shuffle,
        "seed": seed,
        "split_scores": split_scores,
        "cache": str(features["cache_path"]),
    }

"""Inception Score with selectable PANNs or PaSST logits."""

from __future__ import annotations

from pathlib import Path
import typing as tp
import numpy as np

from audio_eval.cache import ensure_audio_feature_cache, is_feature_map, load_feature_map, stack_features
from audio_eval.common import MetricInput
from audio_eval.metrics.panns import get_panns_features

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
    batch_size: int = 8,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    refresh_cache: bool = False,
) -> dict:
    if version not in {"panns", "passt"}:
        raise ValueError("version must be panns or passt")
    filename = "pann_features.pth" if version == "panns" else "passt_logits.pth"
    layer = "logits" if version == "panns" else None
    generated_is_av_cache = sample_rate is None and is_feature_map(
        generated, filename=filename, layer=layer
    )
    if version == "passt" and not generated_is_av_cache:
        generated = ensure_audio_feature_cache(
            generated,
            sample_rate=sample_rate,
            cache_dir=cache_dir,
            batch_size=batch_size,
            device=device,
            refresh_cache=refresh_cache,
        )
        generated_is_av_cache = True
    if version == "passt" or generated_is_av_cache:
        logits = stack_features(load_feature_map(generated, filename=filename, layer=layer))
        if logits.shape[0] < splits:
            splits = 1
        if shuffle:
            permutation = np.random.RandomState(seed).permutation(logits.shape[0])
            logits = logits[permutation]
        logits = logits.double()
        probabilities = logits.softmax(dim=1)
        log_probabilities = logits.log_softmax(dim=1)
        split_scores: list[float] = []
        for index in range(splits):
            start = index * logits.shape[0] // splits
            end = (index + 1) * logits.shape[0] // splits
            probability = probabilities[start:end]
            log_probability = log_probabilities[start:end]
            marginal = probability.mean(dim=0, keepdim=True)
            score = (probability * (log_probability - marginal.log())).sum(dim=1).mean().exp()
            split_scores.append(float(score))
        return {
            "inception_score": float(np.mean(split_scores)),
            "std": float(np.std(split_scores)),
            "version": version,
            "num_samples": logits.shape[0],
            "splits": splits,
            "shuffle": shuffle,
            "seed": seed,
            "split_scores": split_scores,
        }

    features = get_panns_features(
        generated,
        sample_rate=sample_rate,
        backend=backend,
        cache_dir=cache_dir,
        batch_size=batch_size,
        device=device,
        checkpoint_path=checkpoint_path,
        refresh_cache=refresh_cache,
    )
    probabilities = np.maximum(features["probabilities"].astype(np.float64), 1e-12)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    marginal = probabilities.mean(axis=0, keepdims=True)
    divergences = np.sum(probabilities * (np.log(probabilities) - np.log(marginal)), axis=-1)
    return {
        "inception_score": float(np.exp(divergences.mean())),
        "version": version,
        "backend": backend,
        "num_samples": len(features["keys"]),
        "cache": str(features["cache_path"]),
    }

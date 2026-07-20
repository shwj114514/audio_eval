"""ImageBind cosine between mean audio and mean video window embeddings."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from audio_eval.cache import (
    ensure_audio_feature_cache,
    is_feature_map,
    load_feature_map,
    pair_feature_maps,
)
from audio_eval.common import FeatureInput, MetricInput


def _mean_window_embeddings(
    features: dict[str, torch.Tensor],
    *,
    modality: str,
) -> dict[str, torch.Tensor]:
    """Reduce ``[num_windows, embedding_dim]`` features to one vector per item."""
    output: dict[str, torch.Tensor] = {}
    for key, value in features.items():
        if value.ndim == 1:
            embedding = value
        elif value.ndim == 2 and value.shape[0] > 0:
            embedding = value.mean(dim=0)
        else:
            raise ValueError(
                f"ImageBind {modality} feature {key!r} must have shape "
                "[embedding_dim] or [num_windows, embedding_dim], "
                f"got {tuple(value.shape)}"
            )
        if embedding.numel() == 0 or not torch.isfinite(embedding).all():
            raise ValueError(f"Invalid ImageBind {modality} feature {key!r}")
        output[key] = embedding
    return output


def compute_imagebind(
    generated: MetricInput,
    reference: FeatureInput,
    *,
    generated_sample_rate: int | None = None,
    cache_dir: str | Path | None = None,
    batch_size: int = 64,
    device: str | None = None,
    refresh_cache: bool = False,
) -> dict[str, object]:
    if generated_sample_rate is not None or not is_feature_map(
        generated, filename="imagebind_audio.pth", layer=None
    ):
        generated = ensure_audio_feature_cache(
            generated,
            sample_rate=generated_sample_rate,
            cache_dir=cache_dir,
            batch_size=batch_size,
            device=device,
            include_video_metrics=True,
            refresh_cache=refresh_cache,
        )
    generated_map = _mean_window_embeddings(
        load_feature_map(generated, filename="imagebind_audio.pth"),
        modality="audio",
    )
    reference_map = _mean_window_embeddings(
        load_feature_map(reference, filename="imagebind_video.pth"),
        modality="video",
    )
    keys, reference_features, generated_features, unpaired = pair_feature_maps(
        reference_map, generated_map
    )
    per_sample = F.cosine_similarity(reference_features, generated_features, dim=-1)
    score = float(per_sample.mean())
    return {
        "imagebind_score": score,
        "scaled_score": score * 100.0,
        "num_samples": len(keys),
        "unpaired": unpaired,
        "details": [
            {
                "id": key,
                "imagebind_score": float(value),
                "scaled_score": float(value) * 100.0,
            }
            for key, value in zip(keys, per_sample, strict=True)
        ],
    }

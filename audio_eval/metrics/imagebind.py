"""ImageBind cosine similarity between generated audio and reference video."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pathlib import Path

from audio_eval.cache import ensure_audio_feature_cache, is_feature_map, load_feature_map, pair_feature_maps
from audio_eval.common import FeatureInput, MetricInput


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
    generated_map = load_feature_map(generated, filename="imagebind_audio.pth")
    reference_map = load_feature_map(reference, filename="imagebind_video.pth")
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
            {"id": key, "imagebind_score": float(value), "scaled_score": float(value) * 100.0}
            for key, value in zip(keys, per_sample, strict=True)
        ],
    }

"""Synchformer audio-video temporal misalignment score."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from audio_eval.cache import ensure_audio_feature_cache, is_feature_map, load_feature_map, pair_feature_maps
from audio_eval.common import FeatureInput, MetricInput


_SYNCHFORMER_URL = (
    "https://github.com/hkchengrex/MMAudio/releases/download/v0.1/"
    "synchformer_state_dict.pth"
)


def compute_desync(
    generated: MetricInput,
    reference: FeatureInput,
    *,
    generated_sample_rate: int | None = None,
    cache_dir: str | Path | None = None,
    extraction_batch_size: int = 64,
    refresh_cache: bool = False,
    checkpoint_path: str | Path | None = None,
    batch_size: int = 16,
    device: str | None = None,
    model: torch.nn.Module | None = None,
) -> dict[str, object]:
    if generated_sample_rate is not None or not is_feature_map(
        generated, filename="synchformer_audio.pth", layer=None
    ):
        generated = ensure_audio_feature_cache(
            generated,
            sample_rate=generated_sample_rate,
            cache_dir=cache_dir,
            batch_size=extraction_batch_size,
            device=device,
            include_video_metrics=True,
            refresh_cache=refresh_cache,
        )
    generated_map = load_feature_map(generated, filename="synchformer_audio.pth")
    reference_map = load_feature_map(reference, filename="synchformer_video.pth")
    keys, reference_features, generated_features, unpaired = pair_feature_maps(
        reference_map, generated_map
    )
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        try:
            from av_bench.synchformer.synchformer import Synchformer, make_class_grid
        except ImportError as error:
            raise ImportError(
                "DeSync requires the official hkchengrex/av-benchmark package"
            ) from error
        model = Synchformer()
        checkpoint = checkpoint_path or os.environ.get("AUDIO_EVAL_SYNCHFORMER_CHECKPOINT")
        if checkpoint is None:
            checkpoint = Path.home() / ".cache/audio_eval/synchformer_state_dict.pth"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if not checkpoint.is_file():
                torch.hub.download_url_to_file(_SYNCHFORMER_URL, str(checkpoint))
        state = torch.load(Path(checkpoint).expanduser(), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        grid = make_class_grid(-2, 2, 21)
    else:
        grid = torch.from_numpy(np.linspace(-2, 2, 21)).float()
    model = model.to(target_device).eval()

    first_scores: list[float] = []
    last_scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(keys), batch_size):
            video = reference_features[start:start + batch_size].to(target_device)
            audio = generated_features[start:start + batch_size].to(target_device)
            first_ids = model.compare_v_a(video[:, :14], audio[:, :14]).argmax(dim=-1).cpu()
            last_ids = model.compare_v_a(video[:, -14:], audio[:, -14:]).argmax(dim=-1).cpu()
            first_scores.extend(abs(float(grid[index])) for index in first_ids)
            last_scores.extend(abs(float(grid[index])) for index in last_ids)
    per_sample = [(first + last) / 2.0 for first, last in zip(first_scores, last_scores, strict=True)]
    return {
        "desync": float(np.mean(first_scores + last_scores)),
        "unit": "seconds",
        "num_samples": len(keys),
        "unpaired": unpaired,
        "details": [
            {
                "id": key,
                "first_4_8s": first,
                "last_4_8s": last,
                "desync": score,
            }
            for key, first, last, score in zip(
                keys, first_scores, last_scores, per_sample, strict=True
            )
        ],
    }

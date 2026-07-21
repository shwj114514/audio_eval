"""Synchformer audio-video temporal misalignment score."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import typing as tp
import torch

from audio_eval.cache import ensure_audio_feature_cache, is_feature_map, load_feature_map, pair_feature_maps
from audio_eval.common import FeatureInput, MetricInput


_SYNCHFORMER_URL = (
    "https://github.com/hkchengrex/MMAudio/releases/download/v0.1/"
    "synchformer_state_dict.pth"
)

_DESYNC_OPTIONS = {
    # MMAudio (https://openaccess.thecvf.com/content/CVPR2025/html/Cheng_MMAudio_Taming_Multimodal_Joint_Training_for_High-Quality_Video-to-Audio_Synthesis_CVPR_2025_paper.html)
    # the exact first/last 4.8s rule is from its official av-benchmark.  https://github.com/hkchengrex/av-benchmark/blob/main/av_bench/evaluate.py
    "first_last": {"protocol": "first_last"},
    # AS-Synchformer (https://bmvc2025.bmva.org/proceedings/903/) adapted here to score every full 14-segment window with its 2-segment/0.64s hop.
    "sliding_2seg": {"protocol": "sliding_2seg"},
}


def get_desync_options(option: str) -> tp.Dict[str, tp.Any]:
    if not option:
        return {}
    try:
        return dict(_DESYNC_OPTIONS[option])
    except KeyError as error:
        available = ", ".join(sorted(_DESYNC_OPTIONS))
        raise ValueError(
            f"Unknown DeSync option {option!r}. Available: {available}"
        ) from error


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
    protocol: str = "first_last",
) -> tp.Dict[str, tp.Any]:
    if protocol not in _DESYNC_OPTIONS:
        available = ", ".join(sorted(_DESYNC_OPTIONS))
        raise ValueError(f"Unknown DeSync protocol {protocol!r}. Available: {available}")
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

    if protocol == "sliding_2seg":
        window_segments = 14
        hop_segments = 2
        num_segments = min(reference_features.shape[1], generated_features.shape[1])
        if num_segments < window_segments:
            raise ValueError(
                f"DeSync requires at least {window_segments} complete segments; "
                f"found {num_segments}"
            )
        window_starts = list(
            range(0, num_segments - window_segments + 1, hop_segments)
        )
        scores_by_sample: tp.List[tp.List[float]] = [[] for _ in keys]
        with torch.inference_mode():
            for batch_start in range(0, len(keys), batch_size):
                video = reference_features[batch_start:batch_start + batch_size].to(
                    target_device
                )
                audio = generated_features[batch_start:batch_start + batch_size].to(
                    target_device
                )
                for window_start in window_starts:
                    window_end = window_start + window_segments
                    class_ids = model.compare_v_a(
                        video[:, window_start:window_end],
                        audio[:, window_start:window_end],
                    ).argmax(dim=-1).cpu()
                    scores = [abs(float(grid[index])) for index in class_ids]
                    for sample_scores, score in zip(
                        scores_by_sample[batch_start:batch_start + len(scores)],
                        scores,
                        strict=True,
                    ):
                        sample_scores.append(score)
        per_sample = [float(np.mean(scores)) for scores in scores_by_sample]
        return {
            "desync": float(np.mean(per_sample)),
            "unit": "seconds",
            "protocol": protocol,
            "window_segments": window_segments,
            "window_seconds": 4.8,
            "hop_segments": hop_segments,
            "hop_seconds": 0.64,
            "num_samples": len(keys),
            "unpaired": unpaired,
            "details": [
                {
                    "id": key,
                    "desync": score,
                    "windows": [
                        {
                            "start_seconds": round(window_start * 0.32, 10),
                            "end_seconds": round(window_start * 0.32 + 4.8, 10),
                            "desync": window_score,
                        }
                        for window_start, window_score in zip(
                            window_starts, window_scores, strict=True
                        )
                    ],
                }
                for key, score, window_scores in zip(
                    keys, per_sample, scores_by_sample, strict=True
                )
            ],
        }

    first_scores: tp.List[float] = []
    last_scores: tp.List[float] = []
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

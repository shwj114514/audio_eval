"""UTMOS prediction for generated speech."""

from __future__ import annotations

import numpy as np
import torch

from audio_eval.audio import collection_items, load_audio
from audio_eval.common import AudioCollection

_PREDICTORS: dict[tuple[str, str], torch.nn.Module] = {}


def _load_predictor(backend: str, device: str) -> torch.nn.Module:
    key = (backend, device)
    if key not in _PREDICTORS:
        if backend != "utmos22_strong":
            raise ValueError(f"Unsupported UTMOS backend: {backend!r}")
        predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True,
        )
        predictor = predictor.to(device)
        predictor.eval()
        _PREDICTORS[key] = predictor
    return _PREDICTORS[key]


def compute_utmos(
    generated: AudioCollection,
    *,
    sample_rate: int | None = None,
    backend: str = "utmos22_strong",
    device: str | None = None,
) -> dict:
    """Compute mean UTMOS for a file, directory, array, tensor, or collection."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = _load_predictor(backend, device)

    scores: list[float] = []
    details: list[dict] = []
    for key, source in collection_items(generated):
        audio, source_sample_rate = load_audio(source, sample_rate=sample_rate)
        waveform = torch.from_numpy(audio).unsqueeze(0).to(device)
        with torch.no_grad():
            score = float(predictor(waveform, source_sample_rate).reshape(-1).mean().item())
        scores.append(score)
        details.append({"id": key, "utmos": score})

    if not scores:
        raise ValueError("No audio samples were evaluated for UTMOS")
    return {
        "utmos": float(np.mean(scores)),
        "backend": backend,
        "num_samples": len(scores),
        "details": details,
    }

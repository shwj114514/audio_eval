"""Log-spectral distance for paired reconstruction or super-resolution audio."""

from __future__ import annotations

from pathlib import Path
import typing as tp

import numpy as np
from scipy.signal import stft

from audio_eval.audio import load_audio
from audio_eval.common import PairedAudioCollection, load_aligned_pair, mean_or_raise, paired_sources

LSD_OPTIONS: tp.Dict[str, tp.Dict[str, object]] = {
    "standard": {"version": "standard"},
    "ssr_eval": {"version": "ssr_eval"},
}


def get_lsd_options(option: str) -> tp.Dict[str, object]:
    if not option:
        return {}
    try:
        return dict(LSD_OPTIONS[option])
    except KeyError as error:
        available = ", ".join(sorted(LSD_OPTIONS))
        raise ValueError(f"Unknown LSD option {option!r}. Available: {available}") from error


def compute_lsd(
    generated: PairedAudioCollection,
    reference: PairedAudioCollection,
    *,
    generated_sample_rate: int | None = None,
    reference_sample_rate: int | None = None,
    target_sample_rate: int = 24000,
    version: str = "standard",
    n_fft: int = 2048,
    hop_length: int = 512,
    strict: bool = True,
) -> dict:
    if version not in {"standard", "ssr_eval"}:
        raise ValueError("version must be standard or ssr_eval")
    scores: list[float] = []
    details: list[dict] = []
    for key, generated_source, reference_source in paired_sources(generated, reference, strict=strict):
        if version == "ssr_eval":
            try:
                import librosa
            except ImportError as error:
                raise ImportError("ssr_eval LSD requires librosa") from error

            if isinstance(generated_source, (str, Path)):
                generated_audio, _ = librosa.load(
                    generated_source,
                    sr=target_sample_rate,
                    mono=True,
                )
            else:
                generated_audio, _ = load_audio(
                    generated_source,
                    sample_rate=generated_sample_rate,
                    target_sample_rate=target_sample_rate,
                )
            if isinstance(reference_source, (str, Path)):
                reference_audio, _ = librosa.load(
                    reference_source,
                    sr=target_sample_rate,
                    mono=True,
                )
            else:
                reference_audio, _ = load_audio(
                    reference_source,
                    sample_rate=reference_sample_rate,
                    target_sample_rate=target_sample_rate,
                )
            if len(generated_audio) < len(reference_audio):
                generated_audio = np.pad(
                    generated_audio,
                    (0, len(reference_audio) - len(generated_audio)),
                )
            else:
                generated_audio = generated_audio[:len(reference_audio)]
            hop = int(target_sample_rate / 100)
            fft = int(2048 / (44100 / target_sample_rate))
            generated_spectrogram = np.abs(
                librosa.stft(generated_audio, hop_length=hop, n_fft=fft)
            )
            reference_spectrogram = np.abs(
                librosa.stft(reference_audio, hop_length=hop, n_fft=fft)
            )
            squared_distance = np.log10(
                reference_spectrogram**2
                / ((generated_spectrogram + 1e-12) ** 2)
                + 1e-12
            ) ** 2
            score = float(np.mean(np.sqrt(np.mean(squared_distance, axis=0))))
            scores.append(score)
            details.append({"id": key, "lsd": score})
            continue

        generated_audio, reference_audio = load_aligned_pair(
            generated_source,
            reference_source,
            generated_sample_rate=generated_sample_rate,
            reference_sample_rate=reference_sample_rate,
            target_sample_rate=target_sample_rate,
        )
        _, _, generated_stft = stft(
            generated_audio,
            fs=target_sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            nfft=n_fft,
            boundary=None,
        )
        _, _, reference_stft = stft(
            reference_audio,
            fs=target_sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            nfft=n_fft,
            boundary=None,
        )
        generated_db = 10.0 * np.log10(np.maximum(np.abs(generated_stft) ** 2, 1e-10))
        reference_db = 10.0 * np.log10(np.maximum(np.abs(reference_stft) ** 2, 1e-10))
        score = float(np.mean(np.sqrt(np.mean((generated_db - reference_db) ** 2, axis=0))))
        scores.append(score)
        details.append({"id": key, "lsd": score})

    result = {
        "lsd": mean_or_raise(scores, "LSD"),
        "version": version,
        "target_sample_rate": target_sample_rate,
        "num_samples": len(details),
        "details": details,
    }
    if version == "ssr_eval":
        result.update({
            "n_fft": int(2048 / (44100 / target_sample_rate)),
            "hop_length": int(target_sample_rate / 100),
            "generated_length_policy": "pad_or_truncate_to_reference",
        })
    return result

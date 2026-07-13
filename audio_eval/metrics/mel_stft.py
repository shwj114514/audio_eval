"""Multi-scale STFT and mel losses for paired audio."""

from __future__ import annotations

import numpy as np

from audio_eval.common import PairedAudioCollection, load_aligned_pair, mean_or_raise, paired_sources


def _spectral_losses(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    sample_rate: int,
    device: str,
    window_lengths: tuple[int, ...],
    mel_bins: tuple[int, ...],
) -> tuple[float, float]:
    try:
        import librosa
        import torch
        import torch.nn.functional as torch_functional
    except ImportError as error:
        raise ImportError("mel/STFT loss requires `pip install audio-eval[paired]`") from error

    generated_tensor = torch.from_numpy(generated).to(device)
    reference_tensor = torch.from_numpy(reference).to(device)
    stft_loss = torch.zeros((), device=device)
    mel_loss = torch.zeros((), device=device)

    for window_length, n_mels in zip(window_lengths, mel_bins, strict=True):
        hop_length = window_length // 4
        window = torch.hann_window(window_length, device=device)
        generated_stft = torch.stft(
            generated_tensor,
            n_fft=window_length,
            hop_length=hop_length,
            win_length=window_length,
            window=window,
            return_complex=True,
        ).abs()
        reference_stft = torch.stft(
            reference_tensor,
            n_fft=window_length,
            hop_length=hop_length,
            win_length=window_length,
            window=window,
            return_complex=True,
        ).abs()

        stft_loss += torch_functional.l1_loss(generated_stft, reference_stft)
        stft_loss += torch_functional.l1_loss(
            generated_stft.clamp_min(1e-5).square().log10(),
            reference_stft.clamp_min(1e-5).square().log10(),
        )

        mel_filter = librosa.filters.mel(
            sr=sample_rate,
            n_fft=window_length,
            n_mels=n_mels,
            fmin=0.0,
            fmax=sample_rate / 2,
        )
        mel_filter_tensor = torch.from_numpy(mel_filter).to(device=device, dtype=generated_stft.dtype)
        generated_mel = mel_filter_tensor @ generated_stft
        reference_mel = mel_filter_tensor @ reference_stft
        mel_loss += torch_functional.l1_loss(generated_mel, reference_mel)
        mel_loss += torch_functional.l1_loss(
            generated_mel.clamp_min(1e-5).square().log10(),
            reference_mel.clamp_min(1e-5).square().log10(),
        )

    return float(stft_loss.item()), float(mel_loss.item())


def compute_mel_stft_loss(
    generated: PairedAudioCollection,
    reference: PairedAudioCollection,
    *,
    generated_sample_rate: int | None = None,
    reference_sample_rate: int | None = None,
    target_sample_rate: int = 16000,
    device: str | None = None,
    window_lengths: tuple[int, ...] = (2048, 512),
    mel_bins: tuple[int, ...] = (150, 80),
    strict: bool = True,
) -> dict:
    if len(window_lengths) != len(mel_bins):
        raise ValueError("window_lengths and mel_bins must have equal lengths")
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    stft_scores: list[float] = []
    mel_scores: list[float] = []
    details: list[dict] = []
    for key, generated_source, reference_source in paired_sources(generated, reference, strict=strict):
        generated_audio, reference_audio = load_aligned_pair(
            generated_source,
            reference_source,
            generated_sample_rate=generated_sample_rate,
            reference_sample_rate=reference_sample_rate,
            target_sample_rate=target_sample_rate,
        )
        stft_score, mel_score = _spectral_losses(
            generated_audio,
            reference_audio,
            sample_rate=target_sample_rate,
            device=device,
            window_lengths=window_lengths,
            mel_bins=mel_bins,
        )
        stft_scores.append(stft_score)
        mel_scores.append(mel_score)
        details.append({"id": key, "stft_loss": stft_score, "mel_loss": mel_score})

    return {
        "stft_loss": mean_or_raise(stft_scores, "STFT loss"),
        "mel_loss": mean_or_raise(mel_scores, "mel loss"),
        "num_samples": len(details),
        "details": details,
    }

from __future__ import annotations

import os
from pathlib import Path
import typing as tp

import numpy as np
import torch

from audio_eval.audio import load_audio
from audio_eval.utils import audio_file_map

# audio loaded in memory
AudioArray = tp.Union[np.ndarray, torch.Tensor]
# path AudioArray  AudioArray&sr
AudioInput = tp.Union[str, Path, AudioArray, tp.Tuple[AudioArray, int]]
# Path("audio.wav")         [ Path("a.wav"),]       {"sample_001": Path("a.wav"),}
AudioCollection = tp.Union[AudioInput,tp.Sequence[AudioInput],tp.Mapping[str, AudioInput],]
#  { "001": Path("gen/001.wav"), "002": Path("gen/002.wav")}          
#  { "001": Path("ref/001.wav"), "002": Path("ref/002.wav")}
PairedAudioCollection = tp.Union[AudioInput, tp.Mapping[str, AudioInput]]
FeatureValue = tp.Union[AudioArray, tp.Mapping[str, AudioArray]]
FeatureInput = tp.Union[str, Path, AudioArray, tp.Mapping[str, FeatureValue]]
MetricInput = tp.Union[AudioCollection, FeatureInput]


def pair_directories(
    generated_dir: str | Path,
    reference_dir: str | Path,
    *,
    strict: bool = True,
) -> list[tuple[str, Path, Path]]:
    """Pair by relative path without extension, never by global stem."""
    generated = audio_file_map(generated_dir)
    reference = audio_file_map(reference_dir)
    generated_keys = set(generated)
    reference_keys = set(reference)
    missing_generated = sorted(reference_keys - generated_keys)
    missing_reference = sorted(generated_keys - reference_keys)

    if strict and (missing_generated or missing_reference):
        raise ValueError(
            "Directory pairing mismatch: "
            f"missing generated={len(missing_generated)} {missing_generated[:5]}, "
            f"missing reference={len(missing_reference)} {missing_reference[:5]}"
        )

    return [
        (key, generated[key], reference[key])
        for key in sorted(generated_keys & reference_keys)
    ]


def paired_sources(
    generated: PairedAudioCollection,
    reference: PairedAudioCollection,
    *,
    strict: bool = True,
) -> tp.List[tp.Tuple[str, AudioInput, AudioInput]]:
    if isinstance(generated, tp.Mapping) or isinstance(reference, tp.Mapping):
        if not isinstance(generated, tp.Mapping) or not isinstance(reference, tp.Mapping):
            raise TypeError("Generated and reference must both be mappings")
        generated_keys = {str(key) for key in generated}
        reference_keys = {str(key) for key in reference}
        if strict and generated_keys != reference_keys:
            missing_generated = sorted(reference_keys - generated_keys)
            missing_reference = sorted(generated_keys - reference_keys)
            raise ValueError(
                "Audio mapping mismatch: "
                f"missing generated={missing_generated[:5]}, "
                f"missing reference={missing_reference[:5]}"
            )
        generated_by_key = {str(key): value for key, value in generated.items()}
        reference_by_key = {str(key): value for key, value in reference.items()}
        return [
            (key, generated_by_key[key], reference_by_key[key])
            for key in sorted(generated_keys & reference_keys)
        ]
    if isinstance(generated, (str, os.PathLike, Path)) and Path(generated).is_dir():
        if not isinstance(reference, (str, os.PathLike, Path)) or not Path(reference).is_dir():
            raise TypeError("When generated is a directory, reference must also be a directory")
        return pair_directories(generated, reference, strict=strict)
    if isinstance(reference, (str, os.PathLike, Path)) and Path(reference).is_dir():
        raise TypeError("When reference is a directory, generated must also be a directory")
    return [("0", generated, reference)]


def load_aligned_pair(
    generated: AudioInput,
    reference: AudioInput,
    *,
    generated_sample_rate: int | None,
    reference_sample_rate: int | None,
    target_sample_rate: int,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    generated_audio, _ = load_audio(
        generated,
        sample_rate=generated_sample_rate,
        target_sample_rate=target_sample_rate,
    )
    reference_audio, _ = load_audio(
        reference,
        sample_rate=reference_sample_rate,
        target_sample_rate=target_sample_rate,
    )
    length = min(len(generated_audio), len(reference_audio))
    if length == 0:
        raise ValueError("Cannot compare empty aligned audio")
    return generated_audio[:length], reference_audio[:length]


def mean_or_raise(values: tp.List[float], metric: str) -> float:
    if not values:
        raise ValueError(f"No samples were evaluated for {metric}")
    return float(np.mean(values))

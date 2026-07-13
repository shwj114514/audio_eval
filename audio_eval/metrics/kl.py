"""Paired KL divergence with selectable PANNs or PaSST logits."""

from __future__ import annotations

import contextlib
from functools import partial
import os
import pickle
from pathlib import Path
import typing as tp
import numpy as np
import torch
import torch.nn.functional as F

from audio_eval.audio import collection_fingerprint, collection_items, load_audio
from audio_eval.cache import cache_file, is_feature_map, load_feature_map, pair_feature_maps
from audio_eval.common import MetricInput
from audio_eval.metrics.panns import get_panns_features, load_panns_features

_PASST_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "passt_kl"

KL_OPTIONS: tp.Dict[str, tp.Dict[str, object]] = {
    "panns": {"version": "panns"},  # PANNs CNN14   KL(P_generated || P_reference)
    "passt": {"version": "passt"},  # PaSST
    "panns_ref_to_gen": {"version": "panns", "direction": "reference_to_generated"},
    "passt_ref_to_gen": {"version": "passt", "direction": "reference_to_generated"},    # KL(P_reference || P_generated)
}

KL_REFERENCES: tp.Dict[str, tp.Dict[str, Path]] = {
    "passt": {
        "audiocaps": _PASST_ASSETS / "audiocaps-test__collectmean__reference_probabilities.pkl",
        "musiccaps": _PASST_ASSETS / "musiccaps-public__collectmean__reference_probabilities.pkl",
        "musiccaps_nosinging": _PASST_ASSETS / "musiccaps-public-nosinging__collectmean__reference_probabilities.pkl",
        "songdescriber": _PASST_ASSETS / "song_describer__collectmean__reference_probabilities.pkl",
        "songdescriber_nosinging": _PASST_ASSETS / "song_describer-nosinging__collectmean__reference_probabilities.pkl",
    }
}

_PASST_MODELS: tp.Dict[str, torch.nn.Module] = {}


def _load_passt_probabilities(path: str | Path) -> tp.Dict[str, np.ndarray]:
    cache_path = Path(path).expanduser()
    if cache_path.suffix.lower() == ".pkl":
        with cache_path.open("rb") as file:
            values = pickle.load(file)
        return {
            str(key): value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            for key, value in values.items()
        }
    with np.load(cache_path, allow_pickle=False) as loaded:
        return {
            str(key): value
            for key, value in zip(loaded["keys"].astype(str), loaded["probabilities"], strict=True)
        }


def _get_passt_probabilities(
    audio: MetricInput,
    *,
    sample_rate: int | None,
    cache_dir: str | Path | None,
    device: str | None,
    refresh_cache: bool,
) -> tp.Tuple[tp.Dict[str, np.ndarray], Path]:
    target_sample_rate = 32000
    fingerprint = collection_fingerprint(audio, sample_rate=sample_rate)
    output_path = cache_file(
        "features",
        fingerprint,
        backend="passt_probabilities_mean_10s_5s",
        cache_dir=cache_dir,
    )
    if output_path.is_file() and not refresh_cache:
        return _load_passt_probabilities(output_path), output_path

    try:
        import soxr
        from hear21passt.base import get_basic_model
    except ImportError as error:
        raise ImportError("PaSST KL requires hear21passt and soxr") from error

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if target_device not in _PASST_MODELS:
        with open(os.devnull, "w") as output, contextlib.redirect_stdout(output):
            model = get_basic_model(mode="logits")
        model.eval()
        _PASST_MODELS[target_device] = model.to(target_device)
    model = _PASST_MODELS[target_device]

    probabilities: tp.Dict[str, np.ndarray] = {}
    window_size = 10 * target_sample_rate
    step_size = 5 * target_sample_rate
    for key, source in collection_items(audio):
        waveform, source_sample_rate = load_audio(source, sample_rate=sample_rate)
        peak = float(np.max(np.abs(waveform)))
        if peak > 0:
            waveform = waveform * (10.0 ** (-1.0 / 20.0) / peak)
        if source_sample_rate != target_sample_rate:
            waveform = soxr.resample(waveform, source_sample_rate, target_sample_rate)

        window_logits: tp.List[torch.Tensor] = []
        for start in range(0, max(step_size, len(waveform) - step_size), step_size):
            window = waveform[start : start + window_size]
            if len(window) < window_size:
                if len(window) <= int(window_size * 0.15):
                    continue
                padded = np.zeros(window_size, dtype=np.float32)
                padded[: len(window)] = window
                window = padded
            audio_tensor = torch.from_numpy(np.asarray(window, dtype=np.float32)).unsqueeze(0).to(target_device)
            old_stft = torch.stft
            try:
                torch.stft = partial(torch.stft, return_complex=False)
                with open(os.devnull, "w") as output, contextlib.redirect_stdout(output):
                    with torch.no_grad():
                        window_logits.append(torch.squeeze(model(audio_tensor)))
            finally:
                torch.stft = old_stft
        if not window_logits:
            raise ValueError(f"PaSST found no valid 10-second window for {key}")
        probabilities[key] = F.softmax(torch.stack(window_logits).mean(dim=0), dim=0).cpu().numpy()

    np.savez_compressed(
        output_path,
        keys=np.asarray(list(probabilities)),
        probabilities=np.stack(list(probabilities.values())),
    )
    return probabilities, output_path


def get_kl_options(option: str) -> tp.Dict[str, object]:
    if not option:
        return {}
    try:
        return dict(KL_OPTIONS[option])
    except KeyError as error:
        available = ", ".join(sorted(KL_OPTIONS))
        raise ValueError(f"Unknown KL option {option!r}. Available: {available}") from error


def _normalized(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.maximum(probabilities.astype(np.float64), 1e-12)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def compute_kl(
    generated: MetricInput,
    reference: MetricInput | None = None,
    *,
    generated_sample_rate: int | None = None,
    reference_sample_rate: int | None = None,
    backend: str = "panns_cnn14",
    version: str = "panns",
    direction: str = "generated_to_reference",
    cache_dir: str | Path | None = None,
    batch_size: int = 8,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    strict: bool = True,
    refresh_cache: bool = False,
) -> dict:
    if direction not in {"generated_to_reference", "reference_to_generated"}:
        raise ValueError("direction must be generated_to_reference or reference_to_generated")
    if version not in {"panns", "passt"}:
        raise ValueError("version must be panns or passt")

    reference_cache: Path | None = None
    if isinstance(reference, (str, Path)):
        reference_path = Path(reference).expanduser()
        if reference_path.exists():
            if reference_path.is_file() and reference_path.suffix.lower() in {".npz", ".pkl", ".pth"}:
                reference_cache = reference_path
                reference = None
            else:
                reference = reference_path
        elif str(reference) in KL_REFERENCES.get(version, {}):
            reference_cache = KL_REFERENCES[version][str(reference)]
            reference = None
        else:
            available = ", ".join(sorted(KL_REFERENCES.get(version, {})))
            raise FileNotFoundError(
                f"Unknown KL reference {reference!r} for {version}. Available bundled references: {available}"
            )

    filename = "pann_features.pth" if version == "panns" else "passt_logits.pth"
    layer = "logits" if version == "panns" else None
    generated_is_av_cache = generated_sample_rate is None and is_feature_map(
        generated, filename=filename, layer=layer
    )
    if version == "passt" and not generated_is_av_cache:
        generated_probabilities, generated_cache = _get_passt_probabilities(
            generated,
            sample_rate=generated_sample_rate,
            cache_dir=cache_dir,
            device=device,
            refresh_cache=refresh_cache,
        )
        if reference_cache is not None:
            reference_probabilities = _load_passt_probabilities(reference_cache)
        else:
            if reference is None:
                raise ValueError("KL requires reference audio or a reference cache")
            reference_probabilities, reference_cache = _get_passt_probabilities(
                reference,
                sample_rate=reference_sample_rate,
                cache_dir=cache_dir,
                device=device,
                refresh_cache=refresh_cache,
            )
        missing_reference = sorted(set(generated_probabilities) - set(reference_probabilities))
        if strict and missing_reference:
            raise ValueError(f"KL reference cache is missing generated ids: {missing_reference[:5]}")
        keys = sorted(set(generated_probabilities) & set(reference_probabilities))
        if not keys:
            raise ValueError("No matched generated/reference samples for KL")
        generated_array = np.stack([generated_probabilities[key] for key in keys])
        reference_array = np.stack([reference_probabilities[key] for key in keys])
        if direction == "generated_to_reference":
            left, right = generated_array, reference_array
        else:
            left, right = reference_array, generated_array
        left = left.astype(np.float64)
        right = right.astype(np.float64)
        per_sample = np.sum(left * (np.log(left) - np.log(right + 1e-6)), axis=-1)
        return {
            "kl": float(per_sample.mean()),
            "version": version,
            "direction": direction,
            "num_samples": len(keys),
            "details": [
                {"id": key, "kl": float(score)} for key, score in zip(keys, per_sample, strict=True)
            ],
            "generated_cache": str(generated_cache),
            "reference_cache": str(reference_cache),
        }

    if generated_is_av_cache:
        if reference is None and reference_cache is None:
            raise ValueError("KL requires reference features")
        generated_map = load_feature_map(generated, filename=filename, layer=layer)
        reference_map = load_feature_map(
            reference_cache if reference_cache is not None else reference,
            filename=filename,
            layer=layer,
        )
        keys, reference_logits, generated_logits, unpaired = pair_feature_maps(
            reference_map, generated_map
        )
        if direction == "reference_to_generated":
            target_logits, predicted_logits = generated_logits, reference_logits
        else:
            target_logits, predicted_logits = reference_logits, generated_logits
        per_class = F.kl_div(
            F.log_softmax(target_logits.double(), dim=1),
            F.log_softmax(predicted_logits.double(), dim=1),
            reduction="none",
            log_target=True,
        )
        per_sample = per_class.sum(dim=1)
        return {
            "kl": float(per_sample.mean()),
            "version": version,
            "direction": direction,
            "num_samples": len(keys),
            "unpaired": unpaired,
            "details": [
                {"id": key, "kl": float(score)}
                for key, score in zip(keys, per_sample, strict=True)
            ],
        }

    generated_features = get_panns_features(
        generated,
        sample_rate=generated_sample_rate,
        backend=backend,
        cache_dir=cache_dir,
        batch_size=batch_size,
        device=device,
        checkpoint_path=checkpoint_path,
        refresh_cache=refresh_cache,
    )
    if reference_cache is not None:
        reference_data = load_panns_features(reference_cache)
    else:
        if reference is None:
            raise ValueError("KL requires reference audio or a reference cache")
        reference_data = get_panns_features(
            reference,
            sample_rate=reference_sample_rate,
            backend=backend,
            cache_dir=cache_dir,
            batch_size=batch_size,
            device=device,
            checkpoint_path=checkpoint_path,
            refresh_cache=refresh_cache,
        )

    generated_map = {
        key: probability for key, probability in zip(
            generated_features["keys"], generated_features["probabilities"], strict=True
        )
    }
    reference_map = {
        key: probability for key, probability in zip(
            reference_data["keys"], reference_data["probabilities"], strict=True
        )
    }
    generated_keys = set(generated_map)
    reference_keys = set(reference_map)
    if strict and generated_keys != reference_keys:
        raise ValueError(
            "KL pairing mismatch: "
            f"missing generated={sorted(reference_keys - generated_keys)[:5]}, "
            f"missing reference={sorted(generated_keys - reference_keys)[:5]}"
        )
    keys = sorted(generated_keys & reference_keys)
    if not keys:
        raise ValueError("No matched generated/reference samples for KL")

    generated_probabilities = _normalized(np.stack([generated_map[key] for key in keys]))
    reference_probabilities = _normalized(np.stack([reference_map[key] for key in keys]))
    if direction == "generated_to_reference":
        left, right = generated_probabilities, reference_probabilities
    else:
        left, right = reference_probabilities, generated_probabilities
    per_sample = np.sum(left * (np.log(left) - np.log(right)), axis=-1)
    return {
        "kl": float(per_sample.mean()),
        "version": version,
        "backend": backend,
        "direction": direction,
        "num_samples": len(keys),
        "details": [
            {"id": key, "kl": float(score)} for key, score in zip(keys, per_sample, strict=True)
        ],
        "generated_cache": str(generated_features["cache_path"]),
        "reference_cache": str(reference_data["cache_path"]),
    }

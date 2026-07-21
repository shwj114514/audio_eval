"""Paired KL divergence over the 527 AudioSet classes.

The feature extractors use 10-second windows with a 5-second hop.  For example,
an exactly 20-second generated audio produces three overlapping windows::

    G1 = [0 s, 10 s)
    G2 = [5 s, 15 s)
    G3 = [10 s, 20 s)

The examples below use only three classes to keep the arithmetic readable; the
real PANNs and PaSST outputs contain 527 values per window.

KL-PANNs
--------
PANNs returns an already-sigmoid-activated score vector for each window::

    scores_G1 = [0.8, 0.2, 0.1]
    scores_G2 = [0.6, 0.4, 0.2]
    scores_G3 = [0.7, 0.3, 0.3]

Average corresponding classes across G1, G2, and G3::

    mean_scores_G = [(0.8 + 0.6 + 0.7) / 3,
                     (0.2 + 0.4 + 0.3) / 3,
                     (0.1 + 0.2 + 0.3) / 3]
                  = [0.7, 0.3, 0.2]

Sigmoid scores are independent multi-label scores and do not have to sum to
one.  Clamp them to at least ``1e-12`` and divide by their class sum before KL::

    P_G = mean_scores_G / sum(mean_scores_G)
        = [0.7, 0.3, 0.2] / 1.2
        = [0.583333, 0.250000, 0.166667]

Process reference windows R1, R2, and R3 in exactly the same way to obtain P_R,
then calculate one KL value for the complete generated/reference audio pair::

    KL_PANNs_audio = KL(P_G || P_R)

KL-PaSST (Stable Audio ``collect-mean``)
-----------------------------------------
PaSST returns a raw, pre-softmax logit vector for each window::

    logits_G1 = [2.0, 0.5, -1.0]
    logits_G2 = [1.5, 0.7, -0.5]
    logits_G3 = [2.5, 0.3, -0.8]

Average corresponding class logits across G1, G2, and G3 first::

    mean_logits_G = [(2.0 + 1.5 + 2.5) / 3,
                     (0.5 + 0.7 + 0.3) / 3,
                     (-1.0 - 0.5 - 0.8) / 3]
                  = [2.0, 0.5, -0.766667]

Apply softmax once to the averaged logit vector::

    P_G = softmax(mean_logits_G) = [0.777604, 0.173507, 0.048889]

Process reference windows R1, R2, and R3 in exactly the same way to obtain P_R,
then calculate one KL value for the complete generated/reference audio pair::

    KL_PaSST_audio = KL(P_G || P_R)

AudioCraft/MusicGen uses a different PaSST aggregation protocol: it applies
softmax and calculates KL separately for every aligned segment pair::

    p_G1 = softmax(logits_G1)   p_R1 = softmax(logits_R1)
    p_G2 = softmax(logits_G2)   p_R2 = softmax(logits_R2)
    p_G3 = softmax(logits_G3)   p_R3 = softmax(logits_R3)

    KL_segments = (KL(p_G1 || p_R1) + KL(p_G2 || p_R2) + KL(p_G3 || p_R3)) / 3

AudioCraft actually averages over all segment pairs in the dataset, so longer
audio can have more weight.  This file first produces one distribution and one
KL value per source audio, then averages the per-audio KL values, giving every
matched source audio equal weight.

For both PANNs and PaSST, ``KL(P || Q)`` is computed as::

    KL(P || Q) = sum_c P(c) * (log(P(c)) - log(Q(c)))

``direction`` selects which distribution is ``P``.  The default is
``KL(generated || reference)``.

PaSST reference implementation:
https://github.com/Stability-AI/stable-audio-metrics/blob/main/src/passt_kld.py

Stable Audio Open SongDescriber-nosinging KL-PaSST:
* Reported result in the paper: 0.55
* Reproduced result: 
    Using the extracted local reference: 0.5527729759
    Using the officially bundled reference: 0.5529081568
"""
from __future__ import annotations

import pickle
from pathlib import Path
import typing as tp

import numpy as np
import torch

from audio_eval.common import MetricInput
from audio_eval.features.panns import get_panns_features
from audio_eval.features.passt import get_passt_features


_PASST_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "passt_kl"

KL_OPTIONS: tp.Dict[str, tp.Dict[str, object]] = {
    "panns": {"version": "panns"},
    "passt": {"version": "passt"},
    "panns_ref_to_gen": {"version": "panns", "direction": "reference_to_generated"},
    "passt_ref_to_gen": {"version": "passt", "direction": "reference_to_generated"},
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


def get_kl_options(option: str) -> tp.Dict[str, object]:
    if not option:
        return {}
    try:
        return dict(KL_OPTIONS[option])
    except KeyError as error:
        available = ", ".join(sorted(KL_OPTIONS))
        raise ValueError(f"Unknown KL option {option!r}. Available: {available}") from error


def _load_probability_map(path: str | Path) -> tp.Dict[str, np.ndarray]:
    cache_path = Path(path).expanduser()
    if cache_path.suffix.lower() != ".pkl":
        raise ValueError(
            f"KL probability reference must be a bundled .pkl file, got {cache_path}. "
            "Use panns.npz or passt.npz for extracted feature caches."
        )
    with cache_path.open("rb") as file:
        values = pickle.load(file)
    if not isinstance(values, tp.Mapping):
        raise ValueError(f"KL probability cache must contain a mapping: {cache_path}")
    return {
        str(key): value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        for key, value in values.items()
    }


def _mean_by_clip(
    clip_keys: tp.Sequence[str],
    values: np.ndarray,
) -> tp.Dict[str, np.ndarray]:
    grouped: tp.Dict[str, tp.List[np.ndarray]] = {}
    for key, value in zip(clip_keys, values, strict=True):
        grouped.setdefault(str(key), []).append(value)
    return {key: np.stack(rows).mean(axis=0) for key, rows in grouped.items()}


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)

def _normalized(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.maximum(probabilities.astype(np.float64), 1e-12)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)

def compute_kl(
    generated: MetricInput,
    reference: MetricInput | None = None,
    *,
    # Unnecessary for audio file paths and for``(audio, sample_rate)`` tuples
    generated_sample_rate: int | None = None, 
    reference_sample_rate: int | None = None,
    
    backend: str = "panns_cnn14",
    version: str = "panns", #     # "panns"  or "passt"
    direction: str = "generated_to_reference", #  KL direction
    cache_dir: str | Path | None = None,  # Root for automatically fingerprinted feature caches
    generated_cache_dir: str | Path | None = None,
    reference_cache_dir: str | Path | None = None,
    batch_size: int = 8, # Number of 10-second windows processed in each model batch
    device: str | None = None,
    checkpoint_path: str | Path | None = None,  # Optional PANNs Cnn14 checkpoint
    # Require generated and reference sample-ID sets to match exactly. If false, compute the metric only over their intersection.
    strict: bool = True,
    # Re-extract features from the supplied raw audio even if the target cache already exists.
    refresh_cache: bool = False,
) -> tp.Dict[str, tp.Any]:
    """
        Compute paired KL divergence from AudioSet classifier outputs.
        Each generated audio clip
            → split into 10-second windows with a 5-second hop
            → extract PaSST logits for each window / PANNs sigmoid scores for each window
            → average the window-level logits / PANNs sigmoid scores across windows 
            → apply softmax to obtain a 527-dimensional probability vector  / normalize PANNs scores by their class sum
            → compute (KL(\text{gen} \parallel \text{ref})) against the probability vector of the corresponding ground-truth audio (if direction == "generated_to_reference")
        preprocess: mono → resample 32 kHz → -1 dB peak normalize → 10s clip
    """

    if direction not in {"generated_to_reference", "reference_to_generated"}:
        raise ValueError("direction must be generated_to_reference or reference_to_generated")
    if version not in {"panns", "passt"}:
        raise ValueError("version must be panns or passt")

    # Resolve a path/name before feature extraction. Bundled probability maps
    # reproduce published PaSST reference distributions without raw audio.
    probability_reference: Path | None = None
    if isinstance(reference, (str, Path)):
        reference_path = Path(reference).expanduser()
        if reference_path.exists():
            if reference_path.is_file() and reference_path.suffix.lower() == ".pkl":
                probability_reference = reference_path
                reference = None
            elif reference_path.is_file() and reference_path.suffix.lower() == ".pth":
                raise ValueError(
                    "Legacy .pth PANNs/PaSST caches use a different feature protocol and are not supported"
                )
            else:
                reference = reference_path
        elif str(reference) in KL_REFERENCES.get(version, {}):
            probability_reference = KL_REFERENCES[version][str(reference)]
            reference = None
        else:
            available = ", ".join(sorted(KL_REFERENCES.get(version, {})))
            raise FileNotFoundError(
                f"Unknown KL reference {reference!r} for {version}. Available bundled references: {available}"
            )

    if version == "panns":
        # PANNs emits sigmoid probabilities for every window. 
        # Average those probabilities to obtain one classifier vector per source audio.
        generated_features = get_panns_features(
            generated,
            sample_rate=generated_sample_rate,
            backend=backend,
            cache_dir=cache_dir,
            output_dir=generated_cache_dir,
            batch_size=batch_size,
            device=device,
            checkpoint_path=checkpoint_path,
            refresh_cache=refresh_cache,
        )
        reference_source = reference
        if reference_source is None and reference_cache_dir is not None:
            reference_source = Path(reference_cache_dir).expanduser()
        if reference_source is None:
            raise ValueError("KL-PANNs requires reference audio or a panns.npz feature cache")
        reference_features = get_panns_features(
            reference_source,
            sample_rate=reference_sample_rate,
            backend=backend,
            cache_dir=cache_dir,
            output_dir=reference_cache_dir,
            batch_size=batch_size,
            device=device,
            checkpoint_path=checkpoint_path,
            refresh_cache=refresh_cache,
        )
        generated_map = _mean_by_clip(
            generated_features["clip_keys"], generated_features["probabilities"]
        )
        reference_map = _mean_by_clip(
            reference_features["clip_keys"], reference_features["probabilities"]
        )
        generated_cache = generated_features["cache_path"]
        reference_cache = reference_features["cache_path"]
        feature_backend = backend
        num_windows_generated = len(generated_features["clip_keys"])
        num_windows_reference = len(reference_features["clip_keys"])
    else:
        # PaSST emits logits (before softmax). 
        # Average logits across windows first, matching the reference protocol, and apply softmax once per source audio.
        generated_features = get_passt_features(
            generated,
            sample_rate=generated_sample_rate,
            cache_dir=cache_dir,
            output_dir=generated_cache_dir,
            batch_size=batch_size,
            device=device,
            refresh_cache=refresh_cache,
        )
        generated_logits = _mean_by_clip(
            generated_features["clip_keys"], generated_features["logits"]
        )
        generated_map = {key: _softmax(value) for key, value in generated_logits.items()}
        generated_cache = generated_features["cache_path"]
        num_windows_generated = len(generated_features["clip_keys"])
        if probability_reference is not None:
            reference_map = _load_probability_map(probability_reference)
            reference_cache = probability_reference
            num_windows_reference = None
        else:
            reference_source = reference
            if reference_source is None and reference_cache_dir is not None:
                reference_source = Path(reference_cache_dir).expanduser()
            if reference_source is None:
                raise ValueError("KL-PaSST requires reference audio, passt.npz, or a bundled reference")
            reference_features = get_passt_features(
                reference_source,
                sample_rate=reference_sample_rate,
                cache_dir=cache_dir,
                output_dir=reference_cache_dir,
                batch_size=batch_size,
                device=device,
                refresh_cache=refresh_cache,
            )
            reference_logits = _mean_by_clip(
                reference_features["clip_keys"], reference_features["logits"]
            )
            reference_map = {key: _softmax(value) for key, value in reference_logits.items()}
            reference_cache = reference_features["cache_path"]
            num_windows_reference = len(reference_features["clip_keys"])
        feature_backend = "passt_s_swa_p16_128_ap476"

    # KL is paired: only generated/reference records with the same sample ID
    # can be compared. In non-strict mode unmatched records are simply omitted.
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

    # PANNs sigmoid scores do not naturally sum to one, and external referencemaps may contain small numerical errors. 
    # Convert both sides into valid, strictly positive categorical distributions before taking logarithms.
    generated_probabilities = _normalized(np.stack([generated_map[key] for key in keys]))
    reference_probabilities = _normalized(np.stack([reference_map[key] for key in keys]))
    if direction == "generated_to_reference":
        left, right = generated_probabilities, reference_probabilities
    else:
        left, right = reference_probabilities, generated_probabilities

    # KL(P || Q) = sum_c P(c) * (log P(c) - log Q(c))
    # evaluated over the 527 AudioSet classes for every paired sample.
    per_sample = np.sum(left * (np.log(left) - np.log(right)), axis=-1)
    return {
        "kl": float(per_sample.mean()),
        "version": version,
        "backend": feature_backend,
        "direction": direction,
        "num_samples": len(keys),
        "num_windows_generated": num_windows_generated,
        "num_windows_reference": num_windows_reference,
        "details": [
            {"id": key, "kl": float(score)} for key, score in zip(keys, per_sample, strict=True)
        ],
        "generated_cache": str(generated_cache),
        "reference_cache": str(reference_cache),
    }

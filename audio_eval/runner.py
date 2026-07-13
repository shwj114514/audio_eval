"""Folder-level evaluation entry points."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import typing as tp

import numpy as np

from .audio import load_audio
from audio_eval.common import pair_directories
from audio_eval.utils import write_result


def _paired_metric_functions() -> tp.Dict[str, tp.Any]:
    from .metrics.lsd import compute_lsd
    from .metrics.mel_stft import compute_mel_stft_loss
    from .metrics.pesq import compute_pesq
    from .metrics.stoi import compute_stoi

    return {
        "pesq": compute_pesq,
        "stoi": compute_stoi,
        "mel_stft": compute_mel_stft_loss,
        "lsd": compute_lsd,
    }


def _mean_metric_rows(rows: tp.List[tp.Dict[str, tp.Any]]) -> tp.Dict[str, tp.Any]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in {"num_samples", "details"}
            and isinstance(value, (int, float, np.number))
            and not isinstance(value, bool)
        }
    )
    result = {
        key: float(np.mean([row[key] for row in rows if key in row])) for key in numeric_keys
    }
    result["num_samples"] = len(rows)
    return result


def evaluate_paired(
    generated_dir: tp.Union[str, Path],
    reference_dir: tp.Union[str, Path],
    *,
    metrics: tp.Union[tp.List[str], tp.Tuple[str, ...]] = ("pesq", "stoi", "mel_stft"),
    metric_options: tp.Union[tp.List[str], tp.Tuple[str, ...]] = ("", "", ""),
    strict: bool = True,
    results_dir: tp.Union[str, Path] = "results",
    name: tp.Optional[str] = None,
) -> tp.Dict[str, tp.Any]:
    """Evaluate two paired folders while loading every pair only once."""
    if isinstance(metrics, str) or isinstance(metric_options, str):
        raise TypeError("metrics and metric_options must be lists or tuples, not strings")
    if len(metrics) != len(metric_options):
        raise ValueError("metrics and metric_options must have the same length")

    functions = _paired_metric_functions()
    model_metrics = {"speaker_sim", "utmos"}
    unknown = sorted(set(metrics) - set(functions) - model_metrics)
    if unknown:
        raise ValueError(f"Unsupported paired metrics: {unknown}")
    pairs = pair_directories(generated_dir, reference_dir, strict=strict)
    sample_metrics = list(metrics)
    rows: tp.Dict[str, tp.List[tp.Dict[str, tp.Any]]] = {
        metric: [] for metric in sample_metrics
    }
    details: tp.List[tp.Dict[str, tp.Any]] = []

    for key, generated_path, reference_path in pairs:
        generated_audio, generated_sample_rate = load_audio(generated_path)
        reference_audio, reference_sample_rate = load_audio(reference_path)
        sample_detail: tp.Dict[str, tp.Any] = {"id": key}
        for metric, option in zip(sample_metrics, metric_options):
            if metric == "speaker_sim":
                from .metrics.speaker_sim import compute_speaker_sim
                if option:
                    raise ValueError(f"speaker_sim does not define metric option {option!r}")
                metric_result = compute_speaker_sim(
                    generated_audio,
                    reference_audio,
                    generated_sample_rate=generated_sample_rate,
                    reference_sample_rate=reference_sample_rate,
                )
            elif metric == "utmos":
                from .metrics.utmos import compute_utmos
                metric_result = compute_utmos(
                    generated_audio,
                    sample_rate=generated_sample_rate,
                    **({"backend": option} if option else {}),
                )
            else:
                if metric == "lsd":
                    from .metrics.lsd import get_lsd_options
                    options = get_lsd_options(option)
                elif option:
                    raise ValueError(f"{metric} does not define metric option {option!r}")
                else:
                    options = {}
                metric_result = functions[metric](
                    generated_audio,
                    reference_audio,
                    generated_sample_rate=generated_sample_rate,
                    reference_sample_rate=reference_sample_rate,
                    **options,
                )
            compact_result = {
                item_key: item_value
                for item_key, item_value in metric_result.items()
                if item_key not in {"details", "num_samples"}
            }
            rows[metric].append(compact_result)
            sample_detail[metric] = compact_result
        details.append(sample_detail)

    metric_results = {
        metric: _mean_metric_rows(rows[metric]) for metric in sample_metrics
    }
    result: tp.Dict[str, tp.Any] = {
        "mode": "paired",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generated": str(Path(generated_dir).resolve()),
        "reference": str(Path(reference_dir).resolve()),
        "num_samples": len(pairs),
        "metrics": metric_results,
        "details": details,
    }
    output_path = write_result(
        result,
        generated=generated_dir,
        reference=reference_dir,
        results_dir=results_dir,
        name=name,
    )
    result["result_path"] = str(output_path.resolve())
    return result


def evaluate_generation(
    generated_dir: tp.Union[str, Path],
    *,
    metrics: tp.Union[tp.List[str], tp.Tuple[str, ...]],
    metric_options: tp.Union[tp.List[str], tp.Tuple[str, ...]],
    reference_dir: tp.Optional[tp.Union[str, Path]] = None,
    prompts: tp.Optional[tp.Dict[str, str]] = None,
    transcripts: tp.Optional[tp.Dict[str, str]] = None,
    cache_dir: tp.Optional[tp.Union[str, Path]] = None,
    results_dir: tp.Union[str, Path] = "results",
    name: tp.Optional[str] = None,
) -> tp.Dict[str, tp.Any]:
    """Evaluate a generated folder without depending on its generation code."""
    if isinstance(metrics, str) or isinstance(metric_options, str):
        raise TypeError("metrics and metric_options must be lists or tuples, not strings")
    if len(metrics) != len(metric_options):
        raise ValueError("metrics and metric_options must have the same length")

    supported = {"fd", "kl", "inception_score", "clap", "audiobox", "wer", "utmos"}
    unknown = sorted(set(metrics) - supported)
    if unknown:
        raise ValueError(f"Unsupported generation metrics: {unknown}")

    metric_results: tp.Dict[str, tp.Dict[str, tp.Any]] = {}
    for metric, option in zip(metrics, metric_options):
        kwargs: tp.Dict[str, object] = {}
        if metric in {"fd", "kl"}:
            if reference_dir is None:
                raise ValueError(f"{metric} requires reference_dir")
            if metric == "fd":
                from .metrics.fd import compute_fd, get_fd_options
                kwargs = get_fd_options(option)
                compute_function = compute_fd
            else:
                from .metrics.kl import compute_kl, get_kl_options
                kwargs = get_kl_options(option)
                compute_function = compute_kl
            if cache_dir is not None:
                kwargs["cache_dir"] = cache_dir
            metric_results[metric] = compute_function(generated_dir, reference_dir, **kwargs)
        elif metric == "inception_score":
            from .metrics.inception_score import compute_inception_score, get_inception_score_options
            kwargs = get_inception_score_options(option)
            if cache_dir is not None:
                kwargs.setdefault("cache_dir", cache_dir)
            metric_results[metric] = compute_inception_score(generated_dir, **kwargs)
        elif metric == "clap":
            from .metrics.clap import compute_clap
            if option:
                raise ValueError(f"CLAP does not define metric option {option!r}")
            if prompts is None:
                raise ValueError("clap requires prompts")
            if cache_dir is not None:
                kwargs.setdefault("cache_dir", cache_dir)
            metric_results[metric] = compute_clap(generated_dir, prompts, **kwargs)
        elif metric == "audiobox":
            from .metrics.audiobox import compute_audiobox
            if option:
                raise ValueError(f"AudioBox does not define metric option {option!r}")
            metric_results[metric] = compute_audiobox(generated_dir, **kwargs)
        elif metric == "wer":
            from .metrics.wer import compute_wer, get_wer_options
            kwargs = get_wer_options(option)
            if transcripts is None:
                raise ValueError("wer requires transcripts")
            metric_results[metric] = compute_wer(generated_dir, transcripts, **kwargs)
        elif metric == "utmos":
            from .metrics.utmos import compute_utmos
            kwargs = {"backend": option} if option else {}
            metric_results[metric] = compute_utmos(generated_dir, **kwargs)

    result: tp.Dict[str, tp.Any] = {
        "mode": "generation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generated": str(Path(generated_dir).resolve()),
        "reference": (
            str(Path(reference_dir).expanduser().resolve())
            if reference_dir is not None and Path(reference_dir).expanduser().exists()
            else str(reference_dir) if reference_dir is not None else None
        ),
        "metrics": metric_results,
    }
    output_path = write_result(
        result,
        generated=generated_dir,
        reference=reference_dir,
        results_dir=results_dir,
        name=name,
    )
    result["result_path"] = str(output_path.resolve())
    return result

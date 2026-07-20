"""Generated-audio evaluation over the unified JSONL manifest format."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import typing as tp

from .audio import load_audio
from audio_eval.common import MetricInput
from audio_eval.utils import load_manifest, write_result

_SUPPORTED_METRICS = {
    "fd",
    "kl",
    "inception_score",
    "clap",
    "audiobox",
    "utmos",
}
_DISTRIBUTION_METRICS = {"fd", "kl"}


def eval_generation(
    manifest: tp.Union[str, Path],
    *,
    metrics: tp.Union[tp.List[str], tp.Tuple[str, ...]] = (),
    metric_options: tp.Union[tp.List[str], tp.Tuple[str, ...]] = (),
    reference: tp.Optional[tp.Union[str, Path]] = None,
    generated_cache: tp.Optional[tp.Union[str, Path]] = None,
    reference_cache: tp.Optional[tp.Union[str, Path]] = None,
    cache_dir: tp.Optional[tp.Union[str, Path]] = None,
    results_dir: tp.Union[str, Path] = "results",
    name: tp.Optional[str] = None,
    task: str = "generation",
) -> tp.Dict[str, object]:
    """Evaluate generated audio without per-record reference paths."""
    if isinstance(metrics, str) or isinstance(metric_options, str):
        raise TypeError("metrics and metric_options must be lists or tuples, not strings")
    if len(metrics) != len(metric_options):
        raise ValueError("metrics and metric_options must have the same length")

    selected_metrics = list(metrics)
    if not selected_metrics:
        raise ValueError("At least one generation metric is required")
    unknown = sorted(set(selected_metrics) - _SUPPORTED_METRICS)
    if unknown:
        raise ValueError(f"Unsupported generation metrics: {unknown}")

    manifest_path = Path(manifest).expanduser().resolve()
    records = load_manifest(manifest_path)
    if "clap" in selected_metrics:
        missing_prompts = [key for key, record in records.items() if not record["prompt"]]
        if missing_prompts:
            raise ValueError(f"CLAP requires prompt for records: {missing_prompts[:5]}")

    generated_audio = {
        key: load_audio(record["gen_path"], mono=False)
        for key, record in records.items()
    }
    prompts = {
        key: str(record["prompt"])
        for key, record in records.items()
        if record["prompt"] is not None
    }

    reference_records = {
        key: record["ref_path"]
        for key, record in records.items()
        if record["ref_path"] is not None
    }
    if set(selected_metrics) & _DISTRIBUTION_METRICS and reference_records:
        if len(reference_records) != len(records):
            missing = sorted(set(records) - set(reference_records))
            raise ValueError(
                "Distribution metrics require ref_path for every record or no records; "
                f"missing ref_path for {missing[:5]}"
            )
        manifest_reference = {key: load_audio(path, mono=False) for key, path in reference_records.items()}
    else:
        manifest_reference = None

    explicit_reference: tp.Optional[tp.Union[str, Path]] = None
    if reference is not None:
        reference_path = Path(reference).expanduser()
        explicit_reference = reference_path.resolve() if reference_path.exists() else str(reference)
    if explicit_reference is not None:
        reference_label: tp.Optional[str] = str(explicit_reference)
    elif manifest_reference is not None:
        reference_label = "manifest"
    else:
        reference_label = None

    generated_cache_path = (
        Path(generated_cache).expanduser().resolve()
        if generated_cache is not None
        else None
    )
    reference_cache_path = (
        Path(reference_cache).expanduser().resolve()
        if reference_cache is not None
        else None
    )
    if explicit_reference is not None:
        metric_reference: tp.Optional[MetricInput] = explicit_reference
    elif manifest_reference is not None:
        metric_reference = manifest_reference
    else:
        metric_reference = reference_cache_path

    metric_results: tp.Dict[str, object] = {}
    for metric, option in zip(selected_metrics, metric_options):
        options: tp.Dict[str, object] = {}
        metric_generated: MetricInput = generated_audio
        metric_reference_for_metric = metric_reference
        if metric == "fd":
            from .metrics.fd import compute_fd, get_fd_options
            options = get_fd_options(option)
        elif metric == "kl":
            from .metrics.kl import compute_kl, get_kl_options
            options = get_kl_options(option)
        elif metric == "inception_score":
            from .metrics.inception_score import (
                compute_inception_score,
                get_inception_score_options,
            )
            options = get_inception_score_options(option)
        elif metric in {"clap", "audiobox"} and option:
            raise ValueError(f"{metric} does not define metric option {option!r}")
        elif metric == "utmos" and option:
            options = {"backend": option}

        if cache_dir is not None and metric in {"fd", "kl", "inception_score", "clap"}:
            options["cache_dir"] = cache_dir
        if metric in {"fd", "kl", "inception_score"} and generated_cache_path is not None:
            options["generated_cache_dir"] = generated_cache_path
        if metric in {"fd", "kl"} and reference_cache_path is not None:
            options["reference_cache_dir"] = reference_cache_path
        if metric == "fd" and option == "openl3":
            metric_reference_for_metric = (
                explicit_reference if explicit_reference is not None else manifest_reference
            )
            options["cache_dir"] = generated_cache_path or cache_dir
        if metric in _DISTRIBUTION_METRICS:
            if metric_reference_for_metric is None and reference_cache_path is None:
                raise ValueError(
                    f"{metric} requires --reference or ref_path for every JSONL record"
                )
        if metric == "fd":
            metric_result = compute_fd(
                metric_generated,
                metric_reference_for_metric,
                **options,
            )
        elif metric == "kl":
            metric_result = compute_kl(
                metric_generated,
                metric_reference_for_metric,
                **options,
            )
        elif metric == "inception_score":
            metric_result = compute_inception_score(metric_generated, **options)
        elif metric == "clap":
            from .metrics.clap import compute_clap
            metric_result = compute_clap(generated_audio, prompts, **options)
        elif metric == "audiobox":
            from .metrics.audiobox import compute_audiobox
            metric_result = compute_audiobox(generated_audio, **options)
        elif metric == "utmos":
            from .metrics.utmos import compute_utmos
            metric_result = compute_utmos(generated_audio, **options)

        if option:
            metric_result["metric_option"] = option
        if selected_metrics.count(metric) == 1:
            metric_results[metric] = metric_result
        else:
            if not option:
                raise ValueError(f"Repeated metric {metric!r} requires non-empty unique options")
            group = tp.cast(tp.Dict[str, object], metric_results.setdefault(metric, {}))
            if option in group:
                raise ValueError(f"Repeated metric option {metric}={option!r}")
            group[option] = metric_result

    result: tp.Dict[str, object] = {
        "task": task,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "reference": reference_label,
        "generated_cache": str(generated_cache_path) if generated_cache_path is not None else None,
        "reference_cache": str(reference_cache_path) if reference_cache_path is not None else None,
        "num_samples": len(records),
        "metrics": metric_results,
    }
    output_path = write_result(
        result,
        generated=manifest_path,
        results_dir=results_dir,
        name=name or manifest_path.stem,
    )
    result["result_path"] = str(output_path.resolve())
    return result


def eval_tta(
    manifest: tp.Union[str, Path],
    *,
    metrics: tp.Union[tp.List[str], tp.Tuple[str, ...]] = (),
    metric_options: tp.Union[tp.List[str], tp.Tuple[str, ...]] = (),
    reference: tp.Optional[tp.Union[str, Path]] = None,
    generated_cache: tp.Optional[tp.Union[str, Path]] = None,
    reference_cache: tp.Optional[tp.Union[str, Path]] = None,
    cache_dir: tp.Optional[tp.Union[str, Path]] = None,
    results_dir: tp.Union[str, Path] = "results",
    name: tp.Optional[str] = None,
) -> tp.Dict[str, object]:
    return eval_generation(
        manifest,
        metrics=metrics,
        reference=reference,
        metric_options=metric_options,
        cache_dir=cache_dir,
        generated_cache=generated_cache,
        reference_cache=reference_cache,
        results_dir=results_dir,
        name=name,
        task="tta",
    )


def eval_ttm(
    manifest: tp.Union[str, Path],
    *,
    metrics: tp.Union[tp.List[str], tp.Tuple[str, ...]] = (),
    metric_options: tp.Union[tp.List[str], tp.Tuple[str, ...]] = (),
    reference: tp.Optional[tp.Union[str, Path]] = None,
    generated_cache: tp.Optional[tp.Union[str, Path]] = None,
    reference_cache: tp.Optional[tp.Union[str, Path]] = None,
    cache_dir: tp.Optional[tp.Union[str, Path]] = None,
    results_dir: tp.Union[str, Path] = "results",
    name: tp.Optional[str] = None,
) -> tp.Dict[str, object]:
    return eval_generation(
        manifest,
        metrics=metrics,
        reference=reference,
        metric_options=metric_options,
        cache_dir=cache_dir,
        generated_cache=generated_cache,
        reference_cache=reference_cache,
        results_dir=results_dir,
        name=name,
        task="ttm",
    )

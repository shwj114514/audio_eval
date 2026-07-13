"""Audio super-resolution evaluation over the unified JSONL manifest."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import typing as tp

from audio_eval.utils import load_manifest, write_result


_SUPPORTED_METRICS = {"lsd"}


def eval_sr(
    manifest: tp.Union[str, Path],
    *,
    metrics: tp.Union[tp.List[str], tp.Tuple[str, ...]] = ("lsd",),
    metric_options: tp.Union[tp.List[str], tp.Tuple[str, ...]] = ("ssr_eval",),
    results_dir: tp.Union[str, Path] = "results",
    name: tp.Optional[str] = None,
) -> tp.Dict[str, object]:
    if isinstance(metrics, str) or isinstance(metric_options, str):
        raise TypeError("metrics and metric_options must be lists or tuples, not strings")
    if len(metrics) != len(metric_options):
        raise ValueError("metrics and metric_options must have the same length")

    selected_metrics = list(metrics)
    unknown = sorted(set(selected_metrics) - _SUPPORTED_METRICS)
    if unknown:
        raise ValueError(f"Unsupported SR metrics: {unknown}")
    if not selected_metrics:
        raise ValueError("At least one SR metric is required")

    manifest_path = Path(manifest).expanduser().resolve()
    records = load_manifest(manifest_path)
    missing_references = [key for key, record in records.items() if record["ref_path"] is None]
    if missing_references:
        raise ValueError(f"SR requires ref_path for records: {missing_references[:5]}")
    mismatched = [key for key, record in records.items() if Path(record["ref_path"]).stem != key]
    if mismatched:
        raise ValueError(
            "SR requires matching generated/reference filename stems: "
            f"{mismatched[:5]}"
        )

    # List[Path]
    generated = {key: record["gen_path"] for key, record in records.items()}
    reference = {key: record["ref_path"] for key, record in records.items()}
    metric_results: tp.Dict[str, object] = {}
    for metric, option in zip(selected_metrics, metric_options):
        if metric == "lsd":
            from .metrics.lsd import compute_lsd, get_lsd_options
            metric_results["lsd"] = compute_lsd(
                generated,
                reference,
                **get_lsd_options(option),
            )
            if option:
                tp.cast(tp.Dict[str, object], metric_results[metric])["metric_option"] = option

    result: tp.Dict[str, object] = {
        "task": "sr",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
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

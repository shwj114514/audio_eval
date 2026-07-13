"""TTS metric selection over the unified JSONL manifest format."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import typing as tp

from .audio import load_audio
from audio_eval.utils import load_manifest, write_result

_SUPPORTED_METRICS = {
    "wer",
    "speaker_sim",
    "utmos",
    "pesq",
    "stoi",
    "mel_stft",
    "lsd",
}
_REFERENCE_METRICS = {"speaker_sim", "pesq", "stoi", "mel_stft", "lsd"}


def eval_tts(
    manifest: tp.Union[str, Path],
    *,
    metrics: tp.Union[tp.List[str], tp.Tuple[str, ...]] = ("wer", "speaker_sim"),
    metric_options: tp.Union[tp.List[str], tp.Tuple[str, ...]] = ("", ""),
    results_dir: tp.Union[str, Path] = "results",
    name: tp.Optional[str] = None,
) -> tp.Dict[str, object]:
    """Evaluate TTS records containing gen_path, optional ref_path, and prompt."""
    if isinstance(metrics, str) or isinstance(metric_options, str):
        raise TypeError("metrics and metric_options must be lists or tuples, not strings")
    if len(metrics) != len(metric_options):
        raise ValueError("metrics and metric_options must have the same length")

    selected_metrics = list(metrics)
    if not selected_metrics:
        raise ValueError("At least one TTS metric is required")
    unknown = sorted(set(selected_metrics) - _SUPPORTED_METRICS)
    if unknown:
        raise ValueError(f"Unsupported TTS metrics: {unknown}")

    manifest_path = Path(manifest).expanduser().resolve()
    records = load_manifest(manifest_path)
    missing_prompts = [key for key, record in records.items() if not record["prompt"]] # []
    if "wer" in selected_metrics and missing_prompts:
        raise ValueError(f"WER requires prompt for records: {missing_prompts[:5]}")

    # pair metric
    reference_metrics = sorted(set(selected_metrics) & _REFERENCE_METRICS)
    missing_references = [key for key, record in records.items() if record["ref_path"] is None]
    if reference_metrics and missing_references:
        raise ValueError(
            f"Metrics {reference_metrics} require ref_path for records: {missing_references[:5]}"
        )

    generated_audio = { key: load_audio(record["gen_path"]) for key, record in records.items() }
    reference_audio = { key: load_audio(record["ref_path"]) for key, record in records.items() if record["ref_path"] is not None}
    prompts = {key: str(record["prompt"]) for key, record in records.items() if record["prompt"] is not None }
    # dict_keys(['wer', 'speaker_sim'])
    metric_results: tp.Dict[str, object] = {}
    for metric, option in zip(selected_metrics, metric_options):
        if metric == "wer":
            from .metrics.wer import compute_wer, get_wer_options
            options = get_wer_options(option)
            metric_results[metric] = compute_wer(generated_audio, prompts, **options)
        elif metric == "speaker_sim":
            from .metrics.speaker_sim import compute_speaker_sim
            if option:
                raise ValueError(f"speaker_sim does not define metric option {option!r}")
            metric_results[metric] = compute_speaker_sim(generated_audio, reference_audio)
        elif metric == "utmos":
            from .metrics.utmos import compute_utmos
            options = {"backend": option} if option else {}
            metric_results[metric] = compute_utmos(generated_audio, **options)
        elif metric == "pesq":
            from .metrics.pesq import compute_pesq
            if option:
                raise ValueError(f"PESQ does not define metric option {option!r}")
            metric_results[metric] = compute_pesq(generated_audio, reference_audio)
        elif metric == "stoi":
            from .metrics.stoi import compute_stoi
            if option:
                raise ValueError(f"STOI does not define metric option {option!r}")
            metric_results[metric] = compute_stoi(generated_audio, reference_audio)
        elif metric == "mel_stft":
            from .metrics.mel_stft import compute_mel_stft_loss
            if option:
                raise ValueError(f"mel_stft does not define metric option {option!r}")
            metric_results[metric] = compute_mel_stft_loss(generated_audio,reference_audio)
        elif metric == "lsd":
            from .metrics.lsd import compute_lsd, get_lsd_options
            options = get_lsd_options(option)
            metric_results[metric] = compute_lsd(generated_audio, reference_audio, **options)

        if option:
            tp.cast(tp.Dict[str, object], metric_results[metric])["metric_option"] = option

    result: tp.Dict[str, object] = {
        "task": "tts",
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

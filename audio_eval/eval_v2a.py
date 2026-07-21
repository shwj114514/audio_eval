"""Video-to-audio evaluation from a JSONL manifest and reusable feature caches."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import tempfile
import typing as tp

from audio_eval.audio import AUDIO_EXTENSIONS
from audio_eval.cache import (
    IMAGEBIND_OVERLAP_CACHE_MARKER,
    ensure_audio_feature_cache,
    ensure_video_feature_cache,
)
from audio_eval.utils import load_manifest, write_result


_SUPPORTED_METRICS = {"fd", "kl", "inception_score", "imagebind", "desync"}


def eval_v2a(
    manifest: tp.Union[str, Path],
    *,
    generated_cache: tp.Optional[tp.Union[str, Path]] = None,
    reference_cache: tp.Optional[tp.Union[str, Path]] = None,
    reference: tp.Optional[tp.Union[str, Path]] = None,
    metrics: tp.Union[tp.List[str], tp.Tuple[str, ...]] = (
        "fd", "fd", "fd", "kl", "kl", "inception_score", "imagebind", "desync"
    ),
    metric_options: tp.Union[tp.List[str], tp.Tuple[str, ...]] = (
        "passt", "panns", "vggish", "passt_ref_to_gen", "panns_ref_to_gen", "panns", "", ""
    ),
    results_dir: tp.Union[str, Path] = "results",
    name: tp.Optional[str] = None,
) -> tp.Dict[str, object]:
    if isinstance(metrics, str) or isinstance(metric_options, str):
        raise TypeError("metrics and metric_options must be lists or tuples, not strings")
    if len(metrics) != len(metric_options):
        raise ValueError("metrics and metric_options must have the same length")

    selected_metrics = list(metrics)
    if not selected_metrics:
        raise ValueError("At least one V2A metric is required")
    unknown = sorted(set(selected_metrics) - _SUPPORTED_METRICS)
    if unknown:
        raise ValueError(f"Unsupported V2A metrics: {unknown}")

    feature_versions: tp.List[str] = []
    for metric, option in zip(selected_metrics, metric_options):
        if metric == "fd":
            from .metrics.fd import get_fd_options
            version = tp.cast(str, get_fd_options(option).get("version", "panns"))
        elif metric == "kl":
            from .metrics.kl import get_kl_options
            version = tp.cast(str, get_kl_options(option).get("version", "panns"))
        elif metric == "inception_score":
            from .metrics.inception_score import get_inception_score_options
            version = tp.cast(
                str, get_inception_score_options(option).get("version", "panns")
            )
        else:
            continue
        if version != "openl3" and version not in feature_versions:
            feature_versions.append(version)

    manifest_path = Path(manifest).expanduser().resolve()
    records = load_manifest(manifest_path)
    generated_cache_path = (
        Path(generated_cache).expanduser().resolve()
        if generated_cache is not None
        else manifest_path.parent / "generated_cache"
    )
    reference_cache_path = (
        Path(reference_cache).expanduser().resolve()
        if reference_cache is not None
        else manifest_path.parent / "reference_cache"
    )
    generated_cache_path.mkdir(parents=True, exist_ok=True)
    reference_cache_path.mkdir(parents=True, exist_ok=True)

    explicit_reference: tp.Optional[tp.Union[str, Path]] = None
    if reference is not None:
        reference_path = Path(reference).expanduser()
        explicit_reference = reference_path.resolve() if reference_path.exists() else str(reference)

    needs_video_features = bool({"imagebind", "desync"} & set(selected_metrics))
    needs_manifest_reference_audio = (
        explicit_reference is None and bool({"fd", "kl"} & set(selected_metrics))
    )
    reference_audio_records = {
        key: tp.cast(Path, record["ref_path"])
        for key, record in records.items()
        if record["ref_path"] is not None
    }
    video_records = {
        key: tp.cast(Path, record["video_path"] or record["ref_path"])
        for key, record in records.items()
        if record["video_path"] is not None or record["ref_path"] is not None
    }
    with tempfile.TemporaryDirectory(prefix="audio_eval_v2a_") as temp_dir:
        temp_path = Path(temp_dir)
        generated_dir = temp_path / "generated"
        generated_dir.mkdir()
        for key, record in records.items():
            generated_path = tp.cast(Path, record["gen_path"])
            (generated_dir / f"{key}{generated_path.suffix.lower()}").symlink_to(generated_path)
        ensure_audio_feature_cache(
            generated_dir,
            output_dir=generated_cache_path,
            feature_versions=feature_versions,
            include_video_metrics=False,
        )

        if needs_manifest_reference_audio:
            filenames = {"panns": "panns.npz", "passt": "passt.npz", "vggish": "vggish.npz"}
            required_audio_cache = tuple(filenames[version] for version in feature_versions)
            if not all((reference_cache_path / name).is_file() for name in required_audio_cache):
                missing = sorted(set(records) - set(reference_audio_records))
                if missing:
                    raise ValueError(
                        "FD/KL requires --reference, a complete reference cache, or ref_path "
                        f"for every JSONL record; missing ref_path for {missing[:5]}"
                    )
                reference_audio_dir = temp_path / "reference_audio"
                reference_audio_dir.mkdir()
                for key, reference_path in reference_audio_records.items():
                    if reference_path.suffix.lower() in AUDIO_EXTENSIONS:
                        (reference_audio_dir / f"{key}{reference_path.suffix.lower()}").symlink_to(
                            reference_path
                        )
                    else:
                        output_audio = reference_audio_dir / f"{key}.wav"
                        subprocess.run(
                            [
                                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                                "-i", str(reference_path), "-vn", "-ac", "1", "-ar", "16000",
                                str(output_audio),
                            ],
                            check=True,
                        )
                ensure_audio_feature_cache(
                    reference_audio_dir,
                    output_dir=reference_cache_path,
                    feature_versions=feature_versions,
                    include_video_metrics=False,
                )

        if needs_video_features:
            missing = sorted(set(records) - set(video_records))
            video_cache_complete = all(
                (reference_cache_path / name).is_file()
                for name in (
                    "imagebind_video.pth",
                    "synchformer_video.pth",
                    IMAGEBIND_OVERLAP_CACHE_MARKER,
                )
            )
            if missing and not video_cache_complete:
                raise ValueError(
                    "ImageBind/DeSync requires video_path or source-video ref_path for every "
                    f"JSONL record, or a complete reference cache; missing video for {missing[:5]}"
                )
            audio_video_paths = [
                key
                for key, video_path in video_records.items()
                if video_path.suffix.lower() in AUDIO_EXTENSIONS
            ]
            if audio_video_paths and not video_cache_complete:
                raise ValueError(
                    "ImageBind/DeSync video_path, or its ref_path fallback, must point to "
                    f"source video; audio path found for {audio_video_paths[:5]}"
                )

            overlap_by_key: tp.Dict[str, float] = {}
            if missing or audio_video_paths:
                generated_cache_complete = all(
                    (generated_cache_path / name).is_file()
                    for name in (
                        "imagebind_audio.pth",
                        "synchformer_audio.pth",
                        IMAGEBIND_OVERLAP_CACHE_MARKER,
                    )
                )
                if not (video_cache_complete and generated_cache_complete):
                    raise ValueError(
                        "A/V overlap extraction requires generated audio and source video "
                        "for every manifest record"
                    )
            else:
                try:
                    import av
                    import torchaudio
                except ImportError as error:
                    raise ImportError(
                        "ImageBind/DeSync extraction requires `pip install audio-eval[video]`"
                    ) from error

                for key, record in records.items():
                    generated_path = tp.cast(Path, record["gen_path"])
                    video_path = video_records[key]
                    waveform, sample_rate = torchaudio.load(str(generated_path))
                    waveform = waveform.float()
                    if sample_rate != 16000:
                        waveform = torchaudio.functional.resample(
                            waveform,
                            orig_freq=sample_rate,
                            new_freq=16000,
                        )
                    audio_duration = waveform.shape[-1] / 16000
                    with av.open(str(video_path)) as container:
                        video_stream = container.streams.video[0]
                        if video_stream.duration is not None:
                            video_duration = float(
                                video_stream.duration * video_stream.time_base
                            )
                        elif container.duration is not None:
                            video_duration = float(container.duration / av.time_base)
                        else:
                            raise ValueError(
                                f"Cannot determine video duration for {video_path}"
                            )
                    overlap = min(audio_duration, video_duration)
                    if int(0.5 * overlap) < 2:
                        raise ValueError(
                            f"A/V overlap for {key!r} is too short: {overlap:.6f}s"
                        )
                    overlap_by_key[key] = overlap

                ensure_audio_feature_cache(
                    generated_dir,
                    output_dir=generated_cache_path,
                    include_video_metrics=True,
                    duration_by_key=overlap_by_key,
                )
                ensure_video_feature_cache(
                    video_records,
                    output_dir=reference_cache_path,
                    duration_by_key=overlap_by_key,
                )

    distribution_reference: tp.Union[str, Path] = (
        explicit_reference if explicit_reference is not None else reference_cache_path
    )
    metric_results: tp.Dict[str, object] = {}
    for metric, option in zip(selected_metrics, metric_options):
        if metric == "fd":
            from .metrics.fd import compute_fd, get_fd_options
            metric_result = compute_fd(
                generated_cache_path,
                distribution_reference,
                generated_cache_dir=generated_cache_path,
                reference_cache_dir=reference_cache_path,
                **get_fd_options(option),
            )
        elif metric == "kl":
            from .metrics.kl import compute_kl, get_kl_options
            metric_result = compute_kl(
                generated_cache_path,
                distribution_reference,
                generated_cache_dir=generated_cache_path,
                reference_cache_dir=reference_cache_path,
                **get_kl_options(option),
            )
        elif metric == "inception_score":
            from .metrics.inception_score import (
                compute_inception_score,
                get_inception_score_options,
            )
            metric_result = compute_inception_score(
                generated_cache_path,
                generated_cache_dir=generated_cache_path,
                **get_inception_score_options(option),
            )
        elif metric == "imagebind":
            from .metrics.imagebind import compute_imagebind
            if option:
                raise ValueError(f"ImageBind does not define metric option {option!r}")
            metric_result = compute_imagebind(generated_cache_path, reference_cache_path)
        else:
            from .metrics.desync import compute_desync, get_desync_options
            metric_result = compute_desync(
                generated_cache_path,
                reference_cache_path,
                **get_desync_options(option),
            )

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
        "task": "v2a",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "num_samples": len(records),
        "generated_cache": str(generated_cache_path),
        "reference_cache": str(reference_cache_path),
        "reference": str(explicit_reference) if explicit_reference is not None else None,
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

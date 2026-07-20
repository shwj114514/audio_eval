"""Full-duration, distributed V2SFX benchmark support.

This module intentionally does not reuse the legacy eight-second AV caches.
Metadata order is authoritative, GPU feature extraction is sharded by rank,
and every merged cache is paired with protocol metadata.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio_eval.metrics.fd import frechet_distance


PROTOCOL = "full_duration_v1"
EXPECTED_COUNTS = {"moviegen_audio": 527, "cinebench50": 50}
MODEL_COLUMNS = OrderedDict(
    [
        ("sonilo", "sonilo_sfx_v1_0"),
        ("mirelo", "mirelo_v1_6"),
    ]
)
PASST_MODULES = OrderedDict(
    [
        ("hear21passt.base", "hear21passt_base"),
        ("hear21passt.base20sec", "hear21passt_base20sec"),
        ("hear21passt.base30sec", "hear21passt_base30sec"),
    ]
)
SYNCHFORMER_URL = (
    "https://github.com/hkchengrex/MMAudio/releases/download/v0.1/"
    "synchformer_state_dict.pth"
)
PANNS_16K_URL = (
    "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth"
)
PANNS_16K_SIZE = 358668570
DIRECT_CHECKPOINTS = OrderedDict(
    [
        (
            "vggish",
            (
                "https://github.com/harritaylor/torchvggish/releases/download/v0.1/"
                "vggish-10086976.pth",
                "vggish-10086976.pth",
                288567937,
            ),
        ),
        (
            "passt_hear21passt_base",
            (
                "https://github.com/kkoutini/PaSST/releases/download/v0.0.1-audioset/"
                "passt-s-f128-p16-s10-ap.476-swa.pt",
                "passt-s-f128-p16-s10-ap.476-swa.pt",
                344665734,
            ),
        ),
        (
            "passt_hear21passt_base20sec",
            (
                "https://github.com/kkoutini/PaSST/releases/download/v0.0.5/"
                "passt-s-f128-20sec-p16-s10-ap.474-swa.pt",
                "passt-s-f128-20sec-p16-s10-ap.474-swa.pt",
                344975942,
            ),
        ),
        (
            "passt_hear21passt_base30sec",
            (
                "https://github.com/kkoutini/PaSST/releases/download/v0.0.5/"
                "passt-s-f128-30sec-p16-s10-ap.473-swa.pt",
                "passt-s-f128-30sec-p16-s10-ap.473-swa.pt",
                345283206,
            ),
        ),
        (
            "imagebind_huge",
            (
                "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                "imagebind_huge.pth",
                4803584173,
            ),
        ),
        (
            "synchformer",
            (SYNCHFORMER_URL, "synchformer_state_dict.pth", 950058171),
        ),
        (
            "clap_630k_audioset",
            (
                "https://huggingface.co/lukewys/laion_clap/resolve/main/"
                "630k-audioset-best.pt",
                "630k-audioset-best.pt",
                1863587645,
            ),
        ),
    ]
)


@dataclass(frozen=True)
class BenchmarkRecord:
    dataset: str
    sample_id: str
    prompt: str
    gt: Path
    generated: Mapping[str, Path]
    video: Path | None


@dataclass(frozen=True)
class AudioTask:
    sample_id: str
    path: Path
    prompt: str
    video: Path | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize(value: str) -> str:
    return value.replace(".", "_").replace("/", "_").replace("-", "_")


def sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def code_fingerprint() -> str:
    return sha256_file(Path(__file__).resolve())


def git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()

    try:
        status = run("status", "--porcelain")
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status),
            "status": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None, "status": []}


def package_versions() -> dict[str, str | None]:
    names = [
        "torch",
        "torchaudio",
        "torchvision",
        "av_bench",
        "passt",
        "imagebind",
        "laion_clap",
        "audiobox_aesthetics",
        "openl3",
        "tensorflow",
        "numpy",
        "scipy",
        "soundfile",
        "soxr",
    ]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary_name)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def full_coverage_window_starts(
    duration: float,
    window: float,
    hop: float,
) -> list[float]:
    """Return deterministic starts whose windows cover [0, duration]."""
    if duration <= 0 or window <= 0 or hop <= 0:
        raise ValueError("duration, window, and hop must be positive")
    if hop > window:
        raise ValueError("hop must not exceed window or gaps would be uncovered")
    if duration <= window:
        return [0.0]
    last = duration - window
    starts: list[float] = []
    current = 0.0
    while current <= last + 1e-9:
        starts.append(current)
        current += hop
    if not math.isclose(starts[-1], last, rel_tol=0.0, abs_tol=1e-7):
        starts.append(last)
    return starts


def sample_window_starts(length: int, window: int, hop: int) -> list[int]:
    if length <= 0 or window <= 0 or hop <= 0:
        raise ValueError("length, window, and hop must be positive")
    if hop > window:
        raise ValueError("hop must not exceed window")
    if length <= window:
        return [0]
    last = length - window
    starts = list(range(0, last + 1, hop))
    if starts[-1] != last:
        starts.append(last)
    return starts


def load_records(
    pack_root: Path,
    datasets: Sequence[str],
    models: Sequence[str],
    *,
    limit: int | None,
) -> OrderedDict[str, list[BenchmarkRecord]]:
    output: OrderedDict[str, list[BenchmarkRecord]] = OrderedDict()
    for dataset in datasets:
        if dataset not in EXPECTED_COUNTS:
            raise ValueError(f"Unknown dataset: {dataset}")
        dataset_root = pack_root / dataset
        metadata_path = dataset_root / "metadata.csv"
        with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        expected_fields = {"id", "prompt", "gt", *models}
        if dataset == "moviegen_audio":
            expected_fields.add("video")
        missing_fields = expected_fields - set(rows[0] if rows else [])
        if missing_fields:
            raise ValueError(f"{metadata_path} is missing columns {sorted(missing_fields)}")
        if len(rows) != EXPECTED_COUNTS[dataset]:
            raise ValueError(
                f"{dataset} metadata has {len(rows)} rows; expected {EXPECTED_COUNTS[dataset]}"
            )
        ids = [str(row["id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{dataset} metadata IDs are not unique")
        selected_rows = rows[:limit] if limit is not None else rows
        records: list[BenchmarkRecord] = []
        for row in selected_rows:
            sample_id = str(row["id"])
            gt = (dataset_root / row["gt"]).resolve()
            generated = {model: (dataset_root / row[model]).resolve() for model in models}
            video = (
                (dataset_root / row["video"]).resolve()
                if dataset == "moviegen_audio"
                else None
            )
            paths = [gt, *generated.values(), *([video] if video is not None else [])]
            for path in paths:
                if path is None or not path.is_file():
                    raise FileNotFoundError(path)
                if path.stem != sample_id:
                    raise ValueError(
                        f"Metadata ID/path stem mismatch in {dataset}: {sample_id!r} vs {path.stem!r}"
                    )
            records.append(
                BenchmarkRecord(
                    dataset=dataset,
                    sample_id=sample_id,
                    prompt=str(row["prompt"]),
                    gt=gt,
                    generated=generated,
                    video=video,
                )
            )
        output[dataset] = records
    return output


def collection_tasks(
    records: Sequence[BenchmarkRecord],
    collection: str,
) -> list[AudioTask]:
    tasks: list[AudioTask] = []
    for record in records:
        path = record.gt if collection == "gt" else record.generated[collection]
        tasks.append(
            AudioTask(
                sample_id=record.sample_id,
                path=path,
                prompt=record.prompt,
                video=record.video,
            )
        )
    return tasks


def tasks_fingerprint(tasks: Sequence[AudioTask], *, include_video: bool) -> str:
    digest = hashlib.sha256()
    for task in tasks:
        digest.update(task.sample_id.encode())
        digest.update(task.prompt.encode())
        for path in [task.path, *([task.video] if include_video else [])]:
            if path is None:
                digest.update(b"NO_VIDEO")
                continue
            stat = path.stat()
            digest.update(str(path.resolve()).encode())
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def audio_duration(path: Path) -> float:
    info = sf.info(path)
    return float(info.frames / info.samplerate)


def video_duration(path: Path) -> float:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return float(container.duration / 1_000_000)
    raise ValueError(f"Cannot determine video duration for {path}")


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty sequence")
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
    }


def audit_data(
    records_by_dataset: Mapping[str, Sequence[BenchmarkRecord]],
    models: Sequence[str],
) -> dict[str, Any]:
    audit: dict[str, Any] = {"protocol": PROTOCOL, "created_at": utc_now(), "datasets": {}}
    for dataset, records in records_by_dataset.items():
        dataset_audit: dict[str, Any] = {
            "num_samples": len(records),
            "unique_ids": len({record.sample_id for record in records}),
            "collections": {},
            "cinebench_video_expected_absent": dataset == "cinebench50",
        }
        duration_maps: dict[str, dict[str, float]] = {}
        for collection in ["gt", *models]:
            durations = {
                task.sample_id: audio_duration(task.path)
                for task in collection_tasks(records, collection)
            }
            duration_maps[collection] = durations
            infos = [
                sf.info(task.path) for task in collection_tasks(records, collection)
            ]
            dataset_audit["collections"][collection] = {
                "duration_seconds": durations,
                "duration_stats": _stats(list(durations.values())),
                "sample_rates": sorted({int(info.samplerate) for info in infos}),
                "channels": sorted({int(info.channels) for info in infos}),
            }
        if dataset == "moviegen_audio":
            video_durations = {
                record.sample_id: video_duration(record.video)  # type: ignore[arg-type]
                for record in records
            }
            dataset_audit["video"] = {
                "duration_seconds": video_durations,
                "duration_stats": _stats(list(video_durations.values())),
            }
            dataset_audit["av_duration_delta_seconds"] = {}
            for collection, durations in duration_maps.items():
                deltas = {
                    sample_id: durations[sample_id] - video_durations[sample_id]
                    for sample_id in durations
                }
                dataset_audit["av_duration_delta_seconds"][collection] = {
                    "values": deltas,
                    "stats": _stats(list(deltas.values())),
                    "num_abs_gt_0_5": sum(abs(value) > 0.5 for value in deltas.values()),
                    "num_abs_gt_1_0": sum(abs(value) > 1.0 for value in deltas.values()),
                }
        audit["datasets"][dataset] = dataset_audit
    return audit


def load_mono_torch(path: Path, target_sample_rate: int, *, demean: bool) -> torch.Tensor:
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.float().mean(dim=0)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
    if demean:
        waveform = waveform - waveform.mean()
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError(f"Invalid audio: {path}")
    return waveform.contiguous()


def cache_location(
    cache_root: Path,
    dataset: str,
    collection: str,
    backend: str,
    variant: str,
) -> tuple[Path, Path]:
    directory = cache_root / dataset / collection / backend / sanitize(variant)
    if backend == "panns":
        filename = "pann_features.pth"
    elif backend == "vggish":
        filename = "vggish_features.pth"
    elif backend == "passt":
        filename = f"passt_{sanitize(variant)}_logits.pth"
    elif backend == "clap":
        filename = "clap_630k_audioset_scores.pth"
    elif backend == "audiobox":
        filename = "audiobox_aesthetics_scores.pth"
    elif backend == "imagebind":
        filename = "imagebind_full_timeline_scores.pth"
    elif backend == "desync":
        filename = "synchformer_full_timeline_scores.pth"
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return directory / filename, directory / "cache_meta.json"


def cache_valid(
    feature_path: Path,
    metadata_path: Path,
    expected_metadata: Mapping[str, Any],
    expected_ids: Sequence[str],
) -> bool:
    if not feature_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                return False
        values = torch.load(feature_path, map_location="cpu", weights_only=True)
        return isinstance(values, Mapping) and list(values) == list(expected_ids) and _finite(values)
    except Exception:
        return False


def barrier(local_rank: int) -> None:
    dist.barrier(device_ids=[local_rank])


def distributed_extract_collection(
    *,
    cache_root: Path,
    dataset: str,
    collection: str,
    backend: str,
    variant: str,
    tasks: Sequence[AudioTask],
    extractor: Callable[[AudioTask], Any],
    extraction_metadata: Mapping[str, Any],
    rank: int,
    world_size: int,
    local_rank: int,
    repo_root: Path,
    refresh: bool,
    include_video_fingerprint: bool = False,
) -> Path:
    feature_path, metadata_path = cache_location(
        cache_root, dataset, collection, backend, variant
    )
    ids = [task.sample_id for task in tasks]
    expected_metadata = {
        "protocol": PROTOCOL,
        "dataset": dataset,
        "collection": collection,
        "backend": backend,
        "model_variant": variant,
        "num_samples": len(tasks),
        "world_size": world_size,
        "input_fingerprint": tasks_fingerprint(
            tasks, include_video=include_video_fingerprint
        ),
        "code_fingerprint": code_fingerprint(),
    }
    skip_tensor = torch.zeros(1, dtype=torch.int64, device=f"cuda:{local_rank}")
    if rank == 0 and not refresh:
        skip_tensor[0] = int(
            cache_valid(feature_path, metadata_path, expected_metadata, ids)
        )
    dist.broadcast(skip_tensor, src=0)
    if bool(skip_tensor.item()):
        print(
            f"[rank {rank}] cache hit {dataset}/{collection}/{backend}/{variant} "
            f"num_samples={len(tasks)}",
            flush=True,
        )
        barrier(local_rank)
        return feature_path

    rank_tasks = list(tasks[rank::world_size])
    if not rank_tasks and len(tasks) >= world_size:
        raise RuntimeError(
            f"Rank {rank} received no items for {dataset}/{collection}/{backend}/{variant}"
        )
    boundary = (
        f"first={rank_tasks[0].sample_id} last={rank_tasks[-1].sample_id}"
        if rank_tasks
        else "first=N/A last=N/A"
    )
    print(
        f"[rank {rank}] start {dataset}/{collection}/{backend}/{variant} "
        f"assigned={len(rank_tasks)} {boundary}",
        flush=True,
    )
    shard: OrderedDict[str, Any] = OrderedDict()
    for index, task in enumerate(rank_tasks, 1):
        try:
            value = extractor(task)
        except Exception as error:
            raise RuntimeError(
                f"Extraction failed for {dataset}/{collection}/{backend}/{variant}/"
                f"{task.sample_id}: {error}"
            ) from error
        if not _finite(value):
            raise ValueError(
                f"Non-finite output for {dataset}/{collection}/{backend}/{variant}/"
                f"{task.sample_id}"
            )
        shard[task.sample_id] = value
        if index == 1 or index == len(rank_tasks) or index % 10 == 0:
            print(
                f"[rank {rank}] progress {dataset}/{collection}/{backend}/{variant} "
                f"{index}/{len(rank_tasks)} id={task.sample_id}",
                flush=True,
            )

    shard_dir = feature_path.parent / "shards"
    shard_path = shard_dir / f"{feature_path.stem}.rank{rank:03d}.pth"
    shard_metadata_path = shard_dir / f"{feature_path.stem}.rank{rank:03d}.json"
    atomic_torch_save(shard, shard_path)
    atomic_json_dump(
        {
            **expected_metadata,
            "rank": rank,
            "ids": list(shard),
            "num_rank_samples": len(shard),
            "created_at": utc_now(),
        },
        shard_metadata_path,
    )
    print(
        f"[rank {rank}] wrote shard {shard_path} count={len(shard)}",
        flush=True,
    )
    barrier(local_rank)

    if rank == 0:
        merged_unordered: dict[str, Any] = {}
        rank_counts: dict[str, int] = {}
        for shard_rank in range(world_size):
            path = shard_dir / f"{feature_path.stem}.rank{shard_rank:03d}.pth"
            values = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(values, Mapping):
                raise TypeError(f"Shard is not a mapping: {path}")
            duplicate = set(merged_unordered) & set(values)
            if duplicate:
                raise ValueError(f"Duplicate IDs while merging {path}: {sorted(duplicate)[:5]}")
            merged_unordered.update(values)
            rank_counts[str(shard_rank)] = len(values)
        missing = set(ids) - set(merged_unordered)
        extra = set(merged_unordered) - set(ids)
        if missing or extra:
            raise ValueError(
                f"Merged cache ID mismatch: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
            )
        merged = OrderedDict((sample_id, merged_unordered[sample_id]) for sample_id in ids)
        if not _finite(merged):
            raise ValueError(f"Merged cache contains non-finite values: {feature_path}")
        atomic_torch_save(merged, feature_path)
        durations = {task.sample_id: audio_duration(task.path) for task in tasks}
        metadata = {
            **expected_metadata,
            "created_at": utc_now(),
            "feature_path": str(feature_path.resolve()),
            "ids": ids,
            "duration_seconds": durations,
            "duration_stats": _stats(list(durations.values())),
            "rank_counts": rank_counts,
            "git": git_state(repo_root),
            "extraction": dict(extraction_metadata),
        }
        atomic_json_dump(metadata, metadata_path)
        print(
            f"[rank 0] merged {feature_path} count={len(merged)} rank_counts={rank_counts}",
            flush=True,
        )
    barrier(local_rank)
    return feature_path


def coordinated_load_model(
    loader: Callable[[], Any],
    *,
    rank: int,
    local_rank: int,
    label: str,
) -> Any:
    """Let rank zero finish checkpoint preparation before other ranks load."""
    model = None
    if rank == 0:
        print(f"[rank 0] preparing model {label}", flush=True)
        model = loader()
        print(f"[rank 0] model ready {label}", flush=True)
    barrier(local_rank)
    if rank != 0:
        model = loader()
    barrier(local_rank)
    return model


def release_model(model: Any, local_rank: int) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()
    barrier(local_rank)


def parallel_range_download(
    url: str,
    checkpoint: Path,
    total_size: int,
    *,
    connections: int | None = None,
) -> Path:
    """Download a known-size official asset using verified HTTP ranges."""
    checkpoint = checkpoint.expanduser().resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.is_file() and checkpoint.stat().st_size == total_size:
        print(f"Checkpoint ready: {checkpoint} ({total_size} bytes)", flush=True)
        return checkpoint
    if connections is None:
        connections = 16 if total_size >= 1_000_000_000 else 8
    if connections < 1:
        raise ValueError("connections must be positive")
    part_directory = checkpoint.parent / f".{checkpoint.name}.parts"
    part_directory.mkdir(parents=True, exist_ok=True)
    chunk_size = math.ceil(total_size / connections)
    processes: list[tuple[int, int, int, Path, subprocess.Popen[Any]]] = []
    print(
        f"Downloading {checkpoint.name} in {connections} verified ranges "
        f"({total_size} bytes total) from {url}",
        flush=True,
    )
    for index in range(connections):
        start = index * chunk_size
        end = min(total_size - 1, (index + 1) * chunk_size - 1)
        expected_size = end - start + 1
        part = part_directory / f"part{index:02d}.{start}-{end}"
        if part.is_file() and part.stat().st_size == expected_size:
            print(f"Range {index} already complete: {part}", flush=True)
            continue
        part.unlink(missing_ok=True)
        process = subprocess.Popen(
            [
                "curl",
                "-fL",
                "-sS",
                "--retry",
                "5",
                "--retry-all-errors",
                "--connect-timeout",
                "15",
                "--range",
                f"{start}-{end}",
                "--output",
                str(part),
                url,
            ]
        )
        processes.append((index, start, end, part, process))
    for index, start, end, part, process in processes:
        return_code = process.wait()
        expected_size = end - start + 1
        actual_size = part.stat().st_size if part.is_file() else -1
        if return_code != 0 or actual_size != expected_size:
            raise RuntimeError(
                f"PANNs range {index} failed: return_code={return_code}, "
                f"actual_size={actual_size}, expected_size={expected_size}"
            )
        print(
            f"Completed {checkpoint.name} range {index}: bytes {start}-{end}",
            flush=True,
        )
    assembled = checkpoint.with_name(f".{checkpoint.name}.assembling")
    with assembled.open("wb") as destination:
        for index in range(connections):
            start = index * chunk_size
            end = min(total_size - 1, (index + 1) * chunk_size - 1)
            part = part_directory / f"part{index:02d}.{start}-{end}"
            if part.stat().st_size != end - start + 1:
                raise ValueError(f"Invalid range before assembly: {part}")
            with part.open("rb") as source:
                shutil.copyfileobj(source, destination, length=8 << 20)
    if assembled.stat().st_size != total_size:
        raise ValueError(
            f"Assembled checkpoint has {assembled.stat().st_size} bytes; "
            f"expected {total_size}"
        )
    os.replace(assembled, checkpoint)
    for part in part_directory.iterdir():
        part.unlink()
    part_directory.rmdir()
    return checkpoint


def ensure_panns_16k_checkpoint() -> Path:
    checkpoint = Path.home() / ".cache/audioldm_eval/ckpt/Cnn14_16k_mAP=0.438.pth"
    checkpoint.with_suffix(".pth.partial").unlink(missing_ok=True)
    return parallel_range_download(
        PANNS_16K_URL,
        checkpoint,
        PANNS_16K_SIZE,
        connections=8,
    )


def direct_checkpoint_paths(cache_root: Path) -> dict[str, Path]:
    hub_checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
    clap_spec = importlib.util.find_spec("laion_clap")
    if clap_spec is None or clap_spec.origin is None:
        raise ImportError("laion_clap is required for checkpoint prefetch")
    return {
        "vggish": hub_checkpoints / DIRECT_CHECKPOINTS["vggish"][1],
        "passt_hear21passt_base": (
            hub_checkpoints / DIRECT_CHECKPOINTS["passt_hear21passt_base"][1]
        ),
        "passt_hear21passt_base20sec": (
            hub_checkpoints / DIRECT_CHECKPOINTS["passt_hear21passt_base20sec"][1]
        ),
        "passt_hear21passt_base30sec": (
            hub_checkpoints / DIRECT_CHECKPOINTS["passt_hear21passt_base30sec"][1]
        ),
        "imagebind_huge": Path.home() / ".cache/torch/hub/checkpoints/imagebind_huge.pth",
        "synchformer": synchformer_checkpoint_path(cache_root),
        "clap_630k_audioset": (
            Path(clap_spec.origin).parent / DIRECT_CHECKPOINTS["clap_630k_audioset"][1]
        ),
    }


def run_checkpoint_prefetch(args: argparse.Namespace) -> None:
    cache_root = Path(args.cache_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    panns_path = ensure_panns_16k_checkpoint()
    paths = direct_checkpoint_paths(cache_root)
    records: list[dict[str, Any]] = [
        {
            "name": "panns_16k",
            "url": PANNS_16K_URL,
            "path": str(panns_path),
            "size_bytes": PANNS_16K_SIZE,
        }
    ]
    for name, (url, _, size) in DIRECT_CHECKPOINTS.items():
        path = parallel_range_download(url, paths[name], size)
        records.append(
            {
                "name": name,
                "url": url,
                "path": str(path),
                "size_bytes": size,
            }
        )
    atomic_json_dump(
        {
            "protocol": PROTOCOL,
            "completed_at": utc_now(),
            "assets": records,
        },
        results_root / "checkpoint_prefetch_complete.json",
    )
    print("Direct checkpoint prefetch complete.", flush=True)


def panns_model(device: str) -> torch.nn.Module:
    from av_bench.panns import Cnn14

    ensure_panns_16k_checkpoint()
    unused_32k = Path.home() / ".cache/audioldm_eval/ckpt/Cnn14_mAP=0.431.pth"
    valid_unused_checkpoint = (
        unused_32k.is_file() and unused_32k.stat().st_size == 1365409299
    )
    created_placeholder = not valid_unused_checkpoint
    if created_placeholder:
        # The external constructor gates both downloads on this unused file.
        # A temporary placeholder prevents a 1.3 GB download; the 16 kHz
        # checkpoint above is the only state dict loaded by this model.
        unused_32k.parent.mkdir(parents=True, exist_ok=True)
        with unused_32k.open("wb"):
            pass
    try:
        model = Cnn14(
            features_list=["2048", "logits"],
            sample_rate=16000,
            window_size=512,
            hop_size=160,
            mel_bins=64,
            fmin=50,
            fmax=8000,
            classes_num=527,
        ).to(device).eval()
    finally:
        if created_placeholder:
            unused_32k.unlink(missing_ok=True)
    return model


def panns_extractor(model: torch.nn.Module, device: str) -> Callable[[AudioTask], Any]:
    def extract(task: AudioTask) -> Any:
        waveform = load_mono_torch(task.path, 16000, demean=True)
        with torch.inference_mode():
            output = model(waveform.unsqueeze(0).to(device))
        return {
            "2048": output["2048"][0].detach().cpu(),
            "logits": output["logits"][0].detach().cpu(),
        }

    return extract


def vggish_model(device: str) -> torch.nn.Module:
    from av_bench.vggish.vggish import VGGish

    return VGGish(device=torch.device(device), postprocess=False).eval()


def vggish_extractor(model: torch.nn.Module) -> Callable[[AudioTask], Any]:
    def extract(task: AudioTask) -> torch.Tensor:
        waveform = load_mono_torch(task.path, 16000, demean=True)
        with torch.inference_mode():
            features = model(waveform.unsqueeze(0), sample_rate=16000)[0]
        if features.ndim != 2 or features.shape[1] != 128:
            raise ValueError(f"Unexpected VGGish feature shape {tuple(features.shape)}")
        return features.detach().cpu()

    return extract


def passt_model(module_name: str, device: str) -> torch.nn.Module:
    module = importlib.import_module(module_name)
    model = module.get_basic_model(mode="all")
    return model.to(device).eval()


def passt_extractor(model: torch.nn.Module) -> Callable[[AudioTask], Any]:
    def extract(task: AudioTask) -> torch.Tensor:
        waveform = load_mono_torch(task.path, 32000, demean=True)
        with torch.inference_mode():
            all_features = model.get_scene_embeddings(waveform.unsqueeze(0))
        if all_features.shape != (1, 1295):
            raise ValueError(f"Unexpected PaSST feature shape {tuple(all_features.shape)}")
        logits = all_features[0, :527].detach().cpu()
        if logits.shape != (527,):
            raise ValueError(f"Unexpected PaSST logits shape {tuple(logits.shape)}")
        return logits

    return extract


def run_passt_tail_sensitivity(
    model: torch.nn.Module,
    module_name: str,
    output_path: Path,
) -> dict[str, Any]:
    """Verify content after eight seconds changes the full-duration result."""
    sample_rate = 32000
    seconds = 12
    samples = sample_rate * seconds
    time = torch.arange(samples, dtype=torch.float32) / sample_rate
    shared = 0.05 * torch.sin(2 * math.pi * 220.0 * time)
    first = shared.clone()
    second = shared.clone()
    tail = torch.arange(sample_rate * 4, dtype=torch.float32) / sample_rate
    first[sample_rate * 8 :] = 0.0
    second[sample_rate * 8 :] = 0.2 * torch.sin(2 * math.pi * 1760.0 * tail)
    with torch.inference_mode():
        first_output = model.get_scene_embeddings(first.unsqueeze(0)).detach().cpu()
        second_output = model.get_scene_embeddings(second.unsqueeze(0)).detach().cpu()
    difference = (first_output - second_output).abs()
    result = {
        "protocol": PROTOCOL,
        "module": module_name,
        "sample_rate": sample_rate,
        "duration_seconds": seconds,
        "shared_prefix_seconds": 8,
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "passed": bool(float(difference.max()) > 1e-7),
        "created_at": utc_now(),
    }
    if not result["passed"]:
        raise AssertionError(f"PaSST tail-sensitivity failed for {module_name}: {result}")
    atomic_json_dump(result, output_path)
    print(f"PaSST tail sensitivity {module_name}: {result}", flush=True)
    return result


def clap_model(device: str) -> Any:
    import laion_clap

    module = laion_clap.CLAP_Module(enable_fusion=False, device=device)
    text_embeddings = module.model.text_branch.embeddings
    if hasattr(text_embeddings, "position_ids"):
        text_embeddings._non_persistent_buffers_set.add("position_ids")
    module.load_ckpt()
    module.eval()
    return module


def clap_extractor(model: Any, device: str) -> Callable[[AudioTask], Any]:
    def extract(task: AudioTask) -> dict[str, Any]:
        waveform = load_mono_torch(task.path, 48000, demean=False)
        window = 480000
        hop = 240000
        starts = sample_window_starts(waveform.numel(), window, hop)
        windows: list[torch.Tensor] = []
        for start in starts:
            piece = waveform[start : start + window]
            if piece.numel() < window:
                # Model-native repeat/pad is only used for genuinely short samples.
                windows.append(piece)
            else:
                windows.append(piece)
        audio_embeddings: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch_start in range(0, len(windows), 4):
                batch_windows = windows[batch_start : batch_start + 4]
                if len({item.numel() for item in batch_windows}) == 1:
                    batch = torch.stack(batch_windows)
                    embeddings = model.get_audio_embedding_from_data(
                        batch, use_tensor=True
                    )
                    audio_embeddings.append(embeddings.detach().cpu())
                else:
                    for item in batch_windows:
                        embedding = model.get_audio_embedding_from_data(
                            item.unsqueeze(0), use_tensor=True
                        )
                        audio_embeddings.append(embedding.detach().cpu())
            texts = [task.prompt, task.prompt]
            text_embedding = model.get_text_embedding(
                texts, use_tensor=True
            )[0].detach().cpu()
        audio_embedding = torch.cat(audio_embeddings, dim=0).double()
        audio_embedding = F.normalize(audio_embedding, dim=-1)
        text_embedding = F.normalize(text_embedding.double(), dim=-1)
        window_scores = audio_embedding @ text_embedding
        return {
            "score": float(window_scores.mean()),
            "window_scores": window_scores.float(),
            "window_starts_samples": torch.tensor(starts, dtype=torch.int64),
            "num_windows": len(starts),
            "sample_rate": 48000,
            "window_seconds": 10.0,
            "hop_seconds": 5.0,
            "tail_strategy": "append_end_aligned_full_window",
        }

    return extract


def audiobox_model() -> Any:
    from audiobox_aesthetics.infer import initialize_predictor

    return initialize_predictor()


def audiobox_extractor(predictor: Any) -> Callable[[AudioTask], Any]:
    def extract(task: AudioTask) -> dict[str, Any]:
        row = predictor.forward([{"path": str(task.path)}])[0]
        return {
            "CE": float(row["CE"]),
            "CU": float(row["CU"]),
            "PC": float(row["PC"]),
            "PQ": float(row["PQ"]),
            "window_seconds": 10.0,
            "hop_seconds": 10.0,
            "tail_strategy": "zero_pad_with_duration_weighted_aggregation",
        }

    return extract


def _resize_frame(frame: Any, short_side: int = 224) -> torch.Tensor:
    width, height = frame.width, frame.height
    if width <= height:
        target_width = short_side
        target_height = max(short_side, round(height * short_side / width))
    else:
        target_height = short_side
        target_width = max(short_side, round(width * short_side / height))
    array = frame.reformat(
        width=target_width,
        height=target_height,
        format="rgb24",
    ).to_ndarray()
    return torch.from_numpy(np.asarray(array)).permute(2, 0, 1).contiguous()


def decode_video_at_times(path: Path, target_times: Sequence[float]) -> torch.Tensor:
    """Decode nearest source frames for sorted target timestamps."""
    import av

    if not target_times:
        raise ValueError("At least one target timestamp is required")
    indexed = sorted(enumerate(target_times), key=lambda item: item[1])
    sorted_targets = [float(item[1]) for item in indexed]
    selected: list[torch.Tensor | None] = [None] * len(sorted_targets)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        previous_frame = None
        previous_time = None
        target_index = 0
        last_frame = None
        for frame in container.decode(stream):
            frame_time = (
                float(frame.pts * stream.time_base)
                if frame.pts is not None
                else float(frame.time or 0.0)
            )
            last_frame = frame
            while target_index < len(sorted_targets) and sorted_targets[target_index] <= frame_time:
                target = sorted_targets[target_index]
                choose_previous = (
                    previous_frame is not None
                    and previous_time is not None
                    and target - previous_time <= frame_time - target
                )
                chosen = previous_frame if choose_previous else frame
                selected[target_index] = _resize_frame(chosen)
                target_index += 1
            previous_frame = frame
            previous_time = frame_time
            if target_index == len(sorted_targets):
                break
        if last_frame is None:
            raise ValueError(f"No video frames decoded from {path}")
        while target_index < len(sorted_targets):
            selected[target_index] = _resize_frame(last_frame)
            target_index += 1
    if any(frame is None for frame in selected):
        raise RuntimeError(f"Failed to decode all requested frames from {path}")
    sorted_tensor = torch.stack([frame for frame in selected if frame is not None])
    restored: list[torch.Tensor | None] = [None] * len(indexed)
    for sorted_index, (original_index, _) in enumerate(indexed):
        restored[original_index] = sorted_tensor[sorted_index]
    return torch.stack([frame for frame in restored if frame is not None])


def imagebind_model(device: str) -> torch.nn.Module:
    from imagebind.models import imagebind_model as imagebind_factory

    return imagebind_factory.imagebind_huge(pretrained=True).to(device).eval()


def imagebind_extractor(model: torch.nn.Module, device: str) -> Callable[[AudioTask], Any]:
    import torchvision.transforms.v2 as v2
    from av_bench.data.audio_dataset import waveform2melspec
    from av_bench.data.ib_data import SpatialCrop
    from imagebind.models.imagebind_model import ModalityType

    transform = v2.Compose(
        [
            v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    crop = SpatialCrop(224, 3)

    def extract(task: AudioTask) -> dict[str, Any]:
        if task.video is None:
            raise ValueError("ImageBind requires source video")
        waveform = load_mono_torch(task.path, 16000, demean=False).unsqueeze(0)
        audio_seconds = waveform.shape[-1] / 16000
        source_video_seconds = video_duration(task.video)
        overlap = min(audio_seconds, source_video_seconds)
        if overlap < 2.0:
            raise ValueError(
                f"ImageBind overlap is shorter than its native two-second clip: {overlap:.3f}s"
            )
        starts = full_coverage_window_starts(overlap, 2.0, 2.0)
        target_times = [
            start + fraction * 2.0
            for start in starts
            for fraction in (0.25, 0.75)
        ]
        decoded = decode_video_at_times(task.video, target_times)
        scores: list[float] = []
        with torch.inference_mode():
            for index, start in enumerate(starts):
                start_sample = round(start * 16000)
                audio_piece = waveform[:, start_sample : start_sample + 32000].clone()
                if audio_piece.shape[-1] != 32000:
                    raise ValueError(
                        f"ImageBind tail window is incomplete for {task.sample_id}: "
                        f"{audio_piece.shape[-1]} samples"
                    )
                mel = waveform2melspec(
                    audio_piece,
                    sample_rate=16000,
                    num_mel_bins=128,
                    target_length=204,
                )
                mel = (mel - (-4.268)) / 9.138
                audio_input = mel.unsqueeze(0).unsqueeze(0).to(device)

                frames = decoded[index * 2 : index * 2 + 2]
                frames = transform(frames)
                video_cthw = frames.permute(1, 0, 2, 3)
                video_crops = torch.stack(crop([video_cthw])).unsqueeze(0).to(device)
                output = model(
                    {
                        ModalityType.AUDIO: audio_input,
                        ModalityType.VISION: video_crops,
                    }
                )
                audio_embedding = output[ModalityType.AUDIO]
                video_embedding = output[ModalityType.VISION]
                score = F.cosine_similarity(audio_embedding, video_embedding, dim=-1)
                scores.append(float(score[0].detach().cpu()))
        return {
            "score": float(np.mean(scores)),
            "window_scores": torch.tensor(scores, dtype=torch.float32),
            "window_starts_seconds": torch.tensor(starts, dtype=torch.float64),
            "num_windows": len(starts),
            "audio_duration_seconds": audio_seconds,
            "video_duration_seconds": source_video_seconds,
            "overlap_duration_seconds": overlap,
            "window_seconds": 2.0,
            "hop_seconds": 2.0,
            "tail_strategy": "append_end_aligned_window",
            "aggregation": "mean_window_cosine_then_mean_samples",
        }

    return extract


def synchformer_checkpoint_path(cache_root: Path) -> Path:
    configured = os.environ.get("AUDIO_EVAL_SYNCHFORMER_CHECKPOINT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (cache_root.parent / "model_checkpoints" / "synchformer_state_dict.pth").resolve()


def ensure_synchformer_checkpoint(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(SYNCHFORMER_URL, str(path))


def synchformer_model(device: str, checkpoint_path: Path) -> torch.nn.Module:
    from av_bench.synchformer.synchformer import Synchformer

    ensure_synchformer_checkpoint(checkpoint_path)
    model = Synchformer()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def desync_extractor(model: torch.nn.Module, device: str) -> Callable[[AudioTask], Any]:
    import torchaudio
    import torchvision.transforms.v2 as v2
    from av_bench.data.audio_dataset import pad_or_truncate
    from av_bench.synchformer.synchformer import make_class_grid

    transform = v2.Compose(
        [
            v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        win_length=400,
        hop_length=160,
        n_fft=1024,
        n_mels=128,
    ).to(device)
    grid = make_class_grid(-2, 2, 21).cpu()

    def extract(task: AudioTask) -> dict[str, Any]:
        if task.video is None:
            raise ValueError("DeSync requires source video")
        waveform = load_mono_torch(task.path, 16000, demean=False)
        audio_seconds = waveform.numel() / 16000
        source_video_seconds = video_duration(task.video)
        overlap = min(audio_seconds, source_video_seconds)
        if overlap < 8.0:
            raise ValueError(
                f"Synchformer overlap is shorter than its native eight-second clip: {overlap:.3f}s"
            )
        starts = full_coverage_window_starts(overlap, 8.0, 4.0)
        target_times = [
            start + frame_index / 25.0
            for start in starts
            for frame_index in range(200)
        ]
        decoded = decode_video_at_times(task.video, target_times)
        window_scores: list[float] = []
        first_scores: list[float] = []
        last_scores: list[float] = []
        with torch.inference_mode():
            for index, start in enumerate(starts):
                start_sample = round(start * 16000)
                audio_piece = waveform[start_sample : start_sample + 128000]
                if audio_piece.numel() != 128000:
                    raise ValueError(
                        f"Synchformer tail window is incomplete for {task.sample_id}: "
                        f"{audio_piece.numel()} samples"
                    )
                audio_piece = audio_piece.unsqueeze(0).to(device)
                audio_segments = torch.stack(
                    [
                        audio_piece[:, segment_start : segment_start + 10240]
                        for segment_start in range(0, 117761, 5120)
                    ],
                    dim=1,
                )
                audio_features = mel(audio_segments)
                audio_features = torch.log(audio_features + 1e-6)
                audio_features = pad_or_truncate(audio_features, 66)
                audio_features = (audio_features + 4.2677393) / (2 * 4.5689974)
                audio_features = model.extract_afeats(audio_features.unsqueeze(2))

                frames = decoded[index * 200 : (index + 1) * 200]
                frames = transform(frames).to(device)
                video_segments = torch.stack(
                    [frames[segment_start : segment_start + 16] for segment_start in range(0, 185, 8)],
                    dim=0,
                ).unsqueeze(0)
                video_features = model.extract_vfeats(video_segments)
                first_id = int(
                    model.compare_v_a(video_features[:, :14], audio_features[:, :14])
                    .argmax(dim=-1)
                    .cpu()[0]
                )
                last_id = int(
                    model.compare_v_a(video_features[:, -14:], audio_features[:, -14:])
                    .argmax(dim=-1)
                    .cpu()[0]
                )
                first = abs(float(grid[first_id]))
                last = abs(float(grid[last_id]))
                first_scores.append(first)
                last_scores.append(last)
                window_scores.append((first + last) / 2.0)
        return {
            "score": float(np.mean(window_scores)),
            "window_scores": torch.tensor(window_scores, dtype=torch.float32),
            "first_scores": torch.tensor(first_scores, dtype=torch.float32),
            "last_scores": torch.tensor(last_scores, dtype=torch.float32),
            "window_starts_seconds": torch.tensor(starts, dtype=torch.float64),
            "num_windows": len(starts),
            "audio_duration_seconds": audio_seconds,
            "video_duration_seconds": source_video_seconds,
            "overlap_duration_seconds": overlap,
            "window_seconds": 8.0,
            "hop_seconds": 4.0,
            "tail_strategy": "append_end_aligned_window",
            "aggregation": "mean_absolute_first_last_offset_per_window_then_mean_samples",
            "unit": "seconds",
        }

    return extract


def _all_audio_collections(
    records_by_dataset: Mapping[str, Sequence[BenchmarkRecord]],
    models: Sequence[str],
) -> Iterable[tuple[str, str, list[AudioTask]]]:
    for dataset, records in records_by_dataset.items():
        for collection in ["gt", *models]:
            yield dataset, collection, collection_tasks(records, collection)


def _generated_collections(
    records_by_dataset: Mapping[str, Sequence[BenchmarkRecord]],
    models: Sequence[str],
) -> Iterable[tuple[str, str, list[AudioTask]]]:
    for dataset, records in records_by_dataset.items():
        for collection in models:
            yield dataset, collection, collection_tasks(records, collection)


def write_execution_manifest(
    *,
    path: Path,
    repo_root: Path,
    pack_root: Path,
    cache_root: Path,
    results_root: Path,
    records_by_dataset: Mapping[str, Sequence[BenchmarkRecord]],
    models: Sequence[str],
    world_size: int,
) -> None:
    manifest = {
        "protocol": PROTOCOL,
        "created_at": utc_now(),
        "command_argv": sys.argv,
        "cwd": str(Path.cwd()),
        "repo_root": str(repo_root),
        "pack_root": str(pack_root),
        "cache_root": str(cache_root),
        "results_root": str(results_root),
        "datasets": {key: len(value) for key, value in records_by_dataset.items()},
        "models": list(models),
        "world_size": world_size,
        "rank_devices": {
            str(rank): {
                "local_rank": rank,
                "cuda_device": f"cuda:{rank}",
                "gpu_name": torch.cuda.get_device_name(rank),
            }
            for rank in range(world_size)
        },
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": package_versions(),
        "git": git_state(repo_root),
        "code_fingerprint": code_fingerprint(),
        "proxy": {
            "http_proxy": os.environ.get("http_proxy"),
            "https_proxy": os.environ.get("https_proxy"),
        },
    }
    atomic_json_dump(manifest, path)


def run_distributed_extraction(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed V2SFX extraction requires CUDA")
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world_size != 8:
        raise RuntimeError(f"This benchmark requires exactly 8 ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    torch.manual_seed(2020 + rank)
    np.random.seed(2020 + rank)

    repo_root = Path(args.repo_root).expanduser().resolve()
    pack_root = Path(args.pack_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    records_by_dataset = load_records(
        pack_root,
        args.datasets,
        args.models,
        limit=args.limit,
    )
    if rank == 0:
        write_execution_manifest(
            path=results_root / "execution_manifest.json",
            repo_root=repo_root,
            pack_root=pack_root,
            cache_root=cache_root,
            results_root=results_root,
            records_by_dataset=records_by_dataset,
            models=args.models,
            world_size=world_size,
        )
    barrier(local_rank)

    selected = set(args.backends)
    print(
        f"[rank {rank}] initialized device={device} world_size={world_size} "
        f"backends={args.backends}",
        flush=True,
    )

    if "panns" in selected:
        model = coordinated_load_model(
            lambda: panns_model(device), rank=rank, local_rank=local_rank, label="PANNs Cnn14 16k"
        )
        extractor = panns_extractor(model, device)
        for dataset, collection, tasks in _all_audio_collections(records_by_dataset, args.models):
            distributed_extract_collection(
                cache_root=cache_root,
                dataset=dataset,
                collection=collection,
                backend="panns",
                variant="Cnn14_16k_mAP_0_438",
                tasks=tasks,
                extractor=extractor,
                extraction_metadata={
                    "sample_rate": 16000,
                    "mono": "channel_mean",
                    "normalization": "subtract_waveform_mean",
                    "batch_size": 1,
                    "input": "complete_waveform",
                    "aggregation": "model_global_max_plus_mean_pool",
                    "checkpoint": "Cnn14_16k_mAP=0.438.pth",
                    "outputs": {"logits": 527, "embedding": 2048},
                },
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                repo_root=repo_root,
                refresh=args.refresh,
            )
        release_model(model, local_rank)

    if "vggish" in selected:
        model = coordinated_load_model(
            lambda: vggish_model(device), rank=rank, local_rank=local_rank, label="VGGish"
        )
        extractor = vggish_extractor(model)
        for dataset, collection, tasks in _all_audio_collections(records_by_dataset, args.models):
            distributed_extract_collection(
                cache_root=cache_root,
                dataset=dataset,
                collection=collection,
                backend="vggish",
                variant="harritaylor_torchvggish",
                tasks=tasks,
                extractor=extractor,
                extraction_metadata={
                    "sample_rate": 16000,
                    "mono": "channel_mean",
                    "normalization": "subtract_waveform_mean",
                    "batch_size": 1,
                    "input": "complete_waveform",
                    "windowing": "native waveform_to_examples",
                    "aggregation": "retain_all_valid_128d_windows_for_FAD",
                    "tail_strategy": "native_VGGish_preprocessor",
                },
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                repo_root=repo_root,
                refresh=args.refresh,
            )
        release_model(model, local_rank)

    for module_name, variant in PASST_MODULES.items():
        if "passt" not in selected and module_name not in selected:
            continue
        model = coordinated_load_model(
            lambda module_name=module_name: passt_model(module_name, device),
            rank=rank,
            local_rank=local_rank,
            label=module_name,
        )
        if rank == 0:
            run_passt_tail_sensitivity(
                model,
                module_name,
                results_root / "validation" / f"tail_sensitivity_{variant}.json",
            )
        barrier(local_rank)
        extractor = passt_extractor(model)
        max_window_seconds = float(model.max_model_window / 32000)
        scene_hop_seconds = float(model.scene_hop / 32000)
        for dataset, collection, tasks in _all_audio_collections(records_by_dataset, args.models):
            distributed_extract_collection(
                cache_root=cache_root,
                dataset=dataset,
                collection=collection,
                backend="passt",
                variant=variant,
                tasks=tasks,
                extractor=extractor,
                extraction_metadata={
                    "module": module_name,
                    "constructor": "get_basic_model(mode='all')",
                    "sample_rate": 32000,
                    "mono": "channel_mean",
                    "normalization": "subtract_waveform_mean",
                    "batch_size": 1,
                    "input": "complete_waveform_via_get_scene_embeddings",
                    "native_max_window_seconds": max_window_seconds,
                    "native_scene_hop_seconds": scene_hop_seconds,
                    "tail_strategy": "official_wrapper_reflect_padding_and_scene_windows",
                    "aggregation": "official_wrapper_scene_embedding_mean",
                    "logits_dimension": 527,
                },
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                repo_root=repo_root,
                refresh=args.refresh,
            )
        release_model(model, local_rank)

    if "clap" in selected:
        model = coordinated_load_model(
            lambda: clap_model(device),
            rank=rank,
            local_rank=local_rank,
            label="LAION CLAP 630k-audioset",
        )
        extractor = clap_extractor(model, device)
        for dataset, collection, tasks in _generated_collections(records_by_dataset, args.models):
            distributed_extract_collection(
                cache_root=cache_root,
                dataset=dataset,
                collection=collection,
                backend="clap",
                variant="630k_audioset_nonfusion",
                tasks=tasks,
                extractor=extractor,
                extraction_metadata={
                    "model": "630k-audioset",
                    "enable_fusion": False,
                    "sample_rate": 48000,
                    "window_seconds": 10.0,
                    "hop_seconds": 5.0,
                    "tail_strategy": "append_end_aligned_full_window",
                    "short_sample_strategy": "model_native_repeatpad_only",
                    "aggregation": "mean_window_text_audio_cosine_per_sample",
                    "prompt_source": "dataset_metadata.csv",
                },
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                repo_root=repo_root,
                refresh=args.refresh,
            )
        release_model(model, local_rank)

    if "audiobox" in selected:
        predictor = coordinated_load_model(
            audiobox_model,
            rank=rank,
            local_rank=local_rank,
            label="facebook/audiobox-aesthetics",
        )
        extractor = audiobox_extractor(predictor)
        for dataset, collection, tasks in _generated_collections(records_by_dataset, args.models):
            distributed_extract_collection(
                cache_root=cache_root,
                dataset=dataset,
                collection=collection,
                backend="audiobox",
                variant="facebook_audiobox_aesthetics",
                tasks=tasks,
                extractor=extractor,
                extraction_metadata={
                    "model": "facebook/audiobox-aesthetics",
                    "sample_rate": 16000,
                    "window_seconds": 10.0,
                    "hop_seconds": 10.0,
                    "tail_strategy": "zero_pad_with_valid_duration_weight",
                    "aggregation": "duration_weighted_window_mean_per_sample",
                    "axes": ["CE", "CU", "PC", "PQ"],
                },
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                repo_root=repo_root,
                refresh=args.refresh,
            )
        release_model(predictor, local_rank)

    movie_records = records_by_dataset.get("moviegen_audio")
    if "imagebind" in selected and movie_records is not None:
        model = coordinated_load_model(
            lambda: imagebind_model(device),
            rank=rank,
            local_rank=local_rank,
            label="ImageBind Huge",
        )
        extractor = imagebind_extractor(model, device)
        for collection in args.models:
            tasks = collection_tasks(movie_records, collection)
            distributed_extract_collection(
                cache_root=cache_root,
                dataset="moviegen_audio",
                collection=collection,
                backend="imagebind",
                variant="imagebind_huge_full_timeline",
                tasks=tasks,
                extractor=extractor,
                extraction_metadata={
                    "model": "ImageBind Huge",
                    "audio_sample_rate": 16000,
                    "window_seconds": 2.0,
                    "hop_seconds": 2.0,
                    "video_frames_per_window": 2,
                    "video_frame_times": "25% and 75% of each aligned window",
                    "spatial_crops": 3,
                    "tail_strategy": "append_end_aligned_window",
                    "timeline": "real_audio_video_overlap",
                    "aggregation": "mean_window_cosine_then_mean_samples",
                },
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                repo_root=repo_root,
                refresh=args.refresh,
                include_video_fingerprint=True,
            )
        release_model(model, local_rank)

    if "desync" in selected and movie_records is not None:
        checkpoint = synchformer_checkpoint_path(cache_root)
        model = coordinated_load_model(
            lambda: synchformer_model(device, checkpoint),
            rank=rank,
            local_rank=local_rank,
            label="hkchengrex/av-benchmark Synchformer",
        )
        extractor = desync_extractor(model, device)
        for collection in args.models:
            tasks = collection_tasks(movie_records, collection)
            distributed_extract_collection(
                cache_root=cache_root,
                dataset="moviegen_audio",
                collection=collection,
                backend="desync",
                variant="synchformer_full_timeline",
                tasks=tasks,
                extractor=extractor,
                extraction_metadata={
                    "model": "hkchengrex/av-benchmark Synchformer",
                    "checkpoint": str(checkpoint),
                    "audio_sample_rate": 16000,
                    "video_fps": 25.0,
                    "window_seconds": 8.0,
                    "hop_seconds": 4.0,
                    "tail_strategy": "append_end_aligned_window",
                    "timeline": "real_audio_video_overlap",
                    "offset_grid_seconds": [-2.0, 2.0, 21],
                    "aggregation": "mean_absolute_first_last_offset_per_window_then_mean_samples",
                },
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                repo_root=repo_root,
                refresh=args.refresh,
                include_video_fingerprint=True,
            )
        release_model(model, local_rank)

    if rank == 0:
        atomic_json_dump(
            {
                "protocol": PROTOCOL,
                "completed_at": utc_now(),
                "backends": args.backends,
                "world_size": world_size,
            },
            results_root / "distributed_extraction_complete.json",
        )
    barrier(local_rank)
    dist.destroy_process_group()


def load_merged_cache(
    cache_root: Path,
    dataset: str,
    collection: str,
    backend: str,
    variant: str,
    expected_ids: Sequence[str],
) -> OrderedDict[str, Any]:
    feature_path, metadata_path = cache_location(
        cache_root, dataset, collection, backend, variant
    )
    if not feature_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete cache: {feature_path} / {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("protocol") != PROTOCOL:
        raise ValueError(f"Wrong cache protocol in {metadata_path}")
    if metadata.get("num_samples") != len(expected_ids):
        raise ValueError(f"Wrong sample count in {metadata_path}")
    if metadata.get("world_size") != 8:
        raise ValueError(f"Cache was not created with 8 ranks: {metadata_path}")
    values = torch.load(feature_path, map_location="cpu", weights_only=True)
    if not isinstance(values, Mapping) or list(values) != list(expected_ids):
        raise ValueError(f"Cache ID/order mismatch: {feature_path}")
    if not _finite(values):
        raise ValueError(f"Cache contains non-finite values: {feature_path}")
    return OrderedDict(values)


def paired_kl(
    reference_logits: Mapping[str, torch.Tensor],
    generated_logits: Mapping[str, torch.Tensor],
) -> tuple[float, list[dict[str, Any]]]:
    if list(reference_logits) != list(generated_logits):
        raise ValueError("KL cache keys are not exactly paired")
    keys = list(reference_logits)
    reference = torch.stack([reference_logits[key] for key in keys]).double()
    generated = torch.stack([generated_logits[key] for key in keys]).double()
    if reference.shape != generated.shape or reference.shape[1] != 527:
        raise ValueError(
            f"Unexpected KL logits shapes: reference={reference.shape}, generated={generated.shape}"
        )
    log_reference = F.log_softmax(reference, dim=1)
    log_generated = F.log_softmax(generated, dim=1)
    probabilities = log_reference.exp()
    per_sample = (probabilities * (log_reference - log_generated)).sum(dim=1)
    return float(per_sample.mean()), [
        {"id": key, "kl": float(value)}
        for key, value in zip(keys, per_sample, strict=True)
    ]


def inception_score_from_logits(
    logits_by_id: Mapping[str, torch.Tensor],
    *,
    splits: int = 10,
    shuffle: bool = True,
    seed: int = 2020,
) -> dict[str, Any]:
    logits = torch.stack(list(logits_by_id.values())).double()
    effective_splits = splits if logits.shape[0] >= splits else 1
    if shuffle:
        permutation = np.random.RandomState(seed).permutation(logits.shape[0])
        logits = logits[permutation]
    probabilities = logits.softmax(dim=1)
    log_probabilities = logits.log_softmax(dim=1)
    scores: list[float] = []
    for index in range(effective_splits):
        start = index * logits.shape[0] // effective_splits
        end = (index + 1) * logits.shape[0] // effective_splits
        probability = probabilities[start:end]
        log_probability = log_probabilities[start:end]
        marginal = probability.mean(dim=0, keepdim=True)
        value = (
            (probability * (log_probability - marginal.log())).sum(dim=1).mean().exp()
        )
        scores.append(float(value))
    result = {
        "inception_score": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "num_samples": logits.shape[0],
        "splits": effective_splits,
        "requested_splits": splits,
        "shuffle": shuffle,
        "seed": seed,
        "split_scores": scores,
    }
    if not _finite(result):
        raise ValueError(f"Non-finite Inception Score: {result}")
    return result


def openl3_expected_metadata(
    tasks: Sequence[AudioTask], dataset: str, collection: str
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "dataset": dataset,
        "collection": collection,
        "backend": "openl3",
        "model_variant": "mel256_music_512_stereo",
        "num_samples": len(tasks),
        "input_fingerprint": tasks_fingerprint(tasks, include_video=False),
        "code_fingerprint": code_fingerprint(),
        "sample_rate": 44100,
        "channels": 2,
        "content_type": "music",
        "hop_seconds": 0.5,
        "input": "complete_waveform",
    }


def openl3_collection_embeddings(
    *,
    cache_root: Path,
    dataset: str,
    collection: str,
    tasks: Sequence[AudioTask],
    refresh: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    from audio_eval.metrics.fd import _openl3_embeddings

    directory = cache_root / dataset / collection / "openl3" / "mel256_music_512_stereo"
    metadata_path = directory / "cache_meta.json"
    expected = openl3_expected_metadata(tasks, dataset, collection)
    if metadata_path.is_file() and not refresh:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if all(metadata.get(key) == value for key, value in expected.items()):
                feature_path = Path(metadata["feature_path"])
                if feature_path.is_file():
                    with np.load(feature_path, allow_pickle=False) as loaded:
                        embeddings = loaded["embeddings"]
                    if embeddings.ndim == 2 and embeddings.shape[1] == 1024 and np.isfinite(embeddings).all():
                        return embeddings, metadata
        except Exception:
            pass
    directory.mkdir(parents=True, exist_ok=True)
    source = OrderedDict((task.sample_id, task.path) for task in tasks)
    embeddings, feature_path, num_audio = _openl3_embeddings(
        source,
        sample_rate=None,
        target_sample_rate=44100,
        channels=2,
        content_type="music",
        hop_size=0.5,
        batch_size=4,
        cache_dir=directory,
        refresh_cache=True,
    )
    if num_audio != len(tasks) or embeddings.ndim != 2 or embeddings.shape[1] != 1024:
        raise ValueError(
            f"Unexpected OpenL3 output for {dataset}/{collection}: "
            f"num_audio={num_audio}, shape={embeddings.shape}"
        )
    metadata = {
        **expected,
        "created_at": utc_now(),
        "feature_path": str(feature_path.resolve()),
        "num_embeddings": int(embeddings.shape[0]),
        "embedding_dimension": int(embeddings.shape[1]),
        "duration_seconds": {task.sample_id: audio_duration(task.path) for task in tasks},
        "windowing": "OpenL3 native complete-audio windows",
        "aggregation": "all valid windows enter distribution statistics",
    }
    atomic_json_dump(metadata, metadata_path)
    return embeddings, metadata


def checkpoint_inventory(cache_root: Path) -> list[dict[str, Any]]:
    patterns = [
        Path.home() / ".cache/audioldm_eval/ckpt/Cnn14_16k_mAP=0.438.pth",
        Path.home() / ".cache/audioldm_eval/ckpt/Cnn14_mAP=0.431.pth",
        Path.home() / ".cache/torch/hub/checkpoints/imagebind_huge.pth",
        synchformer_checkpoint_path(cache_root),
    ]
    try:
        clap_spec = importlib.util.find_spec("laion_clap")
        if clap_spec is not None and clap_spec.origin is not None:
            patterns.append(Path(clap_spec.origin).parent / "630k-audioset-best.pt")
    except Exception:
        pass
    roots = [
        Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch")) / "hub/checkpoints",
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub",
    ]
    candidates: list[Path] = list(patterns)
    for root in roots:
        if not root.exists():
            continue
        candidates.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".pth", ".pt", ".ckpt", ".bin", ".safetensors"}
        )
    unique: OrderedDict[str, Path] = OrderedDict()
    for path in candidates:
        if path.is_file():
            resolved = path.resolve()
            unique[str(resolved)] = resolved
    inventory: list[dict[str, Any]] = []
    for path in unique.values():
        stat = path.stat()
        inventory.append(
            {
                "path": str(path),
                "size_bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def _mean_cached_score(cache: Mapping[str, Mapping[str, Any]], key: str) -> float:
    values = [float(item[key]) for item in cache.values()]
    result = float(np.mean(values))
    if not math.isfinite(result):
        raise ValueError(f"Non-finite mean for cache key {key}")
    return result


def run_aggregation(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).expanduser().resolve()
    pack_root = Path(args.pack_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not (results_root / "distributed_extraction_complete.json").is_file():
        raise FileNotFoundError(
            results_root / "distributed_extraction_complete.json"
        )
    records_by_dataset = load_records(
        pack_root,
        args.datasets,
        args.models,
        limit=args.limit,
    )
    results_root.mkdir(parents=True, exist_ok=True)
    audit = audit_data(records_by_dataset, args.models)
    atomic_json_dump(audit, results_root / "data_audit.json")

    openl3_by_collection: dict[tuple[str, str], np.ndarray] = {}
    openl3_metadata: dict[str, Any] = {}
    for dataset, collection, tasks in _all_audio_collections(records_by_dataset, args.models):
        embeddings, metadata = openl3_collection_embeddings(
            cache_root=cache_root,
            dataset=dataset,
            collection=collection,
            tasks=tasks,
            refresh=args.refresh,
        )
        openl3_by_collection[(dataset, collection)] = embeddings
        openl3_metadata[f"{dataset}/{collection}"] = metadata

    raw: dict[str, Any] = {
        "protocol": PROTOCOL,
        "created_at": utc_now(),
        "git": git_state(repo_root),
        "datasets": {},
        "openl3_caches": openl3_metadata,
        "csv_output": str(output_path),
    }
    csv_rows: list[dict[str, Any]] = []
    metric_order = [
        "CLAP",
        "FAD-VGGish",
        "FD-openl3",
        "KL-PANNs",
        *[f"KL-PASST-{name}" for name in PASST_MODULES],
        "IS-PANNs",
        *[f"IS-PASST-{name}" for name in PASST_MODULES],
        "Audiobox-CE",
        "Audiobox-CU",
        "Audiobox-PC",
        "Audiobox-PQ",
        "IB-Score",
        "DeSync",
    ]
    better = {
        "CLAP": "higher",
        "FAD-VGGish": "lower",
        "FD-openl3": "lower",
        "KL-PANNs": "lower",
        "IS-PANNs": "higher",
        "Audiobox-CE": "higher",
        "Audiobox-CU": "higher",
        "Audiobox-PC": "higher",
        "Audiobox-PQ": "higher",
        "IB-Score": "higher",
        "DeSync": "lower",
    }
    for module_name in PASST_MODULES:
        better[f"KL-PASST-{module_name}"] = "lower"
        better[f"IS-PASST-{module_name}"] = "higher"

    for dataset, records in records_by_dataset.items():
        ids = [record.sample_id for record in records]
        dataset_raw: dict[str, Any] = {"num_samples": len(ids), "models": {}}
        gt_panns_raw = load_merged_cache(
            cache_root,
            dataset,
            "gt",
            "panns",
            "Cnn14_16k_mAP_0_438",
            ids,
        )
        gt_panns = OrderedDict(
            (key, value["logits"]) for key, value in gt_panns_raw.items()
        )
        gt_vggish = load_merged_cache(
            cache_root,
            dataset,
            "gt",
            "vggish",
            "harritaylor_torchvggish",
            ids,
        )
        gt_vggish_array = torch.cat(list(gt_vggish.values()), dim=0).numpy()
        gt_passt = {
            module_name: load_merged_cache(
                cache_root,
                dataset,
                "gt",
                "passt",
                variant,
                ids,
            )
            for module_name, variant in PASST_MODULES.items()
        }
        values_by_metric: dict[str, dict[str, float | None]] = {
            metric: {} for metric in metric_order
        }
        for model_name in args.models:
            model_column = MODEL_COLUMNS[model_name]
            model_raw: dict[str, Any] = {}
            generated_panns_raw = load_merged_cache(
                cache_root,
                dataset,
                model_name,
                "panns",
                "Cnn14_16k_mAP_0_438",
                ids,
            )
            generated_panns = OrderedDict(
                (key, value["logits"]) for key, value in generated_panns_raw.items()
            )
            kl_panns, kl_panns_details = paired_kl(gt_panns, generated_panns)
            is_panns = inception_score_from_logits(generated_panns)
            model_raw["KL-PANNs"] = {
                "value": kl_panns,
                "direction": "KL(P_GT || P_generated)",
                "num_samples": len(ids),
                "details": kl_panns_details,
            }
            model_raw["IS-PANNs"] = is_panns
            values_by_metric["KL-PANNs"][model_column] = kl_panns
            values_by_metric["IS-PANNs"][model_column] = is_panns["inception_score"]

            for module_name, variant in PASST_MODULES.items():
                generated = load_merged_cache(
                    cache_root,
                    dataset,
                    model_name,
                    "passt",
                    variant,
                    ids,
                )
                kl_value, kl_details = paired_kl(gt_passt[module_name], generated)
                is_value = inception_score_from_logits(generated)
                kl_metric = f"KL-PASST-{module_name}"
                is_metric = f"IS-PASST-{module_name}"
                model_raw[kl_metric] = {
                    "value": kl_value,
                    "direction": "KL(P_GT || P_generated)",
                    "num_samples": len(ids),
                    "details": kl_details,
                }
                model_raw[is_metric] = is_value
                values_by_metric[kl_metric][model_column] = kl_value
                values_by_metric[is_metric][model_column] = is_value["inception_score"]

            generated_vggish = load_merged_cache(
                cache_root,
                dataset,
                model_name,
                "vggish",
                "harritaylor_torchvggish",
                ids,
            )
            generated_vggish_array = torch.cat(list(generated_vggish.values()), dim=0).numpy()
            fad = frechet_distance(generated_vggish_array, gt_vggish_array)
            model_raw["FAD-VGGish"] = {
                "value": fad,
                "num_generated_windows": int(generated_vggish_array.shape[0]),
                "num_reference_windows": int(gt_vggish_array.shape[0]),
            }
            values_by_metric["FAD-VGGish"][model_column] = fad

            fd_openl3 = frechet_distance(
                openl3_by_collection[(dataset, model_name)],
                openl3_by_collection[(dataset, "gt")],
            )
            model_raw["FD-openl3"] = {
                "value": fd_openl3,
                "configuration": "mel256/music/512d/stereo/44100Hz/hop0.5s",
                "num_generated_windows": int(
                    openl3_by_collection[(dataset, model_name)].shape[0]
                ),
                "num_reference_windows": int(
                    openl3_by_collection[(dataset, "gt")].shape[0]
                ),
            }
            values_by_metric["FD-openl3"][model_column] = fd_openl3

            clap = load_merged_cache(
                cache_root,
                dataset,
                model_name,
                "clap",
                "630k_audioset_nonfusion",
                ids,
            )
            clap_value = _mean_cached_score(clap, "score")
            model_raw["CLAP"] = {
                "value": clap_value,
                "model": "630k-audioset",
                "num_samples": len(ids),
                "details": [
                    {
                        "id": key,
                        "score": float(value["score"]),
                        "num_windows": int(value["num_windows"]),
                    }
                    for key, value in clap.items()
                ],
            }
            values_by_metric["CLAP"][model_column] = clap_value

            audiobox = load_merged_cache(
                cache_root,
                dataset,
                model_name,
                "audiobox",
                "facebook_audiobox_aesthetics",
                ids,
            )
            for axis in ["CE", "CU", "PC", "PQ"]:
                metric = f"Audiobox-{axis}"
                value = _mean_cached_score(audiobox, axis)
                model_raw[metric] = {
                    "value": value,
                    "num_samples": len(ids),
                    "details": [
                        {"id": key, axis: float(item[axis])}
                        for key, item in audiobox.items()
                    ],
                }
                values_by_metric[metric][model_column] = value

            if dataset == "moviegen_audio":
                imagebind = load_merged_cache(
                    cache_root,
                    dataset,
                    model_name,
                    "imagebind",
                    "imagebind_huge_full_timeline",
                    ids,
                )
                imagebind_value = _mean_cached_score(imagebind, "score")
                model_raw["IB-Score"] = {
                    "value": imagebind_value,
                    "num_samples": len(ids),
                    "details": [
                        {
                            "id": key,
                            "score": float(item["score"]),
                            "num_windows": int(item["num_windows"]),
                            "overlap_duration_seconds": float(
                                item["overlap_duration_seconds"]
                            ),
                        }
                        for key, item in imagebind.items()
                    ],
                }
                values_by_metric["IB-Score"][model_column] = imagebind_value
                desync = load_merged_cache(
                    cache_root,
                    dataset,
                    model_name,
                    "desync",
                    "synchformer_full_timeline",
                    ids,
                )
                desync_value = _mean_cached_score(desync, "score")
                model_raw["DeSync"] = {
                    "value": desync_value,
                    "unit": "seconds",
                    "num_samples": len(ids),
                    "details": [
                        {
                            "id": key,
                            "score": float(item["score"]),
                            "num_windows": int(item["num_windows"]),
                            "overlap_duration_seconds": float(
                                item["overlap_duration_seconds"]
                            ),
                        }
                        for key, item in desync.items()
                    ],
                }
                values_by_metric["DeSync"][model_column] = desync_value
            else:
                model_raw["IB-Score"] = {
                    "value": None,
                    "reason": "CineBench50 has no source video",
                }
                model_raw["DeSync"] = {
                    "value": None,
                    "reason": "CineBench50 has no source video",
                }
                values_by_metric["IB-Score"][model_column] = None
                values_by_metric["DeSync"][model_column] = None
            dataset_raw["models"][model_column] = model_raw

        for metric in metric_order:
            row: dict[str, Any] = {
                "dataset": dataset,
                "metric": metric,
                "better": better[metric],
            }
            for model_name in args.models:
                column = MODEL_COLUMNS[model_name]
                row[column] = values_by_metric[metric].get(column)
            csv_rows.append(row)
        raw["datasets"][dataset] = dataset_raw

    if not _finite(raw):
        raise ValueError("Raw results contain non-finite values")
    inventory = checkpoint_inventory(cache_root)
    raw["checkpoints"] = inventory
    atomic_json_dump(inventory, results_root / "checkpoint_inventory.json")
    atomic_json_dump(raw, results_root / "raw_results.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "dataset",
                "metric",
                "better",
                "sonilo_sfx_v1_0",
                "mirelo_v1_6",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                formatted = dict(row)
                for column in ["sonilo_sfx_v1_0", "mirelo_v1_6"]:
                    value = formatted.get(column)
                    formatted[column] = "" if value is None else f"{float(value):.4f}"
                writer.writerow(formatted)
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    atomic_json_dump(
        {
            "protocol": PROTOCOL,
            "completed_at": utc_now(),
            "csv": str(output_path),
            "raw_results": str(results_root / "raw_results.json"),
            "num_rows": len(csv_rows),
        },
        results_root / "aggregation_complete.json",
    )
    print(f"Wrote final CSV: {output_path}", flush=True)


def comma_separated(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value")
    return values


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    pack_root = repo_root.parent / "benchmark_audio_pack"
    parser = argparse.ArgumentParser(
        description="Full-duration CineBench50/MovieGen V2SFX evaluation"
    )
    parser.add_argument(
        "--mode", choices=["prefetch", "extract", "aggregate"], required=True
    )
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument("--pack-root", default=str(pack_root))
    parser.add_argument(
        "--cache-root",
        default=str(pack_root / ".eval_cache" / PROTOCOL),
    )
    parser.add_argument(
        "--results-root",
        default=str(pack_root / "results_v2sfx_full_duration"),
    )
    parser.add_argument(
        "--output",
        default=str(pack_root / "metrics_v2sfx_full_duration.csv"),
    )
    parser.add_argument(
        "--datasets",
        type=comma_separated,
        default=["moviegen_audio", "cinebench50"],
    )
    parser.add_argument(
        "--models",
        type=comma_separated,
        default=["sonilo", "mirelo"],
    )
    parser.add_argument(
        "--backends",
        type=comma_separated,
        default=[
            "panns",
            "vggish",
            "passt",
            "clap",
            "audiobox",
            "imagebind",
            "desync",
        ],
        help="GPU extraction backends; PaSST runs all three required modules",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use the first N metadata rows per dataset (smoke testing only)",
    )
    parser.add_argument("--refresh", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    unknown_datasets = set(args.datasets) - set(EXPECTED_COUNTS)
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {sorted(unknown_datasets)}")
    unknown_models = set(args.models) - set(MODEL_COLUMNS)
    if unknown_models:
        raise ValueError(f"Unknown models: {sorted(unknown_models)}")
    if args.models != ["sonilo", "mirelo"]:
        raise ValueError("Final CSV protocol requires models ordered as sonilo,mirelo")
    if args.limit is not None and not 1 <= args.limit <= 4:
        raise ValueError("Smoke --limit must be between 1 and 4")
    known_backends = {
        "panns",
        "vggish",
        "passt",
        "clap",
        "audiobox",
        "imagebind",
        "desync",
        *PASST_MODULES.keys(),
    }
    unknown_backends = set(args.backends) - known_backends
    if unknown_backends:
        raise ValueError(f"Unknown backends: {sorted(unknown_backends)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    if args.mode == "prefetch":
        run_checkpoint_prefetch(args)
    elif args.mode == "extract":
        run_distributed_extraction(args)
    else:
        run_aggregation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

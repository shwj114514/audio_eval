#!/usr/bin/env python3
"""Re-evaluate full-duration MovieGen IB-Score with av-benchmark semantics.

This is a full-overlap extension of the official av-benchmark ImageBind path:

* audio: three uniformly distributed two-second clips;
* video: decode at 0.5 FPS, form adjacent-frame clips, and use three crops;
* ImageBind averages each modality's clip/crop embeddings internally;
* one cosine is computed per sample, followed by a mean over samples.

CineBench50 remains N/A because benchmark_audio_pack has no aligned videos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from v2a_tmp.v2sfx_full_duration import (
    MODEL_COLUMNS,
    AudioTask,
    _finite,
    atomic_json_dump,
    atomic_torch_save,
    barrier,
    collection_tasks,
    coordinated_load_model,
    git_state,
    load_records,
    package_versions,
    release_model,
    sha256_file,
    tasks_fingerprint,
    utc_now,
    video_duration,
)


PROTOCOL = "ibscore_avbenchmark_full_overlap_v1"
DATASET = "moviegen_audio"
CINEBENCH = "cinebench50"
EXPECTED_COUNT = 527
METRIC = "IB-Score"
VARIANT = "imagebind_huge_avbenchmark_full_overlap"

OFFICIAL_REPOSITORY = "https://github.com/hkchengrex/av-benchmark.git"
OFFICIAL_COMMIT = "f351b9a6fc6abde746d5f8e1d4c47c883319cb41"
IMAGEBIND_SAMPLE_RATE = 16000
AUDIO_CLIP_SECONDS = 2.0
AUDIO_CLIPS_PER_SAMPLE = 3
VIDEO_FPS = 0.5
VIDEO_FRAMES_PER_CLIP = 2
SPATIAL_CROPS = 3
EMBEDDING_DIMENSION = 1024


def script_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_imagebind_model(device: str, checkpoint: Path) -> torch.nn.Module:
    from imagebind.models import imagebind_model

    model = imagebind_model.imagebind_huge(pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def cache_paths(cache_root: Path, model_name: str) -> tuple[Path, Path]:
    directory = cache_root / DATASET / model_name / "imagebind" / VARIANT
    return directory / "scores.pth", directory / "cache_meta.json"


def expected_cache_metadata(
    tasks: Sequence[AudioTask], model_name: str, world_size: int
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "collection": model_name,
        "backend": "imagebind",
        "model_variant": VARIANT,
        "num_samples": len(tasks),
        "world_size": world_size,
        "input_fingerprint": tasks_fingerprint(tasks, include_video=True),
        "code_fingerprint": script_fingerprint(),
        "official_av_benchmark_commit": OFFICIAL_COMMIT,
    }


def cache_valid(
    feature_path: Path,
    metadata_path: Path,
    expected: Mapping[str, Any],
    ids: Sequence[str],
) -> bool:
    if not feature_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        values = torch.load(feature_path, map_location="cpu", weights_only=True)
        return isinstance(values, Mapping) and list(values) == list(ids) and _finite(values)
    except Exception:
        return False


def load_audio_for_imagebind(path: Path) -> tuple[torch.Tensor, float]:
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.float()
    if sample_rate != IMAGEBIND_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=IMAGEBIND_SAMPLE_RATE,
        )
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError(f"Invalid audio: {path}")
    return waveform.contiguous(), waveform.shape[-1] / IMAGEBIND_SAMPLE_RATE


def audio_clip_timepoints(duration_seconds: float) -> tuple[tuple[float, float], ...]:
    from av_bench.data.audio_dataset import get_clip_timepoints
    from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler

    if duration_seconds < AUDIO_CLIP_SECONDS:
        raise ValueError(
            f"ImageBind audio requires at least {AUDIO_CLIP_SECONDS}s, got "
            f"{duration_seconds:.6f}s"
        )
    sampler = ConstantClipsPerVideoSampler(
        clip_duration=AUDIO_CLIP_SECONDS,
        clips_per_video=AUDIO_CLIPS_PER_SAMPLE,
    )
    points = get_clip_timepoints(sampler, duration_seconds)
    values = tuple((float(start), float(end)) for start, end in points)
    if len(values) != AUDIO_CLIPS_PER_SAMPLE:
        raise ValueError(f"Expected three ImageBind audio clips, got {values}")
    return values


def build_official_preprocessors() -> tuple[
    Callable[[torch.Tensor, float], tuple[torch.Tensor, tuple[tuple[float, float], ...], float]],
    Callable[[Path, float], tuple[torch.Tensor, int, int]],
]:
    import torchvision.transforms.v2 as v2
    from av_bench.data.audio_dataset import waveform2melspec
    from av_bench.data.ib_data import SpatialCrop
    from av_bench.data.video_dataset import VideoDataset

    video_transform = v2.Compose(
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
    spatial_crop = SpatialCrop(224, SPATIAL_CROPS)
    audio_normalize = v2.Normalize(mean=[-4.268], std=[9.138])

    def audio_input(
        waveform: torch.Tensor, overlap_seconds: float
    ) -> tuple[torch.Tensor, tuple[tuple[float, float], ...], float]:
        overlap_samples = min(
            waveform.shape[-1], int(overlap_seconds * IMAGEBIND_SAMPLE_RATE)
        )
        truncated = waveform[:, :overlap_samples]
        effective_duration = truncated.shape[-1] / IMAGEBIND_SAMPLE_RATE
        points = audio_clip_timepoints(effective_duration)
        clips: list[torch.Tensor] = []
        for start, end in points:
            piece = truncated[
                :,
                int(start * IMAGEBIND_SAMPLE_RATE) : int(end * IMAGEBIND_SAMPLE_RATE),
            ].clone()
            mel = waveform2melspec(
                piece,
                sample_rate=IMAGEBIND_SAMPLE_RATE,
                num_mel_bins=128,
                target_length=204,
            )
            clips.append(audio_normalize(mel))
        value = torch.stack(clips, dim=0).unsqueeze(0)
        if value.shape != (1, AUDIO_CLIPS_PER_SAMPLE, 1, 128, 204):
            raise ValueError(f"Unexpected ImageBind audio input shape: {tuple(value.shape)}")
        return value, points, effective_duration

    def video_input(video_path: Path, overlap_seconds: float) -> tuple[torch.Tensor, int, int]:
        expected_frames = int(VIDEO_FPS * overlap_seconds)
        if expected_frames < VIDEO_FRAMES_PER_CLIP:
            raise ValueError(
                f"ImageBind video needs at least two 0.5-FPS frames, got "
                f"duration={overlap_seconds:.6f}s expected_frames={expected_frames}"
            )
        # torio imports in this environment but its FFmpeg extension cannot be
        # initialized. Use av-benchmark's own exact PyAV fallback instead.
        official_dataset = VideoDataset([video_path], duration_sec=overlap_seconds)
        frames = official_dataset._sample_with_pyav(
            video_path, VIDEO_FPS, expected_frames
        )
        if frames is None or frames.shape[0] < expected_frames:
            raise ValueError(
                f"Video too short at {VIDEO_FPS} FPS: {video_path}; "
                f"expected={expected_frames}, got={None if frames is None else frames.shape[0]}"
            )
        frames = frames[:expected_frames]
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"Unexpected decoded frame shape: {tuple(frames.shape)}")
        frames = video_transform(frames)

        # This layout and ordering exactly mirror VideoDataset + extract_video.py:
        # (3 crops, T, C, H, W) -> adjacent pairs -> (1, clips*crops, C, 2, H, W).
        cropped = torch.stack(spatial_crop([frames]))
        adjacent = torch.cat(
            [cropped[:, index : index + 2] for index in range(expected_frames - 1)],
            dim=0,
        )
        value = adjacent.permute(0, 2, 1, 3, 4).unsqueeze(0).contiguous()
        temporal_clips = expected_frames - 1
        expected_shape = (
            1,
            SPATIAL_CROPS * temporal_clips,
            3,
            VIDEO_FRAMES_PER_CLIP,
            224,
            224,
        )
        if value.shape != expected_shape:
            raise ValueError(
                f"Unexpected ImageBind video input shape: {tuple(value.shape)} "
                f"expected={expected_shape}"
            )
        return value, expected_frames, temporal_clips

    return audio_input, video_input


def build_extractor(
    model: torch.nn.Module, device: str
) -> Callable[[AudioTask], dict[str, Any]]:
    from imagebind.models.imagebind_model import ModalityType

    make_audio_input, make_video_input = build_official_preprocessors()

    def extract(task: AudioTask) -> dict[str, Any]:
        if task.video is None:
            raise ValueError("IB-Score requires a source video")
        waveform, audio_seconds = load_audio_for_imagebind(task.path)
        source_video_seconds = video_duration(task.video)
        overlap = min(audio_seconds, source_video_seconds)
        audio_input, audio_points, effective_audio_seconds = make_audio_input(
            waveform, overlap
        )
        video_input, num_video_frames, num_video_temporal_clips = make_video_input(
            task.video, overlap
        )
        with torch.inference_mode():
            output = model(
                {
                    ModalityType.AUDIO: audio_input.to(device),
                    ModalityType.VISION: video_input.to(device),
                }
            )
            audio_embedding = output[ModalityType.AUDIO][0].detach().cpu()
            video_embedding = output[ModalityType.VISION][0].detach().cpu()
        if audio_embedding.shape != (EMBEDDING_DIMENSION,):
            raise ValueError(f"Unexpected audio embedding: {audio_embedding.shape}")
        if video_embedding.shape != (EMBEDDING_DIMENSION,):
            raise ValueError(f"Unexpected video embedding: {video_embedding.shape}")
        score = float(
            F.cosine_similarity(
                audio_embedding.unsqueeze(0), video_embedding.unsqueeze(0), dim=-1
            )[0]
        )
        return {
            "score": score,
            "audio_embedding": audio_embedding,
            "video_embedding": video_embedding,
            "audio_embedding_norm": float(audio_embedding.norm()),
            "video_embedding_norm": float(video_embedding.norm()),
            "audio_clip_timepoints_seconds": torch.tensor(
                audio_points, dtype=torch.float64
            ),
            "num_audio_clips": len(audio_points),
            "num_video_frames": num_video_frames,
            "num_video_temporal_clips": num_video_temporal_clips,
            "num_spatial_crops": SPATIAL_CROPS,
            "num_video_clip_crops": num_video_temporal_clips * SPATIAL_CROPS,
            "audio_duration_seconds": audio_seconds,
            "video_duration_seconds": source_video_seconds,
            "overlap_duration_seconds": overlap,
            "effective_audio_duration_seconds": effective_audio_seconds,
            "video_sampling_fps": VIDEO_FPS,
            "aggregation": (
                "cosine(mean_audio_clip_embeddings, "
                "mean_video_temporal_clip_and_crop_embeddings)"
            ),
            "unit": "cosine_similarity",
        }

    return extract


def extract_collection(
    *,
    cache_root: Path,
    tasks: Sequence[AudioTask],
    model_name: str,
    extractor: Callable[[AudioTask], Any],
    checkpoint_info: Mapping[str, Any] | None,
    rank: int,
    world_size: int,
    local_rank: int,
    repo_root: Path,
    refresh: bool,
) -> None:
    feature_path, metadata_path = cache_paths(cache_root, model_name)
    ids = [task.sample_id for task in tasks]
    expected = expected_cache_metadata(tasks, model_name, world_size)
    skip = torch.zeros(1, dtype=torch.int64, device=f"cuda:{local_rank}")
    if rank == 0 and not refresh:
        skip[0] = int(cache_valid(feature_path, metadata_path, expected, ids))
    dist.broadcast(skip, src=0)
    if bool(skip.item()):
        print(f"[rank {rank}] cache hit {model_name} count={len(tasks)}", flush=True)
        barrier(local_rank)
        return

    rank_tasks = list(tasks[rank::world_size])
    if len(tasks) >= world_size and not rank_tasks:
        raise RuntimeError(f"Rank {rank} received no tasks for {model_name}")
    boundary = (
        f"first={rank_tasks[0].sample_id} last={rank_tasks[-1].sample_id}"
        if rank_tasks
        else "first=N/A last=N/A"
    )
    print(
        f"[rank {rank}] start {model_name} assigned={len(rank_tasks)} {boundary}",
        flush=True,
    )
    shard: OrderedDict[str, Any] = OrderedDict()
    for index, task in enumerate(rank_tasks, 1):
        try:
            value = extractor(task)
        except Exception as error:
            raise RuntimeError(
                f"ImageBind extraction failed for {model_name}/{task.sample_id}: {error}"
            ) from error
        if not _finite(value):
            raise ValueError(f"Non-finite result for {model_name}/{task.sample_id}")
        shard[task.sample_id] = value
        if index == 1 or index == len(rank_tasks) or index % 10 == 0:
            print(
                f"[rank {rank}] progress {model_name} {index}/{len(rank_tasks)} "
                f"id={task.sample_id} score={value['score']:.6f}",
                flush=True,
            )

    shard_dir = feature_path.parent / "shards"
    shard_path = shard_dir / f"scores.rank{rank:03d}.pth"
    shard_meta_path = shard_dir / f"scores.rank{rank:03d}.json"
    atomic_torch_save(shard, shard_path)
    atomic_json_dump(
        {
            **expected,
            "rank": rank,
            "num_rank_samples": len(shard),
            "ids": list(shard),
            "created_at": utc_now(),
        },
        shard_meta_path,
    )
    print(f"[rank {rank}] wrote {shard_path} count={len(shard)}", flush=True)
    barrier(local_rank)

    if rank == 0:
        merged_unordered: dict[str, Any] = {}
        rank_counts: dict[str, int] = {}
        for shard_rank in range(world_size):
            path = shard_dir / f"scores.rank{shard_rank:03d}.pth"
            values = torch.load(path, map_location="cpu", weights_only=True)
            duplicate = set(merged_unordered) & set(values)
            if duplicate:
                raise ValueError(
                    f"Duplicate IDs in rank {shard_rank}: {sorted(duplicate)[:5]}"
                )
            merged_unordered.update(values)
            rank_counts[str(shard_rank)] = len(values)
        missing = set(ids) - set(merged_unordered)
        extra = set(merged_unordered) - set(ids)
        if missing or extra:
            raise ValueError(
                f"Merged ID mismatch: missing={sorted(missing)[:5]} "
                f"extra={sorted(extra)[:5]}"
            )
        merged = OrderedDict((sample_id, merged_unordered[sample_id]) for sample_id in ids)
        if not _finite(merged):
            raise ValueError(f"Merged cache is non-finite: {model_name}")
        atomic_torch_save(merged, feature_path)
        if checkpoint_info is None:
            raise AssertionError("Rank zero is missing checkpoint provenance")
        atomic_json_dump(
            {
                **expected,
                "created_at": utc_now(),
                "ids": ids,
                "rank_counts": rank_counts,
                "feature_path": str(feature_path.resolve()),
                "checkpoint": dict(checkpoint_info),
                "official_source": {
                    "repository": OFFICIAL_REPOSITORY,
                    "commit": OFFICIAL_COMMIT,
                    "video_dataset": "av_bench/data/video_dataset.py",
                    "video_encoder": "extract_video.py::encode_video_with_imagebind",
                    "audio_dataset": "av_bench/data/audio_dataset.py::ImageBindAudioDataset",
                    "metric": "av_bench/evaluate.py::evaluate",
                },
                "git": git_state(repo_root),
                "extraction": {
                    "model": "ImageBind Huge",
                    "timeline": "real audio/video overlap, not fixed at 8 seconds",
                    "audio_sample_rate": IMAGEBIND_SAMPLE_RATE,
                    "audio_clip_seconds": AUDIO_CLIP_SECONDS,
                    "audio_clips_per_sample": AUDIO_CLIPS_PER_SAMPLE,
                    "audio_clip_selection": "ConstantClipsPerVideoSampler",
                    "video_fps": VIDEO_FPS,
                    "video_temporal_clips": "all adjacent decoded frame pairs",
                    "video_frames_per_clip": VIDEO_FRAMES_PER_CLIP,
                    "spatial_crops": SPATIAL_CROPS,
                    "modality_aggregation": "ImageBind internal mean after per-clip postprocessing",
                    "sample_score": "one cosine between aggregated audio/video embeddings",
                    "dataset_aggregation": "unweighted mean over samples",
                },
            },
            metadata_path,
        )
        print(
            f"[rank 0] merged {feature_path} count={len(merged)} "
            f"rank_counts={rank_counts}",
            flush=True,
        )
    barrier(local_rank)


def run_extract(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("IB-Score extraction requires CUDA")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world_size != 8:
        raise RuntimeError(f"This evaluation requires exactly 8 ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    torch.manual_seed(2020 + rank)
    np.random.seed(2020 + rank)

    repo_root = args.repo_root.expanduser().resolve()
    pack_root = args.pack_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cache_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    records = load_records(pack_root, [DATASET], args.models, limit=args.limit)[DATASET]
    if args.limit is None and len(records) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} MovieGen samples, got {len(records)}")

    checkpoint_info = None
    if rank == 0:
        checkpoint_info = {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        }
        atomic_json_dump(
            {
                "protocol": PROTOCOL,
                "created_at": utc_now(),
                "repo_root": str(repo_root),
                "pack_root": str(pack_root),
                "cache_root": str(cache_root),
                "results_root": str(results_root),
                "checkpoint": checkpoint_info,
                "official_repository": OFFICIAL_REPOSITORY,
                "official_commit": OFFICIAL_COMMIT,
                "num_samples": len(records),
                "models": args.models,
                "world_size": world_size,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "rank_devices": {
                    str(index): {
                        "local_rank": index,
                        "cuda_device": f"cuda:{index}",
                        "gpu_name": torch.cuda.get_device_name(index),
                    }
                    for index in range(world_size)
                },
                "python_packages": package_versions(),
                "git": git_state(repo_root),
                "code_fingerprint": script_fingerprint(),
                "command_argv": list(os.sys.argv),
            },
            results_root / "execution_manifest.json",
        )
    barrier(local_rank)

    model = coordinated_load_model(
        lambda: load_imagebind_model(device, checkpoint),
        rank=rank,
        local_rank=local_rank,
        label="ImageBind Huge av-benchmark IB-Score",
    )
    extractor = build_extractor(model, device)
    for model_name in args.models:
        extract_collection(
            cache_root=cache_root,
            tasks=collection_tasks(records, model_name),
            model_name=model_name,
            extractor=extractor,
            checkpoint_info=checkpoint_info,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            repo_root=repo_root,
            refresh=args.refresh,
        )
    release_model(model, local_rank)
    if rank == 0:
        atomic_json_dump(
            {
                "protocol": PROTOCOL,
                "completed_at": utc_now(),
                "num_samples": len(records),
                "models": args.models,
                "world_size": world_size,
            },
            results_root / "distributed_extraction_complete.json",
        )
    barrier(local_rank)
    dist.destroy_process_group()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv_atomic(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_cache(
    cache_root: Path, model_name: str, ids: Sequence[str]
) -> OrderedDict[str, Any]:
    feature_path, metadata_path = cache_paths(cache_root, model_name)
    if not feature_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(feature_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("protocol") != PROTOCOL:
        raise ValueError(f"Wrong cache protocol: {metadata_path}")
    values = torch.load(feature_path, map_location="cpu", weights_only=True)
    if list(values) != list(ids) or not _finite(values):
        raise ValueError(f"Invalid cache: {feature_path}")
    return OrderedDict(values)


def target_rows(metric_values: Mapping[str, float]) -> dict[str, dict[str, str]]:
    return {
        DATASET: {
            "dataset": DATASET,
            "metric": METRIC,
            "better": "higher",
            **{column: f"{value:.4f}" for column, value in metric_values.items()},
        },
        CINEBENCH: {
            "dataset": CINEBENCH,
            "metric": METRIC,
            "better": "higher",
            **{column: "" for column in MODEL_COLUMNS.values()},
        },
    }


def merge_metric_rows(
    rows: Sequence[Mapping[str, str]], metric_values: Mapping[str, float]
) -> list[dict[str, str]]:
    clean_rows = [dict(row) for row in rows if row["metric"] != METRIC]
    additions = target_rows(metric_values)
    output: list[dict[str, str]] = []
    for index, row in enumerate(clean_rows):
        output.append(row)
        dataset = row["dataset"]
        next_dataset = (
            clean_rows[index + 1]["dataset"] if index + 1 < len(clean_rows) else None
        )
        if dataset != next_dataset and dataset in additions:
            output.append(additions[dataset])
    missing = set(additions) - {row["dataset"] for row in clean_rows}
    if missing:
        raise ValueError(f"Base CSV is missing datasets: {sorted(missing)}")
    return output


def run_aggregate(args: argparse.Namespace) -> None:
    pack_root = args.pack_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    base_csv = args.base_csv.expanduser().resolve()
    records = load_records(pack_root, [DATASET], args.models, limit=args.limit)[DATASET]
    ids = [record.sample_id for record in records]

    raw_models: dict[str, Any] = {}
    metric_values: dict[str, float] = {}
    for model_name in args.models:
        values = load_cache(cache_root, model_name, ids)
        scores = np.asarray([float(item["score"]) for item in values.values()])
        metric = float(scores.mean())
        if not math.isfinite(metric):
            raise ValueError(f"Non-finite IB-Score for {model_name}")
        column = MODEL_COLUMNS[model_name]
        metric_values[column] = metric
        raw_models[column] = {
            "value": metric,
            "num_samples": len(values),
            "sample_score_min": float(scores.min()),
            "sample_score_max": float(scores.max()),
            "sample_score_std": float(scores.std()),
            "details": [
                {
                    "id": sample_id,
                    "ib_score": float(item["score"]),
                    "audio_clip_timepoints_seconds": item[
                        "audio_clip_timepoints_seconds"
                    ].tolist(),
                    "num_audio_clips": int(item["num_audio_clips"]),
                    "num_video_frames": int(item["num_video_frames"]),
                    "num_video_temporal_clips": int(
                        item["num_video_temporal_clips"]
                    ),
                    "num_spatial_crops": int(item["num_spatial_crops"]),
                    "num_video_clip_crops": int(item["num_video_clip_crops"]),
                    "audio_embedding_norm": float(item["audio_embedding_norm"]),
                    "video_embedding_norm": float(item["video_embedding_norm"]),
                    "audio_duration_seconds": float(item["audio_duration_seconds"]),
                    "video_duration_seconds": float(item["video_duration_seconds"]),
                    "overlap_duration_seconds": float(item["overlap_duration_seconds"]),
                    "effective_audio_duration_seconds": float(
                        item["effective_audio_duration_seconds"]
                    ),
                }
                for sample_id, item in values.items()
            ],
        }

    fieldnames, rows = read_csv(base_csv)
    expected_fields = [
        "dataset",
        "metric",
        "better",
        "sonilo_sfx_v1_0",
        "mirelo_v1_6",
    ]
    if fieldnames != expected_fields:
        raise ValueError(f"Unexpected base CSV fields: {fieldnames}")
    output_rows = merge_metric_rows(rows, metric_values)
    write_csv_atomic(output, fieldnames, output_rows)
    write_csv_atomic(
        results_root / "ibscore_only.csv",
        fieldnames,
        list(target_rows(metric_values).values()),
    )
    atomic_json_dump(
        {
            "protocol": PROTOCOL,
            "created_at": utc_now(),
            "official_source": {
                "repository": OFFICIAL_REPOSITORY,
                "commit": OFFICIAL_COMMIT,
            },
            "base_csv": str(base_csv),
            "base_csv_sha256": sha256_file(base_csv),
            "output_csv": str(output),
            "metric": METRIC,
            "definition": {
                "timeline": "complete real A/V overlap",
                "audio": "three uniformly distributed 2-second clips",
                "video": "0.5 FPS adjacent-frame clips, each with three spatial crops",
                "modality_aggregation": "mean embeddings within ImageBind",
                "sample_aggregation": "one cosine per sample",
                "dataset_aggregation": "unweighted mean over 527 samples",
            },
            "datasets": {
                DATASET: {"models": raw_models},
                CINEBENCH: {
                    "models": {
                        column: None for column in MODEL_COLUMNS.values()
                    },
                    "reason": (
                        "benchmark_audio_pack/cinebench50 has no local source video; "
                        "its YouTube URLs have no aligned clip timestamps"
                    ),
                },
            },
        },
        results_root / "raw_results.json",
    )
    atomic_json_dump(
        {
            "protocol": PROTOCOL,
            "completed_at": utc_now(),
            "csv": str(output),
            "metric": METRIC,
            "moviegen_values": metric_values,
            "cinebench": "N/A: source video unavailable",
        },
        results_root / "aggregation_complete.json",
    )
    print(f"Wrote {output}", flush=True)
    print(json.dumps(metric_values, indent=2), flush=True)


def run_parity_test(args: argparse.Namespace) -> None:
    """Demand exact input-tensor parity with official av-benchmark datasets."""
    from av_bench.data.audio_dataset import ImageBindAudioDataset
    from av_bench.data.video_dataset import VideoDataset

    pack_root = args.pack_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    record = load_records(pack_root, [DATASET], ["sonilo"], limit=1)[DATASET][0]
    task = collection_tasks([record], "sonilo")[0]
    if task.video is None:
        raise AssertionError("Parity test requires MovieGen video")
    waveform, audio_seconds = load_audio_for_imagebind(task.path)
    overlap = min(audio_seconds, video_duration(task.video))
    make_audio_input, make_video_input = build_official_preprocessors()
    ours_audio, points, _ = make_audio_input(waveform, overlap)
    ours_video, frames, temporal_clips = make_video_input(task.video, overlap)

    official_audio, official_name = ImageBindAudioDataset([task.path])[0]
    if official_name != task.sample_id:
        raise AssertionError("Official audio dataset returned the wrong ID")
    if not torch.equal(ours_audio, official_audio):
        difference = float((ours_audio - official_audio).abs().max())
        raise AssertionError(f"Official audio input parity failed: max_diff={difference}")

    official_video_dataset = VideoDataset([task.video], duration_sec=overlap)
    official_frames = official_video_dataset._sample_with_pyav(
        task.video, VIDEO_FPS, int(VIDEO_FPS * overlap)
    )
    official_frames = official_video_dataset.ib_transform(official_frames)
    official_ib_video = torch.stack(
        official_video_dataset.crop([official_frames])
    ).unsqueeze(0)
    official_video_clips = torch.cat(
        [
            official_ib_video[:, :, index : index + 2]
            for index in range(official_ib_video.shape[2] - 1)
        ],
        dim=1,
    ).permute(0, 1, 3, 2, 4, 5).contiguous()
    if not torch.equal(ours_video, official_video_clips):
        difference = float((ours_video - official_video_clips).abs().max())
        raise AssertionError(f"Official video input parity failed: max_diff={difference}")

    report = {
        "protocol": PROTOCOL,
        "status": "PASS",
        "sample_id": task.sample_id,
        "official_commit": OFFICIAL_COMMIT,
        "audio_tensor_exact_equal": True,
        "video_tensor_exact_equal": True,
        "audio_shape": list(ours_audio.shape),
        "audio_clip_timepoints_seconds": [list(value) for value in points],
        "video_shape": list(ours_video.shape),
        "num_video_frames": frames,
        "num_video_temporal_clips": temporal_clips,
        "num_video_clip_crops": temporal_clips * SPATIAL_CROPS,
    }
    atomic_json_dump(report, results_root / "official_input_parity.json")
    print(json.dumps(report, indent=2), flush=True)


def run_audit(args: argparse.Namespace) -> None:
    pack_root = args.pack_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    base_csv = args.base_csv.expanduser().resolve()
    parity = json.loads(
        (results_root / "official_input_parity.json").read_text(encoding="utf-8")
    )
    if parity.get("status") != "PASS":
        raise AssertionError("Official input parity is not PASS")
    records = load_records(pack_root, [DATASET], args.models, limit=args.limit)[DATASET]
    ids = [record.sample_id for record in records]
    expected_rank_counts = {str(rank): len(ids[rank::8]) for rank in range(8)}
    raw = json.loads((results_root / "raw_results.json").read_text(encoding="utf-8"))
    audit_models: dict[str, Any] = {}

    for model_name in args.models:
        values = load_cache(cache_root, model_name, ids)
        _, metadata_path = cache_paths(cache_root, model_name)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["rank_counts"] != expected_rank_counts:
            raise AssertionError(f"Rank coverage mismatch for {model_name}")
        if metadata["official_av_benchmark_commit"] != OFFICIAL_COMMIT:
            raise AssertionError(f"Official commit mismatch for {model_name}")
        scores: list[float] = []
        for sample_id, item in values.items():
            audio_embedding = item["audio_embedding"].to(torch.float32)
            video_embedding = item["video_embedding"].to(torch.float32)
            if audio_embedding.shape != (EMBEDDING_DIMENSION,):
                raise AssertionError(f"Wrong audio embedding shape: {sample_id}")
            if video_embedding.shape != (EMBEDDING_DIMENSION,):
                raise AssertionError(f"Wrong video embedding shape: {sample_id}")
            expected_score = float(
                F.cosine_similarity(
                    audio_embedding.unsqueeze(0),
                    video_embedding.unsqueeze(0),
                    dim=-1,
                )[0]
            )
            if not math.isclose(float(item["score"]), expected_score, abs_tol=1e-7):
                raise AssertionError(f"Embedding/cosine mismatch: {sample_id}")
            if int(item["num_audio_clips"]) != AUDIO_CLIPS_PER_SAMPLE:
                raise AssertionError(f"Wrong audio clip count: {sample_id}")
            overlap = float(item["overlap_duration_seconds"])
            expected_frames = int(VIDEO_FPS * overlap)
            if int(item["num_video_frames"]) != expected_frames:
                raise AssertionError(f"Wrong video frame count: {sample_id}")
            if int(item["num_video_temporal_clips"]) != expected_frames - 1:
                raise AssertionError(f"Wrong video temporal clip count: {sample_id}")
            if int(item["num_spatial_crops"]) != SPATIAL_CROPS:
                raise AssertionError(f"Wrong spatial crop count: {sample_id}")
            if int(item["num_video_clip_crops"]) != (expected_frames - 1) * SPATIAL_CROPS:
                raise AssertionError(f"Wrong video clip/crop count: {sample_id}")
            effective_audio_seconds = float(item["effective_audio_duration_seconds"])
            expected_points = torch.tensor(
                audio_clip_timepoints(effective_audio_seconds), dtype=torch.float64
            )
            if not torch.allclose(
                item["audio_clip_timepoints_seconds"],
                expected_points,
                atol=1e-9,
                rtol=0,
            ):
                raise AssertionError(f"Wrong audio clip positions: {sample_id}")
            scores.append(float(item["score"]))

        column = MODEL_COLUMNS[model_name]
        raw_model = raw["datasets"][DATASET]["models"][column]
        expected_value = float(np.mean(scores))
        if not math.isclose(float(raw_model["value"]), expected_value, abs_tol=1e-12):
            raise AssertionError(f"Dataset mean mismatch for {model_name}")
        audit_models[column] = {
            "value": expected_value,
            "num_samples": len(scores),
            "audio_clips_per_sample": AUDIO_CLIPS_PER_SAMPLE,
            "video_frames_per_sample": sorted(
                {int(item["num_video_frames"]) for item in values.values()}
            ),
            "video_temporal_clips_per_sample": sorted(
                {int(item["num_video_temporal_clips"]) for item in values.values()}
            ),
            "video_clip_crops_per_sample": sorted(
                {int(item["num_video_clip_crops"]) for item in values.values()}
            ),
        }

    output_fields, output_rows = read_csv(output)
    base_fields, base_rows = read_csv(base_csv)
    if output_fields != base_fields:
        raise AssertionError("Output/base field mismatch")
    base_without = [row for row in base_rows if row["metric"] != METRIC]
    output_without = [row for row in output_rows if row["metric"] != METRIC]
    if output_without != base_without:
        raise AssertionError("A non-IB-Score CSV row changed")
    selected_rows = [row for row in output_rows if row["metric"] == METRIC]
    if len(selected_rows) != 2 or {row["dataset"] for row in selected_rows} != {
        DATASET,
        CINEBENCH,
    }:
        raise AssertionError("Expected one IB-Score row for each dataset")
    by_dataset = {row["dataset"]: row for row in selected_rows}
    for model_name in args.models:
        column = MODEL_COLUMNS[model_name]
        if by_dataset[DATASET][column] != f"{audit_models[column]['value']:.4f}":
            raise AssertionError(f"CSV/raw mismatch for {column}")
        if by_dataset[CINEBENCH][column] != "":
            raise AssertionError(f"CineBench IB-Score must be N/A for {column}")

    manifest = json.loads(
        (results_root / "execution_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["world_size"] != 8 or len(manifest["rank_devices"]) != 8:
        raise AssertionError("Execution manifest does not prove 8-GPU execution")
    if manifest["official_commit"] != OFFICIAL_COMMIT:
        raise AssertionError("Manifest official commit mismatch")
    report = {
        "protocol": PROTOCOL,
        "status": "PASS",
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "official_input_parity": parity,
        "num_moviegen_samples": len(ids),
        "world_size": 8,
        "rank_counts": expected_rank_counts,
        "output_csv": str(output),
        "models": audit_models,
        "cinebench": {
            "value": None,
            "reason": "source video and aligned clip timestamps unavailable",
        },
        "checks": [
            "exact audio input parity with official ImageBindAudioDataset",
            "exact video input parity with official VideoDataset + extract_video.py",
            "complete real A/V overlap rather than fixed 8 seconds",
            "three official uniformly distributed 2-second audio clips",
            "official 0.5-FPS adjacent-frame video clips and three spatial crops",
            "one cosine after per-modality embedding aggregation",
            "unweighted mean over samples",
            "exact 8-rank sample coverage",
            "only the two requested CSV rows were added/replaced",
        ],
    }
    atomic_json_dump(report, results_root / "audit.json")
    print(json.dumps(report, indent=2), flush=True)


def comma_separated(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected comma-separated values")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["parity-test", "extract", "aggregate", "audit"], required=True
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--pack-root",
        type=Path,
        default=Path("/m2v_intern/leijiahe/codex_workplace/benchmark_audio_pack"),
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path("/tmp/v2a_ibscore_avbenchmark/cache")
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results_v2a_ibscore_avbenchmark_full_duration"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/root/.cache/torch/hub/checkpoints/imagebind_huge.pth"),
    )
    parser.add_argument(
        "--base-csv", type=Path, default=Path("metrics_v2a_benchmark_10s_clip.csv")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/tmp/v2a_ibscore_avbenchmark/metrics_v2a_benchmark_with_ibscore.csv"
        ),
    )
    parser.add_argument("--models", type=comma_separated, default=["sonilo", "mirelo"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--refresh", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if any(model not in MODEL_COLUMNS for model in args.models):
        raise ValueError(f"Unknown models: {args.models}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.mode in {"aggregate", "audit"} and not args.base_csv.is_file():
        raise FileNotFoundError(args.base_csv)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.mode == "parity-test":
        run_parity_test(args)
    elif args.mode == "extract":
        run_extract(args)
    elif args.mode == "aggregate":
        run_aggregate(args)
    else:
        run_audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

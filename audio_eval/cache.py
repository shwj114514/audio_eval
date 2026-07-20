"""Cache paths and reusable audio/video feature caches."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import typing as tp

import numpy as np
import torch

from audio_eval.audio import collection_fingerprint, materialize_audio_collection
from audio_eval.common import FeatureInput, MetricInput


_SYNCHFORMER_URL = (
    "https://github.com/hkchengrex/MMAudio/releases/download/v0.1/"
    "synchformer_state_dict.pth"
)
_FULL_LENGTH_CACHE_MARKER = ".full_length_v1"
IMAGEBIND_WINDOW_CACHE_MARKER = ".imagebind_window_embeddings_v1"
IMAGEBIND_OVERLAP_CACHE_MARKER = ".imagebind_overlap_window_embeddings_v1"


def default_cache_dir() -> Path:
    return Path(os.environ.get("AUDIO_EVAL_CACHE", "~/.cache/audio_eval")).expanduser()


def cache_file(
    namespace: str,
    fingerprint: str,
    *,
    backend: str,
    suffix: str = ".npz",
    cache_dir: str | Path | None = None,
) -> Path:
    root = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
    directory = root / namespace / backend
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{fingerprint}{suffix}"


def feature_cache_file(
    filename: str,
    fingerprint: str,
    *,
    backend: str,
    cache_dir: str | Path | None,
    output_dir: str | Path | None,
) -> Path:
    """Return a direct explicit cache file or a fingerprinted automatic cache file."""
    if output_dir is not None:
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        return directory / filename
    return cache_file("features", fingerprint, backend=backend, cache_dir=cache_dir)


def resolve_feature_cache(source: MetricInput, filename: str) -> Path | None:
    """Resolve a feature cache passed either as its file or containing directory."""
    if not isinstance(source, (str, Path)):
        return None
    path = Path(source).expanduser()
    if path.is_dir() and (path / filename).is_file():
        return path / filename
    if path.is_file() and path.suffix.lower() == ".npz":
        return path
    return None


def clean_sample_name(sample_name: str) -> str:
    """Match MMAudio's generated filename variants to the VGGSound cache key."""
    targets = (
        ("000000014_zxpo56cpUBU_000007-0", "zxpo56cpUBU_000007"),
        ("zxpo56cpUBU_000007-0", "zxpo56cpUBU_000007"),
        ("Y---g-f_I2yQ_000001_0", "---g-f_I2yQ_000001"),
    )
    for example, example_target in targets:
        if sample_name == example:
            return example_target
    return sample_name


def _tensor(value: torch.Tensor | np.ndarray) -> torch.Tensor:
    return value.detach().cpu() if isinstance(value, torch.Tensor) else torch.from_numpy(value)


def load_feature_map(
    source: FeatureInput,
    *,
    filename: str,
    layer: str | None = None,
) -> dict[str, torch.Tensor]:
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if path.is_dir():
            path = path / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        data = torch.load(path, map_location="cpu", weights_only=True)
    elif isinstance(source, torch.Tensor):
        data = source
    elif isinstance(source, np.ndarray):
        data = source
    else:
        data = source

    if isinstance(data, (torch.Tensor, np.ndarray)):
        tensor = _tensor(data)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return {str(index): row for index, row in enumerate(tensor)}
    if not isinstance(data, tp.Mapping):
        raise TypeError(f"Unsupported feature cache type: {type(data).__name__}")

    output: dict[str, torch.Tensor] = {}
    for key, value in data.items():
        if layer is not None:
            if not isinstance(value, tp.Mapping) or layer not in value:
                raise ValueError(f"Feature {key!r} does not contain layer {layer!r}")
            value = value[layer]
        if isinstance(value, tp.Mapping):
            raise ValueError(f"Feature {key!r} requires a layer selection")
        output[str(key)] = _tensor(value)
    if not output:
        raise ValueError(f"No features found in {source}")
    return output


def is_feature_map(source: MetricInput, *, filename: str, layer: str | None) -> bool:
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        return (path.is_dir() and (path / filename).is_file()) or (
            path.is_file() and path.suffix == ".pth"
        )
    if isinstance(source, tp.Mapping) and source:
        value = next(iter(source.values()))
        if layer is None:
            return isinstance(value, (torch.Tensor, np.ndarray))
        return isinstance(value, tp.Mapping) and layer in value
    return False


def _extract_av_audio_features(
    audio_path: Path,
    output: Path,
    *,
    batch_size: int,
    device: str,
    duration_by_key: tp.Mapping[str, float] | None = None,
) -> None:
    try:
        import torchaudio
        from av_bench.data.audio_dataset import (
            ImageBindAudioDataset,
            get_clip_timepoints,
            waveform2melspec,
        )
        from av_bench.data.audio_dataset import SynchformerAudioDataset, pad_or_truncate
        from av_bench.synchformer.synchformer import Synchformer
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
        from torch.utils.data import ConcatDataset, DataLoader
        import torchvision.transforms.v2 as v2
    except ImportError as error:
        raise ImportError(
            "ImageBind/DeSync extraction requires `pip install audio-eval[video]`"
        ) from error

    audios = sorted(
        list(audio_path.glob("*.wav")) + list(audio_path.glob("*.flac")),
        key=lambda path: path.stem,
    )
    if not audios:
        raise FileNotFoundError(f"No WAV or FLAC files found in {audio_path}")
    output.mkdir(parents=True, exist_ok=True)
    audio_durations = {
        path: torchaudio.info(path).num_frames / torchaudio.info(path).sample_rate
        for path in audios
    }
    if duration_by_key is not None:
        missing_durations = sorted(
            path.stem for path in audios if path.stem not in duration_by_key
        )
        if missing_durations:
            raise ValueError(f"Missing A/V overlap for {missing_durations[:5]}")
        audio_durations = {
            path: min(duration, float(duration_by_key[path.stem]))
            for path, duration in audio_durations.items()
        }

    imagebind = imagebind_model.imagebind_huge(pretrained=True).to(device).eval()
    imagebind_features: tp.Dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        if duration_by_key is None:
            imagebind_loader = DataLoader(
                ImageBindAudioDataset(audios),
                batch_size=batch_size,
                num_workers=0,
                pin_memory=True,
            )
            for wav, names in imagebind_loader:
                audio_windows = wav.squeeze(1).to(device)
                batch_size_value, num_windows = audio_windows.shape[:2]
                flat_windows = audio_windows.reshape(
                    batch_size_value * num_windows,
                    *audio_windows.shape[2:],
                )
                features = imagebind(
                    {ModalityType.AUDIO: flat_windows}
                )[ModalityType.AUDIO].reshape(
                    batch_size_value,
                    num_windows,
                    -1,
                ).cpu()
                for index, name in enumerate(names):
                    imagebind_features[str(name)] = features[index]
        else:
            normalize = v2.Normalize(mean=[-4.268], std=[9.138])
            for batch_start in range(0, len(audios), batch_size):
                batch_paths = audios[batch_start : batch_start + batch_size]
                batch_windows = []
                for path in batch_paths:
                    waveform, sample_rate = torchaudio.load(str(path))
                    waveform = waveform.float()
                    if sample_rate != 16000:
                        waveform = torchaudio.functional.resample(
                            waveform,
                            orig_freq=sample_rate,
                            new_freq=16000,
                        )
                    overlap_samples = min(
                        waveform.shape[-1],
                        int(float(duration_by_key[path.stem]) * 16000),
                    )
                    waveform = waveform[:, :overlap_samples]
                    duration = waveform.shape[-1] / 16000
                    sampler = ConstantClipsPerVideoSampler(
                        clip_duration=2.0,
                        clips_per_video=3,
                    )
                    points = get_clip_timepoints(sampler, duration)
                    clips = []
                    for start, end in points:
                        piece = waveform[
                            :,
                            int(start * 16000) : int(end * 16000),
                        ].clone()
                        mel = waveform2melspec(
                            piece,
                            sample_rate=16000,
                            num_mel_bins=128,
                            target_length=204,
                        )
                        clips.append(normalize(mel))
                    batch_windows.append(torch.stack(clips, dim=0))
                audio_windows = torch.stack(batch_windows, dim=0).to(device)
                batch_size_value, num_windows = audio_windows.shape[:2]
                flat_windows = audio_windows.reshape(
                    batch_size_value * num_windows,
                    *audio_windows.shape[2:],
                )
                features = imagebind(
                    {ModalityType.AUDIO: flat_windows}
                )[ModalityType.AUDIO].reshape(
                    batch_size_value,
                    num_windows,
                    -1,
                ).cpu()
                for index, path in enumerate(batch_paths):
                    imagebind_features[path.stem] = features[index]
    torch.save(imagebind_features, output / "imagebind_audio.pth")
    del imagebind, imagebind_features

    checkpoint = os.environ.get("AUDIO_EVAL_SYNCHFORMER_CHECKPOINT")
    checkpoint_path = (
        Path(checkpoint).expanduser()
        if checkpoint is not None
        else default_cache_dir() / "synchformer_state_dict.pth"
    )
    if not checkpoint_path.is_file():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(_SYNCHFORMER_URL, str(checkpoint_path))
    synchformer = Synchformer().to(device).eval()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    synchformer.load_state_dict(state)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        win_length=400,
        hop_length=160,
        n_fft=1024,
        n_mels=128,
    ).to(device)
    synchformer_loader = DataLoader(
        ConcatDataset([
            SynchformerAudioDataset([path], duration=audio_durations[path])
            for path in audios
        ]),
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    )
    synchformer_features: tp.Dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for wav, names in synchformer_loader:
            wav = wav.to(device)
            starts = range(0, wav.shape[-1] - 10240 + 1, 5120)
            segments = torch.stack(
                [wav[:, start:start + 10240] for start in starts],
                dim=1,
            )
            features = mel(segments)
            features = torch.log(features + 1e-6)
            features = pad_or_truncate(features, 66)
            features = (features + 4.2677393) / (2 * 4.5689974)
            features = synchformer.extract_afeats(features.unsqueeze(2)).cpu()
            for index, name in enumerate(names):
                synchformer_features[str(name)] = features[index]
    torch.save(synchformer_features, output / "synchformer_audio.pth")
    del synchformer, synchformer_features

    (output / _FULL_LENGTH_CACHE_MARKER).touch()
    marker = (
        IMAGEBIND_OVERLAP_CACHE_MARKER
        if duration_by_key is not None
        else IMAGEBIND_WINDOW_CACHE_MARKER
    )
    (output / marker).touch()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_audio_feature_cache(
    source: MetricInput,
    *,
    sample_rate: int | None = None,
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    batch_size: int = 64,
    device: str | None = None,
    feature_versions: tp.Sequence[str] = (),
    include_video_metrics: bool = False,
    duration_by_key: tp.Mapping[str, float] | None = None,
    refresh_cache: bool = False,
) -> Path:
    if isinstance(feature_versions, str):
        raise TypeError("feature_versions must be a list or tuple, not a string")
    unknown = sorted(set(feature_versions) - {"panns", "passt", "vggish"})
    if unknown:
        raise ValueError(f"Unknown audio feature versions: {unknown}")
    filenames = {
        "panns": "panns.npz",
        "passt": "passt.npz",
        "vggish": "vggish.npz",
    }
    required = [filenames[version] for version in feature_versions]
    if include_video_metrics:
        marker = (
            IMAGEBIND_OVERLAP_CACHE_MARKER
            if duration_by_key is not None
            else IMAGEBIND_WINDOW_CACHE_MARKER
        )
        required.extend(
            [
                "imagebind_audio.pth",
                "synchformer_audio.pth",
                _FULL_LENGTH_CACHE_MARKER,
                marker,
            ]
        )
    if output_dir is not None:
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        if not refresh_cache and all((output / name).is_file() for name in required):
            return output
    if isinstance(source, (str, Path)):
        source_path = Path(source).expanduser()
        if output_dir is None and source_path.is_dir() and all(
            (source_path / name).is_file() for name in required
        ):
            return source_path

    if output_dir is None:
        fingerprint = collection_fingerprint(source, sample_rate=sample_rate)
        output = cache_file(
            "features",
            fingerprint,
            backend="feature_bundle_v2",
            suffix="",
            cache_dir=cache_dir,
        )
    if not refresh_cache and all((output / name).is_file() for name in required):
        return output
    for version in feature_versions:
        if version == "panns":
            from audio_eval.features.panns import get_panns_features
            get_panns_features(
                source,
                sample_rate=sample_rate,
                output_dir=output,
                batch_size=batch_size,
                device=device,
                refresh_cache=refresh_cache,
            )
        elif version == "passt":
            from audio_eval.features.passt import get_passt_features
            get_passt_features(
                source,
                sample_rate=sample_rate,
                output_dir=output,
                batch_size=batch_size,
                device=device,
                refresh_cache=refresh_cache,
            )
        else:
            from audio_eval.features.vggish import get_vggish_features
            get_vggish_features(
                source,
                sample_rate=sample_rate,
                output_dir=output,
                device=device,
                refresh_cache=refresh_cache,
            )
    if include_video_metrics:
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(source, (str, Path)) and Path(source).expanduser().is_dir():
            _extract_av_audio_features(
                Path(source).expanduser(),
                output,
                batch_size=batch_size,
                device=target_device,
                duration_by_key=duration_by_key,
            )
        else:
            with materialize_audio_collection(source, sample_rate=sample_rate) as audio_path:
                _extract_av_audio_features(
                    audio_path,
                    output,
                    batch_size=batch_size,
                    device=target_device,
                    duration_by_key=duration_by_key,
                )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Audio feature extraction did not create: {missing}")
    return output


def ensure_video_feature_cache(
    source: tp.Mapping[str, tp.Union[str, Path]],
    *,
    output_dir: tp.Union[str, Path],
    batch_size: int = 2,
    device: tp.Optional[str] = None,
    duration_by_key: tp.Mapping[str, float] | None = None,
    refresh_cache: bool = False,
) -> Path:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    required = [
        "imagebind_video.pth",
        "synchformer_video.pth",
        _FULL_LENGTH_CACHE_MARKER,
        (
            IMAGEBIND_OVERLAP_CACHE_MARKER
            if duration_by_key is not None
            else IMAGEBIND_WINDOW_CACHE_MARKER
        ),
    ]
    if not refresh_cache and all((output / name).is_file() for name in required):
        return output
    if not source:
        raise ValueError("Video feature extraction requires at least one ref_path")

    try:
        import av
        from av_bench.data.video_dataset import VideoDataset, error_avoidance_collate
        from av_bench.synchformer.synchformer import Synchformer
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        from torch.utils.data import ConcatDataset, DataLoader
    except ImportError as error:
        raise ImportError(
            "ImageBind/DeSync extraction requires `pip install audio-eval[video]`"
        ) from error

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = os.environ.get("AUDIO_EVAL_SYNCHFORMER_CHECKPOINT")
    if checkpoint is None:
        checkpoint_path = default_cache_dir() / "synchformer_state_dict.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint_path.is_file():
            torch.hub.download_url_to_file(_SYNCHFORMER_URL, str(checkpoint_path))
    else:
        checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    with tempfile.TemporaryDirectory(prefix="audio_eval_video_") as temp_dir:
        video_datasets: tp.List[VideoDataset] = []
        for key, value in sorted(source.items()):
            source_path = Path(value).expanduser().resolve()
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            with av.open(str(source_path)) as container:
                video_stream = container.streams.video[0]
                if video_stream.duration is not None:
                    duration = float(video_stream.duration * video_stream.time_base)
                elif container.duration is not None:
                    duration = float(container.duration / av.time_base)
                elif video_stream.frames and video_stream.average_rate:
                    duration = float(video_stream.frames / video_stream.average_rate)
                else:
                    raise ValueError(f"Cannot determine video duration for {source_path}")
            if duration <= 0:
                raise ValueError(f"Invalid video duration for {source_path}: {duration}")
            if duration_by_key is not None:
                if key not in duration_by_key:
                    raise ValueError(f"Missing A/V overlap duration for {key!r}")
                duration = min(duration, float(duration_by_key[key]))
            staged_path = Path(temp_dir) / f"{key}{source_path.suffix.lower()}"
            staged_path.symlink_to(source_path)
            video_datasets.append(VideoDataset([staged_path], duration_sec=duration))

        dataset = ConcatDataset(video_datasets)
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            collate_fn=error_avoidance_collate,
        )
        imagebind = imagebind_model.imagebind_huge(pretrained=True).to(target_device).eval()
        synchformer = Synchformer().to(target_device).eval()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        synchformer.load_state_dict(state)

        imagebind_features: tp.Dict[str, torch.Tensor] = {}
        synchformer_features: tp.Dict[str, torch.Tensor] = {}
        with torch.inference_mode():
            for batch in loader:
                names = batch["name"]
                imagebind_video = batch["ib_video"].to(target_device)
                imagebind_clips = torch.cat(
                    [
                        imagebind_video[:, :, start:start + 2]
                        for start in range(imagebind_video.shape[2] - 1)
                    ],
                    dim=1,
                ).permute(0, 1, 3, 2, 4, 5)
                batch_size_value, num_imagebind_windows = imagebind_clips.shape[:2]
                flat_imagebind_clips = imagebind_clips.reshape(
                    batch_size_value * num_imagebind_windows,
                    1,
                    *imagebind_clips.shape[2:],
                )
                imagebind_batch = imagebind(
                    {ModalityType.VISION: flat_imagebind_clips}
                )[ModalityType.VISION].reshape(
                    batch_size_value,
                    num_imagebind_windows,
                    -1,
                ).cpu()

                sync_video = batch["sync_video"].to(target_device)
                starts = range(0, sync_video.shape[1] - 16 + 1, 8)
                segments = torch.stack(
                    [sync_video[:, start:start + 16] for start in starts],
                    dim=1,
                )
                batch_size_value, num_segments = segments.shape[:2]
                segments = segments.reshape(
                    batch_size_value * num_segments,
                    1,
                    *segments.shape[2:],
                )
                synchformer_batch = synchformer.extract_vfeats(segments).reshape(
                    batch_size_value,
                    num_segments,
                    8,
                    768,
                ).cpu()
                for index, name in enumerate(names):
                    imagebind_features[str(name)] = imagebind_batch[index]
                    synchformer_features[str(name)] = synchformer_batch[index]

    if len(imagebind_features) != len(source) or len(synchformer_features) != len(source):
        raise RuntimeError(
            "Video feature extraction skipped one or more files; inspect decoder errors above"
        )
    torch.save(imagebind_features, output / "imagebind_video.pth")
    torch.save(synchformer_features, output / "synchformer_video.pth")
    (output / _FULL_LENGTH_CACHE_MARKER).touch()
    marker = (
        IMAGEBIND_OVERLAP_CACHE_MARKER
        if duration_by_key is not None
        else IMAGEBIND_WINDOW_CACHE_MARKER
    )
    (output / marker).touch()
    return output


def pair_feature_maps(
    reference: tp.Mapping[str, torch.Tensor],
    generated: tp.Mapping[str, torch.Tensor],
) -> tuple[list[str], torch.Tensor, torch.Tensor, list[str]]:
    generated_to_reference = {key: clean_sample_name(key) for key in generated}
    unpaired = set(reference) ^ set(generated_to_reference.values())
    keys: list[str] = []
    reference_values: list[torch.Tensor] = []
    generated_values: list[torch.Tensor] = []
    for generated_key, reference_key in generated_to_reference.items():
        if reference_key in unpaired:
            continue
        keys.append(reference_key)
        reference_values.append(reference[reference_key])
        generated_values.append(generated[generated_key])
    if not keys:
        raise ValueError("No matched generated/reference feature keys")
    return (
        keys,
        torch.stack(reference_values, dim=0),
        torch.stack(generated_values, dim=0),
        sorted(unpaired),
    )

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


def _extract_audio_features(
    audio_path: Path,
    output: Path,
    *,
    batch_size: int,
    device: str,
    include_video_metrics: bool,
) -> None:
    try:
        import torchaudio
        from av_bench.data.audio_dataset import AudioDataset, ImageBindAudioDataset
        from av_bench.data.audio_dataset import SynchformerAudioDataset, pad_or_truncate
        from av_bench.panns import Cnn14
        from av_bench.synchformer.synchformer import Synchformer
        from av_bench.vggish.vggish import VGGish
        from hear21passt.base import get_basic_model
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise ImportError(
            "PaSST/VGGish/ImageBind/DeSync extraction requires `pip install audio-eval[video]`"
        ) from error

    audios = sorted(
        list(audio_path.glob("*.wav")) + list(audio_path.glob("*.flac")),
        key=lambda path: path.stem,
    )
    if not audios:
        raise FileNotFoundError(f"No WAV or FLAC files found in {audio_path}")
    output.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        AudioDataset(audios, audio_length=8.0, sr=16000),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
    )
    unused_32k_checkpoint = Path.home() / ".cache/audioldm_eval/ckpt/Cnn14_mAP=0.431.pth"
    created_placeholder = not unused_32k_checkpoint.exists()
    if created_placeholder:
        unused_32k_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        unused_32k_checkpoint.touch()
    try:
        panns = Cnn14(
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
            unused_32k_checkpoint.unlink()
    pann_features: tp.Dict[str, tp.Dict[str, torch.Tensor]] = {}
    with torch.inference_mode():
        for wav, names in loader:
            features = {key: value.cpu() for key, value in panns(wav.squeeze(1).float().to(device)).items()}
            for index, name in enumerate(names):
                pann_features[str(name)] = {key: value[index] for key, value in features.items()}
    torch.save(pann_features, output / "pann_features.pth")
    del panns, pann_features

    vggish = VGGish(postprocess=False).eval()
    vggish_features: tp.Dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for wav, names in loader:
            features = vggish(wav.squeeze(1).float()).cpu()
            for index, name in enumerate(names):
                vggish_features[str(name)] = features[index]
    torch.save(vggish_features, output / "vggish_features.pth")
    del vggish, vggish_features

    passt = get_basic_model(mode="all").to(device).eval()
    passt_loader = DataLoader(
        AudioDataset(audios, audio_length=8.0, sr=32000),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
    )
    passt_features: tp.Dict[str, torch.Tensor] = {}
    passt_logits: tp.Dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for wav, names in passt_loader:
            wav = wav.squeeze(1).float().to(device)
            wav = wav[..., :320000]
            if wav.shape[-1] < 320000:
                wav = torch.nn.functional.pad(wav, (0, 320000 - wav.shape[-1]))
            features = passt(wav).cpu()
            for index, name in enumerate(names):
                passt_logits[str(name)] = features[index, :527]
                passt_features[str(name)] = features[index, 527:]
    torch.save(passt_features, output / "passt_features_embed.pth")
    torch.save(passt_logits, output / "passt_logits.pth")
    del passt, passt_features, passt_logits

    if include_video_metrics:
        imagebind = imagebind_model.imagebind_huge(pretrained=True).to(device).eval()
        imagebind_loader = DataLoader(
            ImageBindAudioDataset(audios),
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
        )
        imagebind_features: tp.Dict[str, torch.Tensor] = {}
        with torch.inference_mode():
            for wav, names in imagebind_loader:
                features = imagebind(
                    {ModalityType.AUDIO: wav.squeeze(1).to(device)}
                )[ModalityType.AUDIO].cpu()
                for index, name in enumerate(names):
                    imagebind_features[str(name)] = features[index]
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
            SynchformerAudioDataset(audios, duration=8.0),
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
        )
        synchformer_features: tp.Dict[str, torch.Tensor] = {}
        with torch.inference_mode():
            for wav, names in synchformer_loader:
                wav = wav.to(device)
                segments = torch.stack(
                    [wav[:, start:start + 10240] for start in range(0, 117761, 5120)],
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
    include_video_metrics: bool = False,
    refresh_cache: bool = False,
) -> Path:
    required = [
        "pann_features.pth",
        "passt_features_embed.pth",
        "passt_logits.pth",
        "vggish_features.pth",
    ]
    if include_video_metrics:
        required.extend(["imagebind_audio.pth", "synchformer_audio.pth"])
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
            backend="av_benchmark_video" if include_video_metrics else "av_benchmark_audio",
            suffix="",
            cache_dir=cache_dir,
        )
    if not refresh_cache and all((output / name).is_file() for name in required):
        return output
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(source, (str, Path)) and Path(source).expanduser().is_dir():
        _extract_audio_features(
            Path(source).expanduser(),
            output,
            batch_size=batch_size,
            device=target_device,
            include_video_metrics=include_video_metrics,
        )
    else:
        with materialize_audio_collection(source, sample_rate=sample_rate) as audio_path:
            _extract_audio_features(
                audio_path,
                output,
                batch_size=batch_size,
                device=target_device,
                include_video_metrics=include_video_metrics,
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
    refresh_cache: bool = False,
) -> Path:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    required = ["imagebind_video.pth", "synchformer_video.pth"]
    if not refresh_cache and all((output / name).is_file() for name in required):
        return output
    if not source:
        raise ValueError("Video feature extraction requires at least one ref_path")

    try:
        from av_bench.data.video_dataset import VideoDataset, error_avoidance_collate
        from av_bench.synchformer.synchformer import Synchformer
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        from torch.utils.data import DataLoader
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
        staged_paths: tp.List[Path] = []
        for key, value in sorted(source.items()):
            source_path = Path(value).expanduser().resolve()
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            staged_path = Path(temp_dir) / f"{key}{source_path.suffix.lower()}"
            staged_path.symlink_to(source_path)
            staged_paths.append(staged_path)

        dataset = VideoDataset(staged_paths, duration_sec=8.0)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
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
                imagebind_batch = imagebind(
                    {ModalityType.VISION: imagebind_clips}
                )[ModalityType.VISION].cpu()

                sync_video = batch["sync_video"].to(target_device)
                segments = torch.stack(
                    [sync_video[:, start:start + 16] for start in range(0, 185, 8)],
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
    return output


def stack_features(features: tp.Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(features.values()), dim=0)


def cat_features(features: tp.Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(list(features.values()), dim=0)


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

"""Text-audio CLAP similarity with cached audio embeddings."""

from __future__ import annotations

import hashlib
from pathlib import Path
import typing as tp

import numpy as np
import torch

from audio_eval.audio import collection_fingerprint, collection_items, load_audio
from audio_eval.cache import cache_file
from audio_eval.common import AudioCollection

_MODELS: tp.Dict[tp.Tuple[str, str, tp.Optional[str], bool], tp.Any] = {}


def _model(
    *,
    device: str,
    model_name: str,
    checkpoint_path: str | Path | None,
    enable_fusion: bool,
) -> tp.Any:
    key = (device, model_name, str(checkpoint_path) if checkpoint_path else None, enable_fusion)
    if key not in _MODELS:
        try:
            import laion_clap
        except ImportError as error:
            raise ImportError("CLAP requires the `laion-clap` package") from error
        module = laion_clap.CLAP_Module(enable_fusion=enable_fusion, device=device)
        text_embeddings = module.model.text_branch.embeddings
        if hasattr(text_embeddings, "position_ids"):
            text_embeddings._non_persistent_buffers_set.add("position_ids")
        if checkpoint_path is None:
            if model_name != "630k-audioset":
                raise ValueError(
                    "Non-default CLAP models require an explicit checkpoint_path"
                )
            module.load_ckpt()
        else:
            module.load_ckpt(str(checkpoint_path))
        _MODELS[key] = module
    return _MODELS[key]


def _prompt_map(
    prompts: tp.Union[tp.Mapping[str, str], tp.Sequence[str]],
    keys: tp.List[str],
) -> tp.Dict[str, str]:
    if isinstance(prompts, tp.Mapping):
        prompt_mapping = {str(key): str(value) for key, value in prompts.items()}
    else:
        if len(prompts) != len(keys):
            raise ValueError("Prompt sequence length must equal the number of audio samples")
        prompt_mapping = dict(zip(keys, map(str, prompts), strict=True))
    missing = sorted(set(keys) - set(prompt_mapping))
    if missing:
        raise ValueError(f"Missing prompts for {len(missing)} audio samples: {missing[:5]}")
    return prompt_mapping


def compute_clap(
    generated: AudioCollection,
    prompts: tp.Union[tp.Mapping[str, str], tp.Sequence[str]],
    *,
    sample_rate: int | None = None,
    backend: str = "laion_clap",
    model_name: str = "630k-audioset",
    checkpoint_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    batch_size: int = 8,
    device: str | None = None,
    enable_fusion: bool = False,
    refresh_cache: bool = False,
) -> dict:
    if backend != "laion_clap":
        raise ValueError(f"Unsupported CLAP backend: {backend!r}")
    try:
        import torch
    except ImportError as error:
        raise ImportError("CLAP requires PyTorch") from error
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    items = collection_items(generated)
    keys = [key for key, _ in items]
    prompt_mapping = _prompt_map(prompts, keys)
    digest = hashlib.sha256()
    digest.update(collection_fingerprint(generated, sample_rate=sample_rate).encode())
    digest.update(f"{backend}:{model_name}:{enable_fusion}".encode())
    audio_cache = cache_file(
        "features",
        digest.hexdigest(),
        backend=f"clap_{model_name}",
        suffix=".npz",
        cache_dir=cache_dir,
    )
    model = _model(
        device=device,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        enable_fusion=enable_fusion,
    )

    if audio_cache.exists() and not refresh_cache:
        loaded = np.load(audio_cache, allow_pickle=False)
        audio_embeddings = loaded["embeddings"]
        cached_keys = loaded["keys"].astype(str).tolist()
        if cached_keys != keys:
            raise ValueError("CLAP cache keys do not match the current audio collection")
    else:
        batches: list[np.ndarray] = []
        for start in range(0, len(items), batch_size):
            waveforms = [
                load_audio(source, sample_rate=sample_rate, target_sample_rate=48000)[0]
                for _, source in items[start : start + batch_size]
            ]
            max_length = max(map(len, waveforms))
            batch = np.zeros((len(waveforms), max_length), dtype=np.float32)
            for index, waveform in enumerate(waveforms):
                batch[index, : len(waveform)] = waveform
            with torch.no_grad():
                embeddings = model.get_audio_embedding_from_data(
                    torch.from_numpy(batch), use_tensor=True
                )
            batches.append(embeddings.detach().cpu().numpy())
        audio_embeddings = np.concatenate(batches, axis=0)
        np.savez_compressed(audio_cache, keys=np.asarray(keys), embeddings=audio_embeddings)

    text_inputs = [prompt_mapping[key] for key in keys]
    if len(text_inputs) == 1:
        text_inputs.append(text_inputs[0])
    with torch.no_grad():
        text_embeddings = model.get_text_embedding(
            text_inputs, use_tensor=True
        ).detach().cpu().numpy()[: len(keys)]
    audio_embeddings = audio_embeddings / np.maximum(
        np.linalg.norm(audio_embeddings, axis=-1, keepdims=True), 1e-12
    )
    text_embeddings = text_embeddings / np.maximum(
        np.linalg.norm(text_embeddings, axis=-1, keepdims=True), 1e-12
    )
    scores = np.sum(audio_embeddings * text_embeddings, axis=-1)
    return {
        "clap": float(scores.mean()),
        "backend": backend,
        "model": model_name,
        "num_samples": len(keys),
        "details": [
            {"id": key, "clap": float(score)} for key, score in zip(keys, scores, strict=True)
        ],
        "cache": str(audio_cache),
    }

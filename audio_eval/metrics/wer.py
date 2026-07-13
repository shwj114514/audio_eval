"""WER with backend selection in one consistently named module."""

from __future__ import annotations

import re
import tempfile
import typing as tp
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal

from audio_eval.audio import collection_items, load_audio
from audio_eval.common import AudioCollection, AudioInput

WER_OPTIONS: tp.Dict[str, tp.Dict[str, object]] = {
    "seedtts_en": {
        "backend": "whisper_hf",
        "model_name": "openai/whisper-large-v3",
        "language": "en",
        "aggregation": "mean_utterance",
        "text_normalization": "seedtts",
        "resample_method": "scipy_fft",
    },
    "seedtts_zh": {
        "backend": "paraformer",
        "model_name": "paraformer-zh",
        "language": "zh",
        "aggregation": "mean_utterance",
        "text_normalization": "seedtts",
        "resample_method": "backend",
    },
    "seedtts_zh_hard": {
        "backend": "paraformer",
        "model_name": "paraformer-zh",
        "language": "zh",
        "aggregation": "mean_utterance",
        "text_normalization": "seedtts",
        "resample_method": "backend",
    },
    "librispeech_test_clean": {
        "backend": "hubert_large",
        "model_name": "facebook/hubert-large-ls960-ft",
        "language": "en",
        "aggregation": "corpus",
    },
}


def get_wer_options(option: str) -> tp.Dict[str, object]:
    if not option:
        return {}
    try:
        return dict(WER_OPTIONS[option])
    except KeyError as error:
        available = ", ".join(sorted(WER_OPTIONS))
        raise ValueError(f"Unknown WER option {option!r}. Available: {available}") from error


def _prepare_audio(
    source: AudioInput,
    *,
    sample_rate: int | None,
    resample_method: str,
) -> tuple[np.ndarray, int]:
    if resample_method == "polyphase":
        return load_audio(source, sample_rate=sample_rate, target_sample_rate=16000)

    audio, source_rate = load_audio(source, sample_rate=sample_rate)
    if resample_method == "backend":
        return audio, source_rate
    if resample_method == "scipy_fft":
        if source_rate != 16000:
            audio = signal.resample(audio, int(len(audio) * 16000 / source_rate))
        return np.ascontiguousarray(audio, dtype=np.float32), 16000
    raise ValueError(f"Unsupported WER resample method: {resample_method!r}")


def _references(
    references: tp.Union[tp.Mapping[str, str], tp.Sequence[str]],
    keys: tp.List[str],
) -> tp.Dict[str, str]:
    if isinstance(references, tp.Mapping):
        mapping = {str(key): str(value) for key, value in references.items()}
    else:
        if len(references) != len(keys):
            raise ValueError("Reference sequence length must equal the number of audio samples")
        mapping = dict(zip(keys, map(str, references), strict=True))
    missing = sorted(set(keys) - set(mapping))
    if missing:
        raise ValueError(f"Missing transcripts for {len(missing)} audio samples: {missing[:5]}")
    return mapping


def _text_normalizer(language: str, protocol: str):
    if protocol == "seedtts":
        def normalize_seedtts(text: str) -> str:
            text = "".join(
                character
                for character in text.strip()
                if character == "'" or not unicodedata.category(character).startswith("P")
            )
            if language.startswith("zh"):
                return " ".join("".join(text.split()))
            return " ".join(text.lower().split())

        return normalize_seedtts
    if protocol != "default":
        raise ValueError(f"Unsupported WER text normalization: {protocol!r}")
    if language.startswith("en"):
        try:
            from whisper.normalizers import EnglishTextNormalizer
            return EnglishTextNormalizer()
        except ImportError:
            pass

    def normalize(text: str) -> str:
        text = text.strip().lower()
        if language.startswith("en"):
            text = re.sub(r"[^a-z0-9' ]+", " ", text)
        return " ".join(text.split())

    return normalize


def _whisper_hf_transcribe(
    items: list[tuple[str, AudioInput]],
    *,
    sample_rate: int | None,
    model_name: str,
    language: str,
    device: str,
    resample_method: str,
) -> list[str]:
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError as error:
        raise ImportError("Hugging Face Whisper requires the `transformers` package") from error

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    decoder_language = "english" if language.startswith("en") else language
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=decoder_language,
        task="transcribe",
    )
    hypotheses: list[str] = []
    for _, source in items:
        audio, audio_sample_rate = _prepare_audio(
            source,
            sample_rate=sample_rate,
            resample_method=resample_method,
        )
        if audio_sample_rate != 16000:
            raise ValueError("Whisper WER requires audio resampled to 16000 Hz")
        input_features = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
            )
        hypotheses.append(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])
    return hypotheses


def _hubert_transcribe(
    items: list[tuple[str, AudioInput]],
    *,
    sample_rate: int | None,
    model_name: str,
    device: str,
    resample_method: str,
) -> list[str]:
    try:
        from transformers import HubertForCTC, Wav2Vec2Processor
    except ImportError as error:
        raise ImportError("HuBERT WER requires the `transformers` package") from error

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = HubertForCTC.from_pretrained(model_name).to(device)
    model.eval()
    hypotheses: list[str] = []
    for _, source in items:
        audio, audio_sample_rate = _prepare_audio(
            source,
            sample_rate=sample_rate,
            resample_method=resample_method,
        )
        if audio_sample_rate != 16000:
            raise ValueError("HuBERT WER requires audio resampled to 16000 Hz")
        inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        hypotheses.append(processor.decode(predicted_ids[0]))
    return hypotheses


def _path_backend_transcribe(
    items: list[tuple[str, AudioInput]],
    *,
    sample_rate: int | None,
    backend: str,
    model_name: str | None,
    language: str,
    batch_size: int,
    device: str,
    resample_method: str,
) -> tuple[list[str], str]:
    with tempfile.TemporaryDirectory(prefix="audio_eval_wer_") as temp_dir:
        paths: list[str] = []
        for index, (_, source) in enumerate(items):
            audio, audio_sample_rate = _prepare_audio(
                source,
                sample_rate=sample_rate,
                resample_method=resample_method,
            )
            path = Path(temp_dir) / f"{index:08d}.wav"
            sf.write(path, audio, audio_sample_rate)
            paths.append(str(path))

        if backend == "nemo_conformer":
            try:
                from nemo.collections.asr.models import ASRModel
            except ImportError as error:
                raise ImportError("NeMo WER requires `pip install audio-eval[speech]`") from error
            model_name = model_name or "nvidia/stt_en_conformer_transducer_large"
            model = ASRModel.from_pretrained(model_name=model_name, map_location=device)
            hypotheses = model.transcribe(paths2audio_files=paths, batch_size=batch_size)
            return [item.text if hasattr(item, "text") else str(item) for item in hypotheses], model_name

        if backend == "whisper":
            try:
                import whisper
            except ImportError as error:
                raise ImportError("Whisper WER requires `pip install audio-eval[speech]`") from error
            model_name = model_name or "large-v3"
            model = whisper.load_model(model_name, device=device)
            return [model.transcribe(path, language=language)["text"] for path in paths], model_name

        if backend == "paraformer":
            try:
                from funasr import AutoModel
            except ImportError as error:
                raise ImportError("Paraformer WER requires `pip install audio-eval[speech]`") from error
            model_name = model_name or "paraformer-zh"
            model_source = model_name
            try:
                from funasr.download.name_maps_from_hub import name_maps_ms
                from modelscope.hub.snapshot_download import snapshot_download
                cached_model = snapshot_download(
                    name_maps_ms.get(model_name, model_name),
                    revision="master",
                    local_files_only=True,
                )
                if (Path(cached_model) / "model.pt").is_file():
                    model_source = cached_model
            except Exception:
                pass
            model = AutoModel(
                model=model_source,
                device=device,
                disable_update=True,
                check_latest=False,
            )
            hypotheses = []
            for path in paths:
                result = model.generate(input=path, batch_size_s=300)
                text = str(result[0]["text"])
                try:
                    import zhconv
                except ImportError as error:
                    raise ImportError(
                        "Paraformer Seed-TTS WER requires the `zhconv` package"
                    ) from error
                text = zhconv.convert(text, "zh-cn")
                hypotheses.append(text)
            return hypotheses, model_name

    raise ValueError(f"Unsupported WER backend: {backend!r}")


def compute_wer(
    generated: AudioCollection,
    references: tp.Union[tp.Mapping[str, str], tp.Sequence[str]],
    *,
    sample_rate: int | None = None,
    backend: str = "nemo_conformer",
    model_name: str | None = None,
    language: str = "en",
    batch_size: int = 8,
    device: str | None = None,
    aggregation: str = "corpus",
    text_normalization: str = "default",
    resample_method: str = "polyphase",
) -> dict:
    """Transcribe generated speech and compute corpus or mean-utterance WER.

    ``backend='hubert_large'`` matches the LLaSA codec evaluation protocol.
    """
    try:
        from jiwer import wer as jiwer_wer
    except ImportError as error:
        raise ImportError("WER requires `pip install audio-eval[speech]`") from error

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    items = collection_items(generated)
    keys = [key for key, _ in items]
    reference_map = _references(references, keys)

    if backend == "hubert_large":
        model_name = model_name or "facebook/hubert-large-ls960-ft"
        hypotheses = _hubert_transcribe(
            items,
            sample_rate=sample_rate,
            model_name=model_name,
            device=device,
            resample_method=resample_method,
        )
    elif backend == "whisper_hf":
        model_name = model_name or "openai/whisper-large-v3"
        hypotheses = _whisper_hf_transcribe(
            items,
            sample_rate=sample_rate,
            model_name=model_name,
            language=language,
            device=device,
            resample_method=resample_method,
        )
    else:
        hypotheses, model_name = _path_backend_transcribe(
            items,
            sample_rate=sample_rate,
            backend=backend,
            model_name=model_name,
            language=language,
            batch_size=batch_size,
            device=device,
            resample_method=resample_method,
        )

    normalize = _text_normalizer(language, text_normalization)
    normalized_references = [normalize(reference_map[key]) for key in keys]
    normalized_hypotheses = [normalize(text) for text in hypotheses]
    utterance_scores = [
        float(jiwer_wer(reference, hypothesis))
        for reference, hypothesis in zip(
            normalized_references,
            normalized_hypotheses,
            strict=True,
        )
    ]
    if aggregation == "corpus":
        score = float(jiwer_wer(normalized_references, normalized_hypotheses))
    elif aggregation == "mean_utterance":
        score = float(np.mean(utterance_scores))
    else:
        raise ValueError(f"Unsupported WER aggregation: {aggregation!r}")
    return {
        "wer": score,
        "wer_percent": score * 100.0,
        "backend": backend,
        "model": model_name,
        "language": language,
        "aggregation": aggregation,
        "text_normalization": text_normalization,
        "resample_method": resample_method,
        "num_samples": len(keys),
        "details": [
            {"id": key, "reference": reference, "hypothesis": hypothesis, "wer": item_wer}
            for key, reference, hypothesis, item_wer in zip(
                keys,
                normalized_references,
                normalized_hypotheses,
                utterance_scores,
                strict=True,
            )
        ],
    }

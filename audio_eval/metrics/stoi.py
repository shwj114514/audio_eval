"""STOI/ESTOI for one audio pair or two strictly paired directories."""

from __future__ import annotations

from audio_eval.common import PairedAudioCollection, load_aligned_pair, mean_or_raise, paired_sources


def compute_stoi(
    generated: PairedAudioCollection,
    reference: PairedAudioCollection,
    *,
    generated_sample_rate: int | None = None,
    reference_sample_rate: int | None = None,
    target_sample_rate: int = 16000,
    extended: bool = False,
    strict: bool = True,
) -> dict:
    try:
        from pystoi import stoi
    except ImportError as error:
        raise ImportError("STOI requires `pip install audio-eval[paired]`") from error

    scores: list[float] = []
    details: list[dict] = []
    for key, generated_source, reference_source in paired_sources(generated, reference, strict=strict):
        generated_audio, reference_audio = load_aligned_pair(
            generated_source,
            reference_source,
            generated_sample_rate=generated_sample_rate,
            reference_sample_rate=reference_sample_rate,
            target_sample_rate=target_sample_rate,
        )
        score = float(stoi(reference_audio, generated_audio, target_sample_rate, extended=extended))
        scores.append(score)
        details.append({"id": key, "stoi": score})

    return {
        "stoi": mean_or_raise(scores, "STOI"),
        "extended": extended,
        "num_samples": len(details),
        "details": details,
    }

"""PESQ for one audio pair or two strictly paired directories."""

from __future__ import annotations

from audio_eval.common import PairedAudioCollection, load_aligned_pair, mean_or_raise, paired_sources


def compute_pesq(
    generated: PairedAudioCollection,
    reference: PairedAudioCollection,
    *,
    generated_sample_rate: int | None = None,
    reference_sample_rate: int | None = None,
    target_sample_rate: int = 16000,
    strict: bool = True,
) -> dict:
    """Compute narrowband and wideband PESQ.

    Arrays and tensors require their sample rates. Paths are loaded directly.
    Directories are paired by relative path without extension.
    """
    if target_sample_rate not in {8000, 16000}:
        raise ValueError("PESQ target_sample_rate must be 8000 or 16000")
    try:
        from pesq import pesq
    except ImportError as error:
        raise ImportError("PESQ requires `pip install audio-eval[paired]`") from error

    nb_scores: list[float] = []
    wb_scores: list[float] = []
    details: list[dict] = []
    for key, generated_source, reference_source in paired_sources(generated, reference, strict=strict):
        generated_audio, reference_audio = load_aligned_pair(
            generated_source,
            reference_source,
            generated_sample_rate=generated_sample_rate,
            reference_sample_rate=reference_sample_rate,
            target_sample_rate=target_sample_rate,
        )
        nb = float(pesq(target_sample_rate, reference_audio, generated_audio, "nb"))
        detail = {"id": key, "pesq_nb": nb}
        nb_scores.append(nb)
        if target_sample_rate == 16000:
            wb = float(pesq(target_sample_rate, reference_audio, generated_audio, "wb"))
            wb_scores.append(wb)
            detail["pesq_wb"] = wb
        details.append(detail)

    result = {
        "pesq_nb": mean_or_raise(nb_scores, "PESQ-NB"),
        "num_samples": len(details),
        "details": details,
    }
    if wb_scores:
        result["pesq_wb"] = mean_or_raise(wb_scores, "PESQ-WB")
    return result

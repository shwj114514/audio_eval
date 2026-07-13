"""AudioBox Aesthetics scores for paths, arrays, or tensors."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

from audio_eval.audio import materialize_audio_collection
from audio_eval.common import AudioCollection


def compute_audiobox(
    generated: AudioCollection,
    *,
    sample_rate: int | None = None,
    backend: str = "audiobox_aesthetics",
    batch_size: int = 16,
    checkpoint_path: str | Path | None = None,
) -> dict:
    if backend != "audiobox_aesthetics":
        raise ValueError(f"Unsupported AudioBox backend: {backend!r}")
    try:
        from audiobox_aesthetics.infer import initialize_predictor
    except ImportError as error:
        raise ImportError("AudioBox scores require the `audiobox-aesthetics` package") from error

    predictor = initialize_predictor(
        ckpt=str(checkpoint_path) if checkpoint_path is not None else None
    )
    with ExitStack() as stack:
        directory = stack.enter_context(
            materialize_audio_collection(generated, sample_rate=sample_rate)
        )
        metadata = [{"path": str(path)} for path in sorted(directory.iterdir())]
        rows = []
        for start in range(0, len(metadata), batch_size):
            rows.extend(predictor.forward(metadata[start : start + batch_size]))
    axes = ("CE", "CU", "PC", "PQ")
    result = {
        axis.lower(): float(sum(row[axis] for row in rows) / len(rows)) for axis in axes
    }
    result.update({"backend": backend, "num_samples": len(rows)})
    return result

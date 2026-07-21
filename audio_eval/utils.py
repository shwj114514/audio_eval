"""JSONL manifest loading and JSON result serialization utilities."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
import typing as tp

from audio_eval.audio import list_audio_files

# TODO eplain keys below
_FIELDS = {"gen_path", "ref_path", "video_path", "prompt"}


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._") or "evaluation"


def _json_value(value: tp.Any) -> tp.Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def audio_file_map(directory: str | Path) -> dict[str, Path]:
    root = Path(directory)
    mapping: dict[str, Path] = {}
    for path in list_audio_files(root):
        key = path.relative_to(root).with_suffix("").as_posix()
        if key in mapping:
            raise ValueError(f"Duplicate audio key {key!r}: {mapping[key]} and {path}")
        mapping[key] = path
    return mapping


def load_manifest(path: str | Path) -> dict[str, dict[str, str | Path | None]]:
    """Load JSONL records keyed internally by the generated filename stem."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    records: dict[str, dict[str, str | Path | None]] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON on line {line_number} of {manifest_path}: {error.msg}"
            ) from error
        if not isinstance(raw, dict):
            raise ValueError(f"Line {line_number} of {manifest_path} must be a JSON object")
        unknown = sorted(set(raw) - _FIELDS)
        if unknown:
            raise ValueError(
                f"Unsupported fields on line {line_number} of {manifest_path}: {unknown}. "
                "Allowed fields are gen_path, ref_path, video_path, prompt"
            )

        gen_value = raw.get("gen_path")
        if not isinstance(gen_value, str) or not gen_value.strip():
            raise ValueError(f"Line {line_number} of {manifest_path} requires gen_path")
        gen_path = Path(gen_value).expanduser()
        if not gen_path.is_absolute():
            gen_path = manifest_path.parent / gen_path
        gen_path = gen_path.resolve()
        if not gen_path.is_file():
            raise FileNotFoundError(
                f"Generated audio not found on line {line_number}: {gen_path}"
            )

        key = gen_path.stem
        if key in records:
            raise ValueError(
                f"Duplicate generated filename stem {key!r} in {manifest_path}; "
                "gen_path stems must be unique"
            )

        ref_value = raw.get("ref_path")
        ref_path: Path | None = None
        if ref_value is not None:
            if not isinstance(ref_value, str) or not ref_value.strip():
                raise ValueError(f"ref_path on line {line_number} must be a path or null")
            ref_path = Path(ref_value).expanduser()
            if not ref_path.is_absolute():
                ref_path = manifest_path.parent / ref_path
            ref_path = ref_path.resolve()
            if not ref_path.is_file():
                raise FileNotFoundError(
                    f"Reference file not found on line {line_number}: {ref_path}"
                )

        video_value = raw.get("video_path")
        video_path: Path | None = None
        if video_value is not None:
            if not isinstance(video_value, str) or not video_value.strip():
                raise ValueError(f"video_path on line {line_number} must be a path or null")
            video_path = Path(video_value).expanduser()
            if not video_path.is_absolute():
                video_path = manifest_path.parent / video_path
            video_path = video_path.resolve()
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"Video file not found on line {line_number}: {video_path}"
                )

        prompt = raw.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(f"prompt on line {line_number} must be a string or null")
        records[key] = {
            "gen_path": gen_path,
            "ref_path": ref_path,
            "video_path": video_path,
            "prompt": prompt,
        }

    if not records:
        raise ValueError(f"Manifest contains no records: {manifest_path}")
    return records


def write_result(
    result: dict[str, tp.Any],
    *,
    generated: str | Path,
    reference: str | Path | None = None,
    results_dir: str | Path = "results",
    name: str | None = None,
) -> Path:
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if name:
        filename = f"{_safe_name(name)}.json"
    else:
        generated_name = _safe_name(Path(generated).name)
        filename = f"{generated_name}.json"
        if reference is not None:
            filename = f"{generated_name}__vs__{_safe_name(Path(reference).name)}.json"
    output_path = output_dir / filename
    output_path.write_text(
        json.dumps(_json_value(result), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path

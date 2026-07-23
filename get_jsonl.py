#!/usr/bin/env python3
"""Build an audio-eval JSONL manifest from generated and reference folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from audio_eval.audio import list_audio_files


PROMPT_ASSETS = {
    "audiocaps": ("audiocaps-test.csv", "audiocap_id", "caption"),
}


def audio_by_stem(folder: Path) -> dict[str, Path]:
    folder = folder.expanduser().resolve()
    files: dict[str, Path] = {}
    for path in list_audio_files(folder):
        if path.stem in files:
            raise ValueError(
                f"Duplicate audio stem {path.stem!r}: {files[path.stem]} and {path}"
            )
        files[path.stem] = path.resolve()
    return files


def load_prompts(dataset: str) -> dict[str, str]:
    filename, id_field, prompt_field = PROMPT_ASSETS[dataset]
    prompt_path = Path(__file__).resolve().parent / "audio_eval/assets/prompts" / filename
    with prompt_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {id_field, prompt_field}.issubset(reader.fieldnames):
            raise ValueError(
                f"{prompt_path} must contain {id_field!r} and {prompt_field!r} columns"
            )
        prompts: dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            sample_id = row[id_field].strip()
            prompt = row[prompt_field].strip()
            if not sample_id or not prompt:
                raise ValueError(f"Empty ID or prompt in {prompt_path} on row {row_number}")
            if sample_id in prompts:
                raise ValueError(f"Duplicate prompt ID {sample_id!r} in {prompt_path}")
            prompts[sample_id] = prompt
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an audio-eval JSONL manifest by matching audio filename stems."
    )
    parser.add_argument(
        "--gen_folder", type=Path, required=True, help="Folder containing generated audio"
    )
    parser.add_argument(
        "--ref_folder", type=Path, required=True, help="Folder containing reference audio"
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(PROMPT_ASSETS),
        help="Use bundled dataset prompts matched by generated filename stem",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manifest.jsonl"),
        help="Output JSONL path (default: manifest.jsonl)",
    )
    args = parser.parse_args()

    generated = audio_by_stem(args.gen_folder)
    references = audio_by_stem(args.ref_folder)
    missing_references = sorted(set(generated) - set(references))
    if missing_references:
        raise ValueError(
            f"Missing reference audio for {len(missing_references)} generated files: "
            f"{missing_references[:5]}"
        )

    prompts = load_prompts(args.dataset) if args.dataset else None
    if prompts is not None:
        missing_prompts = sorted(set(generated) - set(prompts))
        if missing_prompts:
            raise ValueError(
                f"Dataset {args.dataset!r} has no prompt for "
                f"{len(missing_prompts)} generated files: {missing_prompts[:5]}"
            )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample_id in sorted(generated):
            record = {
                "gen_path": str(generated[sample_id]),
                "ref_path": str(references[sample_id]),
            }
            if prompts is not None:
                record["prompt"] = prompts[sample_id]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(generated)} records to {output}")


if __name__ == "__main__":
    main()

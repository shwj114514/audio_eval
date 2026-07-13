"""Command-line interface for folder-level evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import typing as tp

from .runner import evaluate_generation, evaluate_paired


def _metrics(value: str) -> tp.List[str]:
    metrics = [item.strip() for item in value.split(",") if item.strip()]
    if not metrics:
        raise argparse.ArgumentTypeError("At least one metric is required")
    return metrics


def _metric_options(value: str) -> tp.List[str]:
    return [item.strip() for item in value.split(",")]


def _text_mapping(path: tp.Optional[str]) -> tp.Optional[tp.Dict[str, str]]:
    if path is None:
        return None
    input_path = Path(path)
    if input_path.suffix.lower() == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Text JSON must be an {id: text} object")
        return {str(key): str(value) for key, value in data.items()}
    with input_path.open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        if rows.fieldnames is None or not {"id", "text"}.issubset(rows.fieldnames):
            raise ValueError("Text CSV must contain id and text columns")
        return {str(row["id"]): str(row["text"]) for row in rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="audio-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    paired = subparsers.add_parser("paired", help="Evaluate paired generated/reference folders")
    paired.add_argument("generated")
    paired.add_argument("reference")
    paired.add_argument("--metrics", type=_metrics, default=["pesq", "stoi", "mel_stft"])
    paired.add_argument("--metric-options", type=_metric_options, default=["", "", ""])
    paired.add_argument("--results-dir", default="results")
    paired.add_argument("--name")
    paired.add_argument("--no-strict", action="store_true")

    generation = subparsers.add_parser("generation", help="Evaluate a generated folder")
    generation.add_argument("generated", nargs="?")
    generation.add_argument("--manifest", help="JSONL with gen_path and optional prompt")
    generation.add_argument(
        "--reference",
        help="Reference audio path, precomputed cache path, or bundled reference name",
    )
    generation.add_argument("--metrics", type=_metrics, default=[])
    generation.add_argument("--metric-options", type=_metric_options, default=[])
    generation.add_argument("--prompts", help="JSON mapping or CSV with id,text columns")
    generation.add_argument("--transcripts", help="JSON mapping or CSV with id,text columns")
    generation.add_argument("--cache-dir")
    generation.add_argument("--results-dir", default="results")
    generation.add_argument("--name")

    tts = subparsers.add_parser("tts", help="Evaluate a TTS JSONL manifest")
    tts.add_argument("manifest")
    tts.add_argument("--metrics", type=_metrics, default=["wer", "speaker_sim"])
    tts.add_argument("--metric-options", type=_metric_options, default=["", ""])
    tts.add_argument("--results-dir", default="results")
    tts.add_argument("--name")

    recon = subparsers.add_parser("recon", help="Evaluate a reconstruction JSONL manifest")
    recon.add_argument("manifest")
    recon.add_argument("--metrics", type=_metrics, default=["pesq", "stoi", "mel_stft", "lsd"])
    recon.add_argument("--metric-options", type=_metric_options, default=["", "", "", ""])
    recon.add_argument("--results-dir", default="results")
    recon.add_argument("--name")

    sr = subparsers.add_parser("sr", help="Evaluate an audio super-resolution JSONL manifest")
    sr.add_argument("manifest")
    sr.add_argument("--metrics", type=_metrics, default=["lsd"])
    sr.add_argument("--metric-options", type=_metric_options, default=["ssr_eval"])
    sr.add_argument("--results-dir", default="results")
    sr.add_argument("--name")

    v2a = subparsers.add_parser("v2a", help="Evaluate a video-to-audio JSONL manifest")
    v2a.add_argument("manifest")
    v2a.add_argument(
        "--metrics",
        type=_metrics,
        default=["fd", "fd", "fd", "kl", "kl", "inception_score", "imagebind", "desync"],
    )
    v2a.add_argument(
        "--metric-options",
        type=_metric_options,
        default=[
            "passt", "panns", "vggish", "passt_ref_to_gen", "panns_ref_to_gen", "panns", "", ""
        ],
    )
    v2a.add_argument("--generated-cache")
    v2a.add_argument("--reference-cache")
    v2a.add_argument(
        "--reference",
        help="FD/KL reference audio, feature cache, or bundled reference name",
    )
    v2a.add_argument("--results-dir", default="results")
    v2a.add_argument("--name")

    for command in ("tta", "ttm"):
        task_parser = subparsers.add_parser(command, help=f"Evaluate a {command.upper()} JSONL manifest")
        task_parser.add_argument("manifest")
        task_parser.add_argument(
            "--reference",
            help="Reference audio path, precomputed cache path, or bundled reference name",
        )
        task_parser.add_argument("--metrics", type=_metrics, default=[])
        task_parser.add_argument("--metric-options", type=_metric_options, default=[])
        task_parser.add_argument("--generated-cache")
        task_parser.add_argument("--reference-cache")
        task_parser.add_argument("--cache-dir")
        task_parser.add_argument("--results-dir", default="results")
        task_parser.add_argument("--name")
    return parser


def main(argv: tp.Optional[tp.List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "tts":
        from .eval_tts import eval_tts
        result = eval_tts(
            args.manifest,
            metrics=args.metrics,
            metric_options=args.metric_options,
            results_dir=args.results_dir,
            name=args.name,
        )
    elif args.command == "recon":
        from .eval_recon import eval_recon
        result = eval_recon(
            args.manifest,
            metrics=args.metrics,
            metric_options=args.metric_options,
            results_dir=args.results_dir,
            name=args.name,
        )
    elif args.command == "sr":
        from .eval_sr import eval_sr
        result = eval_sr(
            args.manifest,
            metrics=args.metrics,
            metric_options=args.metric_options,
            results_dir=args.results_dir,
            name=args.name,
        )
    elif args.command == "v2a":
        from .eval_v2a import eval_v2a
        result = eval_v2a(
            args.manifest,
            metrics=args.metrics,
            metric_options=args.metric_options,
            generated_cache=args.generated_cache,
            reference_cache=args.reference_cache,
            reference=args.reference,
            results_dir=args.results_dir,
            name=args.name,
        )
    elif args.command in {"tta", "ttm"}:
        from .eval_generation import eval_tta, eval_ttm
        evaluator = eval_tta if args.command == "tta" else eval_ttm
        result = evaluator(
            args.manifest,
            metrics=args.metrics,
            metric_options=args.metric_options,
            reference=args.reference,
            generated_cache=args.generated_cache,
            reference_cache=args.reference_cache,
            cache_dir=args.cache_dir,
            results_dir=args.results_dir,
            name=args.name,
        )
    elif args.command == "paired":
        result = evaluate_paired(
            args.generated,
            args.reference,
            metrics=args.metrics,
            metric_options=args.metric_options,
            strict=not args.no_strict,
            results_dir=args.results_dir,
            name=args.name,
        )
    elif args.command == "generation":
        if args.manifest:
            from .eval_generation import eval_generation
            result = eval_generation(
                args.manifest,
                metrics=args.metrics,
                metric_options=args.metric_options,
                reference=args.reference,
                cache_dir=args.cache_dir,
                results_dir=args.results_dir,
                name=args.name,
            )
        else:
            if args.generated is None:
                raise ValueError("generation requires GENERATED or --manifest")
            if args.metrics is None:
                raise ValueError("folder-based generation requires --metrics")
            result = evaluate_generation(
                args.generated,
                metrics=args.metrics,
                metric_options=args.metric_options,
                reference_dir=args.reference,
                prompts=_text_mapping(args.prompts),
                transcripts=_text_mapping(args.transcripts),
                cache_dir=args.cache_dir,
                results_dir=args.results_dir,
                name=args.name,
            )
    print(f"<<<<<<<<<<<< metric for {args.manifest} <<<<<<<<<<<< ")

    ignored_numeric_fields = {
        "batch_size",
        "channels",
        "hop_length",
        "n_fft",
        "sample_rate",
        "seed",
        "splits",
        "target_sample_rate",
    }
    metric_groups = [("", result["metrics"])]
    for prefix, values in metric_groups:
        for key, value in values.items():
            label = key if not prefix else f"{prefix}.{key}"
            if isinstance(value, dict):
                metric_groups.append((label, value))
            elif (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and not key.startswith("num_")
                and key not in ignored_numeric_fields
            ):
                metric_name = prefix.split(".", 1)[0]
                print(f"{prefix if prefix and key == metric_name else label}: {value}")
    print(f">>>>>>>>>>>> Result saved to: {result['result_path']} >>>>>>>>>>>> ")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

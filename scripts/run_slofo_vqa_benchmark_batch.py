#!/usr/bin/env python3
"""Run a resumable SLoFo benchmark manifest while loading LLaVA only once."""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model-path", default="models/llava-v1.5-7b")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument(
        "--runner", type=Path, default=Path("scripts/run_slofo_scan_locate.py")
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--semantic-rollout-tokens", type=int, default=8)
    parser.add_argument(
        "--scan-capture-mode",
        choices=("all_outputs", "selective_hook"),
        default="all_outputs",
    )
    parser.add_argument(
        "--fusion-normalization", choices=("none", "minmax"), default="minmax"
    )
    parser.add_argument(
        "--locate-coordinate-space", choices=("padded", "original"), default="original"
    )
    parser.add_argument(
        "--semantic-token-aggregation", choices=("mean", "max"), default="mean"
    )
    parser.add_argument(
        "--full-artifact-count",
        type=int,
        default=3,
        help="Keep complete visual artifacts for the first N selected records.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--case-ids", nargs="*", default=())
    return parser.parse_args()


class Tee:
    def __init__(self, *handles):
        self.handles = handles

    def write(self, value: str) -> int:
        for handle in self.handles:
            handle.write(value)
            handle.flush()
        return len(value)

    def flush(self) -> None:
        for handle in self.handles:
            handle.flush()


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("slofo_scan_locate_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_case_id(value: object) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not result:
        raise ValueError(f"Invalid empty case id derived from {value!r}")
    return result


def result_row(record: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    return {
        "benchmark": record.get("benchmark"),
        "question_id": record["qa_id"],
        "image": record["image_file"],
        "question": record["model_query"],
        "ground_truth": record.get("ground_truth"),
        "ground_truth_answers": record.get("ground_truth_answers"),
        "category": record.get("category"),
        "original_answer": result.get("original_answer"),
        "crop_answer": result.get("crop_answer"),
        "legacy_joint_answer": result.get("baseline_joint_answer"),
        "topk_joint_answer": result.get("reranked_joint_answer"),
        "focus_answer": result.get("focused_joint_answer"),
        "selected_bbox": result.get("original_bbox"),
        "legacy_bbox": result.get("legacy_original_bbox"),
        "selected_rank": result.get("topk_reranking", {}).get("selected_rank"),
        "selection_changed": result.get("topk_reranking", {}).get(
            "selection_changed"
        ),
        "artifact_mode": result.get("artifact_mode"),
        "scan_peak_allocated_mib": result.get("scan_peak_allocated_mib"),
        "generation_peak_allocated_mib": result.get(
            "generation_peak_allocated_mib"
        ),
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.full_artifact_count < 0:
        raise ValueError("full_artifact_count must be non-negative")
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    project_root = Path.cwd().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_records = manifest["records"]
    requested_ids = {str(value) for value in args.case_ids}
    records = (
        [row for row in all_records if str(row["qa_id"]) in requested_ids]
        if requested_ids
        else all_records
    )
    known_ids = {str(row["qa_id"]) for row in records}
    if requested_ids - known_ids:
        raise ValueError(f"Unknown case IDs: {sorted(requested_ids - known_ids)}")
    if not records:
        raise ValueError("Manifest selection is empty")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    runner = load_runner(args.runner)
    real_loader = runner.load_pretrained_model
    model_cache: list[object] = []

    def cached_loader(*loader_args, **loader_kwargs):
        if not model_cache:
            print("[batch] Loading LLaVA model once for all requested cases.")
            model_cache.extend(real_loader(*loader_args, **loader_kwargs))
        return tuple(model_cache)

    runner.load_pretrained_model = cached_loader
    batch_started = time.perf_counter()
    case_summaries: list[dict[str, object]] = []
    answer_rows: list[dict[str, object]] = []
    original_argv = sys.argv[:]
    answers_path = args.output_root / "benchmark_answers.jsonl"
    summary_path = args.output_root / "batch_summary.json"
    try:
        for index, record in enumerate(records, 1):
            case_id = safe_case_id(record["qa_id"])
            output_dir = args.output_root / "cases" / case_id
            result_path = output_dir / "result.json"
            log_path = args.log_root / f"{case_id}.log"
            artifact_mode = "full" if index <= args.full_artifact_count else "none"

            if result_path.is_file() and not args.force:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                answer_rows.append(result_row(record, result))
                case_summaries.append({"qa_id": record["qa_id"], "status": "skipped"})
                print(f"[{index:04d}/{len(records):04d}] {case_id}: existing; skipped")
                continue

            image_path = args.image_root / str(record["image_file"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            command_args = [
                str(args.runner),
                "--model-path", str(args.model_path),
                "--image-file", str(image_path),
                "--query", str(record["model_query"]),
                "--output-dir", str(output_dir),
                "--artifact-mode", artifact_mode,
                "--fusion-normalization", args.fusion_normalization,
                "--locate-coordinate-space", args.locate_coordinate_space,
                "--semantic-rollout-tokens", str(args.semantic_rollout_tokens),
                "--semantic-anchor-start-index", "0",
                "--semantic-token-aggregation", args.semantic_token_aggregation,
                "--semantic-score", "log_probability",
                "--semantic-weight", "0.7",
                "--scan-capture-mode", args.scan_capture_mode,
                "--rerank-top-k", str(args.top_k),
                "--rerank-pre-nms-per-scale", "12",
                "--rerank-nms-iou", "0.55",
                "--rerank-scan-weight", "0.15",
                "--rerank-answer-consistency-weight", "0.0",
                "--rerank-verification-mode", "generic-evidence",
                "--rerank-min-improvement", "0.2",
                "--max-new-tokens", str(args.max_new_tokens),
                "--enable-focus",
                "--focus-phases", "4",
                "--focus-prune-ratio", "0.5",
                "--seed", "0",
            ]

            print(f"[{index:04d}/{len(records):04d}] {case_id}: starting")
            output_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            sys.argv = command_args
            with log_path.open("w", encoding="utf-8") as log_handle:
                with contextlib.redirect_stdout(Tee(sys.__stdout__, log_handle)):
                    with contextlib.redirect_stderr(Tee(sys.__stderr__, log_handle)):
                        runner.main()
            elapsed = time.perf_counter() - started
            result = json.loads(result_path.read_text(encoding="utf-8"))
            answer_rows.append(result_row(record, result))
            case_summaries.append(
                {
                    "qa_id": record["qa_id"],
                    "status": "completed",
                    "wall_seconds": elapsed,
                    "artifact_mode": artifact_mode,
                }
            )
            write_jsonl(answers_path, answer_rows)
            summary = {
                "benchmark": args.benchmark,
                "manifest": str(args.manifest),
                "requested_count": len(records),
                "completed_count": sum(
                    row["status"] == "completed" for row in case_summaries
                ),
                "skipped_count": sum(
                    row["status"] == "skipped" for row in case_summaries
                ),
                "elapsed_seconds": time.perf_counter() - batch_started,
                "configuration": {
                    "top_k": args.top_k,
                    "semantic_rollout_tokens": args.semantic_rollout_tokens,
                    "scan_capture_mode": args.scan_capture_mode,
                    "fusion_normalization": args.fusion_normalization,
                    "locate_coordinate_space": args.locate_coordinate_space,
                    "semantic_token_aggregation": args.semantic_token_aggregation,
                },
                "cases": case_summaries,
            }
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"[{index:04d}/{len(records):04d}] {case_id}: "
                f"completed in {elapsed:.1f}s"
            )
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        sys.argv = original_argv

    write_jsonl(answers_path, answer_rows)
    print(
        f"All {len(records)} requested cases completed in "
        f"{time.perf_counter() - batch_started:.1f}s."
    )


if __name__ == "__main__":
    main()

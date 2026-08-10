#!/usr/bin/env python3
"""Run the 2026-08-09 COCO128 diagnostic while loading LLaVA only once."""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", default="models/llava-v1.5-7b")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=Path("scripts/run_slofo_scan_locate.py"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--semantic-rollout-tokens", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--case-ids", nargs="*", default=())
    return parser.parse_args()


class Tee:
    def __init__(self, *handles):
        self.handles = handles

    def write(self, text: str) -> int:
        for handle in self.handles:
            handle.write(text)
            handle.flush()
        return len(text)

    def flush(self) -> None:
        for handle in self.handles:
            handle.flush()


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("slofo_scan_locate_08_09", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    project_root = Path.cwd().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_records = manifest["records"]
    if len(all_records) != 128:
        raise ValueError(f"Expected 128 records, found {len(all_records)}")
    requested_ids = set(args.case_ids)
    records = (
        [record for record in all_records if record["qa_id"] in requested_ids]
        if requested_ids
        else all_records
    )
    known_ids = {record["qa_id"] for record in records}
    if requested_ids - known_ids:
        raise ValueError(f"Unknown case IDs: {sorted(requested_ids - known_ids)}")

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
    original_argv = sys.argv[:]
    try:
        for index, record in enumerate(records, 1):
            case_id = str(record["qa_id"])
            output_dir = args.output_root / case_id
            result_path = output_dir / "result.json"
            log_path = args.log_root / f"{case_id}.log"
            if result_path.is_file() and not args.force:
                print(f"[{index:03d}/{len(records):03d}] {case_id}: existing result; skipped")
                case_summaries.append({"qa_id": case_id, "status": "skipped"})
                continue

            command_args = [
                str(args.runner),
                "--model-path", args.model_path,
                "--image-file", str(args.image_root / str(record["image_file"])),
                "--query", str(record["model_query"]),
                "--output-dir", str(output_dir),
                "--fusion-normalization", "minmax",
                "--compare-fusion-normalizations",
                "--locate-coordinate-space", "original",
                "--semantic-rollout-tokens", str(args.semantic_rollout_tokens),
                "--semantic-anchor-start-index", "0",
                "--semantic-token-aggregation", "mean",
                "--semantic-score", "log_probability",
                "--semantic-weight", "0.7",
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
            evaluation_bbox = record.get("ground_truth_bbox_xyxy")
            if evaluation_bbox is None:
                evaluation_bbox = record.get("primary_target_bbox_xyxy")
            if evaluation_bbox is not None:
                command_args.extend(
                    [
                        "--ground-truth-bbox",
                        *(str(value) for value in evaluation_bbox),
                    ]
                )

            print(f"[{index:03d}/{len(records):03d}] {case_id}: starting")
            output_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            sys.argv = command_args
            with log_path.open("w", encoding="utf-8") as log_handle:
                with contextlib.redirect_stdout(Tee(sys.__stdout__, log_handle)):
                    with contextlib.redirect_stderr(Tee(sys.__stderr__, log_handle)):
                        runner.main()
            elapsed = time.perf_counter() - started
            case_summaries.append(
                {"qa_id": case_id, "status": "completed", "wall_seconds": elapsed}
            )
            summary = {
                "requested_count": len(records),
                "completed_count": sum(row["status"] == "completed" for row in case_summaries),
                "skipped_count": sum(row["status"] == "skipped" for row in case_summaries),
                "elapsed_seconds": time.perf_counter() - batch_started,
                "cases": case_summaries,
            }
            (args.output_root / "batch_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[{index:03d}/{len(records):03d}] {case_id}: completed in {elapsed:.1f}s")
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        sys.argv = original_argv

    print(
        f"All {len(records)} requested cases completed in "
        f"{time.perf_counter() - batch_started:.1f}s."
    )


if __name__ == "__main__":
    main()

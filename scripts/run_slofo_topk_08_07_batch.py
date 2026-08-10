#!/usr/bin/env python3
"""Run the 20-case COCO128 top-k reranking evaluation on one guarded GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", default="models/llava-v1.5-7b")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--runner", default="scripts/run_slofo_scan_locate.py")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--enable-focus", action="store_true")
    parser.add_argument(
        "--case-ids",
        nargs="*",
        default=(),
        help="Optional QA IDs for a smoke/subset run; default runs all 20.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_records = manifest["records"]
    if len(all_records) != 20:
        raise ValueError(f"Expected 20 QA records, found {len(all_records)}")
    requested_ids = set(args.case_ids)
    records = (
        [record for record in all_records if record["qa_id"] in requested_ids]
        if requested_ids
        else all_records
    )
    if requested_ids and requested_ids != {record["qa_id"] for record in records}:
        missing = sorted(requested_ids - {record["qa_id"] for record in records})
        raise ValueError(f"Unknown case IDs: {missing}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})

    for index, record in enumerate(records, 1):
        case_id = str(record["qa_id"])
        output_dir = args.output_root / case_id
        result_path = output_dir / "result.json"
        log_path = args.log_root / f"{case_id}.log"
        if result_path.is_file() and not args.force:
            print(
                f"[{index:02d}/{len(records):02d}] {case_id}: "
                "existing result found; skipping."
            )
            continue
        command = [
            sys.executable,
            args.runner,
            "--model-path",
            args.model_path,
            "--image-file",
            str(args.image_root / record["image_file"]),
            "--query",
            str(record["model_query"]),
            "--output-dir",
            str(output_dir),
            "--fusion-normalization",
            "minmax",
            "--locate-coordinate-space",
            "original",
            "--semantic-rollout-tokens",
            "8",
            "--semantic-anchor-start-index",
            "0",
            "--semantic-token-aggregation",
            "mean",
            "--semantic-score",
            "log_probability",
            "--semantic-weight",
            "0.7",
            "--rerank-top-k",
            str(args.top_k),
            "--rerank-pre-nms-per-scale",
            "12",
            "--rerank-nms-iou",
            "0.55",
            "--rerank-scan-weight",
            "0.15",
            "--rerank-answer-consistency-weight",
            "1.0",
            "--rerank-min-improvement",
            "0.2",
            "--rerank-candidate-answer-tokens",
            "16",
            "--ground-truth-bbox",
            *(str(value) for value in record["ground_truth_bbox_xyxy"]),
            "--max-new-tokens",
            "24",
            "--seed",
            "0",
        ]
        if args.enable_focus:
            command.extend(
                [
                    "--enable-focus",
                    "--focus-phases",
                    "4",
                    "--focus-prune-ratio",
                    "0.5",
                ]
            )
        print(f"[{index:02d}/{len(records):02d}] {case_id}: starting")
        output_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_handle.write(line)
                log_handle.flush()
            return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
        print(f"[{index:02d}/{len(records):02d}] {case_id}: completed")
    print(f"All {len(records)} requested top-k reranking cases completed.")


if __name__ == "__main__":
    main()

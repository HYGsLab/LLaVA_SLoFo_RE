#!/usr/bin/env python3
"""Compare legacy, selective-memory, and paper-default TextVQA runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


ANSWER_FIELDS = (
    "original_answer",
    "crop_answer",
    "legacy_joint_answer",
    "topk_joint_answer",
    "focus_answer",
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def scores(rows: list[dict[str, object]]) -> dict[str, float]:
    evaluator = TextVQAAccuracyEvaluator()
    result: dict[str, float] = {}
    for field in ANSWER_FIELDS:
        total = 0.0
        for row in rows:
            prediction = evaluator.answer_processor(str(row.get(field) or ""))
            answer_scores = evaluator._compute_answer_scores(
                row["ground_truth_answers"]
            )
            total += float(answer_scores.get(prediction, 0.0))
        result[field] = 100.0 * total / len(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--paper-default", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    ordered_ids = [str(row["qa_id"]) for row in manifest["records"]]
    runs = {
        "legacy_all_outputs_8token_minmax_topk": read_jsonl(args.legacy),
        "optimized_selective_8token_minmax_topk": read_jsonl(args.optimized),
        "paper_selective_1token_raw_single": read_jsonl(args.paper_default),
    }
    aligned: dict[str, list[dict[str, object]]] = {}
    for name, rows in runs.items():
        by_id = {str(row["question_id"]): row for row in rows}
        missing = [qa_id for qa_id in ordered_ids if qa_id not in by_id]
        if missing:
            raise ValueError(f"{name} is missing {len(missing)} requested IDs")
        aligned[name] = [by_id[qa_id] for qa_id in ordered_ids]

    legacy = aligned["legacy_all_outputs_8token_minmax_topk"]
    optimized = aligned["optimized_selective_8token_minmax_topk"]
    paper = aligned["paper_selective_1token_raw_single"]
    exact_differences = {
        field: sum(a.get(field) != b.get(field) for a, b in zip(legacy, optimized))
        for field in ANSWER_FIELDS
    }
    exact_differences.update(
        {
            "selected_bbox": sum(
                a.get("selected_bbox") != b.get("selected_bbox")
                for a, b in zip(legacy, optimized)
            ),
            "legacy_bbox": sum(
                a.get("legacy_bbox") != b.get("legacy_bbox")
                for a, b in zip(legacy, optimized)
            ),
            "selected_rank": sum(
                a.get("selected_rank") != b.get("selected_rank")
                for a, b in zip(legacy, optimized)
            ),
        }
    )

    memory = {
        name: distribution(
            [float(row["scan_peak_allocated_mib"]) for row in rows]
        )
        for name, rows in aligned.items()
    }
    old_memory = memory["legacy_all_outputs_8token_minmax_topk"]
    new_memory = memory["optimized_selective_8token_minmax_topk"]
    output = {
        "sample_count": len(ordered_ids),
        "question_id_first_last": [ordered_ids[0], ordered_ids[-1]],
        "scan_memory_mib": memory,
        "optimized_minus_legacy_mib": {
            key: new_memory[key] - old_memory[key] for key in old_memory
        },
        "optimized_reduction_percent": {
            key: 100.0 * (old_memory[key] - new_memory[key]) / old_memory[key]
            for key in old_memory
        },
        "optimized_vs_legacy_exact_differences": exact_differences,
        "textvqa_soft_accuracy_percent": {
            name: scores(rows) for name, rows in aligned.items()
        },
        "paper_vs_optimized_answer_differences": {
            field: sum(a.get(field) != b.get(field) for a, b in zip(paper, optimized))
            for field in ANSWER_FIELDS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

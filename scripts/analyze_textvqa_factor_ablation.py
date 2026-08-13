#!/usr/bin/env python3
"""Analyze the chained TextVQA Token/fusion/Top-k factor ablation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


BRANCHES = (
    "original_answer",
    "crop_answer",
    "legacy_joint_answer",
    "topk_joint_answer",
    "focus_answer",
)

COMPARISONS = (
    ("fusion_at_1token", "t1_raw_single", "t1_minmax_single"),
    ("tokens_at_raw", "t1_raw_single", "t8_raw_single"),
    ("fusion_at_8token", "t8_raw_single", "t8_minmax_single"),
    ("tokens_at_minmax", "t1_minmax_single", "t8_minmax_single"),
    ("topk_at_8token_minmax", "t8_minmax_single", "t8_minmax_topk"),
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
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def paired_interval(differences: list[float]) -> list[float]:
    mean = statistics.fmean(differences)
    if len(differences) < 2:
        return [mean, mean]
    margin = 1.96 * statistics.stdev(differences) / math.sqrt(len(differences))
    return [mean - margin, mean + margin]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=ANSWERS_JSONL",
        help="Repeat for each named factor cell.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    ordered_ids = [str(row["qa_id"]) for row in manifest["records"]]
    run_paths: dict[str, Path] = {}
    for value in args.run:
        if "=" not in value:
            raise ValueError(f"Invalid --run value: {value}")
        name, raw_path = value.split("=", 1)
        run_paths[name] = Path(raw_path)

    if "t1_raw_single" not in run_paths:
        raise ValueError("Missing reference factor cell: t1_raw_single")
    active_comparisons = [
        comparison
        for comparison in COMPARISONS
        if comparison[1] in run_paths and comparison[2] in run_paths
    ]
    if not active_comparisons:
        raise ValueError("No complete factor comparison is available")

    evaluator = TextVQAAccuracyEvaluator()
    aligned: dict[str, list[dict[str, object]]] = {}
    score_rows: dict[str, dict[str, list[float]]] = {}
    summaries: dict[str, dict[str, object]] = {}
    for name, path in run_paths.items():
        rows = read_jsonl(path)
        by_id = {str(row["question_id"]): row for row in rows}
        missing = [qa_id for qa_id in ordered_ids if qa_id not in by_id]
        if missing:
            raise ValueError(f"{name} misses {len(missing)} requested questions")
        aligned[name] = [by_id[qa_id] for qa_id in ordered_ids]
        score_rows[name] = {}
        branch_scores: dict[str, float] = {}
        for branch in BRANCHES:
            values: list[float] = []
            for row in aligned[name]:
                prediction = evaluator.answer_processor(str(row.get(branch) or ""))
                answer_scores = evaluator._compute_answer_scores(
                    row["ground_truth_answers"]
                )
                values.append(float(answer_scores.get(prediction, 0.0)))
            score_rows[name][branch] = values
            branch_scores[branch] = 100.0 * statistics.fmean(values)

        batch_path = path.parent / "batch_summary.json"
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        elapsed = [
            float(row["wall_seconds"])
            for row in batch["cases"]
            if row.get("status") == "completed"
        ]
        scan_memory = [
            float(row["scan_peak_allocated_mib"]) for row in aligned[name]
        ]
        summaries[name] = {
            "configuration": batch.get("configuration"),
            "branch_accuracy_percent": branch_scores,
            "runtime_seconds": {
                "batch_total": float(batch["elapsed_seconds"]),
                **distribution(elapsed),
            },
            "scan_peak_mib": distribution(scan_memory),
            "selection_changed_count": sum(
                bool(row.get("selection_changed")) for row in aligned[name]
            ),
        }

    factor_results: dict[str, dict[str, object]] = {}
    for comparison_name, left_name, right_name in active_comparisons:
        left_rows, right_rows = aligned[left_name], aligned[right_name]
        branch_results: dict[str, object] = {}
        for branch in ("crop_answer", "topk_joint_answer", "focus_answer"):
            left_scores = score_rows[left_name][branch]
            right_scores = score_rows[right_name][branch]
            differences = [
                100.0 * (right - left)
                for left, right in zip(left_scores, right_scores)
            ]
            branch_results[branch] = {
                "right_minus_left_percentage_points": statistics.fmean(differences),
                "paired_95ci_percentage_points": paired_interval(differences),
                "answer_changed": sum(
                    left.get(branch) != right.get(branch)
                    for left, right in zip(left_rows, right_rows)
                ),
                "score_improved": sum(value > 0 for value in differences),
                "score_regressed": sum(value < 0 for value in differences),
            }
        factor_results[comparison_name] = {
            "left": left_name,
            "right": right_name,
            "selected_bbox_changed": sum(
                left.get("selected_bbox") != right.get("selected_bbox")
                for left, right in zip(left_rows, right_rows)
            ),
            "legacy_bbox_changed": sum(
                left.get("legacy_bbox") != right.get("legacy_bbox")
                for left, right in zip(left_rows, right_rows)
            ),
            "branches": branch_results,
        }

    original_reference = aligned["t1_raw_single"]
    original_answer_differences = {
        name: sum(
            reference.get("original_answer") != row.get("original_answer")
            for reference, row in zip(original_reference, rows)
        )
        for name, rows in aligned.items()
    }
    output = {
        "sample_count": len(ordered_ids),
        "runs": summaries,
        "factor_comparisons": factor_results,
        "original_answer_differences_vs_t1_raw_single": original_answer_differences,
        "interpretation_rule": (
            "Each factor result is a paired right-minus-left comparison. "
            "Only the named factor changes inside that comparison."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

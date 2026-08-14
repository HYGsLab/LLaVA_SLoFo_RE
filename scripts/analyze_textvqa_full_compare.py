#!/usr/bin/env python3
"""Compare the full TextVQA baseline and optimized SLoFo engineering runs.

This is deliberately an end-to-end A/E comparison.  The two runs differ in
semantic rollout length, fusion normalization, and candidate-box reranking, so
the resulting delta must not be presented as an isolated single-factor effect.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


BRANCHES = (
    "original_answer",
    "crop_answer",
    "legacy_joint_answer",
    "topk_joint_answer",
    "focus_answer",
)
DEFAULT_PREFIXES = (128, 256, 512, 1024, 2048, 3000, 4000, 5000)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def align_rows(
    ordered_ids: list[str], path: Path, run_name: str
) -> list[dict[str, object]]:
    rows = read_jsonl(path)
    by_id = {str(row["question_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"{run_name} contains duplicate question IDs")
    missing = [qa_id for qa_id in ordered_ids if qa_id not in by_id]
    unexpected = sorted(set(by_id) - set(ordered_ids))
    if missing or unexpected:
        raise ValueError(
            f"{run_name} alignment failed: missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
    return [by_id[qa_id] for qa_id in ordered_ids]


def load_batch_summary(path: Path) -> dict[str, object]:
    return json.loads((path.parent / "batch_summary.json").read_text(encoding="utf-8"))


def score_rows(
    rows: list[dict[str, object]], evaluator: TextVQAAccuracyEvaluator
) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    for branch in BRANCHES:
        values: list[float] = []
        for row in rows:
            answer_scores = evaluator._compute_answer_scores(
                row["ground_truth_answers"]
            )
            prediction = evaluator.answer_processor(str(row.get(branch) or ""))
            values.append(float(answer_scores.get(prediction, 0.0)))
        output[branch] = values
    return output


def run_summary(
    rows: list[dict[str, object]],
    scores: dict[str, list[float]],
    batch: dict[str, object],
) -> dict[str, object]:
    timings = [
        float(case["wall_seconds"])
        for case in batch["cases"]
        if case.get("status") == "completed" and "wall_seconds" in case
    ]
    scan_memory = [float(row["scan_peak_allocated_mib"]) for row in rows]
    generation_memory = [
        float(row["generation_peak_allocated_mib"]) for row in rows
    ]
    ranks = Counter(str(row.get("selected_rank")) for row in rows)
    runtime: dict[str, object] = {
        "batch_elapsed_seconds": float(batch["elapsed_seconds"]),
        "completed_cases_with_timing": len(timings),
    }
    if timings:
        runtime["completed_case_seconds"] = distribution(timings)
    return {
        "configuration": batch.get("configuration"),
        "branch_accuracy_percent": {
            branch: 100.0 * statistics.fmean(values)
            for branch, values in scores.items()
        },
        "runtime": runtime,
        "scan_peak_mib": distribution(scan_memory),
        "generation_peak_mib": distribution(generation_memory),
        "selection_changed_count": sum(
            bool(row.get("selection_changed")) for row in rows
        ),
        "selected_rank_counts": dict(sorted(ranks.items())),
    }


def paired_branch_result(
    baseline_rows: list[dict[str, object]],
    optimized_rows: list[dict[str, object]],
    baseline_scores: list[float],
    optimized_scores: list[float],
    branch: str,
) -> dict[str, object]:
    differences = [
        100.0 * (optimized - baseline)
        for baseline, optimized in zip(baseline_scores, optimized_scores)
    ]
    return {
        "optimized_minus_baseline_percentage_points": statistics.fmean(differences),
        "paired_95ci_percentage_points": paired_interval(differences),
        "answer_changed": sum(
            baseline.get(branch) != optimized.get(branch)
            for baseline, optimized in zip(baseline_rows, optimized_rows)
        ),
        "score_improved": sum(value > 0 for value in differences),
        "score_regressed": sum(value < 0 for value in differences),
        "score_unchanged": sum(value == 0 for value in differences),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-output", type=Path)
    parser.add_argument("--prefix", action="append", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    ordered_ids = [str(row["qa_id"]) for row in manifest["records"]]
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("Manifest contains duplicate question IDs")

    baseline_rows = align_rows(ordered_ids, args.baseline, "baseline")
    optimized_rows = align_rows(ordered_ids, args.optimized, "optimized")
    evaluator = TextVQAAccuracyEvaluator()
    baseline_scores = score_rows(baseline_rows, evaluator)
    optimized_scores = score_rows(optimized_rows, evaluator)

    prefixes = sorted(set(args.prefix or DEFAULT_PREFIXES))
    invalid = [value for value in prefixes if value <= 0 or value > len(ordered_ids)]
    if invalid:
        raise ValueError(f"Invalid prefixes for {len(ordered_ids)} rows: {invalid}")

    paired = {
        branch: paired_branch_result(
            baseline_rows,
            optimized_rows,
            baseline_scores[branch],
            optimized_scores[branch],
            branch,
        )
        for branch in BRANCHES
        if branch != "original_answer"
    }
    convergence: dict[str, object] = {}
    for prefix in prefixes:
        convergence[str(prefix)] = {
            branch: paired_branch_result(
                baseline_rows[:prefix],
                optimized_rows[:prefix],
                baseline_scores[branch][:prefix],
                optimized_scores[branch][:prefix],
                branch,
            )
            for branch in ("crop_answer", "topk_joint_answer", "focus_answer")
        }

    output = {
        "comparison_scope": (
            "End-to-end optimized engineering pipeline versus paper-formula baseline; "
            "semantic rollout, min-max fusion, and top-k reranking change together."
        ),
        "sample_count": len(ordered_ids),
        "manifest_seed": manifest.get("seed"),
        "runs": {
            "baseline_A": run_summary(
                baseline_rows, baseline_scores, load_batch_summary(args.baseline)
            ),
            "optimized_E": run_summary(
                optimized_rows, optimized_scores, load_batch_summary(args.optimized)
            ),
        },
        "paired_optimized_minus_baseline": paired,
        "nested_prefix_convergence": convergence,
        "original_answer_difference_count": sum(
            baseline.get("original_answer") != optimized.get("original_answer")
            for baseline, optimized in zip(baseline_rows, optimized_rows)
        ),
        "selected_bbox_difference_count": sum(
            baseline.get("selected_bbox") != optimized.get("selected_bbox")
            for baseline, optimized in zip(baseline_rows, optimized_rows)
        ),
        "legacy_bbox_difference_count": sum(
            baseline.get("legacy_bbox") != optimized.get("legacy_bbox")
            for baseline, optimized in zip(baseline_rows, optimized_rows)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    case_output = args.case_output or args.output.with_suffix(".csv")
    case_output.parent.mkdir(parents=True, exist_ok=True)
    with case_output.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "question_id",
            "image",
            "question",
            "ground_truth_answers",
            "baseline_selected_bbox",
            "optimized_selected_bbox",
            "optimized_selected_rank",
        ]
        for branch in ("crop_answer", "topk_joint_answer", "focus_answer"):
            fields.extend(
                [
                    f"baseline_{branch}",
                    f"optimized_{branch}",
                    f"baseline_{branch}_score",
                    f"optimized_{branch}_score",
                    f"{branch}_delta",
                ]
            )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (baseline, optimized) in enumerate(
            zip(baseline_rows, optimized_rows)
        ):
            row: dict[str, object] = {
                "question_id": ordered_ids[index],
                "image": baseline.get("image"),
                "question": baseline.get("question"),
                "ground_truth_answers": json.dumps(
                    baseline.get("ground_truth_answers"), ensure_ascii=False
                ),
                "baseline_selected_bbox": json.dumps(baseline.get("selected_bbox")),
                "optimized_selected_bbox": json.dumps(optimized.get("selected_bbox")),
                "optimized_selected_rank": optimized.get("selected_rank"),
            }
            for branch in ("crop_answer", "topk_joint_answer", "focus_answer"):
                baseline_score = baseline_scores[branch][index]
                optimized_score = optimized_scores[branch][index]
                row.update(
                    {
                        f"baseline_{branch}": baseline.get(branch),
                        f"optimized_{branch}": optimized.get(branch),
                        f"baseline_{branch}_score": baseline_score,
                        f"optimized_{branch}_score": optimized_score,
                        f"{branch}_delta": optimized_score - baseline_score,
                    }
                )
            writer.writerow(row)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score paired SLoFo branches on fixed TextVQA, GQA and POPE manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from llava.eval.m4c_evaluator import EvalAIAnswerProcessor, TextVQAAccuracyEvaluator


BRANCHES = {
    "original": "original_answer",
    "crop_only": "crop_answer",
    "legacy_dual": "legacy_joint_answer",
    "topk_dual": "topk_joint_answer",
    "focus": "focus_answer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def normalize_gqa(value: object) -> str:
    return str(value or "").strip().lower().rstrip(".").strip()


def normalize_pope(value: object) -> str:
    text = str(value or "")
    if "." in text:
        text = text.split(".", 1)[0]
    words = text.replace(",", "").split()
    return "no" if any(word in {"No", "no", "not"} for word in words) else "yes"


def binary_metrics(labels: list[str], predictions: list[str]) -> dict[str, float]:
    tp = sum(p == "yes" and y == "yes" for p, y in zip(predictions, labels))
    tn = sum(p == "no" and y == "no" for p, y in zip(predictions, labels))
    fp = sum(p == "yes" and y == "no" for p, y in zip(predictions, labels))
    fn = sum(p == "no" and y == "yes" for p, y in zip(predictions, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": predictions.count("yes") / len(predictions),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def score_textvqa(
    row: dict[str, object], branch_key: str, evaluator: TextVQAAccuracyEvaluator
) -> float:
    prediction = evaluator.answer_processor(str(row.get(branch_key) or ""))
    scores = evaluator._compute_answer_scores(row["ground_truth_answers"])
    return float(scores.get(prediction, 0.0))


def score_gqa(row: dict[str, object], branch_key: str) -> float:
    return float(
        normalize_gqa(row.get(branch_key)) == normalize_gqa(row.get("ground_truth"))
    )


def score_pope(row: dict[str, object], branch_key: str) -> float:
    return float(normalize_pope(row.get(branch_key)) == str(row["ground_truth"]))


def transition_metrics(
    rows: list[dict[str, object]], score_field: str, answer_field: str
) -> dict[str, object]:
    differences = [
        float(row[score_field]) - float(row["score_original"]) for row in rows
    ]
    mean_difference = statistics.fmean(differences)
    standard_error = (
        statistics.stdev(differences) / math.sqrt(len(differences))
        if len(differences) > 1
        else 0.0
    )
    changed = sum(
        str(row.get(answer_field)) != str(row.get(BRANCHES["original"])) for row in rows
    )
    corrected = sum(
        float(row[f"score_original"]) < float(row[score_field]) for row in rows
    )
    regressed = sum(
        float(row[f"score_original"]) > float(row[score_field]) for row in rows
    )
    result: dict[str, object] = {
        "answer_changed": changed,
        "answer_changed_rate": changed / len(rows),
        "score_improved": corrected,
        "score_regressed": regressed,
        "net_improvement": corrected - regressed,
        "paired_score_delta_percentage_points": 100 * mean_difference,
        "paired_score_delta_95ci_percentage_points": [
            100 * (mean_difference - 1.96 * standard_error),
            100 * (mean_difference + 1.96 * standard_error),
        ],
    }
    score_values = [
        float(row["score_original"]) for row in rows
    ] + [float(row[score_field]) for row in rows]
    if all(value in {0.0, 1.0} for value in score_values) and corrected + regressed:
        discordant = corrected + regressed
        tail = sum(
            math.comb(discordant, index)
            for index in range(0, min(corrected, regressed) + 1)
        ) / (2**discordant)
        result["mcnemar_exact_two_sided_p"] = min(1.0, 2 * tail)
    return result


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = {str(row["qa_id"]): row for row in manifest["records"]}
    answers = read_jsonl(args.answers)
    if len(answers) != len(records):
        raise ValueError(f"Expected {len(records)} answers, found {len(answers)}")
    batch_summary = json.loads(args.batch_summary.read_text(encoding="utf-8"))
    benchmark = str(manifest["benchmark"])
    evaluator = TextVQAAccuracyEvaluator()

    rows: list[dict[str, object]] = []
    for answer in answers:
        question_id = str(answer["question_id"])
        record = records[question_id]
        row = dict(answer)
        row["ground_truth"] = record.get("ground_truth")
        row["ground_truth_answers"] = record.get("ground_truth_answers")
        row["category"] = record.get("category")
        for branch_name, branch_key in BRANCHES.items():
            if benchmark == "textvqa":
                score = score_textvqa(row, branch_key, evaluator)
            elif benchmark == "gqa":
                score = score_gqa(row, branch_key)
            elif benchmark == "pope":
                score = score_pope(row, branch_key)
            else:
                raise ValueError(f"Unsupported benchmark: {benchmark}")
            row[f"score_{branch_name}"] = score
        rows.append(row)

    branch_metrics: dict[str, dict[str, object]] = {}
    category_metrics: dict[str, dict[str, float]] = {}
    for branch_name, branch_key in BRANCHES.items():
        score_values = [float(row[f"score_{branch_name}"]) for row in rows]
        metrics: dict[str, object] = {
            "score": statistics.fmean(score_values),
            "score_percent": 100 * statistics.fmean(score_values),
        }
        if benchmark == "pope":
            labels = [str(row["ground_truth"]) for row in rows]
            predictions = [normalize_pope(row.get(branch_key)) for row in rows]
            metrics.update(binary_metrics(labels, predictions))
            metrics["accuracy_percent"] = 100 * float(metrics["accuracy"])
            metrics["f1_percent"] = 100 * float(metrics["f1"])
        branch_metrics[branch_name] = metrics

        grouped_scores: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            grouped_scores[str(row["category"])].append(
                float(row[f"score_{branch_name}"])
            )
        for category, values in grouped_scores.items():
            category_metrics.setdefault(category, {})[branch_name] = 100 * statistics.fmean(
                values
            )

    comparisons = {
        branch_name: transition_metrics(
            rows, f"score_{branch_name}", branch_key
        )
        for branch_name, branch_key in BRANCHES.items()
        if branch_name != "original"
    }
    comparisons["topk_vs_legacy"] = {
        "answer_changed": sum(
            str(row.get(BRANCHES["topk_dual"]))
            != str(row.get(BRANCHES["legacy_dual"]))
            for row in rows
        ),
        "topk_score_improved": sum(
            float(row["score_topk_dual"]) > float(row["score_legacy_dual"])
            for row in rows
        ),
        "topk_score_regressed": sum(
            float(row["score_topk_dual"]) < float(row["score_legacy_dual"])
            for row in rows
        ),
    }
    comparisons["focus_vs_topk"] = {
        "answer_changed": sum(
            str(row.get(BRANCHES["focus"])) != str(row.get(BRANCHES["topk_dual"]))
            for row in rows
        ),
        "focus_score_improved": sum(
            float(row["score_focus"]) > float(row["score_topk_dual"]) for row in rows
        ),
        "focus_score_regressed": sum(
            float(row["score_focus"]) < float(row["score_topk_dual"]) for row in rows
        ),
    }

    elapsed_values = [
        float(row["wall_seconds"])
        for row in batch_summary["cases"]
        if row.get("status") == "completed" and "wall_seconds" in row
    ]
    scan_peak_values = [float(row["scan_peak_allocated_mib"]) for row in rows]
    generation_peak_values = [
        float(row["generation_peak_allocated_mib"]) for row in rows
    ]
    ranks = Counter(str(row.get("selected_rank")) for row in rows)
    localization = {
        "selection_changed": sum(bool(row.get("selection_changed")) for row in rows),
        "selection_changed_rate": sum(bool(row.get("selection_changed")) for row in rows)
        / len(rows),
        "selected_rank_counts": dict(sorted(ranks.items())),
    }
    runtime = {
        "completed_with_timing": len(elapsed_values),
        "total_batch_seconds": float(batch_summary["elapsed_seconds"]),
        "mean_case_seconds": statistics.fmean(elapsed_values),
        "median_case_seconds": statistics.median(elapsed_values),
        "p95_case_seconds": percentile(elapsed_values, 0.95),
        "throughput_questions_per_second": len(rows)
        / float(batch_summary["elapsed_seconds"]),
        "max_scan_peak_mib": max(scan_peak_values),
        "max_generation_peak_mib": max(generation_peak_values),
        "scan_peak_mib": {
            "min": min(scan_peak_values),
            "mean": statistics.fmean(scan_peak_values),
            "median": statistics.median(scan_peak_values),
            "p95": percentile(scan_peak_values, 0.95),
            "max": max(scan_peak_values),
        },
        "generation_peak_mib": {
            "min": min(generation_peak_values),
            "mean": statistics.fmean(generation_peak_values),
            "median": statistics.median(generation_peak_values),
            "p95": percentile(generation_peak_values, 0.95),
            "max": max(generation_peak_values),
        },
    }
    summary = {
        "benchmark": benchmark,
        "sample_count": len(rows),
        "seed": manifest["seed"],
        "sampling": manifest["sampling"],
        "branch_metrics": branch_metrics,
        "category_metrics_percent": dict(sorted(category_metrics.items())),
        "comparisons": comparisons,
        "localization": localization,
        "runtime": runtime,
    }
    (args.output_root / f"{benchmark}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_fields = [
        "benchmark",
        "question_id",
        "image",
        "question",
        "category",
        "ground_truth",
        *BRANCHES.values(),
        *(f"score_{name}" for name in BRANCHES),
        "selected_rank",
        "selection_changed",
        "selected_bbox",
        "legacy_bbox",
    ]
    with (args.output_root / f"{benchmark}_cases.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

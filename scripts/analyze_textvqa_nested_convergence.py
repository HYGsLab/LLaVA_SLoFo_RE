#!/usr/bin/env python3
"""Measure paired TextVQA factor effects on nested manifest prefixes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


BRANCHES = ("crop_answer", "topk_joint_answer", "focus_answer")
COMPARISONS = (
    ("minmax_vs_raw_1token", "t1_raw", "t1_minmax"),
    ("tokens_8_vs_1_minmax", "t1_minmax", "t8_minmax"),
    ("topk5_vs_single_8token_minmax", "t8_minmax", "t8_topk"),
)


def read_jsonl(path: Path) -> dict[str, dict[str, object]]:
    return {
        str(row["question_id"]): row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def paired_ci(differences: list[float]) -> list[float]:
    mean = statistics.fmean(differences)
    if len(differences) < 2:
        return [mean, mean]
    margin = 1.96 * statistics.stdev(differences) / math.sqrt(len(differences))
    return [mean - margin, mean + margin]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=JSONL")
    parser.add_argument("--prefix", action="append", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ordered_ids = [
        str(row["qa_id"])
        for row in json.loads(args.manifest.read_text(encoding="utf-8"))["records"]
    ]
    run_paths = dict(value.split("=", 1) for value in args.run)
    required = {name for _, left, right in COMPARISONS for name in (left, right)}
    if missing := required - run_paths.keys():
        raise ValueError(f"Missing runs: {sorted(missing)}")
    rows = {name: read_jsonl(Path(path)) for name, path in run_paths.items()}
    evaluator = TextVQAAccuracyEvaluator()

    output: dict[str, object] = {"manifest_count": len(ordered_ids), "prefixes": {}}
    for prefix in sorted(set(args.prefix)):
        if prefix <= 0 or prefix > len(ordered_ids):
            raise ValueError(f"Invalid prefix: {prefix}")
        ids = ordered_ids[:prefix]
        prefix_result: dict[str, object] = {}
        for comparison_name, left_name, right_name in COMPARISONS:
            branch_result: dict[str, object] = {}
            for branch in BRANCHES:
                differences: list[float] = []
                answer_changed = 0
                for qa_id in ids:
                    left = rows[left_name][qa_id]
                    right = rows[right_name][qa_id]
                    scores = evaluator._compute_answer_scores(left["ground_truth_answers"])
                    left_answer = evaluator.answer_processor(str(left.get(branch) or ""))
                    right_answer = evaluator.answer_processor(str(right.get(branch) or ""))
                    differences.append(
                        100.0
                        * (
                            scores.get(right_answer, 0.0)
                            - scores.get(left_answer, 0.0)
                        )
                    )
                    answer_changed += left.get(branch) != right.get(branch)
                branch_result[branch] = {
                    "delta_percentage_points": statistics.fmean(differences),
                    "paired_95ci_percentage_points": paired_ci(differences),
                    "answer_changed": answer_changed,
                    "score_improved": sum(value > 0 for value in differences),
                    "score_regressed": sum(value < 0 for value in differences),
                }
            prefix_result[comparison_name] = branch_result
        output["prefixes"][str(prefix)] = prefix_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

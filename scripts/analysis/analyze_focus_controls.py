#!/usr/bin/env python3
"""Aggregate the three matched Focus control groups."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = (
    ROOT
    / "Focus_三组对照实验结果"
    / "experiments"
    / "slofo-08-07"
    / "focus-controls"
)


def aligned_prefix_mean_kl(comparison: dict) -> tuple[float, int]:
    """Use only steps whose decode history is shared, plus the divergence step."""

    if comparison["exact_generated_token_ids"]:
        count = comparison["compared_steps"]
    else:
        count = min(
            comparison["compared_steps"],
            comparison["common_prefix_tokens"] + 1,
        )
    values = [step["symmetric_kl"] for step in comparison["per_step"][:count]]
    return float(np.mean(values)), count


def final_kept_ids(generation: dict) -> set[int]:
    return set(generation["focus"]["stages"][-1]["kept_original_token_ids"])


def jaccard(left: set[int], right: set[int]) -> float:
    return len(left & right) / len(left | right)


def color_signature(answer: str) -> tuple[str, ...]:
    colors = (
        "black|white|gray|grey|blue|green|red|brown|yellow|orange|pink|purple"
    )
    return tuple(re.findall(rf"\b(?:{colors})\b", answer.lower()))


def build_aggregate_chart(summary: dict, output: Path) -> None:
    labels = ["Attention Focus", "Random Focus", "Crop only"]
    values = [
        summary["aggregate"]["attention_focus"]["mean_aligned_symmetric_kl"],
        summary["aggregate"]["random_focus"]["mean_aligned_symmetric_kl"],
        summary["aggregate"]["crop_only"]["mean_aligned_symmetric_kl"],
    ]
    colors = [(51, 120, 209), (237, 125, 49), (112, 173, 71)]
    width, height = 980, 520
    margin_left, margin_bottom, margin_top = 110, 75, 55
    plot_height = height - margin_bottom - margin_top
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 18), "Distribution shift from unpruned original+crop", fill="black")
    draw.text((20, 35), "Mean aligned-prefix symmetric KL (log10 scale)", fill="black")
    log_values = [math.log10(max(value, 1e-8)) for value in values]
    y_min, y_max = -4.5, -0.5
    for tick in (-4, -3, -2, -1):
        y = margin_top + (y_max - tick) / (y_max - y_min) * plot_height
        draw.line((margin_left, y, width - 30, y), fill=(215, 215, 215), width=1)
        draw.text((40, y - 7), f"1e{tick}", fill="black")
    bar_width = 180
    gap = 75
    for index, (label, value, log_value, color) in enumerate(
        zip(labels, values, log_values, colors)
    ):
        x1 = margin_left + 80 + index * (bar_width + gap)
        x2 = x1 + bar_width
        y = margin_top + (y_max - log_value) / (y_max - y_min) * plot_height
        baseline = margin_top + (y_max - y_min) / (y_max - y_min) * plot_height
        draw.rectangle((x1, y, x2, baseline), fill=color)
        draw.text((x1, baseline + 12), label, fill="black")
        draw.text((x1, max(margin_top, y - 18)), f"{value:.6f}", fill="black")
    image.save(output)


def main() -> None:
    rows: list[dict] = []
    attention_kls: list[float] = []
    crop_kls: list[float] = []
    random_kls: list[float] = []
    attention_top1: list[float] = []
    crop_top1: list[float] = []
    random_top1: list[float] = []
    attention_jaccards: list[float] = []
    random_pair_jaccards: list[float] = []
    attention_answer_matches = 0
    crop_answer_matches = 0
    random_answer_matches = 0
    attention_color_matches = 0
    crop_color_matches = 0
    random_color_matches = 0
    attention_below_random_pairs = 0
    attention_below_case_random_mean = 0

    for case_dir in sorted(path for path in RESULT_ROOT.iterdir() if path.is_dir()):
        result = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
        comparisons = result["logit_comparisons"]
        attention_comparison = comparisons["joint_vs_attention_focus"]
        crop_comparison = comparisons["joint_vs_crop_only"]
        attention_kl, attention_steps = aligned_prefix_mean_kl(attention_comparison)
        crop_kl, crop_steps = aligned_prefix_mean_kl(crop_comparison)
        attention_kls.append(attention_kl)
        crop_kls.append(crop_kl)
        attention_top1.append(attention_comparison["top1_agreement_fraction"])
        crop_top1.append(crop_comparison["top1_agreement_fraction"])

        joint_answer = result["joint_answer"]
        attention_answer = result["focused_joint_answer"]
        crop_answer = result["crop_answer"]
        attention_match = attention_answer == joint_answer
        crop_match = crop_answer == joint_answer
        attention_answer_matches += attention_match
        crop_answer_matches += crop_match
        joint_colors = color_signature(joint_answer)
        attention_color_match = color_signature(attention_answer) == joint_colors
        crop_color_match = color_signature(crop_answer) == joint_colors
        attention_color_matches += attention_color_match
        crop_color_matches += crop_color_match

        attention_generation = result["generation"]["original_plus_crop_focus"]
        attention_final = final_kept_ids(attention_generation)
        random_rows: dict[str, dict] = {}
        random_final_sets: list[set[int]] = []
        case_random_kls: list[float] = []
        for seed, random_generation in result["generation"][
            "original_plus_crop_random_focus"
        ].items():
            comparison = comparisons[f"joint_vs_random_focus_seed_{seed}"]
            random_kl, random_steps = aligned_prefix_mean_kl(comparison)
            case_random_kls.append(random_kl)
            random_kls.append(random_kl)
            random_top1.append(comparison["top1_agreement_fraction"])
            random_answer = result["random_focused_joint_answers"][seed]
            answer_match = random_answer == joint_answer
            color_match = color_signature(random_answer) == joint_colors
            random_answer_matches += answer_match
            random_color_matches += color_match
            if attention_kl < random_kl:
                attention_below_random_pairs += 1
            random_final = final_kept_ids(random_generation)
            random_final_sets.append(random_final)
            overlap = jaccard(attention_final, random_final)
            attention_jaccards.append(overlap)
            random_rows[seed] = {
                "answer": random_answer,
                "answer_matches_joint": answer_match,
                "color_matches_joint": color_match,
                "aligned_symmetric_kl": random_kl,
                "aligned_steps": random_steps,
                "top1_agreement_fraction": comparison["top1_agreement_fraction"],
                "attention_random_final_jaccard": overlap,
            }
        for left_index in range(len(random_final_sets)):
            for right_index in range(left_index + 1, len(random_final_sets)):
                random_pair_jaccards.append(
                    jaccard(
                        random_final_sets[left_index],
                        random_final_sets[right_index],
                    )
                )
        case_random_mean = float(np.mean(case_random_kls))
        if attention_kl < case_random_mean:
            attention_below_case_random_mean += 1
        rows.append(
            {
                "case_id": case_dir.name,
                "joint_answer": joint_answer,
                "crop_only": {
                    "answer": crop_answer,
                    "answer_matches_joint": crop_match,
                    "color_matches_joint": crop_color_match,
                    "aligned_symmetric_kl": crop_kl,
                    "aligned_steps": crop_steps,
                    "top1_agreement_fraction": crop_comparison[
                        "top1_agreement_fraction"
                    ],
                },
                "attention_focus": {
                    "answer": attention_answer,
                    "answer_matches_joint": attention_match,
                    "color_matches_joint": attention_color_match,
                    "aligned_symmetric_kl": attention_kl,
                    "aligned_steps": attention_steps,
                    "top1_agreement_fraction": attention_comparison[
                        "top1_agreement_fraction"
                    ],
                },
                "random_focus": random_rows,
                "random_mean_aligned_symmetric_kl": case_random_mean,
            }
        )

    summary = {
        "design": {
            "case_count": len(rows),
            "random_seeds": [0, 1, 2],
            "random_trial_count": len(random_kls),
            "token_schedule": [576, 288, 144, 72],
            "aligned_kl_definition": (
                "Mean symmetric KL over the common generated-token history, "
                "including the first divergence decision step."
            ),
        },
        "aggregate": {
            "attention_focus": {
                "exact_answer_matches": attention_answer_matches,
                "exact_answer_trials": len(rows),
                "color_matches": attention_color_matches,
                "color_trials": len(rows),
                "mean_aligned_symmetric_kl": float(np.mean(attention_kls)),
                "median_aligned_symmetric_kl": float(np.median(attention_kls)),
                "mean_top1_agreement_fraction": float(np.mean(attention_top1)),
            },
            "random_focus": {
                "exact_answer_matches": random_answer_matches,
                "exact_answer_trials": len(random_kls),
                "color_matches": random_color_matches,
                "color_trials": len(random_kls),
                "mean_aligned_symmetric_kl": float(np.mean(random_kls)),
                "median_aligned_symmetric_kl": float(np.median(random_kls)),
                "mean_top1_agreement_fraction": float(np.mean(random_top1)),
                "attention_kl_lower_than_random_pair_count": (
                    attention_below_random_pairs
                ),
                "paired_trial_count": len(random_kls),
                "attention_kl_lower_than_case_random_mean_count": (
                    attention_below_case_random_mean
                ),
                "case_count": len(rows),
            },
            "crop_only": {
                "exact_answer_matches": crop_answer_matches,
                "exact_answer_trials": len(rows),
                "color_matches": crop_color_matches,
                "color_trials": len(rows),
                "mean_aligned_symmetric_kl": float(np.mean(crop_kls)),
                "median_aligned_symmetric_kl": float(np.median(crop_kls)),
                "mean_top1_agreement_fraction": float(np.mean(crop_top1)),
            },
            "retained_token_overlap": {
                "mean_attention_vs_random_final_jaccard": float(
                    np.mean(attention_jaccards)
                ),
                "mean_random_vs_random_final_jaccard": float(
                    np.mean(random_pair_jaccards)
                ),
                "mean_attention_vs_random_final_intersection": float(
                    np.mean(
                        [144.0 * value / (1.0 + value) for value in attention_jaccards]
                    )
                ),
                "mean_random_vs_random_final_intersection": float(
                    np.mean(
                        [144.0 * value / (1.0 + value) for value in random_pair_jaccards]
                    )
                ),
                "expected_random_intersection_tokens": 9.0,
            },
        },
        "rows": rows,
    }
    output = ROOT / "Focus_三组对照实验结果" / "focus_controls_summary.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    build_aggregate_chart(
        summary,
        ROOT / "Focus_三组对照实验结果" / "focus_controls_kl.png",
    )


if __name__ == "__main__":
    main()

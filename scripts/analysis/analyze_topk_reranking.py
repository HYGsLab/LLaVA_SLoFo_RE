#!/usr/bin/env python3
"""Summarize the 20-case SLoFo top-k localization and answer experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MANUAL_ANSWER_JUDGMENTS = {
    "topk_08_07_01": ("pass", "pink and white are visible dominant dress colors"),
    "topk_08_07_02": ("pass", "gray cardigan is correctly identified"),
    "topk_08_07_03": ("pass", "green is a valid main camouflage color"),
    "topk_08_07_04": ("pass", "blue/plaid clothing is correctly identified"),
    "topk_08_07_05": ("pass", "gray shirt is correctly identified"),
    "topk_08_07_06": ("fail", "answer says green; clothing is dark navy/black with tan apron"),
    "topk_08_07_07": ("pass", "dark blue patterned shirt is correctly identified as blue"),
    "topk_08_07_08": ("pass", "white shirt is correctly identified"),
    "topk_08_07_09": ("pass", "black clothing is correctly identified"),
    "topk_08_07_10": ("pass", "pink is a visible dominant part of the white/pink dress"),
    "topk_08_07_11": ("pass", "red jacket is correctly identified"),
    "topk_08_07_12": ("pass", "dark navy top is correctly identified as blue"),
    "topk_08_07_13": ("pass", "small target's red clothing is correctly identified"),
    "topk_08_07_14": ("pass", "white tennis outfit is correctly identified"),
    "topk_08_07_15": ("pass", "light blue shirt is correctly identified as blue"),
    "topk_08_07_16": ("partial", "black pants are visible, but white/gray top is omitted"),
    "topk_08_07_17": ("pass", "very dark suit is correctly identified as black"),
    "topk_08_07_18": ("partial", "blue pattern is visible, but dominant white/pink is omitted"),
    "topk_08_07_19": ("pass", "blue shirt is correctly identified"),
    "topk_08_07_20": ("pass", "dark top is correctly identified as black"),
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_by_id = {record["qa_id"]: record for record in manifest["records"]}
    rows: list[dict[str, object]] = []
    for case_dir in sorted(args.results_root.iterdir()):
        if not case_dir.is_dir():
            continue
        result_path = case_dir / "result.json"
        if not result_path.is_file():
            continue
        case_id = case_dir.name
        result = json.loads(result_path.read_text(encoding="utf-8"))
        reranking = result["topk_reranking"]
        candidates = reranking["candidates"]
        legacy_eval = reranking["legacy_evaluation"]
        selected_eval = reranking["selected_evaluation"]
        oracle = max(
            (candidate["evaluation"] for candidate in candidates),
            key=lambda evaluation: (
                bool(evaluation["target_center_inside"]),
                float(evaluation["target_coverage"]),
                float(evaluation["iou"]),
            ),
        )
        judgment, note = MANUAL_ANSWER_JUDGMENTS[case_id]
        expected = expected_by_id[case_id]
        rows.append(
            {
                "case_id": case_id,
                "dataset_index": expected["dataset_index"],
                "query_zh": expected["query_zh"],
                "expected_answer_zh": expected["expected_answer_zh"],
                "expected_answer_en": expected["expected_answer_en"],
                "legacy_bbox": result["legacy_original_bbox"],
                "selected_bbox": result["original_bbox"],
                "selected_rank": reranking["selected_rank"],
                "selection_changed": reranking["selection_changed"],
                "unconstrained_best_rank": reranking["unconstrained_best_rank"],
                "conservative_fallback": reranking["conservative_fallback_applied"],
                "candidate_count": reranking["actual_candidate_count"],
                "legacy_center_hit": legacy_eval["target_center_inside"],
                "selected_center_hit": selected_eval["target_center_inside"],
                "oracle_center_hit": oracle["target_center_inside"],
                "legacy_iou": legacy_eval["iou"],
                "selected_iou": selected_eval["iou"],
                "oracle_iou": oracle["iou"],
                "legacy_target_coverage": legacy_eval["target_coverage"],
                "selected_target_coverage": selected_eval["target_coverage"],
                "oracle_target_coverage": oracle["target_coverage"],
                "original_answer": result["original_answer"],
                "legacy_joint_answer": result["baseline_joint_answer"],
                "reranked_joint_answer": result["reranked_joint_answer"],
                "answer_changed": (
                    result["baseline_joint_answer"] != result["reranked_joint_answer"]
                ),
                "manual_answer_judgment": judgment,
                "manual_answer_note": note,
                "scan_peak_mib": result["scan_peak_allocated_mib"],
                "generation_peak_mib": result["generation_peak_allocated_mib"],
            }
        )
    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 result rows, found {len(rows)}")

    passes = sum(row["manual_answer_judgment"] == "pass" for row in rows)
    partials = sum(row["manual_answer_judgment"] == "partial" for row in rows)
    failures = sum(row["manual_answer_judgment"] == "fail" for row in rows)
    summary = {
        "case_count": len(rows),
        "candidate_count_mean": mean([float(row["candidate_count"]) for row in rows]),
        "selection_changed_count": sum(bool(row["selection_changed"]) for row in rows),
        "unconstrained_nonlegacy_best_count": sum(
            int(row["unconstrained_best_rank"]) != 1 for row in rows
        ),
        "conservative_fallback_count": sum(
            bool(row["conservative_fallback"]) for row in rows
        ),
        "legacy_center_hits": sum(bool(row["legacy_center_hit"]) for row in rows),
        "selected_center_hits": sum(bool(row["selected_center_hit"]) for row in rows),
        "oracle_topk_center_hits": sum(bool(row["oracle_center_hit"]) for row in rows),
        "legacy_mean_iou": mean([float(row["legacy_iou"]) for row in rows]),
        "selected_mean_iou": mean([float(row["selected_iou"]) for row in rows]),
        "oracle_topk_mean_iou": mean([float(row["oracle_iou"]) for row in rows]),
        "legacy_mean_target_coverage": mean(
            [float(row["legacy_target_coverage"]) for row in rows]
        ),
        "selected_mean_target_coverage": mean(
            [float(row["selected_target_coverage"]) for row in rows]
        ),
        "oracle_topk_mean_target_coverage": mean(
            [float(row["oracle_target_coverage"]) for row in rows]
        ),
        "answer_changed_count": sum(bool(row["answer_changed"]) for row in rows),
        "manual_answer_pass": passes,
        "manual_answer_partial": partials,
        "manual_answer_fail": failures,
        "strict_answer_accuracy": passes / len(rows),
        "partial_credit_answer_accuracy": (passes + 0.5 * partials) / len(rows),
        "lenient_answer_accuracy": (passes + partials) / len(rows),
        "max_scan_peak_mib": max(float(row["scan_peak_mib"]) for row in rows),
        "max_generation_peak_mib": max(float(row["generation_peak_mib"]) for row in rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "topk_rerank_summary.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "topk_rerank_cases.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = ["Target-center hit rate", "Mean target coverage", "Mean IoU"]
    legacy = [
        summary["legacy_center_hits"] / len(rows),
        summary["legacy_mean_target_coverage"],
        summary["legacy_mean_iou"],
    ]
    selected = [
        summary["selected_center_hits"] / len(rows),
        summary["selected_mean_target_coverage"],
        summary["selected_mean_iou"],
    ]
    oracle_values = [
        summary["oracle_topk_center_hits"] / len(rows),
        summary["oracle_topk_mean_target_coverage"],
        summary["oracle_topk_mean_iou"],
    ]
    try:
        import matplotlib.pyplot as plt

        positions = range(len(labels))
        width = 0.25
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.bar([x - width for x in positions], legacy, width, label="Legacy top-1")
        axis.bar(positions, selected, width, label="Conservative reranked")
        axis.bar([x + width for x in positions], oracle_values, width, label="Top-k oracle")
        axis.set_xticks(list(positions), labels)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Score")
        axis.set_title("SLoFo top-k localization comparison (20 images)")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(args.output_dir / "topk_localization_comparison.png", dpi=180)
        plt.close(figure)
    except ImportError:
        from PIL import Image, ImageDraw, ImageFont

        canvas = Image.new("RGB", (1200, 700), "white")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        colors = ((70, 120, 210), (40, 170, 100), (235, 155, 55))
        series = (
            ("Legacy top-1", legacy),
            ("Conservative reranked", selected),
            ("Top-k oracle", oracle_values),
        )
        chart_left, chart_top, chart_bottom = 120, 90, 600
        chart_height = chart_bottom - chart_top
        draw.text((360, 25), "SLoFo top-k localization comparison (20 images)", fill="black", font=font)
        draw.line((chart_left, chart_top, chart_left, chart_bottom), fill="black", width=2)
        draw.line((chart_left, chart_bottom, 1140, chart_bottom), fill="black", width=2)
        for tick in range(6):
            value = tick / 5
            y = chart_bottom - round(value * chart_height)
            draw.line((chart_left, y, 1140, y), fill=(220, 220, 220), width=1)
            draw.text((70, y - 7), f"{value:.1f}", fill="black", font=font)
        group_width = 300
        bar_width = 58
        for metric_index, label in enumerate(labels):
            center = chart_left + 180 + metric_index * group_width
            for series_index, (_, values) in enumerate(series):
                value = float(values[metric_index])
                x1 = center + (series_index - 1) * (bar_width + 10)
                y1 = chart_bottom - round(value * chart_height)
                draw.rectangle((x1, y1, x1 + bar_width, chart_bottom), fill=colors[series_index])
                draw.text((x1 + 8, y1 - 18), f"{value:.3f}", fill="black", font=font)
            draw.text((center - 80, chart_bottom + 22), label, fill="black", font=font)
        for series_index, (name, _) in enumerate(series):
            x = 310 + series_index * 230
            draw.rectangle((x, 650, x + 20, 670), fill=colors[series_index])
            draw.text((x + 28, 652), name, fill="black", font=font)
        canvas.save(args.output_dir / "topk_localization_comparison.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

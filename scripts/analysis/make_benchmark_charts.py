#!/usr/bin/env python3
"""Create dependency-free SVG charts from the three benchmark summaries."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIGURE_ROOT = ROOT / "analysis" / "figures"
BRANCHES = ["original", "crop_only", "legacy_dual", "topk_dual", "focus"]
LABELS = {
    "original": "原图",
    "crop_only": "仅裁切",
    "legacy_dual": "旧双图",
    "topk_dual": "top-k双图",
    "focus": "Focus",
}
COLORS = {
    "original": "#667085",
    "crop_only": "#8E6CEF",
    "legacy_dual": "#4C78A8",
    "topk_dual": "#F58518",
    "focus": "#2EAD72",
}


def load_summary(name: str) -> dict[str, object]:
    path = ROOT / "analysis" / name / f"{name}_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def text(x: float, y: float, value: object, size: int = 16, anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Microsoft YaHei,Noto Sans CJK SC,sans-serif" '
        f'font-size="{size}" fill="#1D2939">{html.escape(str(value))}</text>'
    )


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        text(width / 2, 36, title, 22),
    ]


def write_branch_scores(summaries: dict[str, dict[str, object]]) -> None:
    width, height = 1120, 640
    left, right, top, bottom = 90, 30, 75, 115
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [
        float(summary["branch_metrics"][branch]["score_percent"])
        for summary in summaries.values()
        for branch in BRANCHES
    ]
    y_min = math.floor((min(values) - 5) / 10) * 10
    y_max = math.ceil((max(values) + 2) / 10) * 10
    y_min = max(0, y_min)
    y_max = min(100, y_max)
    lines = svg_start(width, height, "固定子集：各推理分支得分（%）")
    for tick in range(int(y_min), int(y_max) + 1, 5):
        y = top + (y_max - tick) / (y_max - y_min) * plot_h
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#EAECF0"/>')
        lines.append(text(left - 12, y + 5, tick, 13, "end"))

    names = list(summaries)
    group_w = plot_w / len(names)
    bar_w = group_w * 0.13
    for group_index, name in enumerate(names):
        center = left + group_w * (group_index + 0.5)
        for branch_index, branch in enumerate(BRANCHES):
            value = float(summaries[name]["branch_metrics"][branch]["score_percent"])
            x = center + (branch_index - 2) * bar_w - bar_w * 0.42
            y = top + (y_max - value) / (y_max - y_min) * plot_h
            baseline = top + plot_h
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.84:.1f}" height="{baseline-y:.1f}" rx="3" fill="{COLORS[branch]}"/>'
            )
            lines.append(text(x + bar_w * 0.42, y - 7, f"{value:.2f}", 12))
        lines.append(text(center, top + plot_h + 32, name.upper(), 17))

    legend_y = height - 48
    for index, branch in enumerate(BRANCHES):
        x = 230 + index * 150
        lines.append(f'<rect x="{x}" y="{legend_y-14}" width="18" height="18" rx="3" fill="{COLORS[branch]}"/>')
        lines.append(text(x + 26, legend_y + 1, LABELS[branch], 14, "start"))
    lines.append('</svg>')
    (FIGURE_ROOT / "01_branch_scores.svg").write_text("\n".join(lines), encoding="utf-8")


def write_deltas(summaries: dict[str, dict[str, object]]) -> None:
    width, height = 1120, 620
    left, right, top, bottom = 90, 30, 75, 110
    plot_w, plot_h = width - left - right, height - top - bottom
    branches = ["crop_only", "legacy_dual", "topk_dual", "focus"]
    deltas = {
        name: {
            branch: float(summary["branch_metrics"][branch]["score_percent"])
            - float(summary["branch_metrics"]["original"]["score_percent"])
            for branch in branches
        }
        for name, summary in summaries.items()
    }
    all_values = [value for values in deltas.values() for value in values.values()]
    bound = max(3.0, math.ceil(max(abs(value) for value in all_values) + 0.5))
    y_min, y_max = -bound, bound
    lines = svg_start(width, height, "相对原图的配对得分变化（百分点）")
    for tick in range(int(y_min), int(y_max) + 1):
        y = top + (y_max - tick) / (y_max - y_min) * plot_h
        stroke = "#98A2B3" if tick == 0 else "#EAECF0"
        stroke_width = 2 if tick == 0 else 1
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="{stroke}" stroke-width="{stroke_width}"/>')
        lines.append(text(left - 12, y + 5, tick, 13, "end"))
    names = list(summaries)
    group_w = plot_w / len(names)
    bar_w = group_w * 0.16
    zero_y = top + y_max / (y_max - y_min) * plot_h
    for group_index, name in enumerate(names):
        center = left + group_w * (group_index + 0.5)
        for branch_index, branch in enumerate(branches):
            value = deltas[name][branch]
            x = center + (branch_index - 1.5) * bar_w - bar_w * 0.42
            value_y = top + (y_max - value) / (y_max - y_min) * plot_h
            y = min(value_y, zero_y)
            height_value = abs(value_y - zero_y)
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.84:.1f}" height="{height_value:.1f}" rx="3" fill="{COLORS[branch]}"/>'
            )
            label_y = value_y - 8 if value >= 0 else value_y + 18
            lines.append(text(x + bar_w * 0.42, label_y, f"{value:+.2f}", 12))
        lines.append(text(center, top + plot_h + 32, name.upper(), 17))
    legend_y = height - 43
    for index, branch in enumerate(branches):
        x = 280 + index * 160
        lines.append(f'<rect x="{x}" y="{legend_y-14}" width="18" height="18" rx="3" fill="{COLORS[branch]}"/>')
        lines.append(text(x + 26, legend_y + 1, LABELS[branch], 14, "start"))
    lines.append('</svg>')
    (FIGURE_ROOT / "02_delta_from_original.svg").write_text("\n".join(lines), encoding="utf-8")


def write_pope_categories(summary: dict[str, object]) -> None:
    width, height = 1000, 610
    left, right, top, bottom = 90, 30, 75, 110
    plot_w, plot_h = width - left - right, height - top - bottom
    categories = ["adversarial", "popular", "random"]
    values = summary["category_metrics_percent"]
    y_min, y_max = 75.0, 100.0
    lines = svg_start(width, height, "POPE：三种设置下的 Accuracy（%）")
    for tick in range(75, 101, 5):
        y = top + (y_max - tick) / (y_max - y_min) * plot_h
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#EAECF0"/>')
        lines.append(text(left - 12, y + 5, tick, 13, "end"))
    group_w = plot_w / len(categories)
    bar_w = group_w * 0.13
    for group_index, category in enumerate(categories):
        center = left + group_w * (group_index + 0.5)
        for branch_index, branch in enumerate(BRANCHES):
            value = float(values[category][branch])
            x = center + (branch_index - 2) * bar_w - bar_w * 0.42
            y = top + (y_max - value) / (y_max - y_min) * plot_h
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.84:.1f}" height="{top+plot_h-y:.1f}" rx="3" fill="{COLORS[branch]}"/>'
            )
        lines.append(text(center, top + plot_h + 32, category, 17))
    legend_y = height - 43
    for index, branch in enumerate(BRANCHES):
        x = 170 + index * 145
        lines.append(f'<rect x="{x}" y="{legend_y-14}" width="18" height="18" rx="3" fill="{COLORS[branch]}"/>')
        lines.append(text(x + 25, legend_y + 1, LABELS[branch], 14, "start"))
    lines.append('</svg>')
    (FIGURE_ROOT / "03_pope_categories.svg").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = {name: load_summary(name) for name in ("textvqa", "gqa", "pope")}
    write_branch_scores(summaries)
    write_deltas(summaries)
    write_pope_categories(summaries["pope"])
    print(FIGURE_ROOT)


if __name__ == "__main__":
    main()

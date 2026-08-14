from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).parent
ANALYSIS = json.loads(
    (ROOT / "textvqa5000_AE_compare.json").read_text(encoding="utf-8")
)
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

BLUE = "#4878CF"
ORANGE = "#EE854A"
GREEN = "#55A868"
RED = "#C44E52"
GRAY = "#6C757D"
COLORS = (BLUE, ORANGE, GREEN, RED, GRAY)
FONT = "Microsoft YaHei, Noto Sans CJK SC, sans-serif"


def svg_start(width: int, height: int, title: str, subtitle: str = "") -> list[str]:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="{FONT}" font-size="20" font-weight="bold">{html.escape(title)}</text>',
    ]
    if subtitle:
        lines.append(
            f'<text x="{width / 2}" y="54" text-anchor="middle" font-family="{FONT}" font-size="12" fill="#555">{html.escape(subtitle)}</text>'
        )
    return lines


def write_svg(name: str, lines: list[str]) -> None:
    lines.append("</svg>")
    (FIGURES / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def y_position(value: float, y_min: float, y_max: float, top: float, height: float) -> float:
    return top + height * (y_max - value) / (y_max - y_min)


# 1. Branch accuracy for baseline A and optimized E.
runs = ANALYSIS["runs"]
branch_specs = [
    ("original_answer", "原图"),
    ("crop_answer", "裁切图"),
    ("topk_joint_answer", "双图联合"),
    ("focus_answer", "Focus"),
]
lines = svg_start(
    1080,
    620,
    "TextVQA 5,000 题：A/E 分支准确率",
    "A=1 Token + raw + 单框；E=8 Token + min-max + top-k=5",
)
left, top, plot_w, plot_h = 95, 82, 900, 425
all_accuracy = [
    float(runs[run]["branch_accuracy_percent"][branch])
    for run in ("baseline_A", "optimized_E")
    for branch, _ in branch_specs
]
y_min = max(0.0, 5.0 * int((min(all_accuracy) - 5.0) / 5.0))
y_max = min(100.0, 5.0 * (int((max(all_accuracy) + 5.0) / 5.0) + 1))
for tick in range(int(y_min), int(y_max) + 1, 5):
    y = y_position(tick, y_min, y_max, top, plot_h)
    lines.extend(
        [
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e1e1e1"/>',
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="{FONT}" font-size="12">{tick}</text>',
        ]
    )
group_w = plot_w / len(branch_specs)
bar_w = 68
for index, (branch, label) in enumerate(branch_specs):
    center = left + group_w * (index + 0.5)
    for run_index, (run, color, run_label) in enumerate(
        (("baseline_A", BLUE, "A"), ("optimized_E", ORANGE, "E"))
    ):
        value = float(runs[run]["branch_accuracy_percent"][branch])
        x = center + (-0.55 if run_index == 0 else 0.55) * bar_w - bar_w / 2
        y = y_position(value, y_min, y_max, top, plot_h)
        lines.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{top + plot_h - y:.1f}" fill="{color}"/>',
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="{FONT}" font-size="12">{value:.2f}</text>',
            ]
        )
    lines.append(
        f'<text x="{center:.1f}" y="535" text-anchor="middle" font-family="{FONT}" font-size="14">{html.escape(label)}</text>'
    )
lines.extend(
    [
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="24" y="300" transform="rotate(-90 24 300)" text-anchor="middle" font-family="{FONT}" font-size="14">TextVQA accuracy (%)</text>',
        f'<rect x="390" y="572" width="18" height="18" fill="{BLUE}"/><text x="416" y="586" font-family="{FONT}" font-size="13">A 基线</text>',
        f'<rect x="535" y="572" width="18" height="18" fill="{ORANGE}"/><text x="561" y="586" font-family="{FONT}" font-size="13">E 完整方案</text>',
    ]
)
write_svg("01_textvqa5000_accuracy.svg", lines)


# 2. Paired E-A effect as the nested prefix grows.
convergence = ANALYSIS["nested_prefix_convergence"]
prefixes = sorted(int(value) for value in convergence)
branch_series = [
    ("crop_answer", "裁切图", BLUE),
    ("topk_joint_answer", "双图联合", GREEN),
    ("focus_answer", "Focus", ORANGE),
]
values_and_ci = []
for prefix in prefixes:
    for branch, _, _ in branch_series:
        result = convergence[str(prefix)][branch]
        values_and_ci.extend(result["paired_95ci_percentage_points"])
y_min = min(-1.0, float(min(values_and_ci)) - 0.5)
y_max = max(1.0, float(max(values_and_ci)) + 0.5)
lines = svg_start(
    1080,
    620,
    "E−A 整体效果随样本量的收敛趋势",
    "误差棒为配对 95% 置信区间；该差值包含三个同时变化的工程因素",
)
left, top, plot_w, plot_h = 100, 85, 900, 410
tick_low, tick_high = int(y_min), int(y_max) + 1
for tick in range(tick_low, tick_high + 1):
    y = y_position(tick, y_min, y_max, top, plot_h)
    stroke = "#777" if tick == 0 else "#e1e1e1"
    width = 1.5 if tick == 0 else 1
    lines.extend(
        [
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{stroke}" stroke-width="{width}"/>',
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="{FONT}" font-size="12">{tick:+d}</text>',
        ]
    )
for branch, label, color in branch_series:
    points: list[tuple[float, float, float]] = []
    for index, prefix in enumerate(prefixes):
        result = convergence[str(prefix)][branch]
        value = float(result["optimized_minus_baseline_percentage_points"])
        low, high = (float(v) for v in result["paired_95ci_percentage_points"])
        x = left + plot_w * index / (len(prefixes) - 1)
        y = y_position(value, y_min, y_max, top, plot_h)
        low_y = y_position(low, y_min, y_max, top, plot_h)
        high_y = y_position(high, y_min, y_max, top, plot_h)
        lines.extend(
            [
                f'<line x1="{x:.1f}" y1="{high_y:.1f}" x2="{x:.1f}" y2="{low_y:.1f}" stroke="{color}" stroke-opacity="0.45"/>',
                f'<line x1="{x - 4:.1f}" y1="{high_y:.1f}" x2="{x + 4:.1f}" y2="{high_y:.1f}" stroke="{color}" stroke-opacity="0.45"/>',
                f'<line x1="{x - 4:.1f}" y1="{low_y:.1f}" x2="{x + 4:.1f}" y2="{low_y:.1f}" stroke="{color}" stroke-opacity="0.45"/>',
            ]
        )
        points.append((x, y, value))
    lines.append(
        f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)}" fill="none" stroke="{color}" stroke-width="3"/>'
    )
    for x, y, value in points:
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
for index, prefix in enumerate(prefixes):
    x = left + plot_w * index / (len(prefixes) - 1)
    lines.append(
        f'<text x="{x:.1f}" y="522" text-anchor="middle" font-family="{FONT}" font-size="12">{prefix}</text>'
    )
for index, (_, label, color) in enumerate(branch_series):
    x = 280 + index * 190
    lines.extend(
        [
            f'<line x1="{x}" y1="570" x2="{x + 26}" y2="570" stroke="{color}" stroke-width="3"/>',
            f'<text x="{x + 35}" y="575" font-family="{FONT}" font-size="13">{html.escape(label)}</text>',
        ]
    )
lines.extend(
    [
        f'<text x="{left + plot_w / 2}" y="550" text-anchor="middle" font-family="{FONT}" font-size="14">嵌套样本数（题）</text>',
        f'<text x="25" y="300" transform="rotate(-90 25 300)" text-anchor="middle" font-family="{FONT}" font-size="14">E−A 得分变化（百分点）</text>',
    ]
)
write_svg("02_textvqa5000_convergence.svg", lines)


# 3. Runtime and memory use comparable per-case median rather than interrupted total time.
baseline = runs["baseline_A"]
optimized = runs["optimized_E"]
runtime_values = [
    float(baseline["runtime"]["completed_case_seconds"]["median"]),
    float(optimized["runtime"]["completed_case_seconds"]["median"]),
]
memory_values = [
    float(baseline["scan_peak_mib"]["max"]) / 1024.0,
    float(optimized["scan_peak_mib"]["max"]) / 1024.0,
]
lines = svg_start(
    1050,
    570,
    "A/E 运行效率与显存",
    "A 曾断点续跑，因此耗时采用可比较的单题中位数；显存采用 Scan 最大峰值",
)
for panel, (values, title, unit, color) in enumerate(
    (
        (runtime_values, "单题耗时中位数", "s", BLUE),
        (memory_values, "Scan 最大峰值显存", "GiB", ORANGE),
    )
):
    left = 80 + panel * 510
    top, plot_w, plot_h = 110, 390, 325
    maximum = max(values) * 1.25
    lines.append(
        f'<text x="{left + plot_w / 2}" y="88" text-anchor="middle" font-family="{FONT}" font-size="16">{html.escape(title)}</text>'
    )
    for index, (label, value) in enumerate(zip(("A", "E"), values)):
        x = left + 68 + index * 165
        y = top + plot_h * (maximum - value) / maximum
        fill = color if index == 0 else GREEN
        lines.extend(
            [
                f'<rect x="{x}" y="{y:.1f}" width="85" height="{top + plot_h - y:.1f}" fill="{fill}"/>',
                f'<text x="{x + 42.5}" y="{y - 9:.1f}" text-anchor="middle" font-family="{FONT}" font-size="13">{value:.2f} {unit}</text>',
                f'<text x="{x + 42.5}" y="463" text-anchor="middle" font-family="{FONT}" font-size="15">{label}</text>',
            ]
        )
    lines.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>'
    )
lines.append(
    f'<text x="525" y="525" text-anchor="middle" font-family="{FONT}" font-size="13" fill="#555">两组均在真实 24 GiB RTX 3090 空卡上运行</text>'
)
write_svg("03_textvqa5000_runtime_memory.svg", lines)


# 4. Optimized top-k selected-rank distribution.
rank_counts = {
    int(rank): int(count)
    for rank, count in optimized["selected_rank_counts"].items()
    if rank not in {"None", "null"}
}
lines = svg_start(
    920,
    560,
    "E 配置的 Top-k 候选框最终排名分布",
    "rank=1 表示沿用原始最高分框；rank>1 表示重排改变了最终裁切框",
)
left, top, plot_w, plot_h = 100, 90, 730, 350
maximum = max(rank_counts.values())
for rank in sorted(rank_counts):
    count = rank_counts[rank]
    index = rank - 1
    bar_w = 92
    gap = (plot_w - bar_w * len(rank_counts)) / max(1, len(rank_counts) - 1)
    x = left + index * (bar_w + gap)
    y = top + plot_h * (maximum - count) / maximum
    lines.extend(
        [
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{top + plot_h - y:.1f}" fill="{COLORS[index % len(COLORS)]}"/>',
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="{FONT}" font-size="13">{count}</text>',
            f'<text x="{x + bar_w / 2:.1f}" y="470" text-anchor="middle" font-family="{FONT}" font-size="14">rank {rank}</text>',
        ]
    )
lines.extend(
    [
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="25" y="275" transform="rotate(-90 25 275)" text-anchor="middle" font-family="{FONT}" font-size="14">题目数</text>',
        f'<text x="460" y="522" text-anchor="middle" font-family="{FONT}" font-size="13" fill="#555">selection_changed = {optimized["selection_changed_count"]} / {ANALYSIS["sample_count"]}</text>',
    ]
)
write_svg("04_textvqa5000_topk_ranks.svg", lines)

print(FIGURES)

#!/usr/bin/env python3
"""Rebuild the 08-07 SLoFo artifacts from compact JSON/NumPy results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT
    / "08_07"
    / "SLoFo_08_07_实验结果"
    / "experiments"
    / "slofo-08-07"
    / "batch-rollout8-focus4"
)
IMAGE_ROOT = ROOT / "image"


def padded_geometry(image: Image.Image) -> tuple[int, int, int]:
    side = max(image.size)
    return side, (side - image.width) // 2, (side - image.height) // 2


def normalized(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    span = float(values.max() - values.min())
    if span == 0:
        return np.zeros_like(values)
    return (values - values.min()) / span


def colorize(values: np.ndarray) -> Image.Image:
    values = normalized(values)
    rgb = np.stack(
        (values, 1.0 - np.abs(2.0 * values - 1.0), 1.0 - values),
        axis=-1,
    )
    return Image.fromarray(np.uint8(np.clip(rgb * 255.0, 0, 255)), "RGB")


def rebuild_maps(case_dir: Path, image: Image.Image) -> None:
    side, offset_x, offset_y = padded_geometry(image)
    original = image.convert("RGB")
    for name in ("semantic", "structure", "ssim"):
        values = np.load(case_dir / f"{name}_map.npy")
        small = colorize(values)
        small.save(case_dir / f"{name}_map_24x24.png")
        padded = small.resize((side, side), Image.Resampling.BILINEAR)
        heat = padded.crop(
            (offset_x, offset_y, offset_x + image.width, offset_y + image.height)
        )
        heat.save(case_dir / f"{name}_heatmap.png")
        Image.blend(original, heat, 0.45).save(case_dir / f"{name}_overlay.png")


def rebuild_crop(case_dir: Path, image: Image.Image, bbox: list[int]) -> None:
    original = image.convert("RGB")
    boxed = original.copy()
    draw = ImageDraw.Draw(boxed)
    width = max(3, round(min(image.size) / 150))
    draw.rectangle(tuple(bbox), outline=(255, 0, 0), width=width)
    boxed.save(case_dir / "selected_bbox.png")
    original.crop(tuple(bbox)).save(case_dir / "crop.png")


def rebuild_focus(case_dir: Path, image: Image.Image, focus: dict) -> None:
    initial_tokens = int(focus["initial_original_tokens"])
    grid_side = int(round(initial_tokens**0.5))
    side, offset_x, offset_y = padded_geometry(image)
    original = image.convert("RGB")
    dimmed = ImageEnhance.Brightness(original).enhance(0.2)
    for stage in focus["stages"]:
        mask = np.zeros(initial_tokens, dtype=np.uint8)
        mask[np.asarray(stage["kept_original_token_ids"], dtype=np.int64)] = 255
        padded = Image.fromarray(mask.reshape(grid_side, grid_side), "L").resize(
            (side, side), Image.Resampling.NEAREST
        )
        original_mask = padded.crop(
            (offset_x, offset_y, offset_x + image.width, offset_y + image.height)
        )
        Image.composite(original, dimmed, original_mask).save(
            case_dir / f"focus_phase_{stage['phase']}_kept_tokens.png"
        )


def contact_sheet(paths: list[Path], output: Path, title: str) -> None:
    columns = 2
    tile_width, tile_height = 620, 390
    header = 42
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, header + rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), title, fill="black")
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_width - 20, tile_height - 35), Image.Resampling.LANCZOS)
        x = (index % columns) * tile_width + (tile_width - image.width) // 2
        y0 = header + (index // columns) * tile_height
        y = y0 + 24 + (tile_height - 28 - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text((index % columns * tile_width + 8, y0 + 5), path.parent.name, fill="black")
    sheet.save(output, optimize=True)


def main() -> None:
    rows: list[dict] = []
    case_dirs = sorted(path for path in RESULT_ROOT.iterdir() if path.is_dir())
    for case_dir in case_dirs:
        case_id = case_dir.name
        image = Image.open(IMAGE_ROOT / f"{case_id}.jpg").convert("RGB")
        result = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
        focus = result["generation"]["original_plus_crop_focus"]["focus"]
        rebuild_maps(case_dir, image)
        rebuild_crop(case_dir, image, result["original_bbox"])
        rebuild_focus(case_dir, image, focus)
        joint = result["generation"]["original_plus_crop"]
        focused = result["generation"]["original_plus_crop_focus"]
        rows.append(
            {
                "case_id": case_id,
                "bbox": result["original_bbox"],
                "semantic_rollout": result["semantic_rollout"]["text"],
                "original_answer": result["original_answer"],
                "crop_answer": result["crop_answer"],
                "joint_answer": result["joint_answer"],
                "focused_joint_answer": result["focused_joint_answer"],
                "focus_matches_joint": result["focused_joint_answer"] == result["joint_answer"],
                "phase_end_layers": focus["phase_end_layers"],
                "original_token_schedule": [
                    focus["initial_original_tokens"],
                    *[stage["original_tokens_after"] for stage in focus["stages"]],
                ],
                "crop_tokens_kept": focus["crop_tokens_kept"],
                "estimated_prefill_work_reduction": focus[
                    "estimated_prefill_work_reduction"
                ],
                "joint_elapsed_seconds": joint["elapsed_seconds"],
                "focus_elapsed_seconds": focused["elapsed_seconds"],
                "joint_peak_allocated_mib": joint["peak_allocated_mib"],
                "focus_peak_allocated_mib": focused["peak_allocated_mib"],
                "scan_peak_allocated_mib": result["scan_peak_allocated_mib"],
            }
        )

    summary = {
        "case_count": len(rows),
        "focus_matches_joint_count": sum(row["focus_matches_joint"] for row in rows),
        "mean_estimated_prefill_work_reduction": float(
            np.mean([row["estimated_prefill_work_reduction"] for row in rows])
        ),
        "mean_joint_elapsed_seconds": float(
            np.mean([row["joint_elapsed_seconds"] for row in rows])
        ),
        "mean_focus_elapsed_seconds": float(
            np.mean([row["focus_elapsed_seconds"] for row in rows])
        ),
        "mean_joint_peak_allocated_mib": float(
            np.mean([row["joint_peak_allocated_mib"] for row in rows])
        ),
        "mean_focus_peak_allocated_mib": float(
            np.mean([row["focus_peak_allocated_mib"] for row in rows])
        ),
        "max_scan_peak_allocated_mib": max(row["scan_peak_allocated_mib"] for row in rows),
        "rows": rows,
    }
    (RESULT_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    contact_sheet(
        [case_dir / "selected_bbox.png" for case_dir in case_dirs],
        RESULT_ROOT / "selected_bboxes_contact.png",
        "SLoFo 08-07 selected crops (rollout=8, minmax, original coordinates)",
    )
    for phase in (1, 2, 3):
        contact_sheet(
            [case_dir / f"focus_phase_{phase}_kept_tokens.png" for case_dir in case_dirs],
            RESULT_ROOT / f"focus_phase_{phase}_contact.png",
            f"Focus phase {phase}: retained original-image tokens",
        )


if __name__ == "__main__":
    main()

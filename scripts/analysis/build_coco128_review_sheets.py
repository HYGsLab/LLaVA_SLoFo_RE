#!/usr/bin/env python3
"""Render contact sheets with numbered COCO128 person boxes for manual QA review."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def person_boxes(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5 or int(fields[0]) != 0:
            continue
        center_x, center_y, box_width, box_height = map(float, fields[1:])
        boxes.append(
            (
                max(0, round((center_x - box_width / 2) * width)),
                max(0, round((center_y - box_height / 2) * height)),
                min(width, round((center_x + box_width / 2) * width)),
                min(height, round((center_y + box_height / 2) * height)),
            )
        )
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--rows", type=int, default=4)
    args = parser.parse_args()

    manifest = json.loads(
        (args.dataset_dir / "source_manifest.json").read_text(encoding="utf-8")
    )
    records = [
        record
        for record in manifest["records"]
        if int(record["person_annotation_count"]) > 0
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tile_width, tile_height = 360, 300
    capacity = args.columns * args.rows
    font = ImageFont.load_default()
    for sheet_index in range(math.ceil(len(records) / capacity)):
        sheet_records = records[sheet_index * capacity : (sheet_index + 1) * capacity]
        sheet = Image.new(
            "RGB",
            (args.columns * tile_width, args.rows * tile_height),
            "white",
        )
        for position, record in enumerate(sheet_records):
            image_path = args.dataset_dir / "images" / record["file_name"]
            label_path = args.dataset_dir / "labels" / record["label_file"]
            with Image.open(image_path) as source:
                source = source.convert("RGB")
                original_width, original_height = source.size
                scale = min(
                    (tile_width - 8) / original_width,
                    (tile_height - 28) / original_height,
                )
                resized = source.resize(
                    (round(original_width * scale), round(original_height * scale)),
                    Image.Resampling.LANCZOS,
                )
            annotated = resized.copy()
            draw = ImageDraw.Draw(annotated)
            for person_index, (x1, y1, x2, y2) in enumerate(
                person_boxes(label_path, original_width, original_height), 1
            ):
                scaled_box = tuple(round(value * scale) for value in (x1, y1, x2, y2))
                draw.rectangle(scaled_box, outline=(255, 0, 0), width=3)
                draw.text(
                    (scaled_box[0] + 2, scaled_box[1] + 2),
                    f"P{person_index}",
                    fill=(255, 255, 0),
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                    font=font,
                )
            tile_x = (position % args.columns) * tile_width
            tile_y = (position // args.columns) * tile_height
            sheet.paste(annotated, (tile_x + 4, tile_y + 24))
            ImageDraw.Draw(sheet).text(
                (tile_x + 4, tile_y + 4),
                f"#{int(record['index']):03d}  COCO {int(record['source_coco_id']):012d}",
                fill=(0, 0, 0),
                font=font,
            )
        sheet.save(args.output_dir / f"person_review_sheet_{sheet_index + 1:02d}.png")
    print(f"Rendered {len(records)} person-labelled images into {math.ceil(len(records) / capacity)} sheets")


if __name__ == "__main__":
    main()

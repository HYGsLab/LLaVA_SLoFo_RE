#!/usr/bin/env python3
"""Materialize the 20 manually specified posture/color QA records and GT bboxes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def all_person_boxes(label_path: Path, width: int, height: int) -> list[list[int]]:
    boxes: list[list[int]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5 or int(fields[0]) != 0:
            continue
        center_x, center_y, box_width, box_height = map(float, fields[1:])
        boxes.append(
            [
                max(0, round((center_x - box_width / 2) * width)),
                max(0, round((center_y - box_height / 2) * height)),
                min(width, round((center_x + box_width / 2) * width)),
                min(height, round((center_y + box_height / 2) * height)),
            ]
        )
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = json.loads(
        (args.dataset_dir / "source_manifest.json").read_text(encoding="utf-8")
    )
    records_by_index = {
        int(record["index"]): record for record in source_manifest["records"]
    }
    specs = json.loads(args.spec.read_text(encoding="utf-8"))
    if len(specs) != 20 or len({int(item["dataset_index"]) for item in specs}) != 20:
        raise ValueError("The QA spec must contain exactly 20 distinct dataset images.")
    qa_records: list[dict[str, object]] = []
    for qa_index, spec in enumerate(specs, 1):
        dataset_index = int(spec["dataset_index"])
        source = records_by_index[dataset_index]
        image_path = args.dataset_dir / "images" / source["file_name"]
        label_path = args.dataset_dir / "labels" / source["label_file"]
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        person_boxes = all_person_boxes(label_path, width, height)
        person_index = int(spec["person_index"])
        if not 1 <= person_index <= len(person_boxes):
            raise IndexError(
                f"Dataset #{dataset_index}: P{person_index} is absent; "
                f"only {len(person_boxes)} person boxes"
            )
        qa_records.append(
            {
                "qa_id": f"topk_08_07_{qa_index:02d}",
                "dataset_index": dataset_index,
                "image_file": source["file_name"],
                "source_coco_id": source["source_coco_id"],
                "image_size": [width, height],
                "person_index": person_index,
                "ground_truth_bbox_xyxy": person_boxes[person_index - 1],
                "bbox_source": "Ultralytics COCO128 YOLO person label converted to pixels",
                "query_zh": spec["query_zh"],
                "query_en": spec["query_en"],
                "model_query": spec["query_en"],
                "expected_answer_zh": spec["expected_answer_zh"],
                "expected_answer_en": spec["expected_answer_en"],
                "answer_source": "manual visual verification",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    qa_manifest = {
        "name": "SLoFo top-k reranking posture/color evaluation",
        "record_count": len(qa_records),
        "selection_note": (
            "Twenty person-labelled images were manually chosen from the complete "
            "128-image archive for unambiguous posture and clothing-color QA. "
            "All 128 source images remain stored; selection affects only this evaluation."
        ),
        "query_policy": (
            "query_zh follows the requested Chinese template; model_query uses the "
            "equivalent English question for LLaVA-v1.5 consistency."
        ),
        "ground_truth_usage": (
            "Bboxes and expected answers are evaluation-only and are not supplied "
            "to Scan-Locate proposal generation or LLaVA reranking."
        ),
        "records": qa_records,
    }
    (args.output_dir / "qa_bbox_20.json").write_text(
        json.dumps(qa_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    columns, rows = 2, 2
    tile_width, tile_height = 640, 500
    capacity = columns * rows
    font = ImageFont.load_default()
    for sheet_index in range(math.ceil(len(qa_records) / capacity)):
        sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
        for position, qa in enumerate(
            qa_records[sheet_index * capacity : (sheet_index + 1) * capacity]
        ):
            image_path = args.dataset_dir / "images" / qa["image_file"]
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                original_width, original_height = image.size
                scale = min(
                    (tile_width - 12) / original_width,
                    (tile_height - 62) / original_height,
                )
                resized = image.resize(
                    (round(original_width * scale), round(original_height * scale)),
                    Image.Resampling.LANCZOS,
                )
            bbox = [round(value * scale) for value in qa["ground_truth_bbox_xyxy"]]
            draw = ImageDraw.Draw(resized)
            draw.rectangle(bbox, outline=(255, 0, 0), width=5)
            draw.text(
                (bbox[0] + 3, bbox[1] + 3),
                f"GT P{qa['person_index']}",
                fill=(255, 255, 0),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
                font=font,
            )
            tile_x = (position % columns) * tile_width
            tile_y = (position // columns) * tile_height
            sheet.paste(resized, (tile_x + 6, tile_y + 56))
            header = (
                f"{qa['qa_id']} / dataset #{qa['dataset_index']:03d}\n"
                f"{qa['query_en']}  GT: {qa['expected_answer_en']}"
            )
            ImageDraw.Draw(sheet).multiline_text(
                (tile_x + 6, tile_y + 6), header, fill=(0, 0, 0), font=font, spacing=3
            )
        sheet.save(args.output_dir / f"qa_review_sheet_{sheet_index + 1:02d}.png")
    print(f"Wrote {len(qa_records)} QA records and {math.ceil(len(qa_records) / capacity)} review sheets")


if __name__ == "__main__":
    main()

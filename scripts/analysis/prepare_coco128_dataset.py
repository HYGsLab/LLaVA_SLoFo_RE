#!/usr/bin/env python3
"""Extract, validate, and deterministically rename the complete COCO128 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yolo_label(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected five YOLO fields")
        class_id = int(fields[0])
        coordinates = [float(value) for value in fields[1:]]
        if class_id < 0 or any(not 0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"{path}:{line_number}: invalid class or normalized bbox")
        rows.append([class_id, *coordinates])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extract-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    if not zipfile.is_zipfile(args.archive):
        raise ValueError(f"Not a valid ZIP archive: {args.archive}")
    args.extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        archive.testzip()
        archive.extractall(args.extract_dir)

    source_root = args.extract_dir / "coco128"
    source_images = source_root / "images" / "train2017"
    source_labels = source_root / "labels" / "train2017"
    images = sorted(source_images.glob("*.jpg"), key=lambda path: int(path.stem))
    if len(images) != 128:
        raise RuntimeError(f"Expected 128 images, found {len(images)}")

    output_images = args.output_dir / "images"
    output_labels = args.output_dir / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    source_label_files = sorted(source_labels.glob("*.txt"))
    source_image_stems = {path.stem for path in images}
    orphan_source_labels = [
        path.name for path in source_label_files if path.stem not in source_image_stems
    ]
    records: list[dict[str, object]] = []
    for index, source_image in enumerate(images, 1):
        source_label = source_labels / f"{source_image.stem}.txt"
        source_label_exists = source_label.is_file()
        label_rows = read_yolo_label(source_label) if source_label_exists else []
        new_stem = f"coco128_08_07_{index:03d}"
        image_name = f"{new_stem}.jpg"
        label_name = f"{new_stem}.txt"
        shutil.copy2(source_image, output_images / image_name)
        if source_label_exists:
            shutil.copy2(source_label, output_labels / label_name)
        else:
            (output_labels / label_name).write_text("", encoding="utf-8")
        with Image.open(source_image) as image:
            width, height = image.size
            image.verify()
        records.append(
            {
                "index": index,
                "file_name": image_name,
                "label_file": label_name,
                "source_file": source_image.name,
                "source_coco_id": int(source_image.stem),
                "source_label_exists": source_label_exists,
                "width": width,
                "height": height,
                "annotation_count": len(label_rows),
                "person_annotation_count": sum(
                    1 for row in label_rows if int(row[0]) == 0
                ),
                "class_ids": sorted({int(row[0]) for row in label_rows}),
            }
        )

    license_source = source_root / "LICENSE"
    if license_source.is_file():
        shutil.copy2(license_source, args.output_dir / "LICENSE_COCO128.txt")
    manifest = {
        "dataset": "Ultralytics COCO128",
        "description": "Complete first 128 images of COCO train2017; no images filtered.",
        "official_documentation": "https://docs.ultralytics.com/datasets/detect/coco128/",
        "download_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
        "archive_sha256": sha256(args.archive),
        "image_count": len(records),
        "label_count": len(list(output_labels.glob("*.txt"))),
        "source_label_count": len(source_label_files),
        "images_without_matching_source_label": [
            record["source_file"]
            for record in records
            if not bool(record["source_label_exists"])
        ],
        "orphan_source_labels": orphan_source_labels,
        "images_with_person_labels": sum(
            1 for record in records if int(record["person_annotation_count"]) > 0
        ),
        "naming_pattern": "coco128_08_07_NNN.jpg / .txt",
        "records": records,
    }
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()

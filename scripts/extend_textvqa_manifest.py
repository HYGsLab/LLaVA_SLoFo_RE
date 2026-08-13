#!/usr/bin/env python3
"""Deterministically extend a fixed TextVQA manifest without replacing its rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


ANSWER_SUFFIX = "Answer the question using a single word or phrase."


def stable_rng(seed: int, namespace: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def build_official_records(annotation_path: Path) -> list[dict[str, object]]:
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))["data"]
    records: list[dict[str, object]] = []
    for annotation in annotations:
        image_id = str(annotation["image_id"])
        question = str(annotation["question"])
        records.append(
            {
                "benchmark": "textvqa",
                "qa_id": str(annotation["question_id"]),
                "image_file": f"{image_id}.jpg",
                "model_query": f"{question}\n{ANSWER_SUFFIX}",
                "ground_truth": None,
                "ground_truth_answers": annotation["answers"],
                "category": "text_reading",
                "source_image_id": image_id,
            }
        )
    return records


def extend_manifest(
    parent: dict[str, object],
    official_records: list[dict[str, object]],
    count: int,
    seed: int,
) -> dict[str, object]:
    parent_records = list(parent["records"])
    if count < len(parent_records):
        raise ValueError(
            f"Requested {count} rows but parent already has {len(parent_records)}"
        )
    if count > len(official_records):
        raise ValueError(
            f"Requested {count} rows from only {len(official_records)} official rows"
        )

    official_by_id = {str(row["qa_id"]): row for row in official_records}
    if len(official_by_id) != len(official_records):
        raise ValueError("Official TextVQA annotations contain duplicate qa_id values")

    parent_ids = [str(row["qa_id"]) for row in parent_records]
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError("Parent manifest contains duplicate qa_id values")
    unknown = sorted(set(parent_ids) - set(official_by_id))
    if unknown:
        raise ValueError(f"Parent manifest has unknown qa_id values: {unknown[:10]}")

    remaining = [row for row in official_records if str(row["qa_id"]) not in parent_ids]
    additional_count = count - len(parent_records)
    selected_indices = sorted(
        stable_rng(seed, "textvqa-nested-extension").sample(
            range(len(remaining)), additional_count
        )
    )
    additions = [remaining[index] for index in selected_indices]
    records = parent_records + additions

    output = dict(parent)
    output.update(
        {
            "seed": seed,
            "record_count": len(records),
            "sampling": {
                "method": "fixed parent plus deterministic sample from remaining TextVQA val rows",
                "source_count": len(official_records),
                "requested_count": count,
                "parent_count": len(parent_records),
                "added_count": additional_count,
                "extension_namespace": "textvqa-nested-extension",
            },
            "category_counts": dict(
                sorted(Counter(str(row.get("category")) for row in records).items())
            ),
            "records": records,
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    parent = json.loads(args.parent.read_text(encoding="utf-8"))
    official_records = build_official_records(args.annotations)
    output = extend_manifest(parent, official_records, args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(output['records'])} records to {args.output}; "
        f"parent={output['sampling']['parent_count']} "
        f"added={output['sampling']['added_count']}"
    )


if __name__ == "__main__":
    main()

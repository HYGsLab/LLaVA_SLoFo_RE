#!/usr/bin/env python3
"""Create deterministic paired SLoFo manifests for TextVQA, GQA and POPE."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ANSWER_SUFFIX = "Answer the question using a single word or phrase."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--textvqa-size", type=int, default=512)
    parser.add_argument("--gqa-size", type=int, default=512)
    parser.add_argument("--pope-per-cell", type=int, default=100)
    parser.add_argument("--textvqa-question-file", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_rng(seed: int, name: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def deterministic_sample(
    records: list[dict[str, object]], count: int, seed: int, name: str
) -> list[dict[str, object]]:
    if count > len(records):
        raise ValueError(f"Requested {count} records from only {len(records)} in {name}")
    indices = sorted(stable_rng(seed, name).sample(range(len(records)), count))
    return [records[index] for index in indices]


def write_manifest(
    output_path: Path,
    benchmark: str,
    records: list[dict[str, object]],
    seed: int,
    sampling: dict[str, object],
) -> None:
    category_counts = Counter(str(record.get("category")) for record in records)
    payload = {
        "benchmark": benchmark,
        "seed": seed,
        "record_count": len(records),
        "sampling": sampling,
        "category_counts": dict(sorted(category_counts.items())),
        "records": records,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def first_question_line(prompt: str) -> str:
    return prompt.splitlines()[0].strip().lower()


def prepare_textvqa(args: argparse.Namespace) -> Path:
    annotation_path = args.data_root / "textvqa" / "TextVQA_0.5.1_val.json"
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))["data"]
    question_lookup: dict[tuple[str, str], dict[str, object]] = {}
    if args.textvqa_question_file and args.textvqa_question_file.is_file():
        for question in read_jsonl(args.textvqa_question_file):
            key = (str(question["question_id"]), first_question_line(str(question["text"])))
            question_lookup[key] = question

    records: list[dict[str, object]] = []
    official_prompt_matches = 0
    for annotation in annotations:
        image_id = str(annotation["image_id"])
        question_text = str(annotation["question"])
        question_record = question_lookup.get((image_id, question_text.lower()))
        if question_record is None:
            model_query = f"{question_text}\n{ANSWER_SUFFIX}"
            image_file = f"{image_id}.jpg"
        else:
            official_prompt_matches += 1
            model_query = str(question_record["text"])
            image_file = str(question_record["image"])
        records.append(
            {
                "benchmark": "textvqa",
                "qa_id": str(annotation["question_id"]),
                "image_file": image_file,
                "model_query": model_query,
                "ground_truth": None,
                "ground_truth_answers": annotation["answers"],
                "category": "text_reading",
                "source_image_id": image_id,
            }
        )
    selected = deterministic_sample(records, args.textvqa_size, args.seed, "textvqa")
    output = args.output_root / f"textvqa_subset_{args.textvqa_size}.json"
    write_manifest(
        output,
        "textvqa",
        selected,
        args.seed,
        {
            "method": "deterministic uniform sample from the 5,000-question val split",
            "source_count": len(records),
            "requested_count": args.textvqa_size,
            "official_prompt_matches": official_prompt_matches,
        },
    )
    return output


def prepare_gqa(args: argparse.Namespace) -> Path:
    root = args.data_root / "gqa"
    questions = read_jsonl(root / "llava_gqa_testdev_balanced.jsonl")
    ground_truth = json.loads(
        (root / "testdev_balanced_questions.json").read_text(encoding="utf-8")
    )
    records: list[dict[str, object]] = []
    for question in questions:
        question_id = str(question["question_id"])
        annotation = ground_truth[question_id]
        records.append(
            {
                "benchmark": "gqa",
                "qa_id": question_id,
                "image_file": str(question["image"]),
                "model_query": str(question["text"]),
                "ground_truth": str(annotation["answer"]),
                "ground_truth_answers": [str(annotation["answer"])],
                "category": str(annotation["types"]["structural"]),
                "semantic_category": str(annotation["types"]["semantic"]),
                "detailed_category": str(annotation["types"]["detailed"]),
            }
        )
    selected = deterministic_sample(records, args.gqa_size, args.seed, "gqa")
    output = args.output_root / f"gqa_subset_{args.gqa_size}.json"
    write_manifest(
        output,
        "gqa",
        selected,
        args.seed,
        {
            "method": "deterministic uniform sample from testdev-balanced",
            "source_count": len(records),
            "requested_count": args.gqa_size,
        },
    )
    return output


def prepare_pope(args: argparse.Namespace) -> Path:
    root = args.data_root / "pope"
    questions = read_jsonl(root / "llava_pope_test.jsonl")
    labels: dict[tuple[str, int], dict[str, object]] = {}
    for category in ("adversarial", "popular", "random"):
        for annotation in read_jsonl(root / "coco" / f"coco_pope_{category}.json"):
            row = dict(annotation)
            row["category"] = category
            labels[(category, int(row["question_id"]))] = row

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for question in questions:
        question_id = int(question["question_id"])
        category = str(question["category"])
        prefix = {"adversarial": 0, "random": 10_000_000, "popular": 20_000_000}[
            category
        ]
        local_question_id = question_id - prefix
        annotation = labels[(category, local_question_id)]
        if str(annotation["image"]) != str(question["image"]):
            raise ValueError(
                f"POPE image mismatch for {category}/{local_question_id}: "
                f"{annotation['image']} != {question['image']}"
            )
        label = str(annotation["label"])
        grouped[(category, label)].append(
            {
                "benchmark": "pope",
                "qa_id": str(question_id),
                "image_file": str(question["image"]),
                "model_query": str(question["text"]),
                "ground_truth": label,
                "ground_truth_answers": [label],
                "category": category,
                "label": label,
            }
        )

    selected: list[dict[str, object]] = []
    cell_counts: dict[str, int] = {}
    for category in ("adversarial", "popular", "random"):
        for label in ("yes", "no"):
            cell = grouped[(category, label)]
            sampled = deterministic_sample(
                cell,
                args.pope_per_cell,
                args.seed,
                f"pope:{category}:{label}",
            )
            selected.extend(sampled)
            cell_counts[f"{category}/{label}"] = len(sampled)
    selected.sort(key=lambda row: int(str(row["qa_id"])))
    total = args.pope_per_cell * 6
    output = args.output_root / f"pope_stratified_{total}.json"
    write_manifest(
        output,
        "pope",
        selected,
        args.seed,
        {
            "method": "equal deterministic sample from category x yes/no cells",
            "source_count": len(questions),
            "per_cell": args.pope_per_cell,
            "cell_counts": cell_counts,
        },
    )
    return output


def validate_images(manifest_path: Path, image_root: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = [
        str(record["image_file"])
        for record in manifest["records"]
        if not (image_root / str(record["image_file"])).is_file()
    ]
    return {
        "manifest": str(manifest_path),
        "image_root": str(image_root),
        "record_count": len(manifest["records"]),
        "missing_count": len(missing),
        "missing_examples": missing[:20],
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = {
        "textvqa": prepare_textvqa(args),
        "gqa": prepare_gqa(args),
        "pope": prepare_pope(args),
    }
    validations = {
        "textvqa": validate_images(
            manifests["textvqa"], args.data_root / "textvqa" / "train_images"
        ),
        "gqa": validate_images(manifests["gqa"], args.data_root / "gqa" / "images"),
        "pope": validate_images(
            manifests["pope"], args.data_root / "pope" / "val2014"
        ),
    }
    summary = {
        "seed": args.seed,
        "manifests": {name: str(path) for name, path in manifests.items()},
        "validation": validations,
    }
    (args.output_root / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

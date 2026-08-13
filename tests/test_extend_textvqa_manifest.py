from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "extend_textvqa_manifest.py"
SPEC = importlib.util.spec_from_file_location("extend_textvqa_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def official_rows(count: int = 8) -> list[dict[str, object]]:
    return [
        {
            "benchmark": "textvqa",
            "qa_id": str(index),
            "image_file": f"image-{index}.jpg",
            "model_query": f"question {index}",
            "ground_truth": None,
            "ground_truth_answers": [str(index)],
            "category": "text_reading",
            "source_image_id": f"image-{index}",
        }
        for index in range(count)
    ]


def test_extension_is_nested_unique_and_deterministic() -> None:
    official = official_rows()
    parent = {"benchmark": "textvqa", "records": [official[1], official[4]]}

    first = MODULE.extend_manifest(parent, official, count=6, seed=20260813)
    second = MODULE.extend_manifest(parent, official, count=6, seed=20260813)

    first_ids = [row["qa_id"] for row in first["records"]]
    assert first_ids[:2] == ["1", "4"]
    assert len(first_ids) == len(set(first_ids)) == 6
    assert first_ids == [row["qa_id"] for row in second["records"]]
    assert first["sampling"]["parent_count"] == 2
    assert first["sampling"]["added_count"] == 4


def test_extension_rejects_duplicate_parent_ids() -> None:
    official = official_rows()
    parent = {"records": [official[1], official[1]]}
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.extend_manifest(parent, official, count=3, seed=1)


def test_extension_rejects_unknown_parent_id() -> None:
    official = official_rows()
    parent = {"records": [{**official[1], "qa_id": "missing"}]}
    with pytest.raises(ValueError, match="unknown"):
        MODULE.extend_manifest(parent, official, count=3, seed=1)

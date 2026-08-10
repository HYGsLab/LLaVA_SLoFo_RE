#!/usr/bin/env python3
"""Create a deterministic prefix subset from an existing benchmark manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")

    source = json.loads(args.input.read_text(encoding="utf-8"))
    records = list(source["records"])
    if args.count > len(records):
        raise ValueError(f"Requested {args.count} rows from {len(records)} records")

    output = dict(source)
    output["records"] = records[: args.count]
    output["subset_count"] = args.count
    output["subset_method"] = "deterministic prefix of the fixed 2026-08-09 manifest"
    output["parent_manifest"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.count} records to {args.output}")


if __name__ == "__main__":
    main()

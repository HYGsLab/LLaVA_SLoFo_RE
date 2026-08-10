#!/usr/bin/env python3
"""Capture non-sensitive software/GPU metadata for the VQA benchmark report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
from pathlib import Path

import torch
import transformers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def command_output(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def optional_command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        return command_output(command, cwd=cwd)
    except subprocess.CalledProcessError:
        return None


def main() -> None:
    args = parse_args()
    config = json.loads((args.model_path / "config.json").read_text(encoding="utf-8"))
    gpu_rows = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    environment = {
        "captured_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_rows": gpu_rows,
        "project_git_commit": optional_command_output(
            ["git", "rev-parse", "HEAD"], cwd=args.project_root
        ),
        "model_path": str(args.model_path),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "vision_tower": config.get("mm_vision_tower"),
        "image_aspect_ratio": config.get("image_aspect_ratio"),
        "image_grid_pinpoints": config.get("image_grid_pinpoints"),
        "torch_dtype_config": config.get("torch_dtype"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(environment, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

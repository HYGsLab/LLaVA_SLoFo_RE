#!/usr/bin/env python3
"""Point the copied LLaVA checkpoint at the project-local CLIP checkpoint."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path("/data/workspace/Gexuri_Project/HYG_LLaVA_SLoFo")
LLAVA_CONFIG = PROJECT_ROOT / "models/llava-v1.5-7b/config.json"
LOCAL_VISION_TOWER = PROJECT_ROOT / "models/clip-vit-large-patch14-336"
UPSTREAM_VISION_TOWER = "openai/clip-vit-large-patch14-336"


def main() -> None:
    if not LLAVA_CONFIG.is_file():
        raise FileNotFoundError(f"LLaVA config not found: {LLAVA_CONFIG}")
    if not (LOCAL_VISION_TOWER / "config.json").is_file():
        raise FileNotFoundError(f"Local CLIP config not found: {LOCAL_VISION_TOWER}")

    config = json.loads(LLAVA_CONFIG.read_text(encoding="utf-8"))
    current = config.get("mm_vision_tower")
    local_path = str(LOCAL_VISION_TOWER)
    if current not in {UPSTREAM_VISION_TOWER, local_path}:
        raise ValueError(
            "Refusing to replace an unexpected mm_vision_tower: "
            f"{current!r}"
        )

    config["mm_vision_tower"] = local_path
    temporary = LLAVA_CONFIG.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(LLAVA_CONFIG)
    print(f"mm_vision_tower: {current}")
    print(f"              -> {local_path}")


if __name__ == "__main__":
    main()

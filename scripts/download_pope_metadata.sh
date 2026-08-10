#!/usr/bin/env bash
set -euo pipefail

project_root="${1:-$(pwd)}"
pope_root="${project_root}/benchmarks/official/data/pope"
annotation_root="${pope_root}/coco"
log_root="${project_root}/benchmarks/official/logs"
mkdir -p "${annotation_root}" "${log_root}"

wget \
    --continue \
    --no-verbose \
    --timeout=60 \
    --tries=20 \
    --retry-on-http-error=429,500,502,503,504 \
    --output-document="${pope_root}/llava_pope_test.jsonl" \
    "https://github.com/OpenGVLab/InternVL/releases/download/data/llava_pope_test.jsonl"

for split in adversarial popular random; do
    wget \
        --continue \
        --no-verbose \
        --timeout=60 \
        --tries=20 \
        --retry-on-http-error=429,500,502,503,504 \
        --output-document="${annotation_root}/coco_pope_${split}.json" \
        "https://raw.githubusercontent.com/AoiDragon/POPE/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_${split}.json"
done

sha256sum \
    "${pope_root}/llava_pope_test.jsonl" \
    "${annotation_root}"/*.json \
    > "${log_root}/pope_metadata_sha256.txt"

echo "[download] POPE metadata complete"

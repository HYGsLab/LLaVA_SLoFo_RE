#!/usr/bin/env bash
set -euo pipefail

project_root="${1:-$(pwd)}"
download_root="${project_root}/benchmarks/official/downloads"
log_root="${project_root}/benchmarks/official/logs"
mkdir -p "${download_root}" "${log_root}"

download() {
    local url="$1"
    local destination="$2"
    echo "[download] ${destination}"
    wget \
        --continue \
        --no-verbose \
        --timeout=60 \
        --tries=20 \
        --output-document="${download_root}/${destination}" \
        "${url}"
}

pids=()
download \
    "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip" \
    "gqa_questions1.2.zip" &
pids+=("$!")
download \
    "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip" \
    "gqa_images.zip" &
pids+=("$!")
download \
    "http://images.cocodataset.org/zips/val2014.zip" \
    "coco_val2014.zip" &
pids+=("$!")

for pid in "${pids[@]}"; do
    wait "${pid}"
done

echo "[download] GQA and COCO complete"
sha256sum "${download_root}"/* > "${log_root}/download_sha256.txt"
du -sh "${download_root}"

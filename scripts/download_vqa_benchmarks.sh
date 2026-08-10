#!/usr/bin/env bash
set -euo pipefail

# Download the public evaluation metadata and official image/question archives
# required by TextVQA, GQA testdev-balanced, and POPE/COCO.  This script only
# writes below the current project's benchmark directory and supports resume.

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

# This public MobileVLM release mirrors the small LLaVA-format benchmark
# metadata (JSONL questions and evaluation scripts).  Images and ground-truth
# annotations below still come from each benchmark's official host/repository.
download_group() {
    local pids=()
    while (( "$#" )); do
        download "$1" "$2" &
        pids+=("$!")
        shift 2
    done
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
}

# Run at most three transfers at a time so the shared server/network is not
# flooded, while avoiding the several-hour penalty of fully sequential fetches.
download_group \
    "https://github.com/Meituan-AutoML/MobileVLM/releases/download/v0.1/benchmark_data.zip" \
    "benchmark_data.zip" \
    "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json" \
    "TextVQA_0.5.1_val.json" \
    "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip" \
    "textvqa_train_val_images.zip"

download_group \
    "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip" \
    "gqa_questions1.2.zip" \
    "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip" \
    "gqa_images.zip" \
    "http://images.cocodataset.org/zips/val2014.zip" \
    "coco_val2014.zip"

echo "[download] complete"
sha256sum "${download_root}"/* > "${log_root}/download_sha256.txt"
du -sh "${download_root}"

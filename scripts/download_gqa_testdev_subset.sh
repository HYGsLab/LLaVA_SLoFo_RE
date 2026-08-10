#!/usr/bin/env bash
set -euo pipefail

project_root="${1:-$(pwd)}"
download_root="${project_root}/benchmarks/official/downloads/gqa_testdev"
log_root="${project_root}/benchmarks/official/logs"
base_url="https://hf-mirror.com/datasets/AJN-AI/VoQA/resolve/main/test/gqa"

mkdir -p "${download_root}" "${log_root}"

download() {
    local name="$1"
    echo "[download] GQA testdev ${name}"
    wget \
        --continue \
        --no-verbose \
        --timeout=60 \
        --tries=20 \
        --retry-on-http-error=429,500,502,503,504 \
        --output-document="${download_root}/${name}" \
        "${base_url}/${name}"
}

pids=()
for name in \
    images.zip \
    llava_gqa_testdev_balanced.jsonl \
    testdev_balanced_questions.json
do
    download "${name}" &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "${pid}"
done

sha256sum "${download_root}"/* > "${log_root}/gqa_testdev_sha256.txt"
echo "[download] GQA testdev subset complete"
du -sh "${download_root}"

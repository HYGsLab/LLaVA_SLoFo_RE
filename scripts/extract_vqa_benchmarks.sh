#!/usr/bin/env bash
set -euo pipefail

project_root="${1:-$(pwd)}"
download_root="${project_root}/benchmarks/official/downloads"
data_root="${project_root}/benchmarks/official/data"
log_root="${project_root}/benchmarks/official/logs"
mkdir -p "${data_root}" "${log_root}"

extract_if_ready() {
    local archive="$1"
    local destination="$2"
    local marker="$3"
    if [[ -f "${marker}" ]]; then
        echo "[extract] existing marker; skipped: ${archive}"
        return 0
    fi
    if [[ ! -f "${archive}" ]] || ! unzip -Z1 "${archive}" >/dev/null 2>&1; then
        echo "[extract] archive not ready; skipped: ${archive}"
        return 0
    fi
    echo "[extract] ${archive} -> ${destination}"
    mkdir -p "${destination}"
    unzip -q "${archive}" -d "${destination}"
    touch "${marker}"
}

pids=()
extract_if_ready \
    "${download_root}/textvqa_train_val_images.zip" \
    "${data_root}/textvqa" \
    "${data_root}/textvqa/.images_extracted" &
pids+=("$!")
extract_if_ready \
    "${download_root}/gqa_testdev/images.zip" \
    "${data_root}/gqa" \
    "${data_root}/gqa/.images_extracted" &
pids+=("$!")
extract_if_ready \
    "${download_root}/coco_val2014.zip" \
    "${data_root}/pope" \
    "${data_root}/pope/.images_extracted" &
pids+=("$!")
extract_if_ready \
    "${download_root}/benchmark_data.zip" \
    "${data_root}/metadata" \
    "${data_root}/metadata/.benchmark_data_extracted" &
pids+=("$!")

for pid in "${pids[@]}"; do
    wait "${pid}"
done

if [[ -f "${download_root}/TextVQA_0.5.1_val.json" ]]; then
    install -m 0644 \
        "${download_root}/TextVQA_0.5.1_val.json" \
        "${data_root}/textvqa/TextVQA_0.5.1_val.json"
fi
if [[ -f "${download_root}/gqa_testdev/llava_gqa_testdev_balanced.jsonl" ]]; then
    install -m 0644 \
        "${download_root}/gqa_testdev/llava_gqa_testdev_balanced.jsonl" \
        "${data_root}/gqa/llava_gqa_testdev_balanced.jsonl"
fi
if [[ -f "${download_root}/gqa_testdev/testdev_balanced_questions.json" ]]; then
    install -m 0644 \
        "${download_root}/gqa_testdev/testdev_balanced_questions.json" \
        "${data_root}/gqa/testdev_balanced_questions.json"
fi

echo "[extract] pass complete"
du -sh "${data_root}"

#!/usr/bin/env bash
set -euo pipefail

# Guard an explicitly selected physical GPU using aggregate counters only.
# This avoids inspecting other users' commands or project directories.
if (( $# < 2 )); then
    echo "Usage: $0 <physical-gpu-index> <command> [args ...]" >&2
    exit 64
fi

gpu_index="$1"
shift
MAX_USED_MIB="${MAX_USED_MIB:-512}"
MAX_UTIL_PERCENT="${MAX_UTIL_PERCENT:-5}"

IFS=',' read -r raw_used raw_util < <(
    nvidia-smi \
        --id="$gpu_index" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits
)
used="${raw_used//[[:space:]]/}"
util="${raw_util//[[:space:]]/}"
if (( used > MAX_USED_MIB || util > MAX_UTIL_PERCENT )); then
    echo "GPU $gpu_index is not empty: used=${used} MiB, util=${util}%." >&2
    exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu_index"
echo "Selected empty physical GPU $gpu_index; the program will see cuda:0."
exec "$@"

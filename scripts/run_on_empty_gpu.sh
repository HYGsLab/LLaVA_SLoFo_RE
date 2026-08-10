#!/usr/bin/env bash
set -euo pipefail

# Do not inspect other users' processes.  Select only from aggregate GPU
# utilization and memory counters.  Defaults can be tightened via environment.
MAX_USED_MIB="${MAX_USED_MIB:-512}"
MAX_UTIL_PERCENT="${MAX_UTIL_PERCENT:-5}"

selected=""
while IFS=',' read -r raw_index raw_used raw_util; do
    index="${raw_index//[[:space:]]/}"
    used="${raw_used//[[:space:]]/}"
    util="${raw_util//[[:space:]]/}"
    if (( used <= MAX_USED_MIB && util <= MAX_UTIL_PERCENT )); then
        selected="$index"
        break
    fi
done < <(
    nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits
)

if [[ -z "$selected" ]]; then
    echo "No empty GPU found (used <= ${MAX_USED_MIB} MiB, util <= ${MAX_UTIL_PERCENT}%)." >&2
    exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$selected"
echo "Selected physical GPU $selected; the program will see it as cuda:0."

if (( $# == 0 )); then
    echo "Usage: $0 <command> [args ...]" >&2
    exit 64
fi

exec "$@"

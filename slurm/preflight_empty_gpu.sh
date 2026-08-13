#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[gpu-preflight] ERROR: this check must run inside a Slurm allocation." >&2
  exit 40
fi

gpu_list="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "$gpu_list" ]]; then
  echo "[gpu-preflight] ERROR: no allocated GPU is visible." >&2
  exit 41
fi

gpu_id="${gpu_list%%,*}"
used_mib="$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -dc '0-9')"
util_pct="$(nvidia-smi -i "$gpu_id" --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -dc '0-9')"
threshold_mib="${HYG_EMPTY_GPU_MAX_USED_MIB:-512}"

echo "[gpu-preflight] job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-unknown} gpu=$gpu_id used=${used_mib:-unknown}MiB util=${util_pct:-unknown}% threshold=${threshold_mib}MiB"
if [[ -z "$used_mib" || "$used_mib" -gt "$threshold_mib" ]]; then
  echo "[gpu-preflight] ERROR: allocated GPU is not empty enough; aborting before model load." >&2
  exit 42
fi

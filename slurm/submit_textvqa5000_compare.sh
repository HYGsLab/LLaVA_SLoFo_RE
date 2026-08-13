#!/usr/bin/env bash
set -euo pipefail

ROOT=/labmount/users/202400390068/projects/HYG_LLaVA_SLoFo
JOB="$ROOT/slurm/textvqa5000_compare.sbatch"
SCHED_LOG_ROOT=/labmount/users/202400390068/logs/HYG_LLaVA_SLoFo/2026-08-13-textvqa5000/scheduler
mkdir -p "$SCHED_LOG_ROOT"

if squeue -h -u 202400390068 -o '%j' | grep -q '^HYG_TVQA5000_'; then
  echo "A TextVQA-5000 job is already queued or running; refusing duplicate submission." >&2
  exit 65
fi

submit() {
  local config="$1"
  local dependency="${2:-}"
  local args=(
    --parsable
    --job-name="HYG_TVQA5000_${config}"
    --output="$SCHED_LOG_ROOT/${config}_%j.out"
    --error="$SCHED_LOG_ROOT/${config}_%j.err"
    --export="ALL,HYG_FULL_CONFIG=${config}"
  )
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="afterok:${dependency}")
  fi
  sbatch "${args[@]}" "$JOB"
}

job_a=$(submit A)
job_e=$(submit E "$job_a")
printf 'A=%s\nE=%s\n' "$job_a" "$job_e"
squeue -j "$job_a,$job_e" -o '%.18i %.20j %.9T %.10M %R'

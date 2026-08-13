#!/usr/bin/env bash
#SBATCH --job-name=HYG_TVQA1024
#SBATCH --partition=GPUNorm
#SBATCH --account=grads
#SBATCH --qos=q_grads
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --exclude=gn1,gn2,gn3,gn7,gn9,gn10,gn11,gn12,gn13
set -euo pipefail

ROOT=/labmount/users/202400390068/projects/HYG_LLaVA_SLoFo
RESULT_ROOT=/labmount/users/202400390068/results/HYG_LLaVA_SLoFo/2026-08-13-textvqa1024
LOG_ROOT=/labmount/users/202400390068/logs/HYG_LLaVA_SLoFo/2026-08-13-textvqa1024
CONFIG="${HYG_FACTOR_CONFIG:?Set HYG_FACTOR_CONFIG to A, B, D, or E}"

case "$CONFIG" in
  A)
    RUN_NAME=A_1token_raw_single
    ROLLOUT_TOKENS=1
    FUSION=none
    TOP_K=1
    ;;
  B)
    RUN_NAME=B_1token_minmax_single
    ROLLOUT_TOKENS=1
    FUSION=minmax
    TOP_K=1
    ;;
  D)
    RUN_NAME=D_8token_minmax_single
    ROLLOUT_TOKENS=8
    FUSION=minmax
    TOP_K=1
    ;;
  E)
    RUN_NAME=E_8token_minmax_topk5
    ROLLOUT_TOKENS=8
    FUSION=minmax
    TOP_K=5
    ;;
  *)
    echo "Unknown HYG_FACTOR_CONFIG: $CONFIG" >&2
    exit 64
    ;;
esac

mkdir -p "$RESULT_ROOT/$RUN_NAME" "$LOG_ROOT/$RUN_NAME"
"$ROOT/slurm/preflight_empty_gpu.sh"
source "$ROOT/activate_slurm.sh"

echo "started_at=$(date --iso-8601=seconds)"
echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
echo "slurm_node=${SLURMD_NODENAME:-unknown}"
echo "repo_commit=$(git rev-parse HEAD)"
echo "llava_commit=$(git -C LLaVA rev-parse HEAD)"
echo "manifest_sha256=$(sha256sum benchmarks/official/manifests/textvqa_nested_1024.json | awk '{print $1}')"
echo "factor_config=$CONFIG"
echo "run_name=$RUN_NAME"
echo "rollout_tokens=$ROLLOUT_TOKENS fusion=$FUSION top_k=$TOP_K"

python scripts/run_slofo_vqa_benchmark_batch.py \
  --manifest benchmarks/official/manifests/textvqa_nested_1024.json \
  --benchmark textvqa \
  --model-path models/llava-v1.5-7b \
  --image-root benchmarks/official/data/textvqa/train_images \
  --output-root "$RESULT_ROOT/$RUN_NAME" \
  --log-root "$LOG_ROOT/$RUN_NAME" \
  --top-k "$TOP_K" \
  --max-new-tokens 16 \
  --semantic-rollout-tokens "$ROLLOUT_TOKENS" \
  --scan-capture-mode selective_hook \
  --fusion-normalization "$FUSION" \
  --locate-coordinate-space original \
  --semantic-token-aggregation mean \
  --full-artifact-count 3

echo "finished_at=$(date --iso-8601=seconds)"

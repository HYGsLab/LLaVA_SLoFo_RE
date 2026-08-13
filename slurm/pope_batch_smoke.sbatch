#!/usr/bin/env bash
#SBATCH --job-name=HYG_pope_smoke
#SBATCH --partition=GPUNorm
#SBATCH --account=grads
#SBATCH --qos=q_grads
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --exclude=gn1,gn2,gn3,gn7,gn9,gn10,gn11,gn12,gn13
#SBATCH --output=/labmount/users/202400390068/logs/HYG_LLaVA_SLoFo/pope_smoke_%j.log
#SBATCH --error=/labmount/users/202400390068/logs/HYG_LLaVA_SLoFo/pope_smoke_%j.err
set -euo pipefail
/labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/slurm/preflight_empty_gpu.sh
source /labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/activate_slurm.sh
python scripts/run_slofo_vqa_benchmark_batch.py \
 --manifest benchmarks/official/manifests/pope_stratified_600.json \
 --benchmark pope \
 --model-path models/llava-v1.5-7b \
 --image-root benchmarks/official/data/pope/val2014 \
 --output-root /labmount/users/202400390068/results/HYG_LLaVA_SLoFo/slurm_acceptance/pope_batch_smoke \
 --log-root /labmount/users/202400390068/logs/HYG_LLaVA_SLoFo/pope_batch_smoke \
 --top-k 1 \
 --max-new-tokens 16 \
 --semantic-rollout-tokens 1 \
 --scan-capture-mode selective_hook \
 --fusion-normalization none \
 --locate-coordinate-space padded \
 --semantic-token-aggregation mean \
 --full-artifact-count 1 \
 --case-ids 7 \
 --force

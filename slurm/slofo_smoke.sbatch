#!/usr/bin/env bash
#SBATCH --job-name=HYG_slofo_smoke
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
#SBATCH --output=/labmount/users/202400390068/logs/HYG_LLaVA_SLoFo/slofo_smoke_%j.log
#SBATCH --error=/labmount/users/202400390068/logs/HYG_LLaVA_SLoFo/slofo_smoke_%j.err
set -euo pipefail
/labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/slurm/preflight_empty_gpu.sh
source /labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/activate_slurm.sh
OUT=/labmount/users/202400390068/results/HYG_LLaVA_SLoFo/slurm_acceptance/slofo_focus
mkdir -p "$OUT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv,noheader
python scripts/run_slofo_scan_locate.py \
 --model-path models/llava-v1.5-7b \
 --image-file images/test_08_06/test_08_06_01.jpg \
 --query "What color are the clothes worn by the seated person?" \
 --output-dir "$OUT" \
 --artifact-mode full \
 --max-new-tokens 16 \
 --semantic-rollout-tokens 1 \
 --scan-capture-mode selective_hook \
 --fusion-normalization none \
 --rerank-top-k 1 \
 --enable-focus

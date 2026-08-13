#!/usr/bin/env bash
#SBATCH --job-name=HYG_gpu_preflight
#SBATCH --partition=GPUNorm
#SBATCH --account=grads
#SBATCH --qos=q_grads
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --gres=gpu:1
#SBATCH --time=00:02:00
#SBATCH --exclude=gn1,gn2,gn3,gn7,gn9,gn10,gn11,gn12,gn13
#SBATCH --output=/labmount/users/202400390068/logs/HYG_LLaVA_SLoFo/gpu_preflight_%j.log
#SBATCH --error=/labmount/users/202400390068/logs/HYG_LLaVA_SLoFo/gpu_preflight_%j.err
set -euo pipefail
/labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/slurm/preflight_empty_gpu.sh
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

#!/usr/bin/env bash
PROJECT_ROOT="/labmount/users/202400390068/projects/HYG_LLaVA_SLoFo"
REPO_ROOT="$PROJECT_ROOT/repo"
ENV_PREFIX="/labmount/users/202400390068/envs/HYG_LLaVA_SLoFo/llava-slofo-py310"
export PROJECT_ROOT REPO_ROOT ENV_PREFIX
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$REPO_ROOT/LLaVA:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
cd "$REPO_ROOT" || return 1
echo "Project: $PROJECT_ROOT"
echo "Repo:    $REPO_ROOT"
echo "Python:  $ENV_PREFIX/bin/python"
echo "Use sbatch/srun for GPU work; never run GPU inference on the login node."

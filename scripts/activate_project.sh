#!/usr/bin/env bash

# Activate only HYG's isolated LLaVA/SLoFo workspace.
PROJECT_ROOT="/data/workspace/Gexuri_Project/HYG_LLaVA_SLoFo"
ENV_PREFIX="$PROJECT_ROOT/.conda/envs/llava-slofo-paper"

export PROJECT_ROOT
export ENV_PREFIX
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT/LLaVA:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT" || return 1

echo "Project: $PROJECT_ROOT"
echo "Python:  $ENV_PREFIX/bin/python"
echo "Run scripts/run_on_empty_gpu.sh <command> to use an empty GPU safely."

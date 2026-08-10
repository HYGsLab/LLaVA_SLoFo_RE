#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/workspace/Gexuri_Project/HYG_LLaVA_SLoFo"
MODEL_PATH="models/llava-v1.5-7b"
IMAGE_ROOT="images/test_08_06"
OUTPUT_ROOT="experiments/slofo-08-07/batch-rollout8-focus4"
LOG_ROOT="logs/slofo-08-07"

images=(
  "test_08_06_01.jpg"
  "test_08_06_02.jpg"
  "test_08_06_03.jpg"
  "test_08_06_04.jpg"
  "test_08_06_05.jpg"
  "test_08_06_06.jpg"
  "test_08_06_07.jpg"
  "test_08_06_08.jpg"
  "test_08_06_09.jpg"
  "test_08_06_10.jpg"
)

queries=(
  "What color are the clothes worn by the person holding a phone horizontally to take a photo?"
  "What color are the clothes worn by the person standing to the right of the snowman and making a V sign?"
  "What color are the clothes worn by the person standing in front and holding a water bottle with both hands?"
  "What color are the clothes worn by the person with one hand behind his neck and the other arm holding certificates?"
  "What color are the clothes worn by the person leaning forward against the pool table?"
  "What color are the clothes worn by the seated person using chopsticks to pick up food?"
  "What color are the clothes worn by the standing person holding a phone with both hands?"
  "What color are the clothes worn by the person sitting cross-legged on a chair and looking at a phone?"
  "What color are the clothes worn by the person lying down and holding a tablet?"
  "What color are the clothes worn by the person standing in front of the net and holding a badminton racket?"
)

cd "$PROJECT_ROOT"
source scripts/activate_project.sh >/dev/null
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

for index in "${!images[@]}"; do
  image_name="${images[$index]}"
  query="${queries[$index]}"
  case_id="${image_name%.jpg}"
  output_dir="$OUTPUT_ROOT/$case_id"
  log_file="$LOG_ROOT/$case_id.log"

  if [[ -f "$output_dir/result.json" && "${FORCE:-0}" != "1" ]]; then
    echo "[$case_id] existing result found; skipping."
    continue
  fi

  echo "[$case_id] starting"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    scripts/run_on_empty_gpu.sh \
    python scripts/run_slofo_scan_locate.py \
      --model-path "$MODEL_PATH" \
      --image-file "$IMAGE_ROOT/$image_name" \
      --query "$query" \
      --output-dir "$output_dir" \
      --fusion-normalization minmax \
      --locate-coordinate-space original \
      --semantic-rollout-tokens 8 \
      --semantic-anchor-start-index 0 \
      --semantic-token-aggregation mean \
      --semantic-score log_probability \
      --semantic-weight 0.7 \
      --enable-focus \
      --focus-phases 4 \
      --focus-prune-ratio 0.5 \
      --max-new-tokens 24 \
      --seed 0 2>&1 | tee "$log_file"
  echo "[$case_id] completed"
done

echo "All SLoFo 08-07 batch cases completed."

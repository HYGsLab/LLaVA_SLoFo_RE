# HYG LLaVA-SLoFo server workspace

Project root: `/data/workspace/Gexuri_Project/HYG_LLaVA_SLoFo`

Paper-compatible environment:
`/data/workspace/Gexuri_Project/HYG_LLaVA_SLoFo/.conda/envs/llava-slofo-paper`

## Start a terminal

```bash
source scripts/activate_project.sh
```

## Check aggregate GPU availability

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
```

Do not inspect other users' directories or process command lines.  Run GPU work
through the guard script so a currently occupied card is never selected:

```bash
scripts/run_on_empty_gpu.sh python scripts/check_cuda.py
```

## Directory layout

```text
HYG_LLaVA_SLoFo/
|-- LLaVA/          upstream LLaVA source
|-- slofo/          independent Scan-Locate implementation
|-- scripts/        activation, GPU guard, and experiment entry points
|-- models/         explicitly downloaded model files when needed
|-- images/         test images owned by this project
|-- experiments/    reproducible outputs
|-- logs/           run logs
|-- .cache/         Hugging Face cache (hidden by VS Code)
`-- .conda/         isolated environment (hidden by VS Code)
```

## Fixed paper baseline

- LLaVA checkpoint: `liuhaotian/llava-v1.5-7b`
  - revision: `4481d270cc22fd5c4d1bb5df129622006ccd9234`
- Vision tower: `openai/clip-vit-large-patch14-336`
  - revision: `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`
- Precision: FP16 (not 4-bit)
- Paper parameters: semantic layer 14, structure layer 7, PCA dimension 20,
  semantic weight beta 0.7.

The copied LLaVA `config.json` points `mm_vision_tower` to the local CLIP
directory, so both entry points can run with `HF_HUB_OFFLINE=1`.

## Reproduce the full LLaVA baseline

```bash
source scripts/activate_project.sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  scripts/run_on_empty_gpu.sh python scripts/run_llava_fp16.py \
  --model-path models/llava-v1.5-7b \
  --image-file images/validation-002-phone-holder-clothes.png \
  --output-json experiments/llava-fp16-baseline/result.json
```

Expected structural checks are a `336x336` image tensor, 576 CLIP tokens,
`576x4096` projected visual features, and 32 language layers.

## Reproduce Scan-Locate validation 002

```bash
source scripts/activate_project.sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  scripts/run_on_empty_gpu.sh python scripts/run_slofo_scan_locate.py \
  --model-path models/llava-v1.5-7b \
  --image-file images/validation-002-phone-holder-clothes.png \
  --output-dir experiments/slofo-scan-locate/validation-002-minmax-fixed-seed0 \
  --fusion-normalization minmax \
  --locate-coordinate-space original \
  --semantic-weight 0.7 \
  --seed 0
```

The paper equation does not document branch normalization.  The strictly raw
`beta * semantic + (1-beta) * structure` run is retained under
`validation-002-paper-default`; on this image the two branches differ by about
seven orders of magnitude and the crop is structure-dominated.  The explicit
`minmax` experiment balances the branches and is kept separate.  A regression
test protects tiny but non-constant semantic maps from being erased during
normalization.

Scan-Locate is implemented and validated.  The later four-phase Focus token
pruning stage is not yet implemented.

## Reproduce the 2026-08-06 dual-image batch

The experiment entry point now records three answer paths: original image,
crop only, and original plus crop as a true two-image prompt.  The number of
image placeholders is checked against the number of input images, and every
result JSON records the processed tensor shapes.

The semantic saliency product is evaluated in float32 even when LLaVA runs in
float16.  This prevents small positive `attention * gradient` values from
underflowing to zero before branch fusion.  The model itself remains FP16.

Run the ten owned test images sequentially through the empty-GPU guard:

```bash
source scripts/activate_project.sh
scripts/run_slofo_08_06_batch.sh
```

The batch compares raw/min-max branch fusion and original/padded coordinate
spaces.  Outputs are written below `experiments/slofo-08-06/`, and logs below
`logs/slofo-08-06/`.  On the current ten-image set, separate min-max scaling
has a much larger effect than the coordinate-space choice; `minmax + original`
is retained as the working baseline, not claimed as an undocumented paper
default.

#!/usr/bin/env python3
"""Run SLoFo Scan-Locate on the paper-compatible LLaVA-v1.5-7B model."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from slofo import (
    FocusConfig,
    ScanLocateConfig,
    crop_iou,
    fuse_importance,
    locate_from_importance_map,
    scan_locate_from_tensors,
    topk_crop_windows_from_importance_map,
)
from slofo_focus_runtime import focus_generation_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-file", required=True)
    parser.add_argument(
        "--query",
        default="What color are the clothes worn by the person holding a phone?",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--artifact-mode",
        choices=("full", "none"),
        default="full",
        help=(
            "Save all heatmaps/crops/overlays, or retain only result.json for "
            "large benchmark runs. Numerical inference is unchanged."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--semantic-rollout-tokens",
        type=int,
        default=1,
        help=(
            "Number of greedily planned answer tokens used by the semantic "
            "branch. Values above one create response-aware planning anchors."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--semantic-score",
        choices=("log_probability", "logit", "probability", "cross_entropy"),
        default="log_probability",
        help="Scalar pseudo-token target differentiated with respect to attention.",
    )
    parser.add_argument(
        "--diagnose-semantic-scores",
        action="store_true",
        help=(
            "Compute gradients for four plausible pseudo-token scalar definitions "
            "during the same forward pass and record their scales."
        ),
    )
    parser.add_argument("--fusion-normalization", choices=("none", "minmax"), default="none")
    parser.add_argument("--semantic-weight", type=float, default=0.7)
    parser.add_argument(
        "--scan-capture-mode",
        choices=("all_outputs", "selective_hook"),
        default="all_outputs",
        help=(
            "Capture every decoder attention/hidden state (legacy diagnostic path) "
            "or retain only the semantic attention layer and structure hidden layer."
        ),
    )
    parser.add_argument(
        "--semantic-token-aggregation",
        choices=("mean", "max"),
        default="mean",
        help=(
            "How to combine gradient-weighted semantic maps from multiple "
            "response-planning anchors. 'max' preserves action-token peaks."
        ),
    )
    parser.add_argument(
        "--semantic-anchor-start-index",
        type=int,
        default=0,
        help=(
            "Zero-based rollout token at which semantic scoring starts. For "
            "answers beginning with 'The person', 2 isolates action/content anchors."
        ),
    )
    parser.add_argument(
        "--enable-focus",
        action="store_true",
        help="Run the paper's four-phase original-image token pruning.",
    )
    parser.add_argument("--focus-phases", type=int, default=4)
    parser.add_argument("--focus-prune-ratio", type=float, default=0.5)
    parser.add_argument(
        "--random-focus-seeds",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Run matched random-pruning Focus controls for these seeds. "
            "Each control uses the same phase boundaries and token counts."
        ),
    )
    parser.add_argument(
        "--locate-coordinate-space",
        choices=("padded", "original"),
        default="padded",
        help=(
            "Use the square LLaVA padding geometry or reproduce ViCrop's public "
            "helper, which applies the 24x24 map directly to the original size."
        ),
    )
    parser.add_argument(
        "--compare-coordinate-spaces",
        action="store_true",
        help=(
            "Save and evaluate both direct-original and padded-square coordinate "
            "mappings from the same SSIM map."
        ),
    )
    parser.add_argument(
        "--compare-fusion-normalizations",
        action="store_true",
        help=(
            "Evaluate raw paper-equation fusion and per-branch min-max fusion "
            "from the same semantic and structure maps."
        ),
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=1,
        help=(
            "Number of distinct Scan-Locate proposals verified by LLaVA. "
            "Candidate 1 is the legacy single-window result."
        ),
    )
    parser.add_argument("--rerank-pre-nms-per-scale", type=int, default=12)
    parser.add_argument("--rerank-nms-iou", type=float, default=0.55)
    parser.add_argument(
        "--rerank-scan-weight",
        type=float,
        default=0.15,
        help="Weight of normalized Scan-Locate contrast added to Yes/No log-odds.",
    )
    parser.add_argument("--rerank-answer-consistency-weight", type=float, default=1.0)
    parser.add_argument(
        "--rerank-verification-mode",
        choices=("person-clothing", "generic-evidence"),
        default="person-clothing",
        help=(
            "Prompt used to verify top-k crops. The default preserves the original "
            "person/clothing experiments; generic-evidence supports arbitrary "
            "visual questions without assuming a person or clothing attribute."
        ),
    )
    parser.add_argument(
        "--rerank-min-improvement",
        type=float,
        default=0.2,
        help="Keep the legacy crop unless the best proposal exceeds it by this margin.",
    )
    parser.add_argument("--rerank-candidate-answer-tokens", type=int, default=16)
    parser.add_argument(
        "--ground-truth-bbox",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional evaluation-only target-person bbox; never used for reranking.",
    )
    return parser.parse_args()


def image_token_text(model: torch.nn.Module) -> str:
    if model.config.mm_use_im_start_end:
        return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    return DEFAULT_IMAGE_TOKEN


def build_prompt(
    query: str,
    model_name: str,
    model: torch.nn.Module,
    *,
    image_count: int = 1,
) -> str:
    if image_count < 1:
        raise ValueError("image_count must be at least 1.")

    placeholder_count = query.count(IMAGE_PLACEHOLDER)
    if placeholder_count:
        if placeholder_count != image_count:
            raise ValueError(
                f"Query contains {placeholder_count} image placeholders, but "
                f"{image_count} images were supplied."
            )
        query = re.sub(IMAGE_PLACEHOLDER, image_token_text(model), query)
    else:
        prefix = "\n".join(image_token_text(model) for _ in range(image_count))
        query = prefix + "\n" + query

    conv_mode = "llava_v1" if "v1" in model_name.lower() else "llava_v0"
    conversation = conv_templates[conv_mode].copy()
    conversation.append_message(conversation.roles[0], query)
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


def prepare_inputs(
    images: list[Image.Image],
    prompt: str,
    tokenizer,
    image_processor,
    model,
):
    if not images:
        raise ValueError("At least one image is required.")
    image_tensor = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)
    image_positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(
        as_tuple=False
    ).flatten()
    if image_positions.numel() != len(images):
        raise ValueError(
            f"Expected {len(images)} image placeholders, found "
            f"{image_positions.numel()}."
        )
    return input_ids, image_tensor, [image.size for image in images]


def generate_answer(
    images: list[Image.Image],
    query: str,
    model_name: str,
    tokenizer,
    image_processor,
    model,
    max_new_tokens: int,
    *,
    focus_config: FocusConfig | None = None,
    capture_scores: bool = False,
) -> tuple[str, dict[str, object]]:
    prompt = build_prompt(
        query,
        model_name,
        model,
        image_count=len(images),
    )
    input_ids, image_tensor, image_sizes = prepare_inputs(
        images,
        prompt,
        tokenizer,
        image_processor,
        model,
    )
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    focus_trace = None
    with torch.inference_mode():
        if focus_config is None:
            generated = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                return_dict_in_generate=capture_scores,
                output_scores=capture_scores,
            )
        else:
            if len(images) != 2:
                raise ValueError("Focus requires [original_image, cropped_image].")
            with focus_generation_context(
                model,
                input_ids,
                image_token_index=IMAGE_TOKEN_INDEX,
                tokens_per_image=576,
                config=focus_config,
            ) as focus_context:
                generated = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    return_dict_in_generate=capture_scores,
                    output_scores=capture_scores,
                )
            focus_trace = focus_context.trace
    elapsed_seconds = time.perf_counter() - start_time
    output_ids = generated.sequences if capture_scores else generated
    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    metadata: dict[str, object] = {
        "image_count": len(images),
        "image_sizes": [list(size) for size in image_sizes],
        "image_tensor_shape": list(image_tensor.shape),
        "input_ids_shape": list(input_ids.shape),
        "output_ids_shape": list(output_ids.shape),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "elapsed_seconds": round(elapsed_seconds, 4),
    }
    if focus_trace is not None:
        metadata["focus"] = focus_trace
    if capture_scores:
        if not generated.scores:
            raise RuntimeError("Generation score capture produced no decoding steps.")
        score_logits = torch.stack(generated.scores, dim=0)[:, 0, :].detach().float().cpu()
        generated_token_ids = output_ids[0, -score_logits.shape[0] :].detach().cpu()
        score_log_probabilities = F.log_softmax(score_logits, dim=-1)
        top_values, top_ids = torch.topk(score_logits, k=2, dim=-1)
        selected_log_probabilities = score_log_probabilities.gather(
            dim=-1,
            index=generated_token_ids.unsqueeze(-1),
        ).squeeze(-1)
        metadata["score_trace"] = [
            {
                "step": step,
                "generated_token_id": int(generated_token_ids[step].item()),
                "generated_token": tokenizer.decode(
                    [int(generated_token_ids[step].item())],
                    skip_special_tokens=False,
                ),
                "generated_token_log_probability": float(
                    selected_log_probabilities[step].item()
                ),
                "top1_token_id": int(top_ids[step, 0].item()),
                "top1_token": tokenizer.decode(
                    [int(top_ids[step, 0].item())],
                    skip_special_tokens=False,
                ),
                "top2_token_id": int(top_ids[step, 1].item()),
                "top1_logit_margin": float(
                    (top_values[step, 0] - top_values[step, 1]).item()
                ),
            }
            for step in range(score_logits.shape[0])
        ]
        metadata["_score_logits"] = score_logits
        metadata["_generated_token_ids"] = generated_token_ids
        del score_log_probabilities, top_values, top_ids, selected_log_probabilities
    del input_ids, image_tensor, output_ids, generated
    torch.cuda.empty_cache()
    return answer, metadata


def build_candidate_verification_query(
    query: str,
    mode: str = "person-clothing",
) -> str:
    """Ask LLaVA to verify a crop without revealing the expected answer."""

    prefix = (
        "The first image is the full scene. The second image is a candidate crop. "
        f"The target question is: {query} "
    )
    if mode == "person-clothing":
        return (
            prefix
            + "Does the second image contain the specific person referred to by the "
            "target question clearly enough to inspect that person's clothing? "
            "Answer only Yes or No."
        )
    if mode == "generic-evidence":
        return (
            prefix
            + "Does the second image contain the visual evidence needed to answer "
            "the target question clearly? Answer only Yes or No."
        )
    raise ValueError(f"Unsupported verification mode: {mode}")


def _first_token_ids(tokenizer, variants: tuple[str, ...]) -> list[int]:
    token_ids: set[int] = set()
    for text in variants:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded:
            token_ids.add(int(encoded[0]))
    if not token_ids:
        raise RuntimeError(f"Tokenizer produced no IDs for {variants!r}.")
    return sorted(token_ids)


def score_candidate_presence(
    original_image: Image.Image,
    crop_image: Image.Image,
    query: str,
    model_name: str,
    tokenizer,
    image_processor,
    model,
    verification_mode: str = "person-clothing",
) -> dict[str, object]:
    """Return leakage-free Yes-vs-No log-odds for one candidate crop."""

    verification_query = build_candidate_verification_query(query, verification_mode)
    answer, generation = generate_answer(
        [original_image, crop_image],
        verification_query,
        model_name,
        tokenizer,
        image_processor,
        model,
        1,
        capture_scores=True,
    )
    score_logits = generation.pop("_score_logits")[0]
    generation.pop("_generated_token_ids")
    yes_ids = _first_token_ids(tokenizer, ("Yes", " yes", "YES"))
    no_ids = _first_token_ids(tokenizer, ("No", " no", "NO"))
    yes_logit = torch.logsumexp(score_logits[yes_ids], dim=0)
    no_logit = torch.logsumexp(score_logits[no_ids], dim=0)
    log_odds = float((yes_logit - no_logit).item())
    probability = float(torch.sigmoid(yes_logit - no_logit).item())
    del score_logits, yes_logit, no_logit
    return {
        "verifier_answer": answer,
        "yes_token_ids": yes_ids,
        "no_token_ids": no_ids,
        "yes_no_log_odds": log_odds,
        "yes_probability": probability,
        "generation": generation,
    }


def bbox_metrics(
    predicted: tuple[int, int, int, int],
    ground_truth: tuple[int, int, int, int],
) -> dict[str, object]:
    """Compute localization metrics used only after candidate selection."""

    px1, py1, px2, py2 = predicted
    gx1, gy1, gx2, gy2 = ground_truth
    intersection_width = max(0, min(px2, gx2) - max(px1, gx1))
    intersection_height = max(0, min(py2, gy2) - max(py1, gy1))
    intersection = intersection_width * intersection_height
    predicted_area = max(0, px2 - px1) * max(0, py2 - py1)
    ground_truth_area = max(0, gx2 - gx1) * max(0, gy2 - gy1)
    union = predicted_area + ground_truth_area - intersection
    center_x = (gx1 + gx2) / 2.0
    center_y = (gy1 + gy2) / 2.0
    return {
        "iou": float(intersection / union) if union > 0 else 0.0,
        "target_coverage": (
            float(intersection / ground_truth_area) if ground_truth_area > 0 else 0.0
        ),
        "target_center_inside": bool(
            px1 <= center_x <= px2 and py1 <= center_y <= py2
        ),
    }


_COLOR_ALIASES = {
    "black": ("black",),
    "white": ("white",),
    "red": ("red", "crimson", "maroon"),
    "blue": ("blue", "navy", "cyan", "turquoise"),
    "gray": ("gray", "grey", "silver"),
    "brown": ("brown", "tan", "khaki", "beige"),
    "green": ("green", "camouflage", "camo"),
    "pink": ("pink",),
    "purple": ("purple", "violet"),
    "yellow": ("yellow", "gold"),
    "orange": ("orange",),
    "multicolored": ("multicolored", "multi-colored", "colorful", "colourful"),
}


def extract_color_terms(answer: str) -> list[str]:
    """Map common clothing-color words to stable canonical terms."""

    normalized = answer.lower()
    return sorted(
        canonical
        for canonical, aliases in _COLOR_ALIASES.items()
        if any(re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in aliases)
    )


def answer_color_consistency(reference_answer: str, candidate_answer: str) -> float:
    """Measure crop-only color agreement with the original-image pseudo-answer."""

    reference = set(extract_color_terms(reference_answer))
    candidate = set(extract_color_terms(candidate_answer))
    if not reference:
        return 0.0
    return float(len(reference & candidate) / len(reference))


def compare_generation_logits(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    reference_token_ids: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    tokenizer,
) -> dict[str, object]:
    """Compare decoding distributions without storing full-vocabulary logits."""

    reference_steps = int(reference_logits.shape[0])
    candidate_steps = int(candidate_logits.shape[0])
    exact_generated_token_ids = bool(
        reference_token_ids.shape == candidate_token_ids.shape
        and torch.equal(reference_token_ids, candidate_token_ids)
    )
    step_count = min(reference_steps, candidate_steps)
    reference_logits = reference_logits[:step_count].float()
    candidate_logits = candidate_logits[:step_count].float()
    reference_token_ids = reference_token_ids[:step_count]
    candidate_token_ids = candidate_token_ids[:step_count]
    reference_logp = F.log_softmax(reference_logits, dim=-1)
    candidate_logp = F.log_softmax(candidate_logits, dim=-1)
    reference_p = reference_logp.exp()
    candidate_p = candidate_logp.exp()
    kl_reference_candidate = (
        reference_p * (reference_logp - candidate_logp)
    ).sum(dim=-1)
    kl_candidate_reference = (
        candidate_p * (candidate_logp - reference_logp)
    ).sum(dim=-1)
    reference_top_values, reference_top_ids = torch.topk(
        reference_logits, k=2, dim=-1
    )
    candidate_top_values, candidate_top_ids = torch.topk(
        candidate_logits, k=2, dim=-1
    )
    reference_selected_logp = reference_logp.gather(
        -1, reference_token_ids.unsqueeze(-1)
    ).squeeze(-1)
    candidate_reference_token_logp = candidate_logp.gather(
        -1, reference_token_ids.unsqueeze(-1)
    ).squeeze(-1)
    common_prefix = 0
    for reference_id, candidate_id in zip(
        reference_token_ids.tolist(), candidate_token_ids.tolist()
    ):
        if reference_id != candidate_id:
            break
        common_prefix += 1
    symmetric_kl = 0.5 * (kl_reference_candidate + kl_candidate_reference)
    per_step = []
    for step in range(step_count):
        per_step.append(
            {
                "step": step,
                "reference_token": tokenizer.decode(
                    [int(reference_token_ids[step].item())],
                    skip_special_tokens=False,
                ),
                "candidate_token": tokenizer.decode(
                    [int(candidate_token_ids[step].item())],
                    skip_special_tokens=False,
                ),
                "same_generated_token": bool(
                    reference_token_ids[step] == candidate_token_ids[step]
                ),
                "same_top1_token": bool(
                    reference_top_ids[step, 0] == candidate_top_ids[step, 0]
                ),
                "reference_top1_margin": float(
                    (
                        reference_top_values[step, 0]
                        - reference_top_values[step, 1]
                    ).item()
                ),
                "candidate_top1_margin": float(
                    (
                        candidate_top_values[step, 0]
                        - candidate_top_values[step, 1]
                    ).item()
                ),
                "reference_token_log_probability": float(
                    reference_selected_logp[step].item()
                ),
                "candidate_log_probability_for_reference_token": float(
                    candidate_reference_token_logp[step].item()
                ),
                "kl_reference_to_candidate": float(
                    kl_reference_candidate[step].item()
                ),
                "kl_candidate_to_reference": float(
                    kl_candidate_reference[step].item()
                ),
                "symmetric_kl": float(symmetric_kl[step].item()),
            }
        )
    return {
        "compared_steps": step_count,
        "reference_steps": reference_steps,
        "candidate_steps": candidate_steps,
        "exact_generated_token_ids": exact_generated_token_ids,
        "common_prefix_tokens": common_prefix,
        "top1_agreement_fraction": float(
            (reference_top_ids[:, 0] == candidate_top_ids[:, 0]).float().mean().item()
        ),
        "mean_kl_reference_to_candidate": float(
            kl_reference_candidate.mean().item()
        ),
        "max_kl_reference_to_candidate": float(
            kl_reference_candidate.max().item()
        ),
        "mean_kl_candidate_to_reference": float(
            kl_candidate_reference.mean().item()
        ),
        "mean_symmetric_kl": float(symmetric_kl.mean().item()),
        "max_symmetric_kl": float(symmetric_kl.max().item()),
        "mean_reference_top1_margin": float(
            (reference_top_values[:, 0] - reference_top_values[:, 1]).mean().item()
        ),
        "mean_candidate_top1_margin": float(
            (candidate_top_values[:, 0] - candidate_top_values[:, 1]).mean().item()
        ),
        "per_step": per_step,
    }


def padded_geometry(image: Image.Image) -> tuple[int, int, int]:
    width, height = image.size
    side = max(width, height)
    offset_x = (side - width) // 2
    offset_y = (side - height) // 2
    return side, offset_x, offset_y


def remap_crop_to_original(
    padded_bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    offset_x: int,
    offset_y: int,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = padded_bbox
    center_x = (x1 + x2) / 2.0 - offset_x
    center_y = (y1 + y2) / 2.0 - offset_y
    crop_size = min(x2 - x1, y2 - y1, width, height)
    original_x1 = int(round(center_x - crop_size / 2.0))
    original_y1 = int(round(center_y - crop_size / 2.0))
    original_x1 = min(max(original_x1, 0), width - crop_size)
    original_y1 = min(max(original_y1, 0), height - crop_size)
    return (
        original_x1,
        original_y1,
        original_x1 + crop_size,
        original_y1 + crop_size,
    )


def normalized_map(values: torch.Tensor) -> np.ndarray:
    work = values.detach().float().cpu()
    minimum = work.amin()
    span = work.amax() - minimum
    if float(span) <= 1e-12:
        return np.zeros(tuple(work.shape), dtype=np.float32)
    return ((work - minimum) / span).numpy()


def colorize(values: torch.Tensor) -> Image.Image:
    normalized = normalized_map(values)
    red = normalized
    green = 1.0 - np.abs(2.0 * normalized - 1.0)
    blue = 1.0 - normalized
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray(np.uint8(np.clip(rgb * 255.0, 0, 255)), mode="RGB")


def save_map_visualizations(
    image: Image.Image,
    semantic_map: torch.Tensor,
    structure_map: torch.Tensor,
    ssim_map: torch.Tensor,
    original_bbox: tuple[int, int, int, int],
    output_dir: Path,
) -> None:
    side, offset_x, offset_y = padded_geometry(image)
    resampling = Image.Resampling.BILINEAR
    maps = {
        "semantic": semantic_map,
        "structure": structure_map,
        "ssim": ssim_map,
    }
    original_rgb = image.convert("RGB")
    for name, values in maps.items():
        np.save(output_dir / f"{name}_map.npy", values.detach().float().cpu().numpy())
        small = colorize(values)
        small.save(output_dir / f"{name}_map_24x24.png")
        padded_heat = small.resize((side, side), resampling)
        original_heat = padded_heat.crop(
            (offset_x, offset_y, offset_x + image.width, offset_y + image.height)
        )
        original_heat.save(output_dir / f"{name}_heatmap.png")
        Image.blend(original_rgb, original_heat, alpha=0.45).save(
            output_dir / f"{name}_overlay.png"
        )

    boxed = original_rgb.copy()
    draw = ImageDraw.Draw(boxed)
    line_width = max(3, round(min(image.size) / 150))
    draw.rectangle(original_bbox, outline=(255, 0, 0), width=line_width)
    boxed.save(output_dir / "selected_bbox.png")
    original_rgb.crop(original_bbox).save(output_dir / "crop.png")


def save_focus_token_visualizations(
    image: Image.Image,
    focus_trace: dict[str, object],
    output_dir: Path,
    *,
    artifact_prefix: str = "focus",
) -> list[str]:
    """Render the original-image patches retained after each Focus phase."""

    initial_tokens = int(focus_trace["initial_original_tokens"])
    grid_side = int(round(initial_tokens**0.5))
    if grid_side * grid_side != initial_tokens:
        raise ValueError("Focus visualization expects a square original-token grid.")
    padded_side, offset_x, offset_y = padded_geometry(image)
    original_rgb = image.convert("RGB")
    dimmed = ImageEnhance.Brightness(original_rgb).enhance(0.2)
    artifact_names: list[str] = []
    for stage in focus_trace["stages"]:
        kept_ids = np.asarray(stage["kept_original_token_ids"], dtype=np.int64)
        mask = np.zeros(initial_tokens, dtype=np.uint8)
        mask[kept_ids] = 255
        small_mask = Image.fromarray(mask.reshape(grid_side, grid_side), mode="L")
        padded_mask = small_mask.resize(
            (padded_side, padded_side), Image.Resampling.NEAREST
        )
        original_mask = padded_mask.crop(
            (offset_x, offset_y, offset_x + image.width, offset_y + image.height)
        )
        visualized = Image.composite(original_rgb, dimmed, original_mask)
        artifact_name = (
            f"{artifact_prefix}_phase_{int(stage['phase'])}_kept_tokens.png"
        )
        visualized.save(output_dir / artifact_name)
        artifact_names.append(artifact_name)
    return artifact_names


def save_coordinate_crop(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    output_dir: Path,
    name: str,
) -> dict[str, str]:
    original_rgb = image.convert("RGB")
    boxed = original_rgb.copy()
    draw = ImageDraw.Draw(boxed)
    line_width = max(3, round(min(image.size) / 150))
    draw.rectangle(bbox, outline=(255, 0, 0), width=line_width)
    bbox_name = f"{name}_selected_bbox.png"
    crop_name = f"{name}_crop.png"
    boxed.save(output_dir / bbox_name)
    original_rgb.crop(bbox).save(output_dir / crop_name)
    return {"selected_bbox": bbox_name, "crop": crop_name}


def save_topk_candidate_visualizations(
    image: Image.Image,
    candidates: list[dict[str, object]],
    output_dir: Path,
    selected_rank: int,
) -> dict[str, object]:
    """Save each proposal plus a single numbered overview image."""

    original_rgb = image.convert("RGB")
    overview = original_rgb.copy()
    draw = ImageDraw.Draw(overview)
    line_width = max(3, round(min(image.size) / 150))
    colors = (
        (255, 0, 0),
        (0, 170, 255),
        (255, 170, 0),
        (180, 0, 255),
        (0, 200, 90),
        (255, 80, 160),
    )
    artifact_rows: list[dict[str, str]] = []
    for row in candidates:
        rank = int(row["rank"])
        bbox = tuple(int(value) for value in row["original_bbox"])
        color = colors[(rank - 1) % len(colors)]
        draw.rectangle(bbox, outline=color, width=line_width)
        label = f"#{rank}{' SELECTED' if rank == selected_rank else ''}"
        text_xy = (bbox[0] + line_width, max(0, bbox[1] - 16))
        draw.text(text_xy, label, fill=color, stroke_width=2, stroke_fill=(0, 0, 0))
        bbox_name = f"topk_candidate_{rank:02d}_bbox.png"
        crop_name = f"topk_candidate_{rank:02d}_crop.png"
        boxed = original_rgb.copy()
        ImageDraw.Draw(boxed).rectangle(bbox, outline=color, width=line_width)
        boxed.save(output_dir / bbox_name)
        original_rgb.crop(bbox).save(output_dir / crop_name)
        artifact_rows.append({"bbox": bbox_name, "crop": crop_name})
    overview_name = "topk_candidates_overview.png"
    overview.save(output_dir / overview_name)
    return {"overview": overview_name, "candidates": artifact_rows}


def tensor_stats(values: torch.Tensor) -> dict[str, float]:
    work = values.detach().float()
    return {
        "min": float(work.amin().item()),
        "max": float(work.amax().item()),
        "mean": float(work.mean().item()),
        "sum": float(work.sum().item()),
    }


def signed_tensor_stats(values: torch.Tensor) -> dict[str, object]:
    work = values.detach().float()
    return {
        **tensor_stats(work),
        "abs_max": float(work.abs().amax().item()),
        "abs_mean": float(work.abs().mean().item()),
        "positive_fraction": float((work > 0).float().mean().item()),
        "negative_fraction": float((work < 0).float().mean().item()),
        "zero_fraction": float((work == 0).float().mean().item()),
        "dtype": str(values.dtype),
    }


def build_semantic_scores(
    planning_logits: torch.Tensor,
    target_token_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if planning_logits.ndim != 3 or target_token_ids.ndim != 2:
        raise ValueError("Expected logits [batch, anchors, vocab] and token IDs [batch, anchors].")
    if planning_logits.shape[:2] != target_token_ids.shape:
        raise ValueError("Planning logits and target token IDs must share batch/anchor axes.")
    gather_index = target_token_ids.unsqueeze(-1)
    selected_logit = planning_logits.gather(dim=-1, index=gather_index).mean()
    selected_log_probability = F.log_softmax(planning_logits, dim=-1).gather(
        dim=-1, index=gather_index
    ).mean()
    selected_probability = F.softmax(planning_logits, dim=-1).gather(
        dim=-1, index=gather_index
    ).mean()
    return {
        "log_probability": selected_log_probability,
        "logit": selected_logit,
        "probability": selected_probability,
        "cross_entropy": -selected_log_probability,
    }


def generate_semantic_rollout(
    input_ids: torch.Tensor,
    images: torch.Tensor,
    image_sizes: list[tuple[int, int]],
    model,
    *,
    token_count: int,
    bos_token_id: int | None,
) -> tuple[torch.Tensor, float]:
    if token_count < 1:
        raise ValueError("semantic_rollout_tokens must be at least 1.")
    start_time = time.perf_counter()
    with torch.inference_mode():
        rollout_ids = model.generate(
            input_ids,
            images=images,
            image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=token_count,
            use_cache=True,
        )
    # Tensors created by inference_mode carry an inference-only flag even after
    # leaving the context.  Clone once in normal mode so the IDs can safely be
    # used as gather indices by the differentiable teacher-forced pass.
    rollout_ids = rollout_ids.clone()
    if (
        bos_token_id is not None
        and rollout_ids.shape[1] > 1
        and int(rollout_ids[0, 0].item()) == bos_token_id
    ):
        rollout_ids = rollout_ids[:, 1:]
    rollout_ids = rollout_ids[:, :token_count]
    if rollout_ids.ndim != 2 or rollout_ids.shape[0] != 1 or rollout_ids.shape[1] < 1:
        raise RuntimeError("Semantic rollout did not produce any answer token.")
    return rollout_ids, time.perf_counter() - start_time


def decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Return the LLaMA decoder layers without depending on one wrapper name."""
    base_model = getattr(model, "model", None)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        raise TypeError("The loaded model does not expose model.layers.")
    return layers


def configure_selective_semantic_gradient(
    model: torch.nn.Module,
    semantic_layer: int,
) -> None:
    """Keep autograd only from the semantic attention layer onward.

    The semantic attention tensor must require gradients, but gradients for all
    model parameters are unnecessary.  Freezing the model and re-enabling only
    the target q/k projections prevents autograd from retaining the vision tower
    and all decoder layers before the semantic probe.
    """
    model.requires_grad_(False)
    target_attention = decoder_layers(model)[semantic_layer].self_attn
    for projection_name in ("q_proj", "k_proj"):
        projection = getattr(target_attention, projection_name, None)
        if projection is None:
            raise TypeError(
                f"Semantic attention does not expose {projection_name}."
            )
        projection.requires_grad_(True)


def selective_scan_forward(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    images: torch.Tensor,
    image_sizes: list[tuple[int, int]],
    semantic_layer: int,
    structure_layer: int,
) -> tuple[object, torch.Tensor, torch.Tensor]:
    """Run one differentiable forward while retaining only two target tensors."""
    layers = decoder_layers(model)
    if max(semantic_layer, structure_layer) >= len(layers):
        raise ValueError(
            f"Requested layers {semantic_layer}/{structure_layer}, "
            f"but the decoder has {len(layers)} layers."
        )

    configure_selective_semantic_gradient(model, semantic_layer)
    captured: dict[str, torch.Tensor] = {}
    semantic_attention = layers[semantic_layer].self_attn
    original_attention_forward = semantic_attention.forward

    def force_target_attention(*forward_args, **forward_kwargs):
        forward_kwargs["output_attentions"] = True
        return original_attention_forward(*forward_args, **forward_kwargs)

    def capture_attention(_module, _inputs, output):
        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            raise RuntimeError("The semantic attention hook did not receive weights.")
        captured["attention"] = output[1]

    def capture_structure(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["structure"] = hidden.detach()

    attention_handle = semantic_attention.register_forward_hook(capture_attention)
    structure_handle = layers[structure_layer].register_forward_hook(capture_structure)
    semantic_attention.forward = force_target_attention
    try:
        outputs = model(
            input_ids=input_ids,
            images=images,
            image_sizes=image_sizes,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
    finally:
        semantic_attention.forward = original_attention_forward
        attention_handle.remove()
        structure_handle.remove()

    missing = {"attention", "structure"} - captured.keys()
    if missing:
        raise RuntimeError(f"Selective Scan hooks did not capture: {sorted(missing)}")
    return outputs, captured["attention"], captured["structure"]


def main() -> None:
    args = parse_args()
    if args.rerank_top_k < 1:
        raise ValueError("rerank_top_k must be positive.")
    if args.rerank_pre_nms_per_scale < 1:
        raise ValueError("rerank_pre_nms_per_scale must be positive.")
    if not 0.0 <= args.rerank_nms_iou <= 1.0:
        raise ValueError("rerank_nms_iou must be in [0, 1].")
    if args.rerank_candidate_answer_tokens < 1:
        raise ValueError("rerank_candidate_answer_tokens must be positive.")
    if args.rerank_min_improvement < 0.0:
        raise ValueError("rerank_min_improvement must be non-negative.")
    image_path = Path(args.image_file)
    output_dir = Path(args.output_dir)
    save_artifacts = args.artifact_mode == "full"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        model_base=None,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    config = ScanLocateConfig(
        fusion_normalization=args.fusion_normalization,
        semantic_weight=args.semantic_weight,
        semantic_token_aggregation=args.semantic_token_aggregation,
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    image = Image.open(image_path).convert("RGB")
    prompt = build_prompt(args.query, model_name, model, image_count=1)
    input_ids, images, image_sizes = prepare_inputs(
        [image], prompt, tokenizer, image_processor, model
    )
    image_positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
    if image_positions.numel() != 1:
        raise ValueError(f"Expected one image token, found {image_positions.numel()}.")

    visual_start = int(image_positions.item())
    num_visual_tokens = config.patch_grid[0] * config.patch_grid[1]
    visual_end = visual_start + num_visual_tokens
    torch.cuda.reset_peak_memory_stats()

    rollout_ids, rollout_seconds = generate_semantic_rollout(
        input_ids,
        images,
        image_sizes,
        model,
        token_count=args.semantic_rollout_tokens,
        bos_token_id=tokenizer.bos_token_id,
    )
    rollout_count = rollout_ids.shape[1]
    if not 0 <= args.semantic_anchor_start_index < rollout_count:
        raise ValueError(
            "semantic_anchor_start_index must select at least one rollout token; "
            f"got {args.semantic_anchor_start_index} for {rollout_count} tokens."
        )
    # Teacher-force all but the final rollout token.  Each target token is then
    # scored at the anchor immediately before it, giving later content/action
    # words their own response-planning attention instead of relying only on a
    # generic first token such as "The".
    teacher_input_ids = torch.cat([input_ids, rollout_ids[:, :-1]], dim=1)
    structure_tuple_index = config.structure_layer + 1
    if args.scan_capture_mode == "selective_hook":
        outputs, attention_full, captured_structure = selective_scan_forward(
            model,
            input_ids=teacher_input_ids,
            images=images,
            image_sizes=image_sizes,
            semantic_layer=config.semantic_layer,
            structure_layer=config.structure_layer,
        )
        structure_hidden = captured_structure[0, visual_start:visual_end, :]
        del captured_structure
    else:
        outputs = model(
            input_ids=teacher_input_ids,
            images=images,
            image_sizes=image_sizes,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        if len(outputs.attentions) <= config.semantic_layer:
            raise ValueError("The requested semantic layer is not available.")
        if len(outputs.hidden_states) <= structure_tuple_index:
            raise ValueError("The requested structure layer is not available.")
        attention_full = outputs.attentions[config.semantic_layer]
        structure_hidden = outputs.hidden_states[structure_tuple_index][
            0, visual_start:visual_end, :
        ].detach()

    sequence_length = attention_full.shape[-1]
    if visual_end > sequence_length:
        raise ValueError(
            f"Visual span [{visual_start}, {visual_end}) exceeds sequence length "
            f"{sequence_length}."
        )
    base_multimodal_sequence_length = sequence_length - (rollout_count - 1)
    planning_anchor_positions = torch.arange(
        base_multimodal_sequence_length - 1,
        base_multimodal_sequence_length - 1 + rollout_count,
        device=attention_full.device,
    )
    all_planning_attention = attention_full[
        0, :, :, visual_start:visual_end
    ].index_select(1, planning_anchor_positions).detach()
    planning_logits = outputs.logits.index_select(1, planning_anchor_positions)
    anchor_start = args.semantic_anchor_start_index
    scored_planning_logits = planning_logits[:, anchor_start:, :]
    scored_rollout_ids = rollout_ids[:, anchor_start:]
    planning_attention = all_planning_attention[:, anchor_start:, :]
    semantic_scores = build_semantic_scores(
        scored_planning_logits,
        scored_rollout_ids,
    )
    rollout_log_probabilities = F.log_softmax(
        planning_logits.detach().float(), dim=-1
    ).gather(dim=-1, index=rollout_ids.unsqueeze(-1)).squeeze(-1)
    score_names = (
        list(semantic_scores)
        if args.diagnose_semantic_scores
        else [args.semantic_score]
    )
    if args.semantic_score not in score_names:
        score_names.append(args.semantic_score)

    gradient_slices: dict[str, torch.Tensor] = {}
    semantic_score_diagnostics: dict[str, dict[str, object]] = {}
    for score_index, score_name in enumerate(score_names):
        attention_gradient_full = torch.autograd.grad(
            semantic_scores[score_name],
            attention_full,
            retain_graph=score_index < len(score_names) - 1,
            create_graph=False,
            allow_unused=False,
        )[0]
        all_gradient_slice = attention_gradient_full[
            0, :, :, visual_start:visual_end
        ].index_select(1, planning_anchor_positions).detach()
        gradient_slice = all_gradient_slice[:, anchor_start:, :]
        gradient_slices[score_name] = gradient_slice

        semantic_native = (
            planning_attention * gradient_slice.clamp_min(0)
        ).reshape(-1, num_visual_tokens).mean(dim=0)
        semantic_float32 = (
            planning_attention.float() * gradient_slice.float().clamp_min(0)
        ).reshape(-1, num_visual_tokens).mean(dim=0)
        semantic_score_diagnostics[score_name] = {
            "score_value": float(semantic_scores[score_name].detach().float().item()),
            "gradient": signed_tensor_stats(gradient_slice),
            "semantic_native_precision": signed_tensor_stats(semantic_native),
            "semantic_float32": signed_tensor_stats(semantic_float32),
            "native_nonzero_tokens": int((semantic_native != 0).sum().item()),
            "float32_nonzero_tokens": int((semantic_float32 != 0).sum().item()),
        }
        del (
            attention_gradient_full,
            all_gradient_slice,
            gradient_slice,
            semantic_native,
            semantic_float32,
        )

    attention_gradient = gradient_slices[args.semantic_score]
    scan_peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    pseudo_token_id = rollout_ids[:, 0]
    pseudo_token_text = tokenizer.decode(pseudo_token_id.tolist()).strip()
    rollout_text = tokenizer.decode(
        rollout_ids[0].tolist(), skip_special_tokens=True
    ).strip()
    semantic_rollout = {
        "requested_tokens": args.semantic_rollout_tokens,
        "actual_tokens": rollout_count,
        "text": rollout_text,
        "token_ids": rollout_ids[0].detach().cpu().tolist(),
        "tokens": [
            tokenizer.decode([int(token_id)], skip_special_tokens=False)
            for token_id in rollout_ids[0].detach().cpu().tolist()
        ],
        "anchor_start_index": anchor_start,
        "selected_anchor_tokens": [
            tokenizer.decode([int(token_id)], skip_special_tokens=False)
            for token_id in scored_rollout_ids[0].detach().cpu().tolist()
        ],
        "token_log_probabilities": rollout_log_probabilities[0].cpu().tolist(),
        "planning_anchor_positions": planning_anchor_positions[
            anchor_start:
        ].cpu().tolist(),
        "score_aggregation": "mean",
        "map_aggregation": args.semantic_token_aggregation,
        "elapsed_seconds": round(rollout_seconds, 4),
    }

    del (
        outputs,
        attention_full,
        planning_logits,
        semantic_scores,
        gradient_slices,
        teacher_input_ids,
        rollout_log_probabilities,
    )
    torch.cuda.empty_cache()

    padded_side, offset_x, offset_y = padded_geometry(image)
    locate_size = (
        (padded_side, padded_side)
        if args.locate_coordinate_space == "padded"
        else image.size
    )
    result = scan_locate_from_tensors(
        planning_attention,
        structure_hidden,
        locate_size,
        attention_gradient=attention_gradient,
        config=config,
    )
    source_proposals = topk_crop_windows_from_importance_map(
        result.ssim_map,
        locate_size,
        top_k=args.rerank_top_k,
        base_crop_size=config.base_crop_size,
        ratios=config.window_ratios,
        pre_nms_per_scale=args.rerank_pre_nms_per_scale,
        nms_iou_threshold=args.rerank_nms_iou,
    )
    proposal_rows: list[dict[str, object]] = []
    for rank, source_crop in enumerate(source_proposals, start=1):
        proposal_bbox = (
            remap_crop_to_original(
                source_crop.bbox,
                image.size,
                offset_x,
                offset_y,
            )
            if args.locate_coordinate_space == "padded"
            else source_crop.bbox
        )
        if any(tuple(row["original_bbox"]) == tuple(proposal_bbox) for row in proposal_rows):
            continue
        proposal_rows.append(
            {
                "rank": len(proposal_rows) + 1,
                "source_bbox": list(source_crop.bbox),
                "original_bbox": list(proposal_bbox),
                "selected_ratio": source_crop.selected_ratio,
                "evidence_sum": source_crop.evidence_sum,
                "contrast": source_crop.contrast,
                "is_legacy_baseline": rank == 1,
            }
        )
    if not proposal_rows:
        raise RuntimeError("Top-k proposal generation produced no usable crop.")

    contrasts = [float(row["contrast"]) for row in proposal_rows]
    contrast_min = min(contrasts)
    contrast_span = max(contrasts) - contrast_min
    rerank_generation_peaks: list[float] = []
    verification_query = build_candidate_verification_query(
        args.query,
        args.rerank_verification_mode,
    )
    original_answer, original_generation = generate_answer(
        [image],
        args.query,
        model_name,
        tokenizer,
        image_processor,
        model,
        args.max_new_tokens,
    )
    reference_color_terms = extract_color_terms(original_answer)
    for row in proposal_rows:
        normalized_contrast = (
            (float(row["contrast"]) - contrast_min) / contrast_span
            if contrast_span > 1e-12
            else 0.0
        )
        row["normalized_scan_contrast"] = normalized_contrast
        if args.rerank_top_k > 1:
            candidate_crop = image.crop(tuple(row["original_bbox"]))
            presence = score_candidate_presence(
                image,
                candidate_crop,
                args.query,
                model_name,
                tokenizer,
                image_processor,
                model,
                args.rerank_verification_mode,
            )
            rerank_generation_peaks.append(
                float(presence["generation"]["peak_allocated_mib"])
            )
            if args.rerank_answer_consistency_weight != 0:
                candidate_answer, candidate_answer_generation = generate_answer(
                    [candidate_crop],
                    args.query,
                    model_name,
                    tokenizer,
                    image_processor,
                    model,
                    args.rerank_candidate_answer_tokens,
                )
                rerank_generation_peaks.append(
                    float(candidate_answer_generation["peak_allocated_mib"])
                )
                consistency = answer_color_consistency(
                    original_answer,
                    candidate_answer,
                )
                candidate_color_terms = extract_color_terms(candidate_answer)
            else:
                candidate_answer = None
                candidate_answer_generation = None
                consistency = 0.0
                candidate_color_terms = []
            row.update(presence)
            row["candidate_answer"] = candidate_answer
            row["candidate_color_terms"] = candidate_color_terms
            row["answer_color_consistency"] = consistency
            row["candidate_answer_generation"] = candidate_answer_generation
            row["combined_score"] = (
                float(presence["yes_no_log_odds"])
                + args.rerank_scan_weight * normalized_contrast
                + args.rerank_answer_consistency_weight * consistency
            )
        else:
            row["answer_color_consistency"] = None
            row["combined_score"] = normalized_contrast
    best_proposal = max(proposal_rows, key=lambda row: float(row["combined_score"]))
    baseline_score = float(proposal_rows[0]["combined_score"])
    best_improvement = float(best_proposal["combined_score"]) - baseline_score
    selected_proposal = (
        best_proposal
        if int(best_proposal["rank"]) == 1
        or best_improvement >= args.rerank_min_improvement
        else proposal_rows[0]
    )
    selected_rank = int(selected_proposal["rank"])
    legacy_original_bbox = tuple(int(value) for value in proposal_rows[0]["original_bbox"])
    original_bbox = tuple(int(value) for value in selected_proposal["original_bbox"])
    padded_bbox = (
        list(selected_proposal["source_bbox"])
        if args.locate_coordinate_space == "padded"
        else None
    )
    topk_artifacts = (
        save_topk_candidate_visualizations(
            image,
            proposal_rows,
            output_dir,
            selected_rank,
        )
        if args.rerank_top_k > 1 and save_artifacts
        else None
    )
    ground_truth_bbox = (
        tuple(int(value) for value in args.ground_truth_bbox)
        if args.ground_truth_bbox is not None
        else None
    )
    if ground_truth_bbox is not None:
        for row in proposal_rows:
            row["evaluation"] = bbox_metrics(
                tuple(int(value) for value in row["original_bbox"]),
                ground_truth_bbox,
            )

    variant_comparison: dict[str, dict[str, object]] = {}
    normalization_names = (
        ("none", "minmax")
        if args.compare_fusion_normalizations
        else (config.fusion_normalization,)
    )
    coordinate_names = (
        ("original", "padded")
        if args.compare_coordinate_spaces
        else (args.locate_coordinate_space,)
    )
    for normalization_name in normalization_names:
        if normalization_name == config.fusion_normalization:
            comparison_map = result.ssim_map
        else:
            _, _, comparison_scores = fuse_importance(
                result.semantic_importance,
                result.structure_importance,
                semantic_weight=config.semantic_weight,
                normalization=normalization_name,
                eps=config.eps,
            )
            comparison_map = comparison_scores.reshape(config.patch_grid)

        for coordinate_name in coordinate_names:
            coordinate_size = (
                (padded_side, padded_side)
                if coordinate_name == "padded"
                else image.size
            )
            comparison_crop, _ = locate_from_importance_map(
                comparison_map,
                coordinate_size,
                base_crop_size=config.base_crop_size,
                ratios=config.window_ratios,
            )
            comparison_bbox = (
                remap_crop_to_original(
                    comparison_crop.bbox,
                    image.size,
                    offset_x,
                    offset_y,
                )
                if coordinate_name == "padded"
                else comparison_crop.bbox
            )
            variant_name = f"{normalization_name}_{coordinate_name}"
            variant_comparison[variant_name] = {
                "fusion_normalization": normalization_name,
                "coordinate_space": coordinate_name,
                "source_bbox": list(comparison_crop.bbox),
                "original_bbox": list(comparison_bbox),
                "selected_ratio": comparison_crop.selected_ratio,
                "selected_contrast": comparison_crop.contrast,
                "artifacts": (
                    save_coordinate_crop(
                        image,
                        comparison_bbox,
                        output_dir,
                        variant_name,
                    )
                    if save_artifacts
                    else None
                ),
            }
    if save_artifacts:
        save_map_visualizations(
            image,
            result.semantic_map,
            result.structure_map,
            result.ssim_map,
            original_bbox,
            output_dir,
        )

    crop_image = image.crop(original_bbox)
    crop_answer, crop_generation = generate_answer(
        [crop_image],
        args.query,
        model_name,
        tokenizer,
        image_processor,
        model,
        args.max_new_tokens,
        capture_scores=True,
    )
    joint_answer, joint_generation = generate_answer(
        [image, crop_image],
        args.query,
        model_name,
        tokenizer,
        image_processor,
        model,
        args.max_new_tokens,
        capture_scores=True,
    )
    if legacy_original_bbox == original_bbox:
        baseline_joint_answer = joint_answer
        baseline_joint_generation = joint_generation
    else:
        baseline_crop_image = image.crop(legacy_original_bbox)
        baseline_joint_answer, baseline_joint_generation = generate_answer(
            [image, baseline_crop_image],
            args.query,
            model_name,
            tokenizer,
            image_processor,
            model,
            args.max_new_tokens,
        )
    focused_joint_answer = None
    focused_joint_generation = None
    if args.enable_focus:
        focus_config = FocusConfig(
            num_phases=args.focus_phases,
            prune_ratio=args.focus_prune_ratio,
        )
        focused_joint_answer, focused_joint_generation = generate_answer(
            [image, crop_image],
            args.query,
            model_name,
            tokenizer,
            image_processor,
            model,
            args.max_new_tokens,
            focus_config=focus_config,
            capture_scores=True,
        )
        focus_trace = focused_joint_generation["focus"]
        focus_trace["artifacts"] = (
            save_focus_token_visualizations(
                image,
                focus_trace,
                output_dir,
            )
            if save_artifacts
            else None
        )
    random_focus_answers: dict[str, str] = {}
    random_focus_generations: dict[str, dict[str, object]] = {}
    for random_seed in args.random_focus_seeds:
        seed_name = str(random_seed)
        random_focus_config = FocusConfig(
            num_phases=args.focus_phases,
            prune_ratio=args.focus_prune_ratio,
            selection_method="random",
            random_seed=random_seed,
        )
        random_answer, random_generation = generate_answer(
            [image, crop_image],
            args.query,
            model_name,
            tokenizer,
            image_processor,
            model,
            args.max_new_tokens,
            focus_config=random_focus_config,
            capture_scores=True,
        )
        random_trace = random_generation["focus"]
        random_trace["artifacts"] = (
            save_focus_token_visualizations(
                image,
                random_trace,
                output_dir,
                artifact_prefix=f"random_focus_seed_{random_seed}",
            )
            if save_artifacts
            else None
        )
        random_focus_answers[seed_name] = random_answer
        random_focus_generations[seed_name] = random_generation

    joint_logits = joint_generation.pop("_score_logits")
    joint_token_ids = joint_generation.pop("_generated_token_ids")
    crop_logits = crop_generation.pop("_score_logits")
    crop_token_ids = crop_generation.pop("_generated_token_ids")
    logit_comparisons: dict[str, object] = {
        "joint_vs_crop_only": compare_generation_logits(
            joint_logits,
            crop_logits,
            joint_token_ids,
            crop_token_ids,
            tokenizer,
        )
    }
    if focused_joint_generation is not None:
        focused_logits = focused_joint_generation.pop("_score_logits")
        focused_token_ids = focused_joint_generation.pop("_generated_token_ids")
        logit_comparisons["joint_vs_attention_focus"] = compare_generation_logits(
            joint_logits,
            focused_logits,
            joint_token_ids,
            focused_token_ids,
            tokenizer,
        )
        del focused_logits, focused_token_ids
    for seed_name, random_generation in random_focus_generations.items():
        random_logits = random_generation.pop("_score_logits")
        random_token_ids = random_generation.pop("_generated_token_ids")
        logit_comparisons[f"joint_vs_random_focus_seed_{seed_name}"] = (
            compare_generation_logits(
                joint_logits,
                random_logits,
                joint_token_ids,
                random_token_ids,
                tokenizer,
            )
        )
        del random_logits, random_token_ids
    del joint_logits, joint_token_ids, crop_logits, crop_token_ids
    if args.compare_coordinate_spaces or args.compare_fusion_normalizations:
        for variant_name, variant_data in variant_comparison.items():
            variant_bbox = tuple(variant_data["original_bbox"])
            if variant_bbox == tuple(original_bbox):
                variant_answer = joint_answer
                variant_generation = joint_generation
            else:
                variant_crop = image.crop(variant_bbox)
                variant_answer, variant_generation = generate_answer(
                    [image, variant_crop],
                    args.query,
                    model_name,
                    tokenizer,
                    image_processor,
                    model,
                    args.max_new_tokens,
                )
            variant_data["joint_answer"] = variant_answer
            variant_data["joint_generation"] = variant_generation
    generation_peaks = [
        float(original_generation["peak_allocated_mib"]),
        float(crop_generation["peak_allocated_mib"]),
        float(joint_generation["peak_allocated_mib"]),
        float(baseline_joint_generation["peak_allocated_mib"]),
    ]
    generation_peaks.extend(rerank_generation_peaks)
    if focused_joint_generation is not None:
        generation_peaks.append(
            float(focused_joint_generation["peak_allocated_mib"])
        )
    generation_peaks.extend(
        float(generation["peak_allocated_mib"])
        for generation in random_focus_generations.values()
    )
    generation_peaks.extend(
        float(data["joint_generation"]["peak_allocated_mib"])
        for data in variant_comparison.values()
        if "joint_generation" in data
    )
    generation_peak_mib = max(generation_peaks)

    report = {
        "checkpoint": args.model_path,
        "artifact_mode": args.artifact_mode,
        "precision": "float16",
        "vision_tower": model.config.mm_vision_tower,
        "context_length": context_len,
        "query": args.query,
        "original_answer": original_answer,
        "crop_answer": crop_answer,
        "joint_answer": joint_answer,
        "baseline_joint_answer": baseline_joint_answer,
        "reranked_joint_answer": joint_answer,
        "focused_joint_answer": focused_joint_answer,
        "random_focused_joint_answers": random_focus_answers,
        "image_size": list(image.size),
        "padded_side": padded_side,
        "padding_offset": [offset_x, offset_y],
        "semantic_layer_index": config.semantic_layer,
        "semantic_score": args.semantic_score,
        "scan_capture_mode": args.scan_capture_mode,
        "semantic_rollout": semantic_rollout,
        "semantic_score_diagnostics": semantic_score_diagnostics,
        "structure_layer_index": config.structure_layer,
        "structure_hidden_states_tuple_index": structure_tuple_index,
        "pca_components": config.pca_components,
        "semantic_weight_beta": config.semantic_weight,
        "semantic_token_aggregation": config.semantic_token_aggregation,
        "fusion_normalization": config.fusion_normalization,
        "seed": args.seed,
        "locate_coordinate_space": args.locate_coordinate_space,
        "variant_comparison": variant_comparison,
        "logit_comparisons": logit_comparisons,
        "visual_token_span": [visual_start, visual_end],
        "visual_token_count": num_visual_tokens,
        "multimodal_sequence_length": sequence_length,
        "attention_shape": list(planning_attention.shape),
        "attention_gradient_shape": list(attention_gradient.shape),
        "structure_hidden_shape": list(structure_hidden.shape),
        "pseudo_token_id": int(pseudo_token_id.item()),
        "pseudo_token_text": pseudo_token_text,
        "padded_bbox": padded_bbox,
        "original_bbox": list(original_bbox),
        "legacy_original_bbox": list(legacy_original_bbox),
        "ground_truth_bbox": (
            list(ground_truth_bbox) if ground_truth_bbox is not None else None
        ),
        "selected_ratio": selected_proposal["selected_ratio"],
        "selected_contrast": selected_proposal["contrast"],
        "topk_reranking": {
            "enabled": args.rerank_top_k > 1,
            "requested_top_k": args.rerank_top_k,
            "actual_candidate_count": len(proposal_rows),
            "pre_nms_per_scale": args.rerank_pre_nms_per_scale,
            "nms_iou_threshold": args.rerank_nms_iou,
            "scan_weight": args.rerank_scan_weight,
            "answer_consistency_weight": args.rerank_answer_consistency_weight,
            "verification_mode": args.rerank_verification_mode,
            "minimum_improvement": args.rerank_min_improvement,
            "candidate_answer_tokens": args.rerank_candidate_answer_tokens,
            "original_pseudo_answer": original_answer,
            "original_pseudo_answer_color_terms": reference_color_terms,
            "verification_query": (
                verification_query if args.rerank_top_k > 1 else None
            ),
            "selected_rank": selected_rank,
            "unconstrained_best_rank": int(best_proposal["rank"]),
            "best_improvement_over_legacy": best_improvement,
            "conservative_fallback_applied": bool(
                int(best_proposal["rank"]) != 1
                and selected_rank == 1
            ),
            "selection_changed": legacy_original_bbox != original_bbox,
            "legacy_evaluation": (
                bbox_metrics(legacy_original_bbox, ground_truth_bbox)
                if ground_truth_bbox is not None
                else None
            ),
            "selected_evaluation": (
                bbox_metrics(original_bbox, ground_truth_bbox)
                if ground_truth_bbox is not None
                else None
            ),
            "candidates": proposal_rows,
            "artifacts": topk_artifacts,
        },
        "scan_peak_allocated_mib": round(scan_peak_mib, 1),
        "generation_peak_allocated_mib": round(generation_peak_mib, 1),
        "generation": {
            "original": original_generation,
            "crop_only": crop_generation,
            "original_plus_crop": joint_generation,
            "legacy_original_plus_crop": baseline_joint_generation,
            "original_plus_crop_focus": focused_joint_generation,
            "original_plus_crop_random_focus": random_focus_generations,
        },
        "semantic_stats": tensor_stats(result.semantic_importance),
        "structure_stats": tensor_stats(result.structure_importance),
        "ssim_stats": tensor_stats(result.ssim_scores),
        "candidates": [
            {
                "ratio": candidate.ratio,
                "map_xy": [candidate.map_x, candidate.map_y],
                "map_size": [candidate.map_width, candidate.map_height],
                "evidence_sum": candidate.evidence_sum,
                "contrast": candidate.contrast,
            }
            for candidate in result.candidates
        ],
        "artifacts": (
            {
                "crop": "crop.png",
                "selected_bbox": "selected_bbox.png",
                "topk_candidates": topk_artifacts,
                "semantic_overlay": "semantic_overlay.png",
                "structure_overlay": "structure_overlay.png",
                "ssim_overlay": "ssim_overlay.png",
            }
            if save_artifacts
            else None
        ),
    }
    report_path = output_dir / "result.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("semantic layer:", config.semantic_layer)
    print("structure layer:", config.structure_layer)
    print("visual token span:", (visual_start, visual_end))
    print("planning attention:", tuple(planning_attention.shape))
    print("structure hidden:", tuple(structure_hidden.shape))
    print("pseudo token:", pseudo_token_text, int(pseudo_token_id.item()))
    print("semantic rollout:", semantic_rollout["text"])
    print("semantic score:", args.semantic_score)
    print("locate coordinate space:", args.locate_coordinate_space)
    print("legacy located bbox:", legacy_original_bbox)
    print("top-k selected rank:", selected_rank)
    print("original bbox:", original_bbox)
    print("original answer:", original_answer)
    print("crop answer:", crop_answer)
    print("joint original+crop answer:", joint_answer)
    if legacy_original_bbox != original_bbox:
        print("legacy original+crop answer:", baseline_joint_answer)
    if focused_joint_answer is not None:
        print("focused joint answer:", focused_joint_answer)
        print(
            "focus original-token schedule:",
            [
                stage["original_tokens_after"]
                for stage in focused_joint_generation["focus"]["stages"]
            ],
        )
    for seed_name, random_answer in random_focus_answers.items():
        print(f"random-focus seed {seed_name} answer:", random_answer)
    print("scan peak allocated MiB:", report["scan_peak_allocated_mib"])
    print("result JSON:", report_path)


if __name__ == "__main__":
    main()

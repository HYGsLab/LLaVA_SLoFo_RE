#!/usr/bin/env python3
"""Run SLoFo Scan-Locate on the paper-compatible LLaVA-v1.5-7B model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

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
    ScanLocateConfig,
    fuse_importance,
    locate_from_importance_map,
    scan_locate_from_tensors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-file", required=True)
    parser.add_argument(
        "--query",
        default="What color are the clothes worn by the person holding a phone?",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
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
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    metadata: dict[str, object] = {
        "image_count": len(images),
        "image_sizes": [list(size) for size in image_sizes],
        "image_tensor_shape": list(image_tensor.shape),
        "input_ids_shape": list(input_ids.shape),
        "output_ids_shape": list(output_ids.shape),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }
    del input_ids, image_tensor, output_ids
    torch.cuda.empty_cache()
    return answer, metadata


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
    last_logits: torch.Tensor,
    pseudo_token_id: torch.Tensor,
) -> dict[str, torch.Tensor]:
    selected_logit = last_logits.gather(
        dim=-1, index=pseudo_token_id[:, None]
    ).sum()
    selected_log_probability = F.log_softmax(last_logits, dim=-1).gather(
        dim=-1, index=pseudo_token_id[:, None]
    ).sum()
    selected_probability = F.softmax(last_logits, dim=-1).gather(
        dim=-1, index=pseudo_token_id[:, None]
    ).sum()
    return {
        "log_probability": selected_log_probability,
        "logit": selected_logit,
        "probability": selected_probability,
        "cross_entropy": -selected_log_probability,
    }


def main() -> None:
    args = parse_args()
    image_path = Path(args.image_file)
    output_dir = Path(args.output_dir)
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

    outputs = model(
        input_ids=input_ids,
        images=images,
        image_sizes=image_sizes,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    if len(outputs.attentions) <= config.semantic_layer:
        raise ValueError("The requested semantic layer is not available.")
    structure_tuple_index = config.structure_layer + 1
    if len(outputs.hidden_states) <= structure_tuple_index:
        raise ValueError("The requested structure layer is not available.")

    attention_full = outputs.attentions[config.semantic_layer]
    sequence_length = attention_full.shape[-1]
    if visual_end > sequence_length:
        raise ValueError(
            f"Visual span [{visual_start}, {visual_end}) exceeds sequence length "
            f"{sequence_length}."
        )
    planning_attention = attention_full[
        0, :, -1, visual_start:visual_end
    ].detach()
    structure_hidden = outputs.hidden_states[structure_tuple_index][
        0, visual_start:visual_end, :
    ].detach()
    last_logits = outputs.logits[:, -1, :]
    pseudo_token_id = last_logits.detach().argmax(dim=-1)
    semantic_scores = build_semantic_scores(last_logits, pseudo_token_id)
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
        gradient_slice = attention_gradient_full[
            0, :, -1, visual_start:visual_end
        ].detach()
        gradient_slices[score_name] = gradient_slice

        semantic_native = (
            planning_attention * gradient_slice.clamp_min(0)
        ).mean(dim=0)
        semantic_float32 = (
            planning_attention.float() * gradient_slice.float().clamp_min(0)
        ).mean(dim=0)
        semantic_score_diagnostics[score_name] = {
            "score_value": float(semantic_scores[score_name].detach().float().item()),
            "gradient": signed_tensor_stats(gradient_slice),
            "semantic_native_precision": signed_tensor_stats(semantic_native),
            "semantic_float32": signed_tensor_stats(semantic_float32),
            "native_nonzero_tokens": int((semantic_native != 0).sum().item()),
            "float32_nonzero_tokens": int((semantic_float32 != 0).sum().item()),
        }
        del attention_gradient_full, gradient_slice, semantic_native, semantic_float32

    attention_gradient = gradient_slices[args.semantic_score]
    scan_peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    pseudo_token_text = tokenizer.decode(pseudo_token_id.tolist()).strip()

    del outputs, attention_full, last_logits, semantic_scores, gradient_slices
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
    if args.locate_coordinate_space == "padded":
        original_bbox = remap_crop_to_original(
            result.crop.bbox, image.size, offset_x, offset_y
        )
        padded_bbox = list(result.crop.bbox)
    else:
        original_bbox = result.crop.bbox
        padded_bbox = None

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
                "artifacts": save_coordinate_crop(
                    image,
                    comparison_bbox,
                    output_dir,
                    variant_name,
                ),
            }
    save_map_visualizations(
        image,
        result.semantic_map,
        result.structure_map,
        result.ssim_map,
        original_bbox,
        output_dir,
    )

    crop_image = image.crop(original_bbox)
    original_answer, original_generation = generate_answer(
        [image],
        args.query,
        model_name,
        tokenizer,
        image_processor,
        model,
        args.max_new_tokens,
    )
    crop_answer, crop_generation = generate_answer(
        [crop_image],
        args.query,
        model_name,
        tokenizer,
        image_processor,
        model,
        args.max_new_tokens,
    )
    joint_answer, joint_generation = generate_answer(
        [image, crop_image],
        args.query,
        model_name,
        tokenizer,
        image_processor,
        model,
        args.max_new_tokens,
    )
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
    ]
    generation_peaks.extend(
        float(data["joint_generation"]["peak_allocated_mib"])
        for data in variant_comparison.values()
    )
    generation_peak_mib = max(generation_peaks)

    report = {
        "checkpoint": args.model_path,
        "precision": "float16",
        "vision_tower": model.config.mm_vision_tower,
        "context_length": context_len,
        "query": args.query,
        "original_answer": original_answer,
        "crop_answer": crop_answer,
        "joint_answer": joint_answer,
        "image_size": list(image.size),
        "padded_side": padded_side,
        "padding_offset": [offset_x, offset_y],
        "semantic_layer_index": config.semantic_layer,
        "semantic_score": args.semantic_score,
        "semantic_score_diagnostics": semantic_score_diagnostics,
        "structure_layer_index": config.structure_layer,
        "structure_hidden_states_tuple_index": structure_tuple_index,
        "pca_components": config.pca_components,
        "semantic_weight_beta": config.semantic_weight,
        "fusion_normalization": config.fusion_normalization,
        "seed": args.seed,
        "locate_coordinate_space": args.locate_coordinate_space,
        "variant_comparison": variant_comparison,
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
        "selected_ratio": result.crop.selected_ratio,
        "selected_contrast": result.crop.contrast,
        "scan_peak_allocated_mib": round(scan_peak_mib, 1),
        "generation_peak_allocated_mib": round(generation_peak_mib, 1),
        "generation": {
            "original": original_generation,
            "crop_only": crop_generation,
            "original_plus_crop": joint_generation,
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
        "artifacts": {
            "crop": "crop.png",
            "selected_bbox": "selected_bbox.png",
            "semantic_overlay": "semantic_overlay.png",
            "structure_overlay": "structure_overlay.png",
            "ssim_overlay": "ssim_overlay.png",
        },
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
    print("semantic score:", args.semantic_score)
    print("locate coordinate space:", args.locate_coordinate_space)
    print("located bbox:", result.crop.bbox)
    print("original bbox:", original_bbox)
    print("original answer:", original_answer)
    print("crop answer:", crop_answer)
    print("joint original+crop answer:", joint_answer)
    print("scan peak allocated MiB:", report["scan_peak_allocated_mib"])
    print("result JSON:", report_path)


if __name__ == "__main__":
    main()

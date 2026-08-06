"""Run the paper-compatible LLaVA-v1.5-7B baseline in FP16."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--image-file", required=True)
    parser.add_argument(
        "--query",
        default="What color are the clothes worn by the person holding a phone?",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-json")
    return parser.parse_args()


def build_prompt(query: str, model_name: str, model: torch.nn.Module) -> str:
    image_tokens = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in query:
        replacement = image_tokens if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
        query = re.sub(IMAGE_PLACEHOLDER, replacement, query)
    else:
        prefix = image_tokens if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
        query = prefix + "\n" + query

    conv_mode = "llava_v1" if "v1" in model_name.lower() else "llava_v0"
    conversation = conv_templates[conv_mode].copy()
    conversation.append_message(conversation.roles[0], query)
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


def add_shape_tracing(model: torch.nn.Module, observed: dict[str, list[int]]) -> None:
    def vision_hook(_module, _inputs, output):
        value = output[1] if isinstance(output, tuple) else output
        observed["clip_image_features"] = list(value.shape)
        print("[shape] CLIP image_features:", tuple(value.shape))

    def projector_hook(_module, _inputs, output):
        observed["projected_image_features"] = list(output.shape)
        print("[shape] projected image_features:", tuple(output.shape))

    model.get_vision_tower().register_forward_hook(vision_hook)
    model.get_model().mm_projector.register_forward_hook(projector_hook)

    original_prepare = model.prepare_inputs_labels_for_multimodal

    def traced_prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        inputs_embeds = result[4]
        if inputs_embeds is not None:
            observed["multimodal_inputs_embeds"] = list(inputs_embeds.shape)
            print("[shape] multimodal inputs_embeds:", tuple(inputs_embeds.shape))
        return result

    model.prepare_inputs_labels_for_multimodal = traced_prepare


def main() -> None:
    args = parse_args()
    image_path = Path(args.image_file)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    print("checkpoint:", args.model_path)
    print("precision: FP16")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        model_base=None,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
    )

    layers = model.get_model().layers
    vision_tower_name = getattr(model.config, "mm_vision_tower", None)
    print("context length:", context_len)
    print("language layers:", len(layers))
    print("hidden size:", model.config.hidden_size)
    print("vision tower:", vision_tower_name)
    print("model dtype:", next(model.parameters()).dtype)

    prompt = build_prompt(args.query, model_name, model)
    image = Image.open(image_path).convert("RGB")
    image_sizes = [image.size]
    images = process_images([image], image_processor, model.config).to(
        model.device, dtype=torch.float16
    )
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    observed: dict[str, list[int]] = {
        "images": list(images.shape),
        "input_ids": list(input_ids.shape),
    }
    print("[shape] images:", tuple(images.shape))
    print("[shape] input_ids:", tuple(input_ids.shape))
    add_shape_tracing(model, observed)

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images,
            image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    observed["output_ids"] = list(output_ids.shape)
    result = {
        "checkpoint": args.model_path,
        "precision": "float16",
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "vision_tower": vision_tower_name,
        "language_layers": len(layers),
        "hidden_size": model.config.hidden_size,
        "context_length": context_len,
        "image_size": list(image.size),
        "query": args.query,
        "answer": answer,
        "peak_allocated_mib": round(peak_mib, 1),
        "shapes": observed,
    }
    print("[shape] output_ids:", tuple(output_ids.shape))
    print("peak allocated MiB:", result["peak_allocated_mib"])
    print("answer:", answer)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("result JSON:", output_path)


if __name__ == "__main__":
    main()

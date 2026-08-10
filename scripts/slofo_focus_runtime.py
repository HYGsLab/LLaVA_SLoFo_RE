"""LLaVA-1.5 runtime adapter for SLoFo's four-phase Focus stage.

The adapter patches only the inner LlamaModel forward loop.  During the first
multimodal prefill it keeps every cropped-image token and prunes the bottom 50%
of the remaining original-image tokens after layers 7, 15, and 23.  Later
decode steps reuse the phase-specific KV caches created by that prefill.
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch
from torch import Tensor
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast

from slofo import (
    CROPPED_IMAGE_TOKEN,
    ORIGINAL_IMAGE_TOKEN,
    FocusConfig,
    build_multimodal_token_layout,
    prune_original_image_tokens,
)


@dataclass
class FocusRuntimeContext:
    config: FocusConfig
    token_types: Tensor
    original_token_ids: Tensor
    trace: dict[str, object] = field(default_factory=dict)
    prefill_applied: bool = False


def install_focus_runtime(model: torch.nn.Module) -> None:
    """Install the adapter once; inactive calls delegate to the original forward."""

    inner_model = model.get_model()
    if getattr(inner_model, "_slofo_focus_installed", False):
        return
    inner_model._slofo_original_forward = inner_model.forward
    inner_model._slofo_focus_context = None
    inner_model.forward = types.MethodType(_focus_llama_forward, inner_model)
    inner_model._slofo_focus_installed = True


@contextmanager
def focus_generation_context(
    model: torch.nn.Module,
    input_ids: Tensor,
    *,
    image_token_index: int,
    tokens_per_image: int,
    config: Optional[FocusConfig] = None,
) -> Iterator[FocusRuntimeContext]:
    """Activate Focus for one original-plus-crop ``model.generate`` call."""

    install_focus_runtime(model)
    inner_model = model.get_model()
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Focus generation currently supports batch size 1.")
    token_types, original_token_ids = build_multimodal_token_layout(
        input_ids[0],
        image_token_index=image_token_index,
        tokens_per_image=tokens_per_image,
    )
    context = FocusRuntimeContext(
        config=config or FocusConfig(),
        token_types=token_types,
        original_token_ids=original_token_ids,
    )
    if inner_model._slofo_focus_context is not None:
        raise RuntimeError("A Focus generation context is already active.")
    inner_model._slofo_focus_context = context
    try:
        yield context
    finally:
        inner_model._slofo_focus_context = None


def _prepare_layer_mask(
    model,
    supplied_mask: Optional[Tensor],
    hidden_states: Tensor,
    past_key_values_length: int,
    *,
    need_attentions: bool,
) -> Optional[Tensor]:
    batch_size, sequence_length = hidden_states.shape[:2]
    if batch_size != 1:
        raise ValueError("Focus runtime currently supports batch size 1.")

    # The experiment uses an unpadded batch of one.  Generation keeps a global
    # 2-D mask whose length follows the first-layer cache, while Focus creates
    # shorter caches in deeper phases.  All-one masks are therefore rebuilt per
    # layer using that layer's actual cache length.
    if supplied_mask is not None and bool((supplied_mask == 0).any()):
        raise ValueError("Focus runtime does not yet support padded batches.")
    layer_mask = None
    input_shape = (batch_size, sequence_length)
    if model._use_flash_attention_2:
        return None
    if model._use_sdpa and not need_attentions:
        return _prepare_4d_causal_attention_mask_for_sdpa(
            layer_mask,
            input_shape,
            hidden_states,
            past_key_values_length,
        )
    return _prepare_4d_causal_attention_mask(
        layer_mask,
        input_shape,
        hidden_states,
        past_key_values_length,
    )


def _focus_llama_forward(
    self,
    input_ids: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    context: Optional[FocusRuntimeContext] = self._slofo_focus_context
    if context is None:
        return self._slofo_original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Specify input_ids or inputs_embeds, not both.")
    if input_ids is not None:
        batch_size, sequence_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, sequence_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("input_ids or inputs_embeds is required.")
    if batch_size != 1:
        raise ValueError("Focus runtime currently supports batch size 1.")
    if self.gradient_checkpointing and self.training:
        raise RuntimeError("Focus runtime is inference-only.")

    had_past = past_key_values is not None
    past_key_values_length = 0
    use_legacy_cache = False
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(sequence_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            sequence_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    is_focus_prefill = not had_past and sequence_length > 1 and not context.prefill_applied
    phase_end_layers = context.config.phase_end_layers(len(self.layers))
    phase_end_to_number = {
        layer_index: phase_number
        for phase_number, layer_index in enumerate(phase_end_layers, start=1)
    }
    if is_focus_prefill:
        token_types = context.token_types.to(hidden_states.device)
        original_token_ids = context.original_token_ids.to(hidden_states.device)
        if token_types.numel() != sequence_length:
            raise ValueError(
                "Expanded multimodal layout length does not match LLaVA inputs: "
                f"{token_types.numel()} != {sequence_length}."
            )
        initial_original = int(
            (token_types == ORIGINAL_IMAGE_TOKEN).sum().item()
        )
        crop_tokens = int((token_types == CROPPED_IMAGE_TOKEN).sum().item())
        context.trace = {
            "num_layers": len(self.layers),
            "num_phases": context.config.num_phases,
            "phase_end_layers": list(phase_end_layers),
            "prune_ratio": context.config.prune_ratio,
            "selection_method": context.config.selection_method,
            "random_seed": context.config.random_seed,
            "initial_sequence_length": sequence_length,
            "initial_original_tokens": initial_original,
            "crop_tokens_kept": crop_tokens,
            "stages": [],
        }
        token_layer_work = 0
    else:
        token_types = None
        original_token_ids = None
        token_layer_work = 0

    for layer_index, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if is_focus_prefill:
            token_layer_work += hidden_states.shape[1]

        internal_attention = is_focus_prefill and layer_index in phase_end_to_number
        need_attentions = bool(output_attentions or internal_attention)
        layer_past_length = (
            past_key_values.get_usable_length(hidden_states.shape[1], layer_index)
            if use_cache
            else 0
        )
        layer_position_ids = position_ids
        if had_past:
            # Focus creates phase-specific cache lengths.  A single global
            # generation position cannot index every layer's RoPE cache, so
            # continue positions from the actual cache length of this layer.
            layer_position_ids = torch.arange(
                layer_past_length,
                layer_past_length + hidden_states.shape[1],
                dtype=position_ids.dtype,
                device=hidden_states.device,
            ).unsqueeze(0)
        layer_attention_mask = _prepare_layer_mask(
            self,
            attention_mask,
            hidden_states,
            layer_past_length,
            need_attentions=need_attentions,
        )
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=layer_attention_mask,
            position_ids=layer_position_ids,
            past_key_value=past_key_values,
            output_attentions=need_attentions,
            use_cache=use_cache,
        )
        hidden_states = layer_outputs[0]
        layer_attention = layer_outputs[1] if need_attentions else None

        if use_cache:
            next_decoder_cache = layer_outputs[2 if need_attentions else 1]
        if output_attentions:
            all_self_attns += (layer_attention,)

        if internal_attention:
            original_positions = torch.nonzero(
                token_types == ORIGINAL_IMAGE_TOKEN, as_tuple=False
            ).flatten()
            anchor_attention = layer_attention[0, :, -1, :].float().mean(dim=0)
            original_scores = anchor_attention.index_select(0, original_positions)
            if context.config.selection_method == "random":
                generator = torch.Generator(device=original_scores.device)
                generator.manual_seed(
                    context.config.random_seed * 1_000_003
                    + phase_end_to_number[layer_index]
                )
                original_scores = torch.rand(
                    original_scores.shape,
                    dtype=torch.float32,
                    device=original_scores.device,
                    generator=generator,
                )
            before_sequence = hidden_states.shape[1]
            before_original = original_positions.numel()
            prune_result = prune_original_image_tokens(
                hidden_states,
                position_ids,
                token_types,
                original_token_ids,
                original_scores,
                prune_ratio=context.config.prune_ratio,
                min_original_tokens=context.config.min_original_tokens,
            )
            hidden_states = prune_result.hidden_states
            position_ids = prune_result.position_ids
            token_types = prune_result.token_types
            original_token_ids = prune_result.original_token_ids
            after_original = int(
                (token_types == ORIGINAL_IMAGE_TOKEN).sum().item()
            )
            crop_count = int((token_types == CROPPED_IMAGE_TOKEN).sum().item())
            context.trace["stages"].append(
                {
                    "phase": phase_end_to_number[layer_index],
                    "layer_index": layer_index,
                    "sequence_length_before": before_sequence,
                    "sequence_length_after": hidden_states.shape[1],
                    "original_tokens_before": before_original,
                    "original_tokens_after": after_original,
                    "pruned_original_tokens": before_original - after_original,
                    "crop_tokens_after": crop_count,
                    "crop_fraction_of_visual_tokens": (
                        crop_count / (crop_count + after_original)
                    ),
                    "attention_min": float(original_scores.amin().item()),
                    "attention_max": float(original_scores.amax().item()),
                    "attention_mean": float(original_scores.mean().item()),
                    "kept_original_token_ids": (
                        prune_result.kept_original_token_ids.cpu().tolist()
                    ),
                    "pruned_original_token_ids": (
                        prune_result.pruned_original_token_ids.cpu().tolist()
                    ),
                }
            )

        del layer_outputs, layer_attention

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if is_focus_prefill:
        baseline_work = sequence_length * len(self.layers)
        context.trace.update(
            {
                "final_sequence_length": hidden_states.shape[1],
                "final_original_tokens": int(
                    (token_types == ORIGINAL_IMAGE_TOKEN).sum().item()
                ),
                "prefill_token_layer_work": token_layer_work,
                "baseline_token_layer_work": baseline_work,
                "estimated_prefill_work_reduction": 1.0
                - token_layer_work / baseline_work,
            }
        )
        context.prefill_applied = True

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if use_legacy_cache
            else next_decoder_cache
        )
    if not return_dict:
        return tuple(
            value
            for value in [
                hidden_states,
                next_cache,
                all_hidden_states,
                all_self_attns,
            ]
            if value is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

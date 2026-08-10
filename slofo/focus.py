"""Tensor utilities for SLoFo's phase-by-phase Focus token pruning.

The paper keeps every token from the cropped sub-image and progressively
removes low-attention tokens from the original image.  This module contains
the model-independent bookkeeping used by the LLaVA runtime adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


OTHER_TOKEN = 0
ORIGINAL_IMAGE_TOKEN = 1
CROPPED_IMAGE_TOKEN = 2
FocusSelectionMethod = Literal["attention", "random"]


@dataclass(frozen=True)
class FocusConfig:
    """Configuration from Section 2.3 and the implementation details."""

    num_phases: int = 4
    prune_ratio: float = 0.5
    min_original_tokens: int = 1
    selection_method: FocusSelectionMethod = "attention"
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.num_phases < 2:
            raise ValueError("num_phases must be at least 2.")
        if not 0.0 < self.prune_ratio < 1.0:
            raise ValueError("prune_ratio must be strictly between 0 and 1.")
        if self.min_original_tokens < 1:
            raise ValueError("min_original_tokens must be at least 1.")
        if self.selection_method not in ("attention", "random"):
            raise ValueError("selection_method must be 'attention' or 'random'.")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")

    def phase_end_layers(self, num_layers: int) -> tuple[int, ...]:
        """Return zero-based phase ends where pruning is applied.

        For the 32-layer LLaVA-1.5-7B language model and four equal phases,
        this returns ``(7, 15, 23)``.  The final phase is not pruned because
        there is no subsequent phase that can benefit from the shorter input.
        """

        if num_layers < self.num_phases:
            raise ValueError("num_layers must be at least num_phases.")
        return tuple(
            math.ceil(phase_index * num_layers / self.num_phases) - 1
            for phase_index in range(1, self.num_phases)
        )


@dataclass
class FocusPruneResult:
    hidden_states: Tensor
    position_ids: Tensor
    token_types: Tensor
    original_token_ids: Tensor
    kept_sequence_indices: Tensor
    kept_original_token_ids: Tensor
    pruned_original_token_ids: Tensor


def build_multimodal_token_layout(
    input_ids: Tensor,
    *,
    image_token_index: int,
    tokens_per_image: int,
    original_image_index: int = 0,
    cropped_image_index: int = 1,
) -> tuple[Tensor, Tensor]:
    """Expand image placeholders into token-type and original-patch layouts.

    ``input_ids`` is the one-dimensional prompt before LLaVA replaces each
    image placeholder with visual features.  The returned tensors match the
    expanded multimodal sequence.  Original-image patches receive stable IDs
    ``0..tokens_per_image-1``; every other position receives ``-1``.
    """

    if input_ids.ndim != 1:
        raise ValueError("input_ids must be one-dimensional.")
    if tokens_per_image < 1:
        raise ValueError("tokens_per_image must be positive.")
    if original_image_index == cropped_image_index:
        raise ValueError("Original and cropped image indices must differ.")

    device = input_ids.device
    token_type_parts: list[Tensor] = []
    original_id_parts: list[Tensor] = []
    image_index = 0
    for token_id in input_ids.tolist():
        if token_id != image_token_index:
            token_type_parts.append(
                torch.full((1,), OTHER_TOKEN, dtype=torch.int8, device=device)
            )
            original_id_parts.append(
                torch.full((1,), -1, dtype=torch.long, device=device)
            )
            continue

        if image_index == original_image_index:
            token_type = ORIGINAL_IMAGE_TOKEN
            patch_ids = torch.arange(tokens_per_image, device=device)
        elif image_index == cropped_image_index:
            token_type = CROPPED_IMAGE_TOKEN
            patch_ids = torch.full(
                (tokens_per_image,), -1, dtype=torch.long, device=device
            )
        else:
            token_type = OTHER_TOKEN
            patch_ids = torch.full(
                (tokens_per_image,), -1, dtype=torch.long, device=device
            )
        token_type_parts.append(
            torch.full(
                (tokens_per_image,), token_type, dtype=torch.int8, device=device
            )
        )
        original_id_parts.append(patch_ids)
        image_index += 1

    if image_index <= max(original_image_index, cropped_image_index):
        raise ValueError(
            "The prompt does not contain both the original and cropped image "
            "placeholders."
        )
    return torch.cat(token_type_parts), torch.cat(original_id_parts)


def prune_original_image_tokens(
    hidden_states: Tensor,
    position_ids: Tensor,
    token_types: Tensor,
    original_token_ids: Tensor,
    original_attention_scores: Tensor,
    *,
    prune_ratio: float,
    min_original_tokens: int = 1,
) -> FocusPruneResult:
    """Remove the lowest-attention fraction of remaining original patches.

    Only sequence positions tagged ``ORIGINAL_IMAGE_TOKEN`` are eligible for
    removal.  Cropped-image and prompt tokens are always preserved.  Selected
    positions are restored to sequence order after ranking so positional and
    causal relationships remain stable.
    """

    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError("Focus currently supports hidden_states with batch size 1.")
    sequence_length = hidden_states.shape[1]
    if position_ids.shape != (1, sequence_length):
        raise ValueError("position_ids must have shape [1, sequence_length].")
    if token_types.shape != (sequence_length,):
        raise ValueError("token_types must match sequence_length.")
    if original_token_ids.shape != (sequence_length,):
        raise ValueError("original_token_ids must match sequence_length.")
    if not 0.0 < prune_ratio < 1.0:
        raise ValueError("prune_ratio must be strictly between 0 and 1.")
    if min_original_tokens < 1:
        raise ValueError("min_original_tokens must be at least 1.")

    original_positions = torch.nonzero(
        token_types == ORIGINAL_IMAGE_TOKEN, as_tuple=False
    ).flatten()
    original_count = original_positions.numel()
    if original_attention_scores.shape != (original_count,):
        raise ValueError(
            "original_attention_scores must contain one value per remaining "
            "original-image token."
        )
    if original_count <= min_original_tokens:
        keep_indices = torch.arange(sequence_length, device=hidden_states.device)
        kept_ids = original_token_ids[original_positions]
        return FocusPruneResult(
            hidden_states=hidden_states,
            position_ids=position_ids,
            token_types=token_types,
            original_token_ids=original_token_ids,
            kept_sequence_indices=keep_indices,
            kept_original_token_ids=kept_ids,
            pruned_original_token_ids=kept_ids.new_empty((0,)),
        )

    keep_count = max(
        min_original_tokens,
        math.ceil(original_count * (1.0 - prune_ratio)),
    )
    # Stable sorting makes tied attention scores deterministic by retaining the
    # earlier visual patch first.  The chosen patches are then put back into
    # sequence order before slicing the transformer state.
    ranked_local = torch.argsort(
        original_attention_scores.float(), descending=True, stable=True
    )
    kept_original_positions = original_positions[ranked_local[:keep_count]]
    keep_mask = torch.ones(
        sequence_length, dtype=torch.bool, device=hidden_states.device
    )
    keep_mask[original_positions] = False
    keep_mask[kept_original_positions] = True
    keep_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()

    pruned_mask = ~keep_mask[original_positions]
    pruned_original_positions = original_positions[pruned_mask]
    kept_ids = original_token_ids[kept_original_positions].sort().values
    pruned_ids = original_token_ids[pruned_original_positions].sort().values
    return FocusPruneResult(
        hidden_states=hidden_states.index_select(1, keep_indices),
        # Transformers 4.37 sizes LLaMA's RoPE cache from the current KV
        # length.  After token removal, preserving sparse old position numbers
        # can therefore index beyond that cache.  Compact positions for the
        # next phase; decode steps continue from each layer's own cache length.
        position_ids=torch.arange(
            keep_indices.numel(),
            dtype=position_ids.dtype,
            device=position_ids.device,
        ).unsqueeze(0),
        token_types=token_types.index_select(0, keep_indices),
        original_token_ids=original_token_ids.index_select(0, keep_indices),
        kept_sequence_indices=keep_indices,
        kept_original_token_ids=kept_ids,
        pruned_original_token_ids=pruned_ids,
    )

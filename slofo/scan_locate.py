"""Tensor-only implementation of SLoFo's Scan-Locate stage.

The module deliberately does not import LLaVA or load model weights.  A later
adapter only needs to provide three tensors:

1. planning-anchor attention to the visual tokens from language layer 14;
2. the gradient of a scalar pseudo-token score with respect to that attention;
3. visual-token hidden states from language layer 7.

In this paper, SSIM means *Semantic-Structural Importance Map*.  It is unrelated
to the image-quality metric named structural similarity (also abbreviated SSIM).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


Normalization = Literal["none", "minmax"]
PCASolver = Literal["lowrank", "svd"]


@dataclass(frozen=True)
class ScanLocateConfig:
    """Paper defaults plus explicit choices needed by the standalone module."""

    semantic_layer: int = 14
    structure_layer: int = 7
    pca_components: int = 20
    semantic_weight: float = 0.7
    patch_grid: tuple[int, int] = (24, 24)
    base_crop_size: int = 336
    window_ratios: tuple[float, ...] = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
    fusion_normalization: Normalization = "none"
    pca_solver: PCASolver = "lowrank"
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.semantic_layer < 0 or self.structure_layer < 0:
            raise ValueError("Transformer layer indices must be non-negative.")
        if self.pca_components < 1:
            raise ValueError("pca_components must be at least 1.")
        if not 0.0 <= self.semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be between 0 and 1.")
        if len(self.patch_grid) != 2 or min(self.patch_grid) < 1:
            raise ValueError("patch_grid must contain two positive integers.")
        if self.base_crop_size < 1:
            raise ValueError("base_crop_size must be positive.")
        if not self.window_ratios or min(self.window_ratios) <= 0:
            raise ValueError("window_ratios must contain positive values.")
        if self.fusion_normalization not in ("none", "minmax"):
            raise ValueError("fusion_normalization must be 'none' or 'minmax'.")
        if self.pca_solver not in ("lowrank", "svd"):
            raise ValueError("pca_solver must be 'lowrank' or 'svd'.")
        if self.eps <= 0:
            raise ValueError("eps must be positive.")


@dataclass(frozen=True)
class WindowCandidate:
    """Best position found for one sliding-window size."""

    ratio: float
    map_x: int
    map_y: int
    map_width: int
    map_height: int
    evidence_sum: float
    contrast: float


@dataclass(frozen=True)
class CropWindow:
    """Selected crop in source-image pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    selected_ratio: float
    evidence_sum: float
    contrast: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class ScanLocateResult:
    """All intermediate maps are retained so the algorithm can be inspected."""

    semantic_importance: Tensor
    structure_importance: Tensor
    semantic_for_fusion: Tensor
    structure_for_fusion: Tensor
    ssim_scores: Tensor
    semantic_map: Tensor
    structure_map: Tensor
    ssim_map: Tensor
    crop: CropWindow
    candidates: tuple[WindowCandidate, ...]


def _reduce_leading_dimensions(values: Tensor) -> Tensor:
    """Average batch/head dimensions while preserving the visual-token axis."""

    if values.ndim == 0:
        raise ValueError("Expected a visual-token axis, but received a scalar.")
    if values.ndim == 1:
        return values
    return values.reshape(-1, values.shape[-1]).mean(dim=0)


def gradient_weighted_semantic_importance(
    attention: Tensor,
    *,
    gradient: Tensor | None = None,
    planning_score: Tensor | None = None,
) -> Tensor:
    """Compute the paper's semantic branch: ``A_v = A_a2v * ReLU(G_v)``.

    ``attention`` must already be sliced to the visual-token keys, with shape
    ``[..., num_visual_tokens]``.  Callers can either pass a matching ``gradient``
    captured by a model adapter or pass the differentiable scalar
    ``planning_score`` and let this function call ``torch.autograd.grad``.
    Leading batch/head dimensions are averaged after element-wise weighting.
    """

    if not torch.is_floating_point(attention):
        raise TypeError("attention must be a floating-point tensor.")
    if gradient is not None and planning_score is not None:
        raise ValueError("Pass gradient or planning_score, not both.")
    if gradient is None:
        if planning_score is None:
            raise ValueError("Either gradient or planning_score is required.")
        if planning_score.numel() != 1:
            raise ValueError("planning_score must be a scalar tensor.")
        gradient = torch.autograd.grad(
            planning_score,
            attention,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
    if gradient.shape != attention.shape:
        raise ValueError(
            "gradient and attention must have the same shape; "
            f"got {tuple(gradient.shape)} and {tuple(attention.shape)}."
        )

    # Scan-Locate is training-free.  Detaching here avoids retaining the large
    # LLaVA forward graph after the one gradient needed by the semantic branch.
    # LLaVA commonly runs in float16, while attention * gradient values can be
    # around 1e-8 to 1e-7.  Multiplying before an upcast silently underflows most
    # visual tokens to zero.  The ViCrop reference path uses bfloat16, whose
    # exponent range avoids this failure; use float32 explicitly so the tensor
    # implementation is precision-independent.
    attention_work = attention.detach().float()
    gradient_work = gradient.detach().float()
    weighted = attention_work * gradient_work.clamp_min(0)
    return _reduce_leading_dimensions(weighted)


def pca_reconstruction_error(
    visual_hidden_states: Tensor,
    n_components: int = 20,
    *,
    solver: PCASolver = "lowrank",
) -> Tensor:
    """Compute the token-wise structure evidence in Equation (4).

    The input shape is ``[num_visual_tokens, hidden_size]``.  The implementation
    centers tokens, projects them to the principal subspace, reconstructs them,
    and returns ``sum((H - H_hat)^2) / sqrt(hidden_size)`` for every token.

    ``lowrank`` is suitable for real LLaVA features.  ``svd`` is exact and useful
    for small deterministic tests.  PCA is evaluated in float32 for numerical
    stability when the model features use float16/bfloat16.
    """

    if visual_hidden_states.ndim != 2:
        raise ValueError(
            "visual_hidden_states must have shape [num_visual_tokens, hidden_size]."
        )
    if not torch.is_floating_point(visual_hidden_states):
        raise TypeError("visual_hidden_states must be a floating-point tensor.")
    num_tokens, hidden_size = visual_hidden_states.shape
    max_components = min(num_tokens - 1, hidden_size)
    if not 1 <= n_components <= max_components:
        raise ValueError(
            f"n_components must be in [1, {max_components}] for input shape "
            f"{tuple(visual_hidden_states.shape)}."
        )
    if solver not in ("lowrank", "svd"):
        raise ValueError("solver must be 'lowrank' or 'svd'.")

    # The structure branch never back-propagates into the model.
    work = visual_hidden_states.detach().float()
    mean = work.mean(dim=0, keepdim=True)
    centered = work - mean

    if solver == "svd":
        _, _, right_vectors_h = torch.linalg.svd(centered, full_matrices=False)
        principal_axes = right_vectors_h[:n_components].transpose(0, 1)
    else:
        # V contains orthonormal feature-space principal axes [hidden_size, q].
        _, _, principal_axes = torch.pca_lowrank(
            centered, q=n_components, center=False, niter=2
        )

    reconstructed_centered = (centered @ principal_axes) @ principal_axes.transpose(0, 1)
    residual = centered - reconstructed_centered
    return residual.square().sum(dim=-1) / sqrt(hidden_size)


def _normalize(values: Tensor, mode: Normalization, eps: float) -> Tensor:
    if mode == "none":
        return values
    if mode != "minmax":
        raise ValueError("normalization must be 'none' or 'minmax'.")
    # Semantic gradient-attention values can legitimately be around 1e-7 while
    # still carrying strong relative contrast.  Comparing their span against a
    # fixed absolute epsilon would erase the entire semantic branch.  Evaluate
    # normalization in float32 and treat only a truly constant map as empty.
    work = values.float()
    minimum = work.amin()
    span = work.amax() - minimum
    if not bool(torch.isfinite(span)):
        raise ValueError("Cannot normalize a map containing non-finite values.")
    if bool(span == 0):
        return torch.zeros_like(work)
    return (work - minimum) / span


def fuse_importance(
    semantic: Tensor,
    structure: Tensor,
    *,
    semantic_weight: float = 0.7,
    normalization: Normalization = "none",
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fuse semantic and structure evidence using Equation (5).

    Returns ``(semantic_for_fusion, structure_for_fusion, ssim)``.  The paper
    equation does not state a normalization step, so ``none`` is the faithful
    default.  ``minmax`` is exposed as an explicit experiment rather than hidden
    inside the implementation.
    """

    if semantic.shape != structure.shape:
        raise ValueError("semantic and structure tensors must have the same shape.")
    if semantic.ndim != 1:
        raise ValueError("semantic and structure tensors must be one-dimensional.")
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError("semantic_weight must be between 0 and 1.")

    semantic_scaled = _normalize(semantic, normalization, eps)
    structure_scaled = _normalize(structure, normalization, eps)
    ssim = semantic_weight * semantic_scaled + (1.0 - semantic_weight) * structure_scaled
    return semantic_scaled, structure_scaled, ssim


def _reshape_scores(scores: Tensor, grid_shape: tuple[int, int]) -> Tensor:
    height, width = grid_shape
    if scores.ndim != 1 or scores.numel() != height * width:
        raise ValueError(
            f"Expected {height * width} scores for grid {grid_shape}, "
            f"but received shape {tuple(scores.shape)}."
        )
    return scores.reshape(height, width)


def _window_sums(importance_map: Tensor, height: int, width: int) -> Tensor:
    kernel = torch.ones(
        (1, 1, height, width),
        dtype=importance_map.dtype,
        device=importance_map.device,
    )
    return F.conv2d(importance_map[None, None], kernel)[0, 0]


def locate_from_importance_map(
    importance_map: Tensor,
    image_size: tuple[int, int],
    *,
    base_crop_size: int = 336,
    ratios: Sequence[float] = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0),
) -> tuple[CropWindow, tuple[WindowCandidate, ...]]:
    """Locate a crop by multi-scale sliding-window evidence and local contrast.

    ``image_size`` follows Pillow's ``(width, height)`` convention.  Each scale
    first selects the position with maximum cumulative evidence.  Across scales,
    the selected scale maximizes the per-cell difference from its immediate
    left/right/up/down sliding-window neighbors, matching ViCrop's public crop
    rule on which SLoFo is based.
    """

    if importance_map.ndim != 2:
        raise ValueError("importance_map must be two-dimensional.")
    if not torch.is_floating_point(importance_map):
        importance_map = importance_map.float()
    image_width, image_height = image_size
    if image_width < 1 or image_height < 1:
        raise ValueError("image_size must contain positive values.")
    if base_crop_size < 1:
        raise ValueError("base_crop_size must be positive.")
    if not ratios or min(ratios) <= 0:
        raise ValueError("ratios must contain positive values.")

    map_height, map_width = importance_map.shape
    cell_width = image_width / map_width
    cell_height = image_height / map_height
    candidates: list[WindowCandidate] = []

    for ratio in ratios:
        requested_size = base_crop_size * float(ratio)
        block_width = min(max(int(requested_size / cell_width), 1), map_width)
        block_height = min(max(int(requested_size / cell_height), 1), map_height)

        # A full-map window cannot be compared with a neighbor.  If the smallest
        # requested crop already covers the image, the only meaningful crop is full.
        if block_width == map_width and block_height == map_height:
            if not candidates:
                full = CropWindow(
                    0,
                    0,
                    image_width,
                    image_height,
                    float(ratio),
                    float(importance_map.sum().item()),
                    0.0,
                )
                return full, tuple()
            continue

        sums = _window_sums(importance_map, block_height, block_width)
        flat_index = int(torch.argmax(sums).item())
        position_y = flat_index // sums.shape[1]
        position_x = flat_index % sums.shape[1]
        best_sum = sums[position_y, position_x]

        neighbors: list[Tensor] = []
        if position_x > 0:
            neighbors.append(sums[position_y, position_x - 1])
        if position_x + 1 < sums.shape[1]:
            neighbors.append(sums[position_y, position_x + 1])
        if position_y > 0:
            neighbors.append(sums[position_y - 1, position_x])
        if position_y + 1 < sums.shape[0]:
            neighbors.append(sums[position_y + 1, position_x])

        if neighbors:
            neighbor_mean = torch.stack(neighbors).mean()
            contrast_tensor = (best_sum - neighbor_mean) / (block_width * block_height)
            contrast = float(contrast_tensor.item())
        else:
            contrast = 0.0

        candidates.append(
            WindowCandidate(
                ratio=float(ratio),
                map_x=position_x,
                map_y=position_y,
                map_width=block_width,
                map_height=block_height,
                evidence_sum=float(best_sum.item()),
                contrast=contrast,
            )
        )

    if not candidates:
        raise RuntimeError("No valid sliding-window candidate was produced.")
    selected = max(candidates, key=lambda item: item.contrast)

    center_x = (selected.map_x + selected.map_width / 2.0) * cell_width
    center_y = (selected.map_y + selected.map_height / 2.0) * cell_height
    requested_size = int(round(base_crop_size * selected.ratio))
    crop_width = min(requested_size, image_width)
    crop_height = min(requested_size, image_height)

    x1 = int(round(center_x - crop_width / 2.0))
    y1 = int(round(center_y - crop_height / 2.0))
    x1 = min(max(x1, 0), image_width - crop_width)
    y1 = min(max(y1, 0), image_height - crop_height)
    crop = CropWindow(
        x1=x1,
        y1=y1,
        x2=x1 + crop_width,
        y2=y1 + crop_height,
        selected_ratio=selected.ratio,
        evidence_sum=selected.evidence_sum,
        contrast=selected.contrast,
    )
    return crop, tuple(candidates)


def scan_locate_from_tensors(
    planning_attention: Tensor,
    structure_hidden_states: Tensor,
    image_size: tuple[int, int],
    *,
    attention_gradient: Tensor | None = None,
    planning_score: Tensor | None = None,
    config: ScanLocateConfig | None = None,
) -> ScanLocateResult:
    """Run the complete Scan-Locate algorithm on already-extracted tensors."""

    config = config or ScanLocateConfig()
    semantic = gradient_weighted_semantic_importance(
        planning_attention,
        gradient=attention_gradient,
        planning_score=planning_score,
    )
    structure = pca_reconstruction_error(
        structure_hidden_states,
        config.pca_components,
        solver=config.pca_solver,
    )
    semantic_scaled, structure_scaled, ssim = fuse_importance(
        semantic,
        structure,
        semantic_weight=config.semantic_weight,
        normalization=config.fusion_normalization,
        eps=config.eps,
    )

    semantic_map = _reshape_scores(semantic_scaled, config.patch_grid)
    structure_map = _reshape_scores(structure_scaled, config.patch_grid)
    ssim_map = _reshape_scores(ssim, config.patch_grid)
    crop, candidates = locate_from_importance_map(
        ssim_map,
        image_size,
        base_crop_size=config.base_crop_size,
        ratios=config.window_ratios,
    )
    return ScanLocateResult(
        semantic_importance=semantic,
        structure_importance=structure,
        semantic_for_fusion=semantic_scaled,
        structure_for_fusion=structure_scaled,
        ssim_scores=ssim,
        semantic_map=semantic_map,
        structure_map=structure_map,
        ssim_map=ssim_map,
        crop=crop,
        candidates=candidates,
    )


def stitch_importance_maps(tile_rows: Sequence[Sequence[Tensor]]) -> Tensor:
    """Reassemble per-tile SSIM maps for the paper's high-resolution variant.

    Every tile map must have identical ``[height, width]`` shape.  This function
    only stitches maps; extracting each tile's model features belongs in a future
    LLaVA adapter.
    """

    if not tile_rows or any(not row for row in tile_rows):
        raise ValueError("tile_rows must be a non-empty rectangular grid.")
    column_count = len(tile_rows[0])
    if any(len(row) != column_count for row in tile_rows):
        raise ValueError("tile_rows must be rectangular.")
    first_shape = tile_rows[0][0].shape
    if len(first_shape) != 2:
        raise ValueError("Every tile importance map must be two-dimensional.")
    for row in tile_rows:
        for tile in row:
            if tile.shape != first_shape:
                raise ValueError("Every tile importance map must have the same shape.")
    return torch.cat([torch.cat(list(row), dim=1) for row in tile_rows], dim=0)


def crop_sub_image(image: object, window: CropWindow) -> object:
    """Crop a Pillow-compatible image without importing Pillow in this module."""

    crop_method = getattr(image, "crop", None)
    if crop_method is None:
        raise TypeError("image must provide a crop((x1, y1, x2, y2)) method.")
    return crop_method(window.bbox)

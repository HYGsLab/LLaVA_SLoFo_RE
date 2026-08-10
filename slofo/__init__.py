"""SLoFo components implemented independently from the model repositories."""

from .focus import (
    CROPPED_IMAGE_TOKEN,
    ORIGINAL_IMAGE_TOKEN,
    OTHER_TOKEN,
    FocusConfig,
    FocusPruneResult,
    build_multimodal_token_layout,
    prune_original_image_tokens,
)

from .scan_locate import (
    CropWindow,
    ScanLocateConfig,
    ScanLocateResult,
    WindowCandidate,
    crop_sub_image,
    crop_iou,
    fuse_importance,
    gradient_weighted_semantic_importance,
    locate_from_importance_map,
    pca_reconstruction_error,
    scan_locate_from_tensors,
    stitch_importance_maps,
    topk_crop_windows_from_importance_map,
)

__all__ = [
    "CROPPED_IMAGE_TOKEN",
    "CropWindow",
    "FocusConfig",
    "FocusPruneResult",
    "ORIGINAL_IMAGE_TOKEN",
    "OTHER_TOKEN",
    "ScanLocateConfig",
    "ScanLocateResult",
    "WindowCandidate",
    "build_multimodal_token_layout",
    "crop_sub_image",
    "crop_iou",
    "fuse_importance",
    "gradient_weighted_semantic_importance",
    "locate_from_importance_map",
    "pca_reconstruction_error",
    "prune_original_image_tokens",
    "scan_locate_from_tensors",
    "stitch_importance_maps",
    "topk_crop_windows_from_importance_map",
]

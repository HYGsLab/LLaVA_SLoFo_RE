"""CPU-only tests for the standalone SLoFo Scan-Locate implementation."""

from __future__ import annotations

import unittest

import torch

from slofo import (
    CROPPED_IMAGE_TOKEN,
    ORIGINAL_IMAGE_TOKEN,
    OTHER_TOKEN,
    FocusConfig,
    ScanLocateConfig,
    build_multimodal_token_layout,
    fuse_importance,
    gradient_weighted_semantic_importance,
    locate_from_importance_map,
    pca_reconstruction_error,
    prune_original_image_tokens,
    scan_locate_from_tensors,
    stitch_importance_maps,
    topk_crop_windows_from_importance_map,
)


class SemanticBranchTests(unittest.TestCase):
    def test_supplied_gradient_is_relu_weighted_then_heads_are_averaged(self) -> None:
        attention = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        gradient = torch.tensor([[2.0, -1.0, 1.0], [0.0, 3.0, -2.0]])

        actual = gradient_weighted_semantic_importance(
            attention, gradient=gradient
        )

        expected = torch.tensor([1.0, 3.0, 1.5])
        torch.testing.assert_close(actual, expected)

    def test_planning_score_can_drive_autograd(self) -> None:
        attention = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        planning_score = attention[0] * 2 - attention[1] + attention[2] * 0.5

        actual = gradient_weighted_semantic_importance(
            attention, planning_score=planning_score
        )

        torch.testing.assert_close(actual, torch.tensor([2.0, 0.0, 1.5]))

    def test_float16_product_is_upcast_before_it_can_underflow(self) -> None:
        attention = torch.full((2, 4), 1e-4, dtype=torch.float16)
        gradient = torch.full((2, 4), 1e-4, dtype=torch.float16)

        actual = gradient_weighted_semantic_importance(
            attention, gradient=gradient
        )

        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(bool(torch.all(actual > 0)))
        torch.testing.assert_close(
            actual,
            torch.full((4,), 1e-8),
            rtol=2e-3,
            atol=0.0,
        )

    def test_max_token_aggregation_preserves_distinct_anchor_evidence(self) -> None:
        attention = torch.ones(2, 3, 4)
        gradient = torch.zeros_like(attention)
        gradient[:, 0, 0] = 1.0
        gradient[:, 1, 2] = 3.0
        gradient[:, 2, 3] = 2.0

        actual = gradient_weighted_semantic_importance(
            attention,
            gradient=gradient,
            token_aggregation="max",
        )

        torch.testing.assert_close(actual, torch.tensor([1.0, 0.0, 3.0, 2.0]))


class StructureBranchTests(unittest.TestCase):
    def test_pca_error_matches_equation_shape_and_detects_unique_token(self) -> None:
        dominant = torch.arange(-5.0, 5.0).unsqueeze(1)
        hidden = torch.cat([dominant, torch.zeros(10, 3)], dim=1)
        hidden[5, 1] = 2.0

        error = pca_reconstruction_error(hidden, 1, solver="svd")

        self.assertEqual(tuple(error.shape), (10,))
        self.assertEqual(int(torch.argmax(error).item()), 5)
        self.assertTrue(bool(torch.all(error >= 0)))

    def test_lowrank_solver_smoke_test(self) -> None:
        hidden = torch.randn(12, 8, generator=torch.Generator().manual_seed(3))
        error = pca_reconstruction_error(hidden, 3, solver="lowrank")
        self.assertEqual(tuple(error.shape), (12,))
        self.assertTrue(bool(torch.isfinite(error).all()))


class FusionAndLocationTests(unittest.TestCase):
    def test_fusion_follows_paper_equation_without_hidden_normalization(self) -> None:
        semantic = torch.tensor([1.0, 2.0])
        structure = torch.tensor([5.0, 1.0])

        semantic_used, structure_used, ssim = fuse_importance(
            semantic, structure, semantic_weight=0.7
        )

        torch.testing.assert_close(semantic_used, semantic)
        torch.testing.assert_close(structure_used, structure)
        torch.testing.assert_close(ssim, torch.tensor([2.2, 1.7]))

    def test_minmax_preserves_tiny_but_nonconstant_semantic_values(self) -> None:
        semantic = torch.tensor([0.0, 5.0e-7], dtype=torch.float16)
        structure = torch.tensor([0.0, 1.0], dtype=torch.float32)

        semantic_used, _, _ = fuse_importance(
            semantic,
            structure,
            semantic_weight=0.7,
            normalization="minmax",
        )

        self.assertEqual(semantic_used.dtype, torch.float32)
        torch.testing.assert_close(semantic_used, torch.tensor([0.0, 1.0]))

    def test_sliding_window_finds_hot_region(self) -> None:
        importance = torch.zeros(24, 24)
        importance[15:19, 16:20] = 10.0

        crop, candidates = locate_from_importance_map(
            importance,
            (960, 720),
            base_crop_size=240,
            ratios=(1.0,),
        )

        self.assertEqual(len(candidates), 1)
        self.assertLessEqual(crop.x1, int(18 * 40))
        self.assertGreaterEqual(crop.x2, int(16 * 40))
        self.assertLessEqual(crop.y1, int(17 * 30))
        self.assertGreaterEqual(crop.y2, int(15 * 30))

    def test_full_image_is_returned_when_base_crop_covers_it(self) -> None:
        importance = torch.ones(4, 4)
        crop, candidates = locate_from_importance_map(
            importance, (200, 100), base_crop_size=336
        )
        self.assertEqual(crop.bbox, (0, 0, 200, 100))
        self.assertEqual(candidates, tuple())

    def test_topk_keeps_legacy_first_and_adds_distinct_regions(self) -> None:
        importance = torch.zeros(12, 12)
        importance[1:4, 1:4] = 8.0
        importance[8:11, 8:11] = 7.0

        legacy, _ = locate_from_importance_map(
            importance,
            (480, 480),
            base_crop_size=120,
            ratios=(1.0, 1.5),
        )
        crops = topk_crop_windows_from_importance_map(
            importance,
            (480, 480),
            top_k=3,
            base_crop_size=120,
            ratios=(1.0, 1.5),
            pre_nms_per_scale=10,
            nms_iou_threshold=0.3,
        )

        self.assertEqual(crops[0], legacy)
        self.assertGreaterEqual(len(crops), 2)
        self.assertEqual(len({crop.bbox for crop in crops}), len(crops))
        self.assertTrue(any(crop.x1 >= 240 and crop.y1 >= 240 for crop in crops))

    def test_topk_validates_nms_threshold(self) -> None:
        with self.assertRaises(ValueError):
            topk_crop_windows_from_importance_map(
                torch.ones(4, 4),
                (100, 100),
                nms_iou_threshold=1.1,
            )

    def test_high_resolution_tile_maps_are_stitched(self) -> None:
        tiles = [
            [torch.ones(2, 2), torch.full((2, 2), 2.0)],
            [torch.full((2, 2), 3.0), torch.full((2, 2), 4.0)],
        ]
        actual = stitch_importance_maps(tiles)
        self.assertEqual(tuple(actual.shape), (4, 4))
        self.assertEqual(float(actual[0, 3]), 2.0)
        self.assertEqual(float(actual[3, 0]), 3.0)


class FocusStageTests(unittest.TestCase):
    def test_four_equal_phases_end_at_expected_llava_layers(self) -> None:
        self.assertEqual(FocusConfig().phase_end_layers(32), (7, 15, 23))

    def test_random_focus_configuration_is_explicit_and_validated(self) -> None:
        config = FocusConfig(selection_method="random", random_seed=2)
        self.assertEqual(config.selection_method, "random")
        self.assertEqual(config.random_seed, 2)
        with self.assertRaises(ValueError):
            FocusConfig(selection_method="unknown")

    def test_multimodal_layout_separates_original_crop_and_prompt(self) -> None:
        layout, patch_ids = build_multimodal_token_layout(
            torch.tensor([-200, 11, -200, 12]),
            image_token_index=-200,
            tokens_per_image=4,
        )

        self.assertEqual(
            layout.tolist(),
            [
                ORIGINAL_IMAGE_TOKEN,
                ORIGINAL_IMAGE_TOKEN,
                ORIGINAL_IMAGE_TOKEN,
                ORIGINAL_IMAGE_TOKEN,
                OTHER_TOKEN,
                CROPPED_IMAGE_TOKEN,
                CROPPED_IMAGE_TOKEN,
                CROPPED_IMAGE_TOKEN,
                CROPPED_IMAGE_TOKEN,
                OTHER_TOKEN,
            ],
        )
        self.assertEqual(
            patch_ids.tolist(), [0, 1, 2, 3, -1, -1, -1, -1, -1, -1]
        )

    def test_pruning_only_removes_low_attention_original_tokens(self) -> None:
        hidden = torch.arange(10, dtype=torch.float32).view(1, 10, 1)
        position_ids = torch.arange(10).view(1, 10)
        token_types = torch.tensor(
            [
                OTHER_TOKEN,
                ORIGINAL_IMAGE_TOKEN,
                ORIGINAL_IMAGE_TOKEN,
                ORIGINAL_IMAGE_TOKEN,
                ORIGINAL_IMAGE_TOKEN,
                CROPPED_IMAGE_TOKEN,
                CROPPED_IMAGE_TOKEN,
                CROPPED_IMAGE_TOKEN,
                CROPPED_IMAGE_TOKEN,
                OTHER_TOKEN,
            ],
            dtype=torch.int8,
        )
        original_ids = torch.tensor([-1, 0, 1, 2, 3, -1, -1, -1, -1, -1])
        result = prune_original_image_tokens(
            hidden,
            position_ids,
            token_types,
            original_ids,
            torch.tensor([0.1, 0.9, 0.2, 0.8]),
            prune_ratio=0.5,
        )

        self.assertEqual(result.hidden_states.shape[1], 8)
        self.assertEqual(result.kept_original_token_ids.tolist(), [1, 3])
        self.assertEqual(result.pruned_original_token_ids.tolist(), [0, 2])
        self.assertEqual(
            int((result.token_types == CROPPED_IMAGE_TOKEN).sum().item()), 4
        )
        self.assertEqual(int((result.token_types == OTHER_TOKEN).sum().item()), 2)
        self.assertEqual(result.position_ids.tolist(), [[0, 1, 2, 3, 4, 5, 6, 7]])

    def test_three_pruning_boundaries_follow_576_to_72_schedule(self) -> None:
        original_count = 576
        crop_count = 576
        sequence_length = original_count + crop_count + 3
        hidden = torch.zeros((1, sequence_length, 2))
        position_ids = torch.arange(sequence_length).view(1, -1)
        token_types = torch.tensor(
            [ORIGINAL_IMAGE_TOKEN] * original_count
            + [CROPPED_IMAGE_TOKEN] * crop_count
            + [OTHER_TOKEN] * 3,
            dtype=torch.int8,
        )
        original_ids = torch.tensor(
            list(range(original_count)) + [-1] * (crop_count + 3)
        )

        remaining = []
        for _ in range(3):
            scores = torch.arange(
                int((token_types == ORIGINAL_IMAGE_TOKEN).sum()),
                dtype=torch.float32,
            )
            result = prune_original_image_tokens(
                hidden,
                position_ids,
                token_types,
                original_ids,
                scores,
                prune_ratio=0.5,
            )
            hidden = result.hidden_states
            position_ids = result.position_ids
            token_types = result.token_types
            original_ids = result.original_token_ids
            remaining.append(
                int((token_types == ORIGINAL_IMAGE_TOKEN).sum().item())
            )

        self.assertEqual(remaining, [288, 144, 72])
        self.assertEqual(
            int((token_types == CROPPED_IMAGE_TOKEN).sum().item()), crop_count
        )


class EndToEndTensorTests(unittest.TestCase):
    def test_scan_locate_retains_intermediate_maps(self) -> None:
        generator = torch.Generator().manual_seed(7)
        attention = torch.rand((2, 16), generator=generator)
        gradient = torch.rand((2, 16), generator=generator) - 0.25
        hidden = torch.randn((16, 8), generator=generator)
        config = ScanLocateConfig(
            pca_components=3,
            patch_grid=(4, 4),
            base_crop_size=64,
            window_ratios=(1.0, 1.5),
            pca_solver="svd",
        )

        result = scan_locate_from_tensors(
            attention,
            hidden,
            (256, 192),
            attention_gradient=gradient,
            config=config,
        )

        self.assertEqual(tuple(result.semantic_importance.shape), (16,))
        self.assertEqual(tuple(result.structure_importance.shape), (16,))
        self.assertEqual(tuple(result.ssim_map.shape), (4, 4))
        self.assertGreater(result.crop.width, 0)
        self.assertGreater(result.crop.height, 0)


if __name__ == "__main__":
    unittest.main()

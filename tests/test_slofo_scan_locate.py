"""CPU-only tests for the standalone SLoFo Scan-Locate implementation."""

from __future__ import annotations

import unittest

import torch

from slofo import (
    ScanLocateConfig,
    fuse_importance,
    gradient_weighted_semantic_importance,
    locate_from_importance_map,
    pca_reconstruction_error,
    scan_locate_from_tensors,
    stitch_importance_maps,
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

    def test_high_resolution_tile_maps_are_stitched(self) -> None:
        tiles = [
            [torch.ones(2, 2), torch.full((2, 2), 2.0)],
            [torch.full((2, 2), 3.0), torch.full((2, 2), 4.0)],
        ]
        actual = stitch_importance_maps(tiles)
        self.assertEqual(tuple(actual.shape), (4, 4))
        self.assertEqual(float(actual[0, 3]), 2.0)
        self.assertEqual(float(actual[3, 0]), 3.0)


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

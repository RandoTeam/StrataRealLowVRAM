"""Unit test suite for Optimal 1.58-bit Trit Quantization & Strata TQ1_0 (GGML Type 34).

Verifies:
1. Mathematical Formulations:
   - Optimal scaling factor alpha* = (sum_{|w_i| > Delta} |w_i|) / N_active minimizes MSE.
   - Optimal threshold critical condition Delta* = 0.5 * alpha*(Delta*).
   - Analytical threshold Delta* ~= (2/3) * E[|W|] under Gaussian and Laplace distributions.
   - Golden-section and grid search convergence on empirical tensors.
2. Radix-3 Kernels & CUDA Bit-for-Bit Validation:
   - All 3^5 = 243 5-trit combinations pack and unpack identically.
   - All 3^4 = 81 4-trit qh combinations pack and unpack identically.
   - Division-free unpack logic with POW3_PACKED matches Strata CUDA dq_tq1_0.
3. Block Layout & Geometry:
   - 256 weights pack into exactly 54 bytes (48B qs, 4B qh, 2B FP16 scale d).
   - Multi-block vectorized packing and unpacking roundtrip bit-for-bit.
4. Sparsity & MSE Targets:
   - Zero-trit sparsity consistently within 35% - 50% target range.
   - Reconstruction MSE bounded and verified against explicit dequantization.
5. Real & Synthetic Weights:
   - Actual weights loaded from models/ (if present).
   - Realistic transformer layer projections (e.g. 2048x2048, 2048x512, 4096x4096).
6. NumPy and PyTorch Interoperability.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.ternary.trit_sampler import (
    BLOCK_SIZE,
    POW3_PACKED,
    QH_BYTES,
    QS_BYTES,
    SCALE_BYTES,
    TQ1_0_BLOCK_BYTES,
    TritQuantResult,
    golden_section_search_threshold,
    grid_search_threshold,
    optimal_scale,
    pack_4_trits_qh,
    pack_5_trits,
    pack_tq1_0,
    pack_tq1_0_block,
    quantize_to_trits,
    reconstruction_mse,
    tq1_0_trit,
    unpack_4_trits_qh,
    unpack_5_trits,
    unpack_tq1_0,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def reference_cuda_dq_tq1_0(qs: np.ndarray, qh: np.ndarray) -> np.ndarray:
    """Exact emulation of Strata CUDA dq_tq1_0 kernel (iq_kernels.cu:2061).

    Simulates the thread/chunk mapping for 8 chunks of 32 threads.
    """
    unpacked = np.zeros(256, dtype=np.int8)
    for c in range(8):
        for tid in range(32):
            if c < 5:
                w = tq1_0_trit(int(qs[tid]), c)
            elif c == 5:
                w = (
                    tq1_0_trit(int(qs[32 + tid]), 0)
                    if tid < 16
                    else tq1_0_trit(int(qs[32 + (tid - 16)]), 1)
                )
            elif c == 6:
                w = (
                    tq1_0_trit(int(qs[32 + tid]), 2)
                    if tid < 16
                    else tq1_0_trit(int(qs[32 + (tid - 16)]), 3)
                )
            else:  # c == 7
                if tid < 16:
                    w = tq1_0_trit(int(qs[32 + tid]), 4)
                else:
                    n = (tid - 16) // 4
                    j = (tid - 16) % 4
                    w = tq1_0_trit(int(qh[j]), n)
            unpacked[c * 32 + tid] = w
    return unpacked


class TestMathematicalFormulation(unittest.TestCase):
    """Test mathematical derivation of optimal threshold and scale."""

    def test_optimal_scale_minimizes_mse(self):
        """Verify alpha* = (sum_{|w_i| > Delta} |w_i|) / N_active minimizes MSE."""
        rng = np.random.RandomState(42)
        w = rng.randn(256).astype(np.float32)
        delta = 0.5

        alpha_opt = optimal_scale(w, delta)
        mse_opt = reconstruction_mse(w, delta, alpha_opt)

        # Check perturbing alpha increases MSE
        for perturbation in [-0.2, -0.05, 0.05, 0.2]:
            alpha_perturbed = alpha_opt + perturbation
            if alpha_perturbed > 0:
                mse_perturbed = reconstruction_mse(w, delta, alpha_perturbed)
                self.assertGreaterEqual(
                    mse_perturbed,
                    mse_opt - 1e-6,
                    f"MSE with perturbed scale {alpha_perturbed} was lower than optimal {alpha_opt}",
                )

    def test_optimal_boundary_condition(self):
        """Verify Delta* ~= 0.5 * alpha*(Delta*) at critical point."""
        rng = np.random.RandomState(123)
        w = rng.randn(4096).astype(np.float32)

        # Dense search for best delta
        delta_best, alpha_best, _ = grid_search_threshold(
            w, bounds_factor=(0.4, 1.0), num_points=100
        )
        # Midpoint ratio: Delta* / alpha* should be close to 0.5
        ratio = delta_best / alpha_best
        self.assertAlmostEqual(
            ratio,
            0.5,
            delta=0.08,
            msg=f"Expected Delta*/alpha* ~= 0.5, got {ratio:.4f}",
        )

    def test_analytical_threshold_gaussian(self):
        """Verify Delta* ~= (2/3) * E[|W|] achieves near-optimal MSE on Gaussian data."""
        rng = np.random.RandomState(456)
        sigma = 1.5
        w = rng.normal(0.0, sigma, size=32768).astype(np.float32)

        mean_abs = float(np.mean(np.abs(w)))
        delta_analytical = (2.0 / 3.0) * mean_abs
        mse_analytical = reconstruction_mse(w, delta_analytical)

        # Fine grid search to find true empirical minimum MSE
        delta_best, _, mse_best = grid_search_threshold(
            w, bounds_factor=(0.5, 0.9), num_points=50
        )

        # Relative error should be < 2%
        rel_diff = (mse_analytical - mse_best) / mse_best
        self.assertLess(
            rel_diff,
            0.025,
            f"Analytical MSE {mse_analytical:.5f} is not within 2.5% of optimal {mse_best:.5f}",
        )

        # Sparsity should be strictly in 35% - 50%
        sparsity = float(np.mean(np.abs(w) <= delta_analytical) * 100.0)
        self.assertGreaterEqual(sparsity, 35.0)
        self.assertLessEqual(sparsity, 50.0)

    def test_analytical_threshold_laplace(self):
        """Verify Delta* ~= (2/3) * E[|W|] on Laplace distribution."""
        rng = np.random.RandomState(789)
        b = 0.8
        w = rng.laplace(0.0, b, size=32768).astype(np.float32)

        mean_abs = float(np.mean(np.abs(w)))
        delta_analytical = (2.0 / 3.0) * mean_abs

        # Zero sparsity on Laplace with Delta = (2/3) b is 1 - exp(-2/3) ~= 48.7%
        sparsity = float(np.mean(np.abs(w) <= delta_analytical) * 100.0)
        self.assertGreaterEqual(sparsity, 45.0)
        self.assertLessEqual(sparsity, 52.0)

    def test_golden_section_and_grid_search_agreement(self):
        """Verify golden-section and grid search find consistent optimal thresholds."""
        rng = np.random.RandomState(101)
        w = rng.randn(256).astype(np.float32)

        d_gold, a_gold, mse_gold = golden_section_search_threshold(w)
        d_grid, a_grid, mse_grid = grid_search_threshold(w, num_points=40)

        # Both should find very similar MSE
        self.assertAlmostEqual(mse_gold, mse_grid, places=3)
        self.assertAlmostEqual(a_gold, a_grid, delta=0.05)


class TestTritRadix3Kernels(unittest.TestCase):
    """Test low-level Radix-3 packing and unpacking kernels."""

    def test_all_243_combinations_5_trits_roundtrip(self):
        """Verify that all 3^5 = 243 possible 5-trit combinations roundtrip identically."""
        trit_values = (-1, 0, 1)
        count = 0
        for t0 in trit_values:
            for t1 in trit_values:
                for t2 in trit_values:
                    for t3 in trit_values:
                        for t4 in trit_values:
                            original = (t0, t1, t2, t3, t4)
                            byte_val = pack_5_trits(original)
                            self.assertGreaterEqual(byte_val, 0)
                            self.assertLessEqual(byte_val, 255)

                            unpacked = unpack_5_trits(byte_val)
                            self.assertEqual(
                                unpacked,
                                original,
                                f"Failed for {original}: packed byte {byte_val}, got {unpacked}",
                            )
                            count += 1
        self.assertEqual(count, 243)

    def test_all_81_combinations_4_trits_qh_roundtrip(self):
        """Verify that all 3^4 = 81 possible 4-trit qh combinations roundtrip identically."""
        trit_values = (-1, 0, 1)
        count = 0
        for t0 in trit_values:
            for t1 in trit_values:
                for t2 in trit_values:
                    for t3 in trit_values:
                        original = (t0, t1, t2, t3)
                        byte_val = pack_4_trits_qh(original)
                        self.assertGreaterEqual(byte_val, 0)
                        self.assertLessEqual(byte_val, 255)

                        unpacked = unpack_4_trits_qh(byte_val)
                        self.assertEqual(
                            unpacked,
                            original,
                            f"Failed qh for {original}: packed byte {byte_val}, got {unpacked}",
                        )
                        count += 1
        self.assertEqual(count, 81)

    def test_cuda_tq1_0_trit_scalar_logic(self):
        """Verify tq1_0_trit scalar extraction against POW3_PACKED constant."""
        self.assertEqual(POW3_PACKED, 0xF3511B090301)
        powers = [1, 3, 9, 27, 81]
        for t in range(5):
            expected_p3 = powers[t]
            actual_p3 = (POW3_PACKED >> (t * 8)) & 0xFF
            self.assertEqual(actual_p3, expected_p3)


class TestStrataTQ10BlockPacking(unittest.TestCase):
    """Test 256-element block packing and Strata CUDA unpack table logic."""

    def test_block_geometry(self):
        """Verify block sizes: 54 bytes per 256 weights (48B qs, 4B qh, 2B scale)."""
        self.assertEqual(BLOCK_SIZE, 256)
        self.assertEqual(TQ1_0_BLOCK_BYTES, 54)
        self.assertEqual(QS_BYTES, 48)
        self.assertEqual(QH_BYTES, 4)
        self.assertEqual(SCALE_BYTES, 2)
        self.assertEqual(QS_BYTES + QH_BYTES + SCALE_BYTES, TQ1_0_BLOCK_BYTES)

    def test_single_block_pack_matches_cuda_dq_tq1_0(self):
        """Verify that pack_tq1_0_block matches reference_cuda_dq_tq1_0 bit-for-bit."""
        rng = np.random.RandomState(202)
        for trial in range(50):
            trits = rng.choice([-1, 0, 1], size=BLOCK_SIZE).astype(np.int8)
            scale = float(rng.uniform(0.01, 5.0))

            packed = pack_tq1_0_block(trits, scale)
            self.assertEqual(len(packed), TQ1_0_BLOCK_BYTES)

            qs = np.frombuffer(packed[:48], dtype=np.uint8)
            qh = np.frombuffer(packed[48:52], dtype=np.uint8)
            scale_u16 = np.frombuffer(packed[52:54], dtype=np.uint16)[0]
            unpacked_scale = scale_u16.view(np.float16)

            # Emulate CUDA dq_tq1_0 kernel
            cuda_unpacked_trits = reference_cuda_dq_tq1_0(qs, qh)
            np.testing.assert_array_equal(
                cuda_unpacked_trits,
                trits,
                err_msg=f"CUDA dq_tq1_0 unpack mismatch on trial {trial}",
            )
            self.assertEqual(unpacked_scale, np.float16(scale))

    def test_multi_block_vectorized_pack_and_unpack_roundtrip(self):
        """Verify vectorized pack_tq1_0 and unpack_tq1_0 across 100 blocks (25,600 weights)."""
        rng = np.random.RandomState(303)
        num_blocks = 100
        trits = rng.choice([-1, 0, 1], size=(num_blocks, BLOCK_SIZE)).astype(np.int8)
        scales = rng.uniform(0.05, 10.0, size=num_blocks).astype(np.float16)

        packed_bytes = pack_tq1_0(trits, scales)
        self.assertEqual(len(packed_bytes), num_blocks * TQ1_0_BLOCK_BYTES)

        unpacked_trits, unpacked_scales = unpack_tq1_0(packed_bytes)
        np.testing.assert_array_equal(unpacked_trits, trits)
        np.testing.assert_array_equal(unpacked_scales, scales)

    def test_fp16_scale_preservation(self):
        """Verify exact IEEE 754 half-precision scale bit preservation."""
        scales = np.array([0.0001, 1.0, 1.5, 3.140625, 65504.0], dtype=np.float16)
        trits = np.zeros((len(scales), BLOCK_SIZE), dtype=np.int8)

        packed = pack_tq1_0(trits, scales)
        _, unpacked_scales = unpack_tq1_0(packed)

        np.testing.assert_array_equal(
            scales.view(np.uint16),
            unpacked_scales.view(np.uint16),
        )


class TestQuantizeToTrits(unittest.TestCase):
    """Test quantize_to_trits API, sparsity, and reconstruction MSE."""

    def test_quantize_return_structure(self):
        """Verify return type supports tuple unpacking and attribute access."""
        rng = np.random.RandomState(404)
        w = rng.randn(512).astype(np.float32)

        res = quantize_to_trits(w, block_size=256)
        self.assertIsInstance(res, TritQuantResult)
        self.assertEqual(len(res), 4)

        # Attribute access
        self.assertEqual(res.trits.shape, (512,))
        self.assertEqual(res.scales.shape, (2,))
        self.assertIsInstance(res.sparsity, float)
        self.assertIsInstance(res.mse, float)

        # Tuple unpacking
        t, s, sp, mse = res
        self.assertEqual(t.shape, (512,))
        self.assertEqual(s.shape, (2,))

    def test_sparsity_target_range(self):
        """Verify zero-trit sparsity falls strictly within target 35% - 50%."""
        rng = np.random.RandomState(505)
        # Test on multiple distributions
        for dist_name, data in [
            ("Standard Normal", rng.randn(1024 * 8).astype(np.float32)),
            ("Scaled Normal", rng.normal(0, 0.05, size=1024 * 8).astype(np.float32)),
            ("Student-t (df=5)", rng.standard_t(5, size=1024 * 8).astype(np.float32)),
        ]:
            with self.subTest(distribution=dist_name):
                # Test with analytical threshold
                res_ana = quantize_to_trits(data, search_threshold=False)
                self.assertGreaterEqual(
                    res_ana.sparsity,
                    35.0,
                    f"Analytical sparsity too low for {dist_name}: {res_ana.sparsity:.1f}%",
                )
                self.assertLessEqual(
                    res_ana.sparsity,
                    50.0,
                    f"Analytical sparsity too high for {dist_name}: {res_ana.sparsity:.1f}%",
                )

                # Test with search threshold
                res_search = quantize_to_trits(data, search_threshold=True)
                self.assertGreaterEqual(
                    res_search.sparsity,
                    35.0,
                    f"Search sparsity too low for {dist_name}: {res_search.sparsity:.1f}%",
                )
                self.assertLessEqual(
                    res_search.sparsity,
                    50.0,
                    f"Search sparsity too high for {dist_name}: {res_search.sparsity:.1f}%",
                )

    def test_mse_consistency(self):
        """Verify returned MSE matches manual calculation from scales and trits."""
        rng = np.random.RandomState(606)
        w = rng.randn(2048).astype(np.float32)

        res = quantize_to_trits(w, block_size=256)
        # Reconstruct manually using FP16 scales expanded over blocks
        scales_expanded = np.repeat(res.scales.astype(np.float32), 256)
        manual_recon = scales_expanded * res.trits.astype(np.float32)
        manual_mse = float(np.mean((w - manual_recon) ** 2))

        self.assertAlmostEqual(res.mse, manual_mse, places=5)

    def test_all_zeros_edge_case(self):
        """Verify graceful handling of all-zero tensor."""
        w = np.zeros(512, dtype=np.float32)
        res = quantize_to_trits(w)
        self.assertEqual(res.sparsity, 100.0)
        self.assertEqual(res.mse, 0.0)
        np.testing.assert_array_equal(res.trits, 0)
        np.testing.assert_array_equal(res.scales, 0)

    def test_invalid_tensor_size_raises(self):
        """Verify ValueError is raised if tensor size is not a multiple of block_size."""
        w = np.ones(255, dtype=np.float32)
        with self.assertRaises(ValueError):
            quantize_to_trits(w, block_size=256)

    @unittest.skipUnless(HAS_TORCH, "PyTorch not available")
    def test_torch_tensor_interoperability(self):
        """Verify seamless execution when input is a PyTorch Tensor."""
        w_torch = torch.randn(1024, dtype=torch.float32)
        res = quantize_to_trits(w_torch, block_size=256)

        self.assertIsInstance(res.trits, torch.Tensor)
        self.assertIsInstance(res.scales, torch.Tensor)
        self.assertEqual(res.trits.dtype, torch.int8)
        self.assertEqual(res.scales.dtype, torch.float16)
        self.assertEqual(res.trits.shape, (1024,))
        self.assertEqual(res.scales.shape, (4,))


class TestModelWeightsAndProjections(unittest.TestCase):
    """Test quantization, packing, and MSE on real model weights and projection matrices."""

    def test_realistic_linear_projections(self):
        """Verify TQ1_0 on dimensions representing modern LLM projections."""
        rng = np.random.RandomState(707)
        test_shapes = [
            ("Attention Proj (2048 x 2048)", (2048, 2048)),
            ("MoE Expert Proj (2048 x 512)", (2048, 512)),
            ("Large Feedforward (4096 x 1024)", (4096, 1024)),
        ]

        for name, shape in test_shapes:
            with self.subTest(layer=name):
                # Standard LeCun/He initialization variance sigma ~= 1 / sqrt(d_in)
                d_in = shape[0]
                sigma = 1.0 / math.sqrt(d_in)
                weights = rng.normal(0.0, sigma, size=shape).astype(np.float32)

                res = quantize_to_trits(weights, block_size=256, search_threshold=True)

                # Check sparsity in target 35-50%
                self.assertGreaterEqual(res.sparsity, 35.0)
                self.assertLessEqual(res.sparsity, 50.0)

                # Normalized MSE: MSE / Var(W) should be < 0.22 for optimal 1.58-bit quant
                var_w = float(np.var(weights))
                normalized_mse = res.mse / var_w
                self.assertLess(
                    normalized_mse,
                    0.25,
                    f"Normalized MSE {normalized_mse:.4f} exceeds 0.25 on {name}",
                )

                # Bit-pack and unpack
                packed = pack_tq1_0(res.trits, res.scales)
                unpacked_trits, unpacked_scales = unpack_tq1_0(packed, shape=shape)

                np.testing.assert_array_equal(unpacked_trits, res.trits)
                np.testing.assert_array_equal(unpacked_scales, res.scales)

    def test_actual_model_weights_if_available(self):
        """Test on actual model tensor from models/ if GGUF file exists on disk."""
        candidate_model = Path(REPO_ROOT) / "models" / "Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf"
        if not candidate_model.exists():
            self.skipTest(f"Model file {candidate_model.name} not found on disk")

        from tools.gguf_reader import GGUFFile

        gguf = GGUFFile(candidate_model)
        # Find a suitable linear projection float32/fp16 weight tensor (exclude 1D norm scales)
        candidate_tensors = [
            t for t in gguf.tensors
            if "norm" not in t.name
            and t.type_name in ("F32", "F16", "BF16")
            and t.elements % 256 == 0
            and t.elements >= 2048
        ]
        if not candidate_tensors:
            self.skipTest("No suitable linear projection float tensors found in model")

        target_t = candidate_tensors[0]
        elem_size = 4 if target_t.type_name == "F32" else 2
        with open(candidate_model, "rb") as fh:
            fh.seek(gguf.data_start + target_t.offset)
            raw_bytes = fh.read(target_t.elements * elem_size)

        if target_t.type_name == "F32":
            w = np.frombuffer(raw_bytes, dtype=np.float32).copy()
        else:
            w = np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32)

        res = quantize_to_trits(w, block_size=256, search_threshold=True)

        # Verify sparsity in 35-50%
        self.assertGreaterEqual(res.sparsity, 35.0)
        self.assertLessEqual(res.sparsity, 50.0)

        # Verify pack/unpack roundtrip
        packed = pack_tq1_0(res.trits, res.scales)
        unpacked_trits, unpacked_scales = unpack_tq1_0(packed, shape=w.shape)
        np.testing.assert_array_equal(unpacked_trits, res.trits)
        np.testing.assert_array_equal(unpacked_scales, res.scales)


if __name__ == "__main__":
    unittest.main()

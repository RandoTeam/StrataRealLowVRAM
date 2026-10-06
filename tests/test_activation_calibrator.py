"""Unit test suite for Activation-Aware & MoE-Router-Weighted Trit Calibration.

Verifies:
1. Hessian Estimation:
   - Diagonal activation second moments h_j = (1/N) sum X_{i,j}^2.
   - Full covariance H = (1/N) X^T X matches analytic outer product.
   - Online streaming minibatch updates match single-batch calculations bit-for-bit.
   - High-variance outlier channel identification and ratio estimation.
2. Weighted Trit Calibration vs Unweighted MSE:
   - Mathematical guarantee: activation-weighted calibration achieves lower
     weighted reconstruction error Tr((W - W_hat) H (W - W_hat)^T) than unweighted MSE.
   - Analytical weighted scale alpha* = (sum h_i |w_i|) / (sum h_i) minimizes weighted loss.
   - Uniform activations recover standard unweighted MSE identically.
   - Fallback triggered cleanly when activations are None or uniform.
   - Full-Hessian matrix calibration.
   - Zero-trit sparsity consistently within target range [35%, 50%].
3. MoE Router Weighting across 8 Active Experts:
   - Top-8 routing across 256 experts (Qwen 3.6 35B A3B, Ornith 1.5 35B A3B) and 512 experts (Qwen 3.8 Flash Next).
   - Gating weights strictly sum to 1.0 per token across the 8 active experts.
   - Routing distribution entropy and expert frequency metrics.
   - Simultaneous multi-expert Hessian accumulation G_sq^T @ X^2.
   - Rare-expert shrinkage and zero-fired expert fallback without NaNs or infinities.
4. Synthetic Calibration Generator:
   - Multilingual text corpus (English, Chinese, Japanese, German, French, Spanish, Russian).
   - Code syntax corpus (Python decorators/types, C++ templates, CUDA kernels).
   - Realistic synthetic activation generation with heavy tails and outlier channels.
   - Router gate projection generation with power-law popularity.
5. Model Profiles:
   - Qwen 3.8 Flash Next (125B MoE, 512 experts, 8 routed).
   - Qwen 3.6 35B A3B (256 experts, 8 routed).
   - Ornith 1.5 35B A3B (256 experts, 8 routed).
   - Profile retrieval via exact key and aliases.
6. Strata TQ1_0 Bit-Packing Integration:
   - Calibrated trits and FP16 scales pack into 54-byte blocks and roundtrip bit-for-bit.
7. PyTorch Interoperability:
   - Seamless support for PyTorch Tensors on CPU/GPU.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.ternary.activation_calibrator import (
    ActivationCalibrator,
    ActivationHessian,
    CalibratedTritResult,
    CalibrationConfig,
    MoEModelProfile,
    MODEL_PROFILES,
    MODEL_ALIASES,
    RouterGatingResult,
    SyntheticCalibrationGenerator,
    SyntheticDataset,
    compute_activation_hessian,
    compute_expert_hessians,
    compute_router_gates,
    detect_outlier_channels,
    get_model_profile,
    update_activation_hessian,
    weighted_optimal_scale,
    weighted_quantize_block,
    weighted_quantize_full_hessian,
    weighted_reconstruction_error,
)
from tools.ternary.trit_sampler import (
    BLOCK_SIZE,
    quantize_to_trits,
    reconstruction_mse,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestHessianEstimation(unittest.TestCase):
    """Test activation covariance and diagonal Hessian estimation."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.n_tokens = 256
        self.dim = 128
        self.activations = self.rng.randn(self.n_tokens, self.dim).astype(np.float32)

    def test_diagonal_hessian_calculation(self):
        """Verify diagonal Hessian matches expected second moments."""
        reg = 1e-4
        hess = compute_activation_hessian(self.activations, compute_full=False, regularize_lambda=reg)
        expected_diag = np.mean(self.activations ** 2, axis=0) + reg

        self.assertIsInstance(hess, ActivationHessian)
        self.assertEqual(hess.diag.shape, (self.dim,))
        self.assertIsNone(hess.full)
        self.assertEqual(hess.sample_count, self.n_tokens)
        np.testing.assert_allclose(hess.diag, expected_diag, rtol=1e-5, atol=1e-6)

    def test_full_hessian_calculation(self):
        """Verify full covariance matches analytic (1/N) * X^T X."""
        reg = 1e-4
        hess = compute_activation_hessian(self.activations, compute_full=True, regularize_lambda=reg)
        expected_full = (self.activations.T @ self.activations) / float(self.n_tokens)
        np.fill_diagonal(expected_full, expected_full.diagonal() + reg)

        self.assertIsNotNone(hess.full)
        self.assertEqual(hess.full.shape, (self.dim, self.dim))
        np.testing.assert_allclose(hess.full, expected_full, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(hess.diag, np.diag(hess.full), rtol=1e-5, atol=1e-6)

    def test_streaming_online_update_matches_batch(self):
        """Verify streaming minibatches produce identical Hessian to single batch."""
        batch1 = self.activations[:128]
        batch2 = self.activations[128:]

        single_hess = compute_activation_hessian(self.activations, compute_full=True, regularize_lambda=1e-4)

        h_stream = update_activation_hessian(None, batch1, compute_full=True, regularize_lambda=1e-4)
        h_stream = update_activation_hessian(h_stream, batch2, compute_full=True, regularize_lambda=1e-4)

        self.assertEqual(h_stream.sample_count, self.n_tokens)
        np.testing.assert_allclose(h_stream.diag, single_hess.diag, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(h_stream.full, single_hess.full, rtol=1e-5, atol=1e-6)

    def test_outlier_channel_detection(self):
        """Verify outlier channels are correctly identified."""
        acts = self.rng.randn(200, 64).astype(np.float32)
        # Inject huge variance into channels 7, 23, 45
        acts[:, [7, 23, 45]] *= 30.0

        hess = compute_activation_hessian(acts)
        outliers = hess.outlier_channels

        self.assertIn(7, outliers)
        self.assertIn(23, outliers)
        self.assertIn(45, outliers)
        self.assertGreater(hess.outlier_ratio, 50.0)


class TestWeightedTritCalibration(unittest.TestCase):
    """Test activation-weighted trit threshold and scale calibration."""

    def setUp(self):
        self.rng = np.random.RandomState(123)

    def test_weighted_calibration_achieves_lower_weighted_error(self):
        """Key Theorem: Activation-weighted calibration MUST achieve <= weighted error than unweighted MSE."""
        dim = 256
        weights = self.rng.randn(dim).astype(np.float32)

        # Activations with significant variance skew (first 32 channels have 20x variance)
        h_diag = np.ones(dim, dtype=np.float32)
        h_diag[:32] = 20.0

        # 1. Unweighted MSE calibration
        unw_res = quantize_to_trits(weights, block_size=256, search_threshold=True)
        unw_trits = unw_res.trits
        unw_scale = float(unw_res.scales[0])
        unw_recon = unw_scale * unw_trits.astype(np.float32)
        unw_weighted_err = float(np.sum(h_diag * ((weights - unw_recon) ** 2)))

        # 2. Weighted calibration
        w_res = weighted_quantize_block(
            weights,
            h_diag=h_diag,
            block_size=256,
            search_threshold=True,
            target_sparsity=None,  # pure loss comparison
        )
        w_trits = w_res.trits
        w_scale = float(w_res.scales[0])
        w_recon = w_scale * w_trits.astype(np.float32)
        w_weighted_err = float(np.sum(h_diag * ((weights - w_recon) ** 2)))

        # Weighted calibration must be strictly lower or equal to unweighted on weighted metric
        self.assertLessEqual(w_weighted_err, unw_weighted_err + 1e-5)
        improvement_pct = (unw_weighted_err - w_weighted_err) / unw_weighted_err * 100.0
        self.assertGreater(improvement_pct, 0.0)

    def test_optimal_weighted_scale_minimizes_weighted_error(self):
        """Verify analytical scale alpha*(Delta; h) = sum(h*|w|) / sum(h) minimizes weighted error."""
        w = self.rng.randn(256).astype(np.float32)
        h = self.rng.uniform(0.1, 5.0, size=256).astype(np.float32)
        delta = 0.5 * float(np.mean(np.abs(w)))

        alpha_opt = weighted_optimal_scale(w, delta, h)

        trits = np.zeros_like(w, dtype=np.int8)
        trits[w > delta] = 1
        trits[w < -delta] = -1

        def eval_loss(scale):
            return float(np.sum(h * ((w - scale * trits.astype(np.float32)) ** 2)))

        loss_opt = eval_loss(alpha_opt)
        loss_lower = eval_loss(alpha_opt * 0.95)
        loss_upper = eval_loss(alpha_opt * 1.05)

        self.assertLessEqual(loss_opt, loss_lower)
        self.assertLessEqual(loss_opt, loss_upper)

    def test_uniform_activations_recover_unweighted_mse(self):
        """When activations are uniform, weighted calibration must match unweighted MSE."""
        weights = self.rng.randn(512).astype(np.float32)
        h_uniform = np.full(512, 2.5, dtype=np.float32)

        res_unw = quantize_to_trits(weights, block_size=256, search_threshold=True)
        res_weighted = weighted_quantize_block(weights, h_diag=h_uniform, block_size=256, search_threshold=True)

        self.assertTrue(res_weighted.is_fallback)
        np.testing.assert_array_equal(res_weighted.trits, res_unw.trits)
        np.testing.assert_allclose(res_weighted.scales, res_unw.scales, rtol=1e-3)
        self.assertAlmostEqual(res_weighted.sparsity, res_unw.sparsity, places=3)

    def test_fallback_when_activations_none(self):
        """When activations/hessian are None, fallback flag is set and standard MSE is used."""
        weights = self.rng.randn(256).astype(np.float32)
        res = weighted_quantize_block(weights, h_diag=None, block_size=256)

        self.assertTrue(res.is_fallback)
        self.assertEqual(res.trits.shape, (256,))
        self.assertGreaterEqual(res.sparsity, 30.0)
        self.assertLessEqual(res.sparsity, 55.0)

    def test_sparsity_target_range(self):
        """Verify zero-trit sparsity is strictly in target 35% - 50% range."""
        weights = self.rng.randn(2048).astype(np.float32)
        h_diag = self.rng.uniform(0.5, 2.0, size=2048).astype(np.float32)

        res = weighted_quantize_block(
            weights,
            h_diag=h_diag,
            block_size=256,
            target_sparsity=(35.0, 50.0),
        )

        self.assertGreaterEqual(res.sparsity, 35.0)
        self.assertLessEqual(res.sparsity, 50.0)

    def test_full_hessian_matrix_calibration(self):
        """Verify exact full-Hessian calibration on 2D weight matrix."""
        d_out, d_in = 8, 32
        W = self.rng.randn(d_out, d_in).astype(np.float32)
        X = self.rng.randn(100, d_in).astype(np.float32)
        H = (X.T @ X) / 100.0

        res = weighted_quantize_full_hessian(W, H, target_sparsity=None)

        self.assertEqual(res.trits.shape, (d_out, d_in))
        self.assertEqual(len(res.scales), 1)
        self.assertFalse(res.is_fallback)
        self.assertGreater(float(res.scales[0]), 0.0)
        self.assertGreater(res.sparsity, 0.0)
        self.assertLess(res.weighted_error, 1.0)


class TestMoERouterWeighting(unittest.TestCase):
    """Test MoE router gating computation across top-8 active experts."""

    def setUp(self):
        self.rng = np.random.RandomState(999)
        self.num_tokens = 200
        self.dim = 128

    def test_router_gating_qwen36_256_experts_top8(self):
        """Test top-8 router gating for Qwen 3.6 35B / Ornith 1.5 35B (256 experts, 8 routed)."""
        num_experts = 256
        top_k = 8

        activations = self.rng.randn(self.num_tokens, self.dim).astype(np.float32)
        w_gate = self.rng.randn(num_experts, self.dim).astype(np.float32) * 0.05

        gating_res = compute_router_gates(
            router_weights=w_gate,
            activations=activations,
            top_k=top_k,
        )

        self.assertIsInstance(gating_res, RouterGatingResult)
        self.assertEqual(gating_res.gating_weights.shape, (self.num_tokens, num_experts))
        self.assertEqual(gating_res.active_expert_indices.shape, (self.num_tokens, top_k))

        # Each token must have exactly top_k non-zero weights
        non_zero_per_token = np.sum(gating_res.gating_weights > 0.0, axis=1)
        np.testing.assert_array_equal(non_zero_per_token, np.full(self.num_tokens, top_k))

        # Sum of gating weights per token must equal 1.0
        row_sums = np.sum(gating_res.gating_weights, axis=1)
        np.testing.assert_allclose(row_sums, np.ones(self.num_tokens), rtol=1e-5, atol=1e-6)

        # Routing entropy must be strictly positive
        self.assertGreater(gating_res.routing_entropy, 0.0)

    def test_router_gating_qwen38_512_experts_top8(self):
        """Test top-8 router gating for Qwen 3.8 Flash Next (512 experts, 8 routed)."""
        num_experts = 512
        top_k = 8

        activations = self.rng.randn(self.num_tokens, self.dim).astype(np.float32)
        w_gate = self.rng.randn(num_experts, self.dim).astype(np.float32) * 0.05

        gating_res = compute_router_gates(
            router_weights=w_gate,
            activations=activations,
            top_k=top_k,
        )

        self.assertEqual(gating_res.gating_weights.shape, (self.num_tokens, 512))
        self.assertEqual(gating_res.expert_total_weights.shape, (512,))

        # Verify sum of total weights equals N
        total_mass = float(np.sum(gating_res.expert_total_weights))
        self.assertAlmostEqual(total_mass, float(self.num_tokens), places=3)

    def test_simultaneous_expert_hessians_and_shrinkage(self):
        """Test simultaneous G_sq^T @ X^2 Hessian estimation with shrinkage."""
        num_experts = 16
        top_k = 4
        acts = self.rng.randn(100, 32).astype(np.float32)
        w_gate = self.rng.randn(num_experts, 32).astype(np.float32) * 0.05

        # Artificially suppress experts 14 and 15 via bias so they are never selected
        bias = np.zeros(num_experts, dtype=np.float32)
        bias[14] = -1e9
        bias[15] = -1e9

        gating_res = compute_router_gates(w_gate, acts, top_k=top_k, bias=bias)
        h_matrix, masses = compute_expert_hessians(
            activations=acts,
            gating_result=gating_res,
            shrinkage_lambda=1.0,
        )

        self.assertEqual(h_matrix.shape, (num_experts, 32))
        self.assertEqual(masses.shape, (num_experts,))

        # Unfired experts 14 and 15 should have 0 routing mass
        self.assertEqual(masses[14], 0.0)
        self.assertEqual(masses[15], 0.0)

        # Shrinkage guarantees positive, well-behaved Hessian even for unfired experts
        self.assertTrue(np.all(h_matrix[14] > 0))
        self.assertTrue(np.all(h_matrix[15] > 0))
        self.assertFalse(np.any(np.isnan(h_matrix)))

    def test_rare_and_unfired_expert_fallback_in_layer_calibration(self):
        """Verify ActivationCalibrator cleanly falls back for rarely/unfired experts."""
        calibrator = ActivationCalibrator(profile="qwen3.6-35b-a3b")
        acts = self.rng.randn(50, 4096).astype(np.float32)

        # 256 router weights with an expert that never gets selected via bias
        w_gate = self.rng.randn(256, 4096).astype(np.float32) * 0.01
        bias = np.zeros(256, dtype=np.float32)
        bias[42] = -1e9  # expert 42 never selected

        expert_weights = {
            0: self.rng.randn(512, 4096).astype(np.float32),   # popular expert
            42: self.rng.randn(512, 4096).astype(np.float32),  # unfired expert
        }

        results = calibrator.route_and_calibrate_moe(
            router_weights=w_gate,
            expert_weights=expert_weights,
            activations=acts,
            router_bias=bias,
        )

        self.assertIn(0, results)
        self.assertIn(42, results)

        # Unfired expert 42 must trigger fallback
        self.assertTrue(results[42].is_fallback)
        self.assertEqual(results[42].expert_weight, 0.0)
        self.assertFalse(np.any(np.isnan(results[42].trits)))
        self.assertFalse(np.any(np.isnan(results[42].scales)))


class TestSyntheticCalibrationGenerator(unittest.TestCase):
    """Test synthetic multilingual, code syntax, and activation generation."""

    def setUp(self):
        self.gen = SyntheticCalibrationGenerator(seed=777)

    def test_multilingual_text_corpus(self):
        """Verify multilingual text corpus contains required languages."""
        corpus = self.gen.generate_text_corpus()
        self.assertGreaterEqual(len(corpus), 10)

        # Verify presence of multilingual scripts
        full_text = " ".join(corpus)
        has_chinese = any("\u4e00" <= char <= "\u9fff" for char in full_text)
        has_japanese = any("\u3040" <= char <= "\u30ff" for char in full_text)
        has_russian = any("\u0400" <= char <= "\u04ff" for char in full_text)

        self.assertTrue(has_chinese, "Must contain Chinese text")
        self.assertTrue(has_japanese, "Must contain Japanese text")
        self.assertTrue(has_russian, "Must contain Russian text")

    def test_code_syntax_corpus(self):
        """Verify code syntax samples include Python, C++, and CUDA."""
        code = self.gen.CODE_SAMPLES
        code_str = "\n".join(code)

        self.assertIn("def calibrate_block", code_str)
        self.assertIn("#include <vector>", code_str)
        self.assertIn("__global__ void", code_str)

    def test_token_sequences(self):
        """Verify token sequence generation structure and bounds."""
        seqs = self.gen.generate_token_sequences(num_sequences=8, seq_len=32)
        self.assertEqual(len(seqs), 8)
        for s in seqs:
            self.assertEqual(len(s), 32)
            self.assertEqual(s[0], 151644)   # BOS
            self.assertEqual(s[-1], 151645)  # EOS
            self.assertTrue(all(0 <= tok < self.gen.vocab_size for tok in s))

    def test_synthetic_activation_outlier_properties(self):
        """Verify synthetic activations exhibit realistic heavy-tailed outlier channels."""
        acts = self.gen.generate_activations(
            num_tokens=256,
            dim=1024,
            num_outliers=16,
            outlier_multiplier=30.0,
        )
        self.assertEqual(acts.shape, (256, 1024))

        variances = np.var(acts, axis=0)
        median_var = float(np.median(variances))
        max_var = float(np.max(variances))

        self.assertGreater(max_var / median_var, 15.0)

    def test_complete_synthetic_dataset(self):
        """Verify generate_complete_dataset packages all components."""
        ds = self.gen.generate_complete_dataset(num_tokens=100, dim=256, num_experts=16)
        self.assertIsInstance(ds, SyntheticDataset)
        self.assertEqual(ds.activations.shape, (100, 256))
        self.assertEqual(ds.router_weights.shape, (16, 256))
        self.assertGreater(len(ds.text_samples), 0)
        self.assertGreater(len(ds.code_samples), 0)


class TestModelProfiles(unittest.TestCase):
    """Test predefined MoE model architectural configurations."""

    def test_qwen38_flash_next_profile(self):
        """Verify Qwen 3.8 Flash Next (125B MoE, 512 experts, 8 routed)."""
        p = get_model_profile("qwen3.8-flash-next")
        self.assertEqual(p.total_experts, 512)
        self.assertEqual(p.routed_experts, 8)
        self.assertEqual(p.hidden_size, 4096)
        self.assertEqual(p.intermediate_size, 2048)

    def test_qwen36_35b_a3b_profile(self):
        """Verify Qwen 3.6 35B A3B (256 experts, 8 routed)."""
        p = get_model_profile("qwen3.6-35b-a3b")
        self.assertEqual(p.total_experts, 256)
        self.assertEqual(p.routed_experts, 8)
        self.assertEqual(p.hidden_size, 4096)
        self.assertEqual(p.intermediate_size, 2048)

    def test_ornith15_35b_a3b_profile(self):
        """Verify Ornith 1.5 35B A3B (256 experts, 8 routed)."""
        p = get_model_profile("ornith1.5-35b-a3b")
        self.assertEqual(p.total_experts, 256)
        self.assertEqual(p.routed_experts, 8)
        self.assertEqual(p.hidden_size, 4096)
        self.assertEqual(p.intermediate_size, 2048)

    def test_profile_aliases(self):
        """Verify common alias strings resolve correctly."""
        self.assertEqual(get_model_profile("qwen38").name, "Qwen 3.8 Flash Next")
        self.assertEqual(get_model_profile("qwen36").name, "Qwen 3.6 35B A3B")
        self.assertEqual(get_model_profile("ornith").name, "Ornith 1.5 35B A3B")

    def test_invalid_profile_raises(self):
        """Verify KeyError is raised for unsupported profile name."""
        with self.assertRaises(KeyError):
            get_model_profile("unknown_model_v99")


class TestTQ10PackingIntegration(unittest.TestCase):
    """Test bit-level roundtrip between ActivationCalibrator and Strata TQ1_0 format."""

    def test_calibrated_tq1_0_roundtrip(self):
        """Verify calibrated trits and scales pack and unpack bit-for-bit."""
        calibrator = ActivationCalibrator(profile="qwen3.6-35b-a3b")
        rng = np.random.RandomState(456)

        # 4 blocks of 256 weights = 1024 weights
        weights = rng.randn(1024).astype(np.float32)
        acts = rng.randn(100, 1024).astype(np.float32)

        res = calibrator.calibrate_layer(weights, activations=acts)
        packed_bytes = calibrator.pack_to_tq1_0(res)

        # 4 blocks * 54 bytes/block = 216 bytes
        self.assertEqual(len(packed_bytes), 4 * 54)

        unpacked_trits, unpacked_scales = calibrator.unpack_from_tq1_0(packed_bytes, shape=(1024,))

        np.testing.assert_array_equal(unpacked_trits, res.trits.reshape(-1))
        np.testing.assert_array_equal(unpacked_scales, res.scales)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestPyTorchInteroperability(unittest.TestCase):
    """Test PyTorch Tensor inputs and device handling."""

    def test_torch_tensor_end_to_end(self):
        """Verify PyTorch Tensors flow through calibrator and retain device."""
        calibrator = ActivationCalibrator(profile="qwen3.6-35b-a3b")
        w_torch = torch.randn(512, dtype=torch.float32)
        acts_torch = torch.randn(64, 512, dtype=torch.float32)

        res = calibrator.calibrate_layer(w_torch, activations=acts_torch)

        self.assertIsInstance(res.trits, torch.Tensor)
        self.assertIsInstance(res.scales, torch.Tensor)
        self.assertEqual(res.trits.dtype, torch.int8)
        self.assertEqual(res.scales.dtype, torch.float16)
        self.assertEqual(res.trits.shape, (512,))
        self.assertEqual(res.scales.shape, (2,))


if __name__ == "__main__":
    unittest.main()

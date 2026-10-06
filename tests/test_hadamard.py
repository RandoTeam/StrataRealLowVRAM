"""Unit test suite for Fast Walsh-Hadamard Transform (FWHT) & Outlier Suppression.

Verifies:
1. Orthogonality: H @ H.T = I and (H @ R) @ (H @ R).T = I.
2. Norm preservation (Isometry): ||H * x||_2 = ||x||_2 within 1e-5 relative tolerance.
3. Outlier suppression:
   - Mathematical bound: max |H * x| <= (1 / sqrt(N)) * sum |x|.
   - Outlier magnitude reduction by at least 3x - 6x on Cauchy and extreme normal outliers.
4. Target models:
   - Qwen 3.8 Flash Next (hidden_size 2048, MoE intermediate 512, 512 experts)
   - Qwen 3.6 35B A3B (hidden_size 2048, MoE intermediate 512, 256 experts)
   - Ornith 1.5 35B A3B (hidden_size 2048, MoE intermediate 512, 256 experts)
5. Randomized Hadamard Transform (RHT) coherence breakdown.
6. Arbitrary dimension support (block-diagonal and padded Hadamard).
7. Weight and activation rotation invariance for linear layers: X_rot @ W_rot == X @ W.
"""

import math
import os
import sys
import unittest

import numpy as np
import torch

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.ternary.hadamard import (
    apply_hadamard_activation_rotation,
    apply_hadamard_weight_rotation,
    block_hadamard_transform,
    decompose_dimension,
    fwht,
    generate_random_signs,
    get_hadamard_matrix,
    ifwht,
    inverse_block_hadamard_transform,
    inverse_randomized_hadamard_transform,
    irht,
    is_power_of_two,
    measure_outlier_ratio,
    next_power_of_two,
    outlier_suppression_bound,
    padded_hadamard_inverse,
    padded_hadamard_transform,
    randomized_hadamard_transform,
    rht,
)


class TestOrthogonality(unittest.TestCase):
    """Test orthogonality: H @ H.T = I across power-of-2 and arbitrary dimensions."""

    def test_power_of_two_orthogonality_torch(self):
        """Verify H @ H.T = I for powers of 2 using PyTorch in float64 and float32."""
        for N in [2, 4, 8, 16, 64, 128, 256, 512, 2048]:
            with self.subTest(N=N):
                # Float64: pristine machine precision
                I64 = torch.eye(N, dtype=torch.float64)
                H64 = fwht(I64, normalize=True)
                HTH64 = H64 @ H64.T
                max_err64 = torch.max(torch.abs(HTH64 - I64)).item()
                self.assertLess(max_err64, 1e-14, f"Float64 orthogonality failed for N={N}")

                # Float32: standard deep learning precision
                I32 = torch.eye(N, dtype=torch.float32)
                H32 = fwht(I32, normalize=True)
                HTH32 = H32 @ H32.T
                max_err32 = torch.max(torch.abs(HTH32 - I32)).item()
                self.assertLess(max_err32, 1e-5, f"Float32 orthogonality failed for N={N}")

    def test_power_of_two_orthogonality_numpy(self):
        """Verify H @ H.T = I for powers of 2 using NumPy."""
        for N in [2, 4, 8, 16, 64, 128, 256, 512, 1024]:
            with self.subTest(N=N):
                I = np.eye(N, dtype=np.float64)
                H = fwht(I, normalize=True)
                HTH = H @ H.T
                max_err = np.max(np.abs(HTH - I))
                self.assertLess(max_err, 1e-14, f"NumPy orthogonality failed for N={N}")

    def test_randomized_hadamard_orthogonality(self):
        """Verify (H @ R) @ (H @ R).T = I with random Rademacher sign flipping."""
        for N in [64, 512, 2048]:
            with self.subTest(N=N):
                signs = generate_random_signs(N, seed=42 + N)
                I = torch.eye(N, dtype=torch.float64)
                H_rht = rht(I, signs=signs, normalize=True)
                HTH = H_rht @ H_rht.T
                max_err = torch.max(torch.abs(HTH - I)).item()
                self.assertLess(max_err, 1e-14, f"RHT orthogonality failed for N={N}")

    def test_block_hadamard_orthogonality_non_power_of_two(self):
        """Verify H_block @ H_block.T = I for non-power-of-2 dimensions."""
        # Realistic transformer dimensions that may not be powers of 2
        for D in [192, 768, 1408, 1536, 2560, 5632]:
            with self.subTest(D=D):
                # Greedy power-of-2 decomposition
                I = torch.eye(D, dtype=torch.float64)
                H_block = block_hadamard_transform(I, normalize=True)
                HTH = H_block @ H_block.T
                max_err = torch.max(torch.abs(HTH - I)).item()
                self.assertLess(max_err, 1e-14, f"Block Hadamard orthogonality failed for D={D}")

                # Fixed block_size = 512 (when divisible, e.g. 1536, 2560, 5632)
                if D % 512 == 0:
                    H_b512 = block_hadamard_transform(I, block_size=512, normalize=True)
                    HTH_b512 = H_b512 @ H_b512.T
                    max_err_b512 = torch.max(torch.abs(HTH_b512 - I)).item()
                    self.assertLess(max_err_b512, 1e-14, f"Block-512 Hadamard orthogonality failed for D={D}")


class TestNormPreservation(unittest.TestCase):
    """Test norm preservation (Parseval identity): ||H * x||_2 = ||x||_2."""

    def test_norm_preservation_distributions_torch(self):
        """Test ||H * x||_2 == ||x||_2 within 1e-5 relative tolerance across distributions."""
        torch.manual_seed(101)
        for N in [512, 2048]:
            # 1. Normal distribution
            x_norm = torch.randn(64, N, dtype=torch.float32)
            # 2. Uniform distribution
            x_unif = torch.rand(64, N, dtype=torch.float32) * 2.0 - 1.0
            # 3. Sparse extreme outliers
            x_sparse = torch.zeros(64, N, dtype=torch.float32)
            x_sparse[:, 0] = 100.0
            x_sparse[:, N // 2] = -50.0
            # 4. Cauchy heavy-tailed
            x_cauchy = torch.distributions.Cauchy(0, 1).sample((64, N))

            for dist_name, x in [
                ("Normal", x_norm),
                ("Uniform", x_unif),
                ("Sparse", x_sparse),
                ("Cauchy", x_cauchy),
            ]:
                with self.subTest(N=N, distribution=dist_name):
                    y = fwht(x, normalize=True)
                    norm_x = torch.norm(x, p=2, dim=-1)
                    norm_y = torch.norm(y, p=2, dim=-1)
                    rel_err = torch.max(torch.abs(norm_x - norm_y) / (norm_x + 1e-8)).item()
                    self.assertLess(
                        rel_err,
                        1e-5,
                        f"Norm preservation failed for N={N}, dist={dist_name}, rel_err={rel_err:.2e}",
                    )

    def test_norm_preservation_numpy(self):
        """Test norm preservation with NumPy arrays."""
        np.random.seed(202)
        for N in [512, 2048]:
            x = np.random.randn(32, N)
            y = fwht(x, normalize=True)
            norm_x = np.linalg.norm(x, axis=-1)
            norm_y = np.linalg.norm(y, axis=-1)
            rel_err = np.max(np.abs(norm_x - norm_y) / norm_x)
            self.assertLess(rel_err, 1e-12, f"NumPy norm preservation failed for N={N}")

    def test_norm_preservation_multidimensional_tensors(self):
        """Test norm preservation on 3D and 4D batched tensors."""
        torch.manual_seed(303)
        # 3D: (batch_size, seq_len, hidden_dim) = (4, 32, 2048)
        x_3d = torch.randn(4, 32, 2048)
        y_3d = fwht(x_3d, normalize=True)
        norm_x_3d = torch.norm(x_3d, p=2, dim=-1)
        norm_y_3d = torch.norm(y_3d, p=2, dim=-1)
        rel_err_3d = torch.max(torch.abs(norm_x_3d - norm_y_3d) / norm_x_3d).item()
        self.assertLess(rel_err_3d, 1e-5)

        # 4D: (batch, heads, seq_len, head_dim) = (2, 8, 16, 512)
        x_4d = torch.randn(2, 8, 16, 512)
        y_4d = fwht(x_4d, normalize=True)
        norm_x_4d = torch.norm(x_4d, p=2, dim=-1)
        norm_y_4d = torch.norm(y_4d, p=2, dim=-1)
        rel_err_4d = torch.max(torch.abs(norm_x_4d - norm_y_4d) / norm_x_4d).item()
        self.assertLess(rel_err_4d, 1e-5)

    def test_norm_preservation_cuda_if_available(self):
        """Test norm preservation on GPU device if CUDA is available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        x_cuda = torch.randn(16, 2048, device="cuda", dtype=torch.float32)
        y_cuda = fwht(x_cuda, normalize=True)
        norm_x = torch.norm(x_cuda, p=2, dim=-1)
        norm_y = torch.norm(y_cuda, p=2, dim=-1)
        rel_err = torch.max(torch.abs(norm_x - norm_y) / norm_x).item()
        self.assertLess(rel_err, 1e-5)


class TestOutlierSuppression(unittest.TestCase):
    """Test outlier suppression properties and mathematical bounds."""

    def test_theoretical_upper_bound(self):
        """Prove that max |H * x| <= (1 / sqrt(N)) * sum |x| holds strictly."""
        torch.manual_seed(404)
        for N in [512, 2048]:
            with self.subTest(N=N):
                # 100 trials with varied distributions and outlier scales
                for _ in range(20):
                    x = torch.randn(10, N)
                    # Add random spikes
                    num_spikes = torch.randint(1, 5, (1,)).item()
                    spike_indices = torch.randint(0, N, (num_spikes,))
                    x[:, spike_indices] += torch.randn(10, num_spikes) * 50.0

                    y = fwht(x, normalize=True)

                    max_y = torch.max(torch.abs(y), dim=-1).values
                    l1_x = torch.sum(torch.abs(x), dim=-1)
                    theoretical_bound = l1_x / math.sqrt(N)

                    # Check max |H * x| <= (1 / sqrt(N)) * ||x||_1 (with numerical tolerance)
                    diff = max_y - theoretical_bound
                    max_violation = torch.max(diff).item()
                    self.assertLessEqual(
                        max_violation,
                        1e-5,
                        f"Theoretical upper bound violated for N={N}: max violation = {max_violation}",
                    )

    def test_single_channel_outlier_reduction(self):
        """Verify that single channel outliers are reduced by at least 3x - 6x (typically 8x - 20x)."""
        torch.manual_seed(505)
        # Test target model dimensions: 512 (MoE intermediate) and 2048 (hidden size)
        for N in [512, 2048]:
            with self.subTest(N=N):
                x = torch.randn(50, N)
                # Introduce a massive 50x outlier at a fixed channel
                outlier_channel = 17
                x[:, outlier_channel] = 100.0

                signs = generate_random_signs(N, seed=505 + N)
                y = rht(x, signs=signs, normalize=True)

                max_x = torch.max(torch.abs(x), dim=-1).values
                max_y = torch.max(torch.abs(y), dim=-1).values
                reduction_ratios = max_x / max_y

                mean_ratio = torch.mean(reduction_ratios).item()
                min_ratio = torch.min(reduction_ratios).item()

                # Requirement: outlier magnitude reduced by at least 3x - 6x
                self.assertGreaterEqual(
                    min_ratio,
                    3.0,
                    f"Outlier reduction min ratio {min_ratio:.2f} < 3.0x for N={N}",
                )
                self.assertGreaterEqual(
                    mean_ratio,
                    6.0,
                    f"Outlier reduction mean ratio {mean_ratio:.2f} < 6.0x for N={N}",
                )

    def test_cauchy_heavy_tailed_outlier_reduction(self):
        """Verify outlier suppression on heavy-tailed Cauchy distributions."""
        torch.manual_seed(606)
        for N in [512, 2048]:
            with self.subTest(N=N):
                # Cauchy distribution produces extreme heavy tails
                x = torch.distributions.Cauchy(0, 1).sample((100, N))
                signs = generate_random_signs(N, seed=606 + N)
                y = rht(x, signs=signs, normalize=True)

                max_x = torch.max(torch.abs(x), dim=-1).values
                max_y = torch.max(torch.abs(y), dim=-1).values
                reduction_ratios = max_x / max_y

                # Filter samples where x actually had a substantial outlier (> 5.0)
                mask = max_x > 5.0
                if torch.sum(mask) > 0:
                    ratios_filtered = reduction_ratios[mask]
                    mean_ratio = torch.mean(ratios_filtered).item()
                    # On heavy-tailed samples with outliers, mean reduction exceeds 3x - 6x
                    self.assertGreaterEqual(
                        mean_ratio,
                        3.0,
                        f"Cauchy mean reduction {mean_ratio:.2f} < 3.0x for N={N}",
                    )

    def test_target_models_architectures(self):
        """Simulate realistic activation and MoE expert tensors for the 3 target models:
        1. Qwen 3.8 Flash Next (hidden 2048, expert 512, 512 experts)
        2. Qwen 3.6 35B A3B (hidden 2048, expert 512, 256 experts)
        3. Ornith 1.5 35B A3B (hidden 2048, expert 512, 256 experts)
        """
        torch.manual_seed(707)

        # Target 1: Hidden dimension (N = 2048)
        # Activation tensor: (batch=2, seq_len=16, hidden=2048)
        x_hidden = torch.randn(2, 16, 2048)
        # Inject realistic outlier channels (channels 128 and 1024 with 30x activation spikes)
        x_hidden[:, :, 128] += 45.0
        x_hidden[:, :, 1024] -= 60.0

        signs_hidden = generate_random_signs(2048, seed=707)
        y_hidden = rht(x_hidden, signs=signs_hidden, normalize=True)

        ratio_hidden = measure_outlier_ratio(x_hidden, y_hidden)
        # For N=2048, reduction is typically > 12x - 18x
        self.assertGreaterEqual(
            ratio_hidden,
            6.0,
            f"Hidden size 2048 outlier suppression {ratio_hidden:.2f} < 6.0x",
        )

        # Target 2: MoE expert intermediate dimension (N = 512)
        # Expert intermediate activation tensor: (active_experts=8, tokens=16, intermediate=512)
        x_expert = torch.randn(8, 16, 512)
        # Inject outlier in intermediate activations
        x_expert[:, :, 42] += 50.0

        signs_expert = generate_random_signs(512, seed=808)
        y_expert = rht(x_expert, signs=signs_expert, normalize=True)

        ratio_expert = measure_outlier_ratio(x_expert, y_expert)
        # For N=512, reduction is typically > 6x - 10x
        self.assertGreaterEqual(
            ratio_expert,
            5.0,
            f"Expert intermediate 512 outlier suppression {ratio_expert:.2f} < 5.0x",
        )


class TestRandomizedSignFlipping(unittest.TestCase):
    """Test Randomized Sign Flipping (RHT) and anti-coherence properties."""

    def test_coherence_breakdown(self):
        """Verify that RHT prevents coherence collapse on adversarial / worst-case inputs."""
        N = 2048
        # Adversarial input: all ones vector (completely aligned with first Walsh-Hadamard row)
        x_adversarial = torch.ones(N)

        # Without RHT: constructive interference produces a massive spike of sqrt(N) = 45.25
        y_no_rht = fwht(x_adversarial, normalize=True)
        peak_no_rht = torch.max(torch.abs(y_no_rht)).item()
        expected_peak = math.sqrt(N)
        self.assertAlmostEqual(peak_no_rht, expected_peak, places=3)

        # With RHT: signs randomize phases, peak collapses from 45.25 to ~3.0 - 4.0
        signs = generate_random_signs(N, seed=909)
        y_with_rht = rht(x_adversarial, signs=signs, normalize=True)
        peak_with_rht = torch.max(torch.abs(y_with_rht)).item()

        # Coherence breakdown ratio
        coherence_reduction = peak_no_rht / peak_with_rht
        self.assertGreaterEqual(
            coherence_reduction,
            8.0,
            f"RHT failed to break coherence: peak without={peak_no_rht:.2f}, with={peak_with_rht:.2f}",
        )

    def test_rht_roundtrip_inversion(self):
        """Verify exact mathematical inversion: irht(rht(x, signs), signs) == x."""
        torch.manual_seed(1001)
        for N in [512, 2048]:
            x = torch.randn(8, N)
            signs = generate_random_signs(N, seed=1001 + N)
            y = rht(x, signs=signs, normalize=True)
            x_rec = irht(y, signs=signs, normalize=True)
            max_err = torch.max(torch.abs(x - x_rec)).item()
            self.assertLess(max_err, 1e-6, f"RHT inversion roundtrip error {max_err:.2e} for N={N}")


class TestBlockAndPaddedHadamard(unittest.TestCase):
    """Test non-power-of-2 dimension handling via block-diagonal and padded transforms."""

    def test_dimension_decomposition(self):
        """Test greedy power-of-2 decomposition logic."""
        self.assertEqual(decompose_dimension(2048), [2048])
        self.assertEqual(decompose_dimension(512), [512])
        self.assertEqual(decompose_dimension(768), [512, 256])
        self.assertEqual(decompose_dimension(5632), [4096, 1024, 512])
        self.assertEqual(decompose_dimension(5632, block_size=512), [512] * 11)

    def test_block_hadamard_roundtrip(self):
        """Test block Hadamard forward and inverse round-trip on arbitrary dimensions."""
        torch.manual_seed(1101)
        for D in [192, 768, 1408, 2560, 5632]:
            with self.subTest(D=D):
                x = torch.randn(4, D)
                signs = generate_random_signs(D, seed=D)
                y = block_hadamard_transform(x, signs=signs, normalize=True)
                x_rec = inverse_block_hadamard_transform(y, signs=signs, normalize=True)
                max_err = torch.max(torch.abs(x - x_rec)).item()
                self.assertLess(max_err, 1e-6, f"Block roundtrip failed for D={D}")

    def test_padded_hadamard_roundtrip(self):
        """Test padded Hadamard transform and inverse unpadding."""
        torch.manual_seed(1201)
        for D in [300, 768, 1500, 3000]:
            with self.subTest(D=D):
                x = torch.randn(8, D)
                y_padded, orig_dim = padded_hadamard_transform(x, normalize=True)
                self.assertEqual(orig_dim, D)
                self.assertTrue(is_power_of_two(y_padded.shape[-1]))

                x_rec = padded_hadamard_inverse(y_padded, orig_dim=orig_dim, normalize=True)
                self.assertEqual(x_rec.shape, x.shape)
                max_err = torch.max(torch.abs(x - x_rec)).item()
                self.assertLess(max_err, 1e-6, f"Padded roundtrip failed for D={D}")


class TestWeightAndActivationRotation(unittest.TestCase):
    """Test weight and activation rotation invariance: X_rot @ W_rot == X @ W."""

    def test_gemm_invariance(self):
        """Verify that pre-rotated weights and online rotated activations preserve GEMM output."""
        torch.manual_seed(1301)
        B, in_feat, out_feat = 8, 2048, 512
        X = torch.randn(B, in_feat)
        W = torch.randn(in_feat, out_feat)
        signs = generate_random_signs(in_feat, seed=1301)

        # Rotate activations: X_rot = X @ R @ H
        X_rot = apply_hadamard_activation_rotation(X, signs=signs)
        # Rotate weights along contraction dim: W_rot = H @ R @ W
        W_rot = apply_hadamard_weight_rotation(W, signs=signs, rotate_dim=0)

        Y_orig = X @ W
        Y_rot = X_rot @ W_rot

        rel_err = (torch.max(torch.abs(Y_orig - Y_rot)) / torch.max(torch.abs(Y_orig))).item()
        self.assertLess(rel_err, 1e-5, f"GEMM rotation invariance failed: rel_err={rel_err:.2e}")

    def test_nn_linear_invariance(self):
        """Verify invariance in PyTorch nn.Linear layout: weight is (out_features, in_features)."""
        torch.manual_seed(1401)
        B, in_feat, out_feat = 4, 2048, 512
        X = torch.randn(B, in_feat)
        # Linear layer weight shape: (out_features, in_features)
        W_linear = torch.randn(out_feat, in_feat)
        signs = generate_random_signs(in_feat, seed=1401)

        X_rot = apply_hadamard_activation_rotation(X, signs=signs)
        W_linear_rot = apply_hadamard_weight_rotation(W_linear, signs=signs, rotate_dim=-1)

        Y_orig = torch.nn.functional.linear(X, W_linear)
        Y_rot = torch.nn.functional.linear(X_rot, W_linear_rot)

        rel_err = (torch.max(torch.abs(Y_orig - Y_rot)) / torch.max(torch.abs(Y_orig))).item()
        self.assertLess(rel_err, 1e-5, f"nn.Linear rotation invariance failed: rel_err={rel_err:.2e}")

    def test_moe_expert_tensor_rotation(self):
        """Verify invariance on 3D batched MoE expert weights: (num_experts, out_feat, in_feat)."""
        torch.manual_seed(1501)
        num_experts, in_feat, out_feat = 16, 2048, 512
        signs = generate_random_signs(in_feat, seed=1501)

        # Weights for 16 experts
        W_experts = torch.randn(num_experts, out_feat, in_feat)
        W_experts_rot = apply_hadamard_weight_rotation(W_experts, signs=signs, rotate_dim=-1)

        # Random routed token activations: (num_experts, tokens_per_expert=4, in_feat)
        X_tokens = torch.randn(num_experts, 4, in_feat)
        X_tokens_rot = apply_hadamard_activation_rotation(X_tokens, signs=signs)

        # Batch matrix multiply: (num_experts, 4, in_feat) @ (num_experts, in_feat, out_feat)
        Y_orig = torch.bmm(X_tokens, W_experts.transpose(1, 2))
        Y_rot = torch.bmm(X_tokens_rot, W_experts_rot.transpose(1, 2))

        rel_err = (torch.max(torch.abs(Y_orig - Y_rot)) / torch.max(torch.abs(Y_orig))).item()
        self.assertLess(rel_err, 1e-5, f"MoE expert rotation invariance failed: rel_err={rel_err:.2e}")


class TestInplaceEquivalence(unittest.TestCase):
    """Test that inplace=True produces identical results to inplace=False."""

    def test_inplace_torch(self):
        torch.manual_seed(1601)
        x = torch.randn(10, 512)
        y_out = fwht(x, normalize=True, inplace=False)

        x_in = x.clone()
        y_in = fwht(x_in, normalize=True, inplace=True)

        self.assertTrue(torch.allclose(y_out, y_in, atol=1e-7))
        self.assertTrue(torch.allclose(x_in, y_out, atol=1e-7))

    def test_inplace_numpy(self):
        np.random.seed(1701)
        x = np.random.randn(10, 512)
        y_out = fwht(x, normalize=True, inplace=False)

        x_in = x.copy()
        y_in = fwht(x_in, normalize=True, inplace=True)

        self.assertTrue(np.allclose(y_out, y_in, atol=1e-12))
        self.assertTrue(np.allclose(x_in, y_out, atol=1e-12))


if __name__ == "__main__":
    unittest.main()

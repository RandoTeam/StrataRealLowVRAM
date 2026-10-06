"""Activation-Aware & MoE-Router-Weighted Calibration for 1.58-bit Ternary Quantization.

Mathematical Foundation:
========================
1. Activation-Weighted Reconstruction Loss:
   Let W in R^{d_out x d_in} be a weight matrix and X in R^{N x d_in} be an input activation
   matrix across N calibration tokens. The output approximation error is:
       ||Y - Y_hat||_F^2 = ||X W^T - X W_hat^T||_F^2
                          = Tr( (W - W_hat) X^T X (W - W_hat)^T )
                          = Tr( (W - W_hat) H (W - W_hat)^T )
   where H = X^T X in R^{d_in x d_in} is the unnormalized activation covariance / Hessian matrix.

2. Diagonal Hessian Approximation:
   In high-dimensional transformer layers (d_in in {2048, 4096, 8192}), full covariance
   inversion or storage across 512 experts is memory-intensive. The diagonal approximation
   h = diag(H) in R^{d_in} measures feature channel variance:
       h_j = sum_{i=1}^N X_{i, j}^2 = ||X_{:, j}||_2^2
   Under the diagonal Hessian, the loss decomposes across rows and columns:
       L(W, W_hat; h) = sum_{r=1}^{d_out} sum_{c=1}^{d_in} h_c (W_{rc} - W_hat_{rc})^2

3. Hessian-Weighted Optimal Scale alpha*(Delta) for Trits:
   Let W_hat = alpha * T where T = trit(W, Delta) in {-1, 0, +1}^{d_out x d_in}:
       trit(w, Delta) = +1  if w > Delta
                        -1  if w < -Delta
                         0  if |w| <= Delta
   For active indices I_active = {(r, c) : |W_{rc}| > Delta}:
       L(alpha, Delta; h) = sum_{(r,c) not in I_active} h_c W_{rc}^2 + sum_{(r,c) in I_active} h_c (|W_{rc}| - alpha)^2
   Differentiating with respect to alpha and setting to zero:
       dL / d(alpha) = -2 sum_{(r,c) in I_active} h_c (|W_{rc}| - alpha) = 0
       ==> alpha*(Delta; h) = (sum_{(r,c) in I_active} h_c |W_{rc}|) / (sum_{(r,c) in I_active} h_c)
   This is the exact activation-variance-weighted expectation of active weight magnitudes.
   When h_c = 1 (uniform activations), this reduces identically to unweighted MSE:
       alpha*(Delta; 1) = (1 / N_active) sum_{i in I_active} |w_i|

4. General Full-Hessian Optimal Scale:
   For arbitrary positive semi-definite matrix H:
       L(alpha) = Tr( (W - alpha T) H (W - alpha T)^T )
                = Tr(W H W^T) - 2 alpha Tr(W H T^T) + alpha^2 Tr(T H T^T)
       ==> alpha*(H, Delta) = Tr(W H T^T) / Tr(T H T^T)
   Substituting alpha* back into L:
       L(alpha*, Delta; H) = Tr(W H W^T) - (Tr(W H T^T))^2 / Tr(T H T^T)
   Minimizing weighted reconstruction error is therefore equivalent to maximizing:
       J(Delta) = (Tr(W H T^T))^2 / Tr(T H T^T)

5. Mixture-of-Experts (MoE) Router Weighting:
   In MoE architectures (e.g. Qwen 3.8 Flash Next with 512 experts / 8 routed,
   Qwen 3.6 35B A3B with 256 experts / 8 routed, Ornith 1.5 35B A3B with 256 experts / 8 routed),
   token x_i is routed to a top-k subset of experts K_i subset of {0, ..., E-1} with gating
   weights g_e(x_i) in [0, 1] such that sum_{e in K_i} g_e(x_i) = 1.
   The effective input activation entering expert e for token i is:
       x_{e, i} = g_e(x_i) * x_i
   The expert-specific activation Hessian is:
       H_e = sum_{i=1}^N g_e(x_i)^2 x_i x_i^T
   In diagonal form:
       h_{e, j} = sum_{i=1}^N g_e(x_i)^2 X_{i, j}^2 = (G_sq^T S_act)_{e, j}
   where G_sq = G^2 in R^{N x E} and S_act = X^2 in R^{N x d_in}.

6. Rare-Expert Regularization & Fallback:
   Rarely-fired experts have total routing mass M_e = sum_{i=1}^N g_e(x_i) << N.
   To prevent overfitting or degeneracy on few calibration tokens:
   - Bayesian shrinkage towards global activation prior:
       h_e^{reg} = (h_e + lambda * h_global) / (M_e + lambda)
     where h_global = (1 / N) sum_{i=1}^N X_i^2.
   - Fallback: If M_e < epsilon or activations are uniform / unavailable, smoothly fall
     back to fast block-wise unweighted MSE (tools.ternary.trit_sampler.quantize_to_trits).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .trit_sampler import (
    BLOCK_SIZE,
    TritQuantResult,
    pack_tq1_0,
    quantize_to_trits,
    unpack_tq1_0,
)


# ==============================================================================
# Model Architectures & Profiles
# ==============================================================================

@dataclass(frozen=True)
class MoEModelProfile:
    """Architectural profile for Mixture-of-Experts models."""
    name: str
    total_experts: int
    routed_experts: int
    hidden_size: int
    intermediate_size: int
    shared_experts: int = 0
    description: str = ""


MODEL_PROFILES: Dict[str, MoEModelProfile] = {
    "qwen3.8-flash-next": MoEModelProfile(
        name="Qwen 3.8 Flash Next",
        total_experts=512,
        routed_experts=8,
        hidden_size=4096,
        intermediate_size=2048,
        shared_experts=0,
        description="125B MoE with 512 fine-grained routed experts (top-8 active)",
    ),
    "qwen3.6-35b-a3b": MoEModelProfile(
        name="Qwen 3.6 35B A3B",
        total_experts=256,
        routed_experts=8,
        hidden_size=4096,
        intermediate_size=2048,
        shared_experts=0,
        description="35B MoE with 256 routed experts (top-8 active)",
    ),
    "ornith1.5-35b-a3b": MoEModelProfile(
        name="Ornith 1.5 35B A3B",
        total_experts=256,
        routed_experts=8,
        hidden_size=4096,
        intermediate_size=2048,
        shared_experts=0,
        description="35B MoE architecture with 256 routed experts (top-8 active)",
    ),
}

# Alias lookups for convenience
MODEL_ALIASES: Dict[str, str] = {
    "qwen38": "qwen3.8-flash-next",
    "qwen-3.8": "qwen3.8-flash-next",
    "qwen-3.8-flash-next": "qwen3.8-flash-next",
    "qwen36": "qwen3.6-35b-a3b",
    "qwen-3.6": "qwen3.6-35b-a3b",
    "qwen-3.6-35b-a3b": "qwen3.6-35b-a3b",
    "ornith": "ornith1.5-35b-a3b",
    "ornith-1.5": "ornith1.5-35b-a3b",
    "ornith1.5": "ornith1.5-35b-a3b",
    "ornith-1.5-35b-a3b": "ornith1.5-35b-a3b",
}


def get_model_profile(name_or_alias: str) -> MoEModelProfile:
    """Retrieve an MoE model profile by standard key or common alias."""
    key = name_or_alias.strip().lower()
    if key in MODEL_PROFILES:
        return MODEL_PROFILES[key]
    if key in MODEL_ALIASES:
        return MODEL_PROFILES[MODEL_ALIASES[key]]
    raise KeyError(
        f"Unknown model profile '{name_or_alias}'. Supported: {list(MODEL_PROFILES.keys())}"
    )


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class CalibrationConfig:
    """Configuration hyperparameters for activation and MoE calibration."""
    model_name: str = "qwen3.6-35b-a3b"
    block_size: int = BLOCK_SIZE
    bounds_factor: Tuple[float, float] = (0.5, 0.9)
    grid_points: int = 33
    target_sparsity: Optional[Tuple[float, float]] = (35.0, 50.0)
    regularization_lambda: float = 1e-4
    rare_expert_threshold: float = 1e-3
    uniform_threshold: float = 1e-4
    use_diagonal_hessian: bool = True
    dampening: float = 1e-6


@dataclass
class ActivationHessian:
    """Encapsulates activation covariance / second-moment estimates."""
    diag: np.ndarray                            # (d_in,) diagonal elements
    full: Optional[np.ndarray] = None           # Optional (d_in, d_in) full matrix
    sample_count: int = 0                       # Number of tokens processed
    outlier_channels: np.ndarray = field(       # Indices of high-variance channels
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    outlier_ratio: float = 1.0                  # Max variance / median variance


@dataclass
class RouterGatingResult:
    """Output of MoE router gating computation."""
    gating_weights: np.ndarray                  # (N, E) sparse or dense routing weights
    active_expert_indices: np.ndarray           # (N, k) indices of top-k experts per token
    expert_frequencies: np.ndarray              # (E,) fraction of tokens where expert was top-k
    expert_total_weights: np.ndarray            # (E,) sum of gating weights per expert
    routing_entropy: float                      # Routing distribution entropy (load balance metric)
    rare_experts: List[int]                     # Indices of rarely-fired experts
    frequent_experts: List[int]                 # Indices of frequently-fired experts


class CalibratedTritResult(NamedTuple):
    """Result of activation-aware trit calibration."""
    trits: Any                                  # int8 array/tensor {-1, 0, +1}
    scales: Any                                 # float16 array/tensor per block
    thresholds: Any                             # float32 array/tensor of thresholds
    weighted_error: float                       # Tr((W-W_hat)H(W-W_hat)^T) / Tr(WHW^T)
    unweighted_mse: float                       # Standard reconstruction MSE
    sparsity: float                             # Percentage of zero trits (target: 35-50%)
    expert_weight: float                        # Gating mass associated with this expert
    is_fallback: bool                           # True if unweighted fallback was invoked


@dataclass
class SyntheticDataset:
    """Synthetic calibration sequences and activation matrices for offline testing."""
    text_samples: List[str]
    code_samples: List[str]
    token_ids: List[List[int]]
    activations: np.ndarray                      # (N, d_in)
    router_weights: np.ndarray                   # (E, d_in)


# ==============================================================================
# Synthetic Calibration Token & Corpus Generator
# ==============================================================================

class SyntheticCalibrationGenerator:
    """Generates synthetic calibration corpora, tokens, and activations.

    Covers:
    - Multilingual text: English, Chinese, Japanese, German, French, Spanish, Russian.
    - Code syntax: Python (classes, typing, numpy), C++ (templates, RAII), CUDA kernels.
    - Mathematical formulations and structured JSON.
    - Realistic activation distributions with power-law tails and outlier features.
    """

    MULTILINGUAL_SAMPLES: List[str] = [
        # English
        "The transformer architecture relies on scaled dot-product attention and multi-head mechanisms.",
        "In Mixture-of-Experts routing, tokens are dispatched dynamically across sparse expert networks.",
        # Chinese (Simplified)
        "混合专家模型（MoE）通过稀疏门控网络将输入令牌路由到前八个活跃专家中进行计算。",
        "三进制量化采用正一、零和负一表示权重，并在保持模型精度的同时大幅度降低显存占用。",
        # Japanese
        "Walsh-Hadamard変換を用いることで、重みテンソルの異常値を空間全体に均一に分散させます。",
        "スパース活性化行列の共分散を計算し、ヘッセ行列重み付けによる最適な閾値を決定します。",
        # German
        "Die Optimierung der ternären Schwellenwerte minimiert den gewichteten Rekonstruktionsfehler.",
        "Hierarchische Speicherarchitekturen ermöglichen eine hocheffiziente Ausführung auf modernen Beschleunigern.",
        # French
        "La quantification ternaire permet une réduction significative de l'empreinte mémoire des réseaux de neurones.",
        "Les poids de routage déterminent la contribution relative de chaque expert lors de l'inférence.",
        # Spanish
        "El calibrador calcula la matriz de covarianza de las activaciones para preservar los canales críticos.",
        "La transformación Hadamard mitiga los valores atípicos antes de la cuantización a 1.58 bits.",
        # Russian
        "Активационно-взвешенная калибровка оптимизирует масштабы квантования с учетом функции потерь.",
        "Маршрутизатор MoE выбирает наиболее релевантные эксперты на основе скалярного произведения с эмбеддингом.",
    ]

    CODE_SAMPLES: List[str] = [
        # Python
        (
            "import numpy as np\n"
            "from typing import Tuple, Optional\n\n"
            "def calibrate_block(w: np.ndarray, h: np.ndarray) -> Tuple[float, float]:\n"
            "    active = np.abs(w) > 0.5 * np.mean(np.abs(w))\n"
            "    scale = np.sum(h[active] * np.abs(w[active])) / np.sum(h[active])\n"
            "    return scale, float(np.mean((w - scale * np.sign(w)) ** 2))\n"
        ),
        (
            "@dataclass\n"
            "class ExpertGate:\n"
            "    dim: int\n"
            "    num_experts: int = 512\n"
            "    top_k: int = 8\n\n"
            "    def route(self, x: np.ndarray) -> np.ndarray:\n"
            "        logits = x @ self.w.T\n"
            "        topk = np.argpartition(logits, -self.top_k, axis=-1)[:, -self.top_k:]\n"
            "        return topk\n"
        ),
        # C++
        (
            "#include <vector>\n"
            "#include <cmath>\n"
            "#include <algorithm>\n\n"
            "template <typename T>\n"
            "inline T compute_trit(T w, T delta, T scale) {\n"
            "    if (w > delta) return scale;\n"
            "    if (w < -delta) return -scale;\n"
            "    return static_cast<T>(0);\n"
            "}\n"
        ),
        # CUDA
        (
            "__global__ void k_tq1_dot(const uint8_t* __restrict__ qs,\n"
            "                          const uint8_t* __restrict__ qh,\n"
            "                          const half* __restrict__ d,\n"
            "                          const half* __restrict__ x,\n"
            "                          float* __restrict__ y,\n"
            "                          int num_blocks) {\n"
            "    int bid = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    if (bid >= num_blocks) return;\n"
            "    // Radix-3 SIMD unpacking and fused multiply-accumulate\n"
            "}\n"
        ),
    ]

    def __init__(self, vocab_size: int = 152064, seed: int = 42):
        self.vocab_size = vocab_size
        self.rng = np.random.RandomState(seed)

    def generate_text_corpus(self, repeat: int = 1) -> List[str]:
        """Return combined multilingual and code text samples."""
        corpus = list(self.MULTILINGUAL_SAMPLES) + list(self.CODE_SAMPLES)
        return corpus * repeat

    def generate_token_sequences(
        self,
        num_sequences: int = 16,
        seq_len: int = 64,
    ) -> List[List[int]]:
        """Generate synthetic token ID sequences with realistic special tokens."""
        bos_id, eos_id = 151644, 151645
        sequences = []
        for _ in range(num_sequences):
            # Token IDs drawn from vocabulary with Zipf-like distribution
            raw_ids = self.rng.zipf(a=1.3, size=seq_len - 2) % (self.vocab_size - 100) + 10
            seq = [bos_id] + raw_ids.tolist() + [eos_id]
            sequences.append(seq)
        return sequences

    def generate_activations(
        self,
        num_tokens: int = 512,
        dim: int = 4096,
        num_outliers: int = 32,
        outlier_multiplier: float = 25.0,
    ) -> np.ndarray:
        """Generate synthetic activation matrix with authentic LLM characteristics.

        LLM activations typically exhibit:
        1. Heavy-tailed marginal distributions across channels.
        2. A small set (~0.5-2%) of persistent outlier channels with 10x-50x larger variance.
        3. Subtle cross-channel correlations.
        """
        # Base activations: standard normal modulated by log-normal channel scale
        channel_scales = self.rng.lognormal(mean=0.0, sigma=0.5, size=dim).astype(np.float32)

        # Inject outlier channels
        if num_outliers > 0 and num_outliers < dim:
            outlier_indices = self.rng.choice(dim, size=num_outliers, replace=False)
            channel_scales[outlier_indices] *= outlier_multiplier

        base = self.rng.randn(num_tokens, dim).astype(np.float32)
        activations = base * channel_scales[None, :]

        # Add modest cross-channel correlation via low-rank component
        rank = 4
        u = self.rng.randn(num_tokens, rank).astype(np.float32)
        v = self.rng.randn(rank, dim).astype(np.float32) * 0.1
        activations += u @ v

        return activations

    def generate_router_weights(
        self,
        num_experts: int = 512,
        dim: int = 4096,
        skew_factor: float = 1.8,
    ) -> np.ndarray:
        """Generate router gate weights with skewed expert affinities.

        Ensures some experts fire frequently while others fire rarely,
        mirroring empirical MoE routing behavior.
        """
        # Base random projection
        w_gate = self.rng.randn(num_experts, dim).astype(np.float32) * (1.0 / math.sqrt(dim))

        # Apply skew factor across expert norms so popularity varies by power law
        popularity = np.linspace(0.1, 2.5, num_experts) ** skew_factor
        self.rng.shuffle(popularity)
        w_gate = w_gate * popularity[:, None]

        return w_gate

    def generate_complete_dataset(
        self,
        num_tokens: int = 512,
        dim: int = 4096,
        num_experts: int = 256,
    ) -> SyntheticDataset:
        """Create complete synthetic dataset for offline calibration testing."""
        return SyntheticDataset(
            text_samples=self.generate_text_corpus(),
            code_samples=list(self.CODE_SAMPLES),
            token_ids=self.generate_token_sequences(num_sequences=16, seq_len=64),
            activations=self.generate_activations(num_tokens=num_tokens, dim=dim),
            router_weights=self.generate_router_weights(num_experts=num_experts, dim=dim),
        )


# ==============================================================================
# Hessian & Activation Statistics
# ==============================================================================

def detect_outlier_channels(
    h_diag: np.ndarray,
    threshold_factor: float = 6.0,
) -> Tuple[np.ndarray, float]:
    """Detect outlier activation channels whose variance exceeds threshold * median.

    Returns:
        (outlier_indices, outlier_ratio)
    """
    median_val = float(np.median(h_diag))
    if median_val < 1e-12:
        return np.zeros(0, dtype=np.int64), 1.0

    ratios = h_diag / median_val
    outlier_idx = np.where(ratios >= threshold_factor)[0]
    max_ratio = float(np.max(ratios)) if len(ratios) > 0 else 1.0
    return outlier_idx, max_ratio


def compute_activation_hessian(
    activations: np.ndarray,
    compute_full: bool = False,
    regularize_lambda: float = 1e-4,
) -> ActivationHessian:
    """Compute diagonal and optionally full Hessian from an activation matrix.

    Args:
        activations: (N, d_in) array of activation vectors.
        compute_full: If True, computes the full (d_in, d_in) covariance X^T X / N.
        regularize_lambda: Ridge dampening added to diagonal for conditioning.

    Returns:
        ActivationHessian containing diagonal, full matrix (if requested),
        and outlier channel diagnostics.
    """
    if activations.ndim != 2:
        raise ValueError(f"Activations must be 2D (N, d_in), got shape {activations.shape}")

    n_samples, dim = activations.shape
    if n_samples == 0:
        raise ValueError("Activations cannot be empty")

    # Second moments: h_j = (1 / N) sum_{i=1}^N X_{i, j}^2
    sq = activations ** 2
    h_diag = np.mean(sq, axis=0) + regularize_lambda

    outlier_idx, max_ratio = detect_outlier_channels(h_diag)

    h_full = None
    if compute_full:
        # Full covariance: H = (1 / N) * X^T X + lambda * I
        h_full = (activations.T @ activations) / float(n_samples)
        if regularize_lambda > 0.0:
            np.fill_diagonal(h_full, h_full.diagonal() + regularize_lambda)

    return ActivationHessian(
        diag=h_diag,
        full=h_full,
        sample_count=n_samples,
        outlier_channels=outlier_idx,
        outlier_ratio=max_ratio,
    )


def update_activation_hessian(
    current: Optional[ActivationHessian],
    batch_activations: np.ndarray,
    compute_full: bool = False,
    regularize_lambda: float = 1e-4,
) -> ActivationHessian:
    """Online streaming update of ActivationHessian across minibatches.

    Maintains exact numerical running averages without loading all tokens in RAM.
    """
    if current is None or current.sample_count == 0:
        return compute_activation_hessian(
            batch_activations,
            compute_full=compute_full,
            regularize_lambda=regularize_lambda,
        )

    n_old = current.sample_count
    n_new = batch_activations.shape[0]
    n_total = n_old + n_new

    # Update diagonal second moments
    new_sq = np.sum(batch_activations ** 2, axis=0)
    old_sq_sum = (current.diag - regularize_lambda) * n_old
    merged_diag = (old_sq_sum + new_sq) / float(n_total) + regularize_lambda

    # Update full covariance if present
    merged_full = None
    if compute_full:
        new_outer = batch_activations.T @ batch_activations
        if current.full is not None:
            old_outer = (current.full - np.diag(np.full(current.full.shape[0], regularize_lambda))) * n_old
            merged_full = (old_outer + new_outer) / float(n_total)
        else:
            merged_full = new_outer / float(n_total)
        if regularize_lambda > 0.0:
            np.fill_diagonal(merged_full, merged_full.diagonal() + regularize_lambda)

    outlier_idx, max_ratio = detect_outlier_channels(merged_diag)

    return ActivationHessian(
        diag=merged_diag,
        full=merged_full,
        sample_count=n_total,
        outlier_channels=outlier_idx,
        outlier_ratio=max_ratio,
    )


# ==============================================================================
# MoE Router Gating & Expert Weighting
# ==============================================================================

def compute_router_gates(
    router_weights: np.ndarray,
    activations: np.ndarray,
    top_k: int = 8,
    bias: Optional[np.ndarray] = None,
    temperature: float = 1.0,
    rare_threshold: float = 0.005,
) -> RouterGatingResult:
    """Compute top-k router gating weights and expert dispatch statistics.

    Args:
        router_weights: (E, d_in) routing projection matrix.
        activations: (N, d_in) token activation vectors.
        top_k: Number of active experts per token (default: 8).
        bias: Optional (E,) router bias vector.
        temperature: Logit scaling temperature.
        rare_threshold: Frequency threshold below which an expert is classified as rare.

    Returns:
        RouterGatingResult containing gating weights (N, E), active indices (N, k),
        routing entropy, and rare/frequent expert classifications.
    """
    n_tokens, d_in = activations.shape
    n_experts, d_in_r = router_weights.shape

    if d_in != d_in_r:
        raise ValueError(
            f"Dimension mismatch: activations d_in={d_in}, router_weights d_in={d_in_r}"
        )
    if top_k > n_experts:
        raise ValueError(f"top_k ({top_k}) cannot exceed total experts ({n_experts})")

    # Router logits: S = X @ W_gate^T + b  (N, E)
    logits = activations @ router_weights.T
    if bias is not None:
        logits = logits + bias[None, :]

    if temperature != 1.0 and temperature > 0.0:
        logits = logits / temperature

    # Select top-k experts per token via vectorized partition
    topk_indices = np.argpartition(logits, -top_k, axis=1)[:, -top_k:]  # (N, k)
    topk_logits = np.take_along_axis(logits, topk_indices, axis=1)       # (N, k)

    # Numerically stable softmax over top-k
    max_logit = np.max(topk_logits, axis=1, keepdims=True)
    exp_logits = np.exp(topk_logits - max_logit)
    sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
    topk_gates = exp_logits / np.maximum(sum_exp, 1e-12)                # (N, k)

    # Populate sparse/dense gating matrix (N, E)
    gating_matrix = np.zeros((n_tokens, n_experts), dtype=np.float32)
    np.put_along_axis(gating_matrix, topk_indices, topk_gates, axis=1)

    # Expert dispatch statistics
    expert_counts = np.zeros(n_experts, dtype=np.int64)
    for k_col in range(top_k):
        for e in topk_indices[:, k_col]:
            expert_counts[e] += 1
    expert_frequencies = expert_counts.astype(np.float64) / float(n_tokens)

    expert_total_weights = np.sum(gating_matrix, axis=0)  # (E,)

    # Routing distribution entropy: H(P) = -sum p_e log(p_e)
    # Measures how uniformly load is distributed across experts
    p_norm = expert_total_weights / (np.sum(expert_total_weights) + 1e-12)
    p_positive = p_norm[p_norm > 1e-12]
    routing_entropy = -float(np.sum(p_positive * np.log(p_positive)))

    rare_experts = [int(e) for e in range(n_experts) if expert_frequencies[e] < rare_threshold]
    frequent_experts = [int(e) for e in range(n_experts) if expert_frequencies[e] >= rare_threshold]

    return RouterGatingResult(
        gating_weights=gating_matrix,
        active_expert_indices=topk_indices,
        expert_frequencies=expert_frequencies,
        expert_total_weights=expert_total_weights,
        routing_entropy=routing_entropy,
        rare_experts=rare_experts,
        frequent_experts=frequent_experts,
    )


def compute_expert_hessians(
    activations: np.ndarray,
    gating_result: RouterGatingResult,
    regularize_lambda: float = 1e-4,
    shrinkage_lambda: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute activation diagonal Hessians for all experts simultaneously.

    For expert e:
        h_{e, j} = sum_{i=1}^N g_e(x_i)^2 X_{i, j}^2 = (G_sq^T S_act)_{e, j}
    With Bayesian shrinkage towards the global activation prior:
        h_e^{reg} = (h_e + lambda * h_global) / (M_e + lambda)
    where M_e = sum_i g_e(x_i) is the total routing mass.

    Returns:
        (H_diag_matrix (E, d_in), routing_masses (E,))
    """
    n_tokens, d_in = activations.shape
    gating = gating_result.gating_weights  # (N, E)
    n_experts = gating.shape[1]

    # Global activation second moment
    s_act = activations ** 2  # (N, d_in)
    h_global = np.mean(s_act, axis=0)  # (d_in,)

    # Expert-specific raw second moments: (E, d_in) = (E, N) @ (N, d_in)
    g_sq = gating ** 2
    h_raw = g_sq.T @ s_act  # (E, d_in)

    # Routing mass per expert: M_e
    masses = gating_result.expert_total_weights  # (E,)

    # Regularized expert diagonal: shrinkage handles rare/unfired experts safely
    denom = masses[:, None] + shrinkage_lambda
    h_reg = (h_raw + shrinkage_lambda * h_global[None, :]) / denom + regularize_lambda

    return h_reg, masses


# ==============================================================================
# Optimal Weighted Trit Optimization Kernels
# ==============================================================================

def weighted_optimal_scale(
    weights: np.ndarray,
    threshold: float,
    h_diag: Optional[np.ndarray] = None,
) -> float:
    """Compute optimal scale alpha* minimizing weighted MSE for a fixed threshold Delta.

    alpha*(Delta; h) = (sum_{|w_i| > Delta} h_i |w_i|) / (sum_{|w_i| > Delta} h_i)
    """
    abs_w = np.abs(weights)
    active = abs_w > threshold
    if not np.any(active):
        return 0.0

    if h_diag is None:
        return float(np.mean(abs_w[active]))

    h_active = h_diag[active]
    sum_h = float(np.sum(h_active))
    if sum_h < 1e-12:
        return float(np.mean(abs_w[active]))

    return float(np.sum(h_active * abs_w[active]) / sum_h)


def weighted_reconstruction_error(
    weights: np.ndarray,
    trits: np.ndarray,
    scale: float,
    h_diag: Optional[np.ndarray] = None,
) -> float:
    """Compute normalized weighted reconstruction error Tr((W - W_hat) H (W - W_hat)^T) / Tr(W H W^T)."""
    recon = scale * trits.astype(np.float32)
    diff_sq = (weights - recon) ** 2
    if h_diag is None:
        num = float(np.sum(diff_sq))
        den = float(np.sum(weights ** 2)) + 1e-12
    else:
        num = float(np.sum(h_diag * diff_sq))
        den = float(np.sum(h_diag * (weights ** 2))) + 1e-12
    return num / den


def weighted_quantize_block(
    weights: np.ndarray,
    h_diag: Optional[np.ndarray] = None,
    block_size: int = BLOCK_SIZE,
    search_threshold: bool = True,
    bounds_factor: Tuple[float, float] = (0.5, 0.9),
    grid_points: int = 33,
    target_sparsity: Optional[Tuple[float, float]] = (35.0, 50.0),
    uniform_threshold: float = 1e-4,
) -> CalibratedTritResult:
    """Vectorized block-wise trit quantization with diagonal Hessian weighting.

    Minimizes:
        sum_b sum_{i in block_b} h_{b, i} (w_{b, i} - alpha_b * trit(w_{b, i}, Delta_b))^2
    subject to target zero-trit sparsity in [35%, 50%].

    If h_diag is None or uniform (rel_std < uniform_threshold), falls back to
    unweighted block MSE.
    """
    orig_shape = weights.shape
    w_flat = np.asarray(weights, dtype=np.float32).reshape(-1)
    total_elems = w_flat.size

    if total_elems % block_size != 0:
        raise ValueError(
            f"Weight size ({total_elems}) must be a multiple of block_size ({block_size})"
        )

    num_blocks = total_elems // block_size
    blocks = w_flat.reshape(num_blocks, block_size)
    abs_blocks = np.abs(blocks)

    # Check for uniform or absent Hessian -> fallback to standard unweighted MSE
    is_fallback = False
    if h_diag is None:
        is_fallback = True
        h_blocks = np.ones((num_blocks, block_size), dtype=np.float32)
    else:
        h_flat = np.asarray(h_diag, dtype=np.float32).reshape(-1)
        if h_flat.size == block_size:
            # Broadcast identical block importance across all blocks
            h_blocks = np.tile(h_flat, (num_blocks, 1))
        elif h_flat.size == total_elems:
            h_blocks = h_flat.reshape(num_blocks, block_size)
        elif total_elems % h_flat.size == 0:
            # Tile column activation importance across rows of 2D weight matrix (d_out, d_in)
            reps = total_elems // h_flat.size
            h_blocks = np.tile(h_flat, reps).reshape(num_blocks, block_size)
        else:
            raise ValueError(
                f"h_diag size ({h_flat.size}) must match block_size ({block_size}), "
                f"total weights ({total_elems}), or divide total weights."
            )

        # Check relative variation: if uniform, trigger fallback
        h_std = float(np.std(h_blocks))
        h_mean = float(np.mean(h_blocks))
        if h_mean < 1e-12 or (h_std / h_mean) < uniform_threshold:
            is_fallback = True
            h_blocks = np.ones((num_blocks, block_size), dtype=np.float32)

    # Normalize h per block so mean(h_b) = 1.0 (numerical invariance)
    h_means = np.mean(h_blocks, axis=1, keepdims=True) + 1e-12
    h_norm = h_blocks / h_means

    mean_abs = np.mean(abs_blocks, axis=1, keepdims=True)  # (num_blocks, 1)

    if not search_threshold:
        # Analytical optimal threshold: Delta* = (2/3) * E[|W|]
        deltas = (2.0 / 3.0) * mean_abs
    else:
        # Vectorized dense grid search over candidates
        factors = np.linspace(bounds_factor[0], bounds_factor[1], grid_points)  # (K,)
        cand_deltas = mean_abs * factors.reshape(1, -1)                         # (B, K)

        # Broadcast evaluation:
        # abs_exp: (B, block_size, 1), cand_deltas: (B, 1, K)
        abs_exp = abs_blocks[:, :, None]
        active = abs_exp > cand_deltas[:, None, :]                             # (B, block_size, K)

        h_exp = h_norm[:, :, None]                                             # (B, block_size, 1)
        h_active = h_exp * active                                              # (B, block_size, K)

        sum_h_w = np.sum(h_active * abs_exp, axis=1)                          # (B, K)
        sum_h = np.sum(h_active, axis=1)                                       # (B, K)

        # Objective J(Delta) = (sum_{active} h_i |w_i|)^2 / (sum_{active} h_i)
        n_active = np.sum(active, axis=1)                                      # (B, K)
        denom_safe = np.maximum(sum_h, 1e-12)
        j_obj = np.where(sum_h > 0, (sum_h_w ** 2) / denom_safe, 0.0)

        # Sparsity constraint filter
        if target_sparsity is not None:
            low_sp, high_sp = target_sparsity
            zero_ratio = (block_size - n_active) / block_size * 100.0          # (B, K)
            valid_sp = (zero_ratio >= low_sp) & (zero_ratio <= high_sp)
            j_obj_constrained = np.where(valid_sp, j_obj, -1.0)
            has_valid = np.any(valid_sp, axis=1)
            best_indices = np.where(
                has_valid,
                np.argmax(j_obj_constrained, axis=1),
                np.argmax(j_obj, axis=1),
            )
        else:
            best_indices = np.argmax(j_obj, axis=1)

        deltas = np.take_along_axis(cand_deltas, best_indices[:, None], axis=1)

    # Compute optimal trits
    trits_blocks = np.zeros_like(blocks, dtype=np.int8)
    trits_blocks[blocks > deltas] = 1
    trits_blocks[blocks < -deltas] = -1

    # Optimal weighted scale per block: alpha* = (sum h_i |w_i|) / (sum h_i)
    active_mask = (trits_blocks != 0)
    h_act = h_norm * active_mask
    sum_h_act = np.sum(h_act, axis=1)
    sum_hw_act = np.sum(h_act * abs_blocks, axis=1)

    scales = np.zeros(num_blocks, dtype=np.float32)
    valid_blocks = sum_h_act > 0
    scales[valid_blocks] = sum_hw_act[valid_blocks] / sum_h_act[valid_blocks]

    # Convert scales to IEEE 754 half-precision (FP16) as required by TQ1_0
    scales_fp16 = scales.astype(np.float16)

    # Compute reconstruction metrics
    recon_blocks = scales_fp16.astype(np.float32)[:, None] * trits_blocks.astype(np.float32)
    diff_blocks = blocks - recon_blocks
    unweighted_mse = float(np.mean(diff_blocks ** 2))

    w_denom = np.sum(h_norm * (blocks ** 2)) + 1e-12
    weighted_err = float(np.sum(h_norm * (diff_blocks ** 2)) / w_denom)

    zero_count = int(np.count_nonzero(trits_blocks == 0))
    sparsity = float((zero_count / total_elems) * 100.0)

    trits_out = trits_blocks.reshape(orig_shape)
    deltas_out = deltas.reshape(num_blocks)

    return CalibratedTritResult(
        trits=trits_out,
        scales=scales_fp16,
        thresholds=deltas_out,
        weighted_error=weighted_err,
        unweighted_mse=unweighted_mse,
        sparsity=sparsity,
        expert_weight=1.0,
        is_fallback=is_fallback,
    )


def weighted_quantize_full_hessian(
    weights: np.ndarray,
    H: np.ndarray,
    bounds_factor: Tuple[float, float] = (0.5, 0.9),
    grid_points: int = 33,
    target_sparsity: Optional[Tuple[float, float]] = (35.0, 50.0),
) -> CalibratedTritResult:
    """Exact full-Hessian trit quantization for a 2D weight matrix W in R^{d_out x d_in}.

    Minimizes:
        Tr( (W - alpha * trit(W, Delta)) H (W - alpha * trit(W, Delta))^T )
    where:
        alpha*(Delta) = Tr(W H T^T) / Tr(T H T^T)
    """
    d_out, d_in = weights.shape
    if H.shape != (d_in, d_in):
        raise ValueError(f"Hessian shape {H.shape} does not match weight dimension (., {d_in})")

    abs_w = np.abs(weights)
    mean_abs = float(np.mean(abs_w))
    candidates = np.linspace(bounds_factor[0] * mean_abs, bounds_factor[1] * mean_abs, grid_points)

    tr_whw = float(np.trace(weights @ H @ weights.T))
    best_delta = candidates[0]
    best_alpha = 0.0
    best_loss = float("inf")
    best_trits = np.zeros_like(weights, dtype=np.int8)

    for delta in candidates:
        T = np.zeros_like(weights, dtype=np.int8)
        T[weights > delta] = 1
        T[weights < -delta] = -1

        t_fp = T.astype(np.float64)
        w_fp = weights.astype(np.float64)

        # Tr(W H T^T) and Tr(T H T^T)
        num = float(np.trace(w_fp @ H @ t_fp.T))
        den = float(np.trace(t_fp @ H @ t_fp.T))

        if den < 1e-12:
            alpha = 0.0
            loss = tr_whw
        else:
            alpha = num / den
            loss = tr_whw - (num ** 2) / den

        # Check sparsity constraint
        zero_pct = float(np.mean(T == 0) * 100.0)
        if target_sparsity is not None:
            if zero_pct < target_sparsity[0] or zero_pct > target_sparsity[1]:
                continue

        if loss < best_loss:
            best_loss = loss
            best_delta = delta
            best_alpha = alpha
            best_trits = T

    # If constrained search found no valid candidate, evaluate unconstrained
    if math.isinf(best_loss):
        for delta in candidates:
            T = np.zeros_like(weights, dtype=np.int8)
            T[weights > delta] = 1
            T[weights < -delta] = -1
            num = float(np.trace(weights.astype(np.float64) @ H @ T.astype(np.float64).T))
            den = float(np.trace(T.astype(np.float64) @ H @ T.astype(np.float64).T))
            alpha = num / den if den > 1e-12 else 0.0
            loss = tr_whw - (num ** 2) / den if den > 1e-12 else tr_whw
            if loss < best_loss:
                best_loss = loss
                best_delta = delta
                best_alpha = alpha
                best_trits = T

    alpha_fp16 = np.float16(best_alpha)
    recon = float(alpha_fp16) * best_trits.astype(np.float32)
    unweighted_mse = float(np.mean((weights - recon) ** 2))
    weighted_err = best_loss / (tr_whw + 1e-12)
    sparsity = float(np.mean(best_trits == 0) * 100.0)

    return CalibratedTritResult(
        trits=best_trits,
        scales=np.array([alpha_fp16], dtype=np.float16),
        thresholds=np.array([best_delta], dtype=np.float32),
        weighted_error=weighted_err,
        unweighted_mse=unweighted_mse,
        sparsity=sparsity,
        expert_weight=1.0,
        is_fallback=False,
    )


# ==============================================================================
# ActivationCalibrator Main Orchestrator
# ==============================================================================

class ActivationCalibrator:
    """End-to-end calibrator for activation-aware and MoE-router-weighted quantization.

    Supports:
    - Predefined architectures: Qwen 3.8 Flash Next, Qwen 3.6 35B A3B, Ornith 1.5 35B A3B.
    - Diagonal and full Hessian accumulation across calibration batches.
    - Simultaneous MoE router gating computation across top-8 active experts.
    - Rare-expert shrinkage and fallback to fast block-wise MSE.
    - Synthetic multilingual and code token generation.
    - Bit-level packing into Strata TQ1_0 (GGML Type 34).
    """

    def __init__(
        self,
        profile: Optional[Union[str, MoEModelProfile]] = None,
        config: Optional[CalibrationConfig] = None,
    ):
        if profile is None:
            self.profile = MODEL_PROFILES["qwen3.6-35b-a3b"]
        elif isinstance(profile, MoEModelProfile):
            self.profile = profile
        elif isinstance(profile, str):
            self.profile = get_model_profile(profile)
        else:
            raise TypeError(f"Invalid profile type: {type(profile)}")

        self.config = config or CalibrationConfig(model_name=self.profile.name)
        self.generator = SyntheticCalibrationGenerator()
        self._cached_hessian: Optional[ActivationHessian] = None

    def generate_synthetic_dataset(
        self,
        num_tokens: int = 512,
        include_code: bool = True,
        include_multilingual: bool = True,
    ) -> SyntheticDataset:
        """Generate synthetic calibration dataset with code syntax and multilingual text."""
        return self.generator.generate_complete_dataset(
            num_tokens=num_tokens,
            dim=self.profile.hidden_size,
            num_experts=self.profile.total_experts,
        )

    def calibrate_activations(
        self,
        activations: Union[np.ndarray, "torch.Tensor"],
        compute_full: Optional[bool] = None,
    ) -> ActivationHessian:
        """Estimate activation Hessian from a batch of token activation vectors.

        Args:
            activations: (N, d_in) NumPy array or PyTorch Tensor.
            compute_full: If None, defaults to config.use_diagonal_hessian (inverted).

        Returns:
            ActivationHessian containing diagonal, full matrix, and outlier channel statistics.
        """
        if HAS_TORCH and isinstance(activations, torch.Tensor):
            act_np = activations.detach().cpu().to(torch.float32).numpy()
        else:
            act_np = np.asarray(activations, dtype=np.float32)

        do_full = compute_full if compute_full is not None else (not self.config.use_diagonal_hessian)

        self._cached_hessian = compute_activation_hessian(
            act_np,
            compute_full=do_full,
            regularize_lambda=self.config.regularization_lambda,
        )
        return self._cached_hessian

    def update_activations(
        self,
        batch_activations: Union[np.ndarray, "torch.Tensor"],
        compute_full: Optional[bool] = None,
    ) -> ActivationHessian:
        """Streaming online update of activation Hessian across minibatches."""
        if HAS_TORCH and isinstance(batch_activations, torch.Tensor):
            batch_np = batch_activations.detach().cpu().to(torch.float32).numpy()
        else:
            batch_np = np.asarray(batch_activations, dtype=np.float32)

        do_full = compute_full if compute_full is not None else (not self.config.use_diagonal_hessian)

        self._cached_hessian = update_activation_hessian(
            self._cached_hessian,
            batch_np,
            compute_full=do_full,
            regularize_lambda=self.config.regularization_lambda,
        )
        return self._cached_hessian

    def compute_moe_routing(
        self,
        activations: Union[np.ndarray, "torch.Tensor"],
        router_weights: Union[np.ndarray, "torch.Tensor"],
        top_k: Optional[int] = None,
        bias: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    ) -> RouterGatingResult:
        """Compute router gating across experts for the target model."""
        if HAS_TORCH and isinstance(activations, torch.Tensor):
            act_np = activations.detach().cpu().to(torch.float32).numpy()
        else:
            act_np = np.asarray(activations, dtype=np.float32)

        if HAS_TORCH and isinstance(router_weights, torch.Tensor):
            rw_np = router_weights.detach().cpu().to(torch.float32).numpy()
        else:
            rw_np = np.asarray(router_weights, dtype=np.float32)

        bias_np = None
        if bias is not None:
            if HAS_TORCH and isinstance(bias, torch.Tensor):
                bias_np = bias.detach().cpu().to(torch.float32).numpy()
            else:
                bias_np = np.asarray(bias, dtype=np.float32)

        k = top_k or self.profile.routed_experts

        return compute_router_gates(
            router_weights=rw_np,
            activations=act_np,
            top_k=k,
            bias=bias_np,
            rare_threshold=self.config.rare_expert_threshold,
        )

    def calibrate_layer(
        self,
        weights: Union[np.ndarray, "torch.Tensor"],
        activations: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
        hessian: Optional[ActivationHessian] = None,
    ) -> CalibratedTritResult:
        """Calibrate a single weight tensor using activation Hessian weighting.

        If activations and hessian are None, automatically falls back to unweighted block MSE.
        """
        is_torch = HAS_TORCH and isinstance(weights, torch.Tensor)
        if is_torch:
            orig_device = weights.device
            orig_shape = weights.shape
            w_np = weights.detach().cpu().to(torch.float32).numpy()
        else:
            orig_shape = weights.shape
            w_np = np.asarray(weights, dtype=np.float32)

        h_diag = None
        if hessian is not None:
            h_diag = hessian.diag
        elif activations is not None:
            h_est = self.calibrate_activations(activations)
            h_diag = h_est.diag
        elif self._cached_hessian is not None:
            h_diag = self._cached_hessian.diag

        result = weighted_quantize_block(
            weights=w_np,
            h_diag=h_diag,
            block_size=self.config.block_size,
            search_threshold=True,
            bounds_factor=self.config.bounds_factor,
            grid_points=self.config.grid_points,
            target_sparsity=self.config.target_sparsity,
            uniform_threshold=self.config.uniform_threshold,
        )

        if is_torch:
            trits_torch = torch.from_numpy(result.trits).to(device=orig_device)
            scales_torch = torch.from_numpy(result.scales).to(device=orig_device)
            return CalibratedTritResult(
                trits=trits_torch,
                scales=scales_torch,
                thresholds=result.thresholds,
                weighted_error=result.weighted_error,
                unweighted_mse=result.unweighted_mse,
                sparsity=result.sparsity,
                expert_weight=result.expert_weight,
                is_fallback=result.is_fallback,
            )

        return result

    def route_and_calibrate_moe(
        self,
        router_weights: Union[np.ndarray, "torch.Tensor"],
        expert_weights: Dict[int, Union[np.ndarray, "torch.Tensor"]],
        activations: Union[np.ndarray, "torch.Tensor"],
        top_k: Optional[int] = None,
        router_bias: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    ) -> Dict[int, CalibratedTritResult]:
        """Complete MoE routing and expert-specific calibration.

        Computes:
        1. Router gating matrix G for all calibration tokens.
        2. Simultaneous expert diagonal Hessians H_diag (E, d_in) via G_sq^T @ X^2.
        3. Expert calibration with shrinkage for rare experts and fallback for unfired experts.

        Returns:
            Dictionary mapping expert_id -> CalibratedTritResult.
        """
        if HAS_TORCH and isinstance(activations, torch.Tensor):
            act_np = activations.detach().cpu().to(torch.float32).numpy()
        else:
            act_np = np.asarray(activations, dtype=np.float32)

        gating_res = self.compute_moe_routing(
            activations=act_np,
            router_weights=router_weights,
            top_k=top_k,
            bias=router_bias,
        )

        # Compute diagonal Hessians for all experts
        h_matrix, masses = compute_expert_hessians(
            activations=act_np,
            gating_result=gating_res,
            regularize_lambda=self.config.regularization_lambda,
        )

        calibrated_experts: Dict[int, CalibratedTritResult] = {}
        total_tokens = act_np.shape[0]

        for e_idx, w_e in expert_weights.items():
            if e_idx < 0 or e_idx >= self.profile.total_experts:
                raise ValueError(
                    f"Expert index {e_idx} out of range [0, {self.profile.total_experts})"
                )

            is_torch = HAS_TORCH and isinstance(w_e, torch.Tensor)
            if is_torch:
                w_np = w_e.detach().cpu().to(torch.float32).numpy()
            else:
                w_np = np.asarray(w_e, dtype=np.float32)

            mass_e = float(masses[e_idx])
            h_e = h_matrix[e_idx]

            # If expert fired virtually not at all, pass None to trigger explicit fallback
            if mass_e < self.config.rare_expert_threshold * total_tokens:
                # Rare / unfired expert: fallback to unweighted block MSE
                calib = weighted_quantize_block(
                    weights=w_np,
                    h_diag=None,  # triggers fast fallback
                    block_size=self.config.block_size,
                    search_threshold=True,
                    bounds_factor=self.config.bounds_factor,
                    grid_points=self.config.grid_points,
                    target_sparsity=self.config.target_sparsity,
                )
                calib = calib._replace(expert_weight=mass_e, is_fallback=True)
            else:
                # Active expert: activation-weighted calibration
                calib = weighted_quantize_block(
                    weights=w_np,
                    h_diag=h_e,
                    block_size=self.config.block_size,
                    search_threshold=True,
                    bounds_factor=self.config.bounds_factor,
                    grid_points=self.config.grid_points,
                    target_sparsity=self.config.target_sparsity,
                )
                calib = calib._replace(expert_weight=mass_e, is_fallback=False)

            if is_torch:
                orig_dev = w_e.device
                calibrated_experts[e_idx] = CalibratedTritResult(
                    trits=torch.from_numpy(calib.trits).to(device=orig_dev),
                    scales=torch.from_numpy(calib.scales).to(device=orig_dev),
                    thresholds=calib.thresholds,
                    weighted_error=calib.weighted_error,
                    unweighted_mse=calib.unweighted_mse,
                    sparsity=calib.sparsity,
                    expert_weight=calib.expert_weight,
                    is_fallback=calib.is_fallback,
                )
            else:
                calibrated_experts[e_idx] = calib

        return calibrated_experts

    def pack_to_tq1_0(self, result: CalibratedTritResult) -> bytes:
        """Pack calibrated trits and FP16 scales into Strata TQ1_0 binary format."""
        trits = result.trits
        scales = result.scales

        if HAS_TORCH and isinstance(trits, torch.Tensor):
            trits = trits.detach().cpu().numpy()
        if HAS_TORCH and isinstance(scales, torch.Tensor):
            scales = scales.detach().cpu().numpy()

        return pack_tq1_0(trits, scales)

    def unpack_from_tq1_0(
        self,
        packed_bytes: bytes,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack Strata TQ1_0 binary stream into trits and FP16 scales."""
        return unpack_tq1_0(packed_bytes, shape=shape)

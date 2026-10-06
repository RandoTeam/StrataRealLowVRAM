"""Optimal 1.58-bit Ternary Quantization & Strata TQ1_0 (GGML Type 34) Packer.

Mathematical Foundation:
========================
1. Optimal Trit Thresholding Problem:
   Given a continuous/discrete weight tensor W in R^N, find scale alpha in R^+
   and threshold Delta in R^+ minimizing mean squared error (MSE):
       min_{alpha, Delta} L(alpha, Delta) = ||W - alpha * trit(W, Delta)||_F^2
   where:
       trit(w, Delta) = +1  if w > Delta
                        -1  if w < -Delta
                         0  if |w| <= Delta

2. Derivation of Optimal Scale alpha*(Delta):
   Partition indices into active I_active = {i : |w_i| > Delta} and zero I_0 = {i : |w_i| <= Delta}.
   L(alpha, Delta) = sum_{i in I_0} w_i^2 + sum_{i in I_active} (|w_i| - alpha)^2
   Differentiating with respect to alpha:
       dL / d(alpha) = -2 * sum_{i in I_active} (|w_i| - alpha) = 0
       ==> alpha*(Delta) = (1 / N_active) * sum_{i in I_active} |w_i|
   where N_active = |I_active|. If N_active == 0, alpha* = 0.

3. Objective Function in Delta:
   Substituting alpha*(Delta) into L(alpha, Delta):
       L(alpha*, Delta) = ||W||_F^2 - N_active * (alpha*(Delta))^2
   Minimizing reconstruction MSE is therefore mathematically equivalent to maximizing:
       J(Delta) = N_active * (alpha*(Delta))^2 = (sum_{|w_i| > Delta} |w_i|)^2 / N_active

4. Critical Point Condition & Exact Boundary Theorem:
   For continuous symmetric distribution f(w) with X = |W| and density g(x) = 2 f(x):
       E(Delta) = E[X^2] - p(Delta) * (alpha(Delta))^2
       dE / d(Delta) = alpha(Delta) * g(Delta) * [2 * Delta - alpha(Delta)] = 0
   Since alpha > 0 and g(Delta) > 0, the exact necessary condition for optimality is:
       Delta* = 0.5 * alpha*(Delta*)
   This confirms the half-distance midpoint boundary between quantizing to 0 and quantizing to alpha.

5. Optimal Analytical Threshold Delta* ~= (2/3) E[|W|]:
   - Gaussian Distribution W ~ N(0, sigma^2):
     E[|W|] = sigma * sqrt(2 / pi) ~= 0.7979 sigma.
     Numerical minimization yields Delta* ~= 0.6120 sigma ~= 0.7670 E[|W|].
     Setting Delta = (2/3) E[|W|] ~= 0.5319 sigma yields zero sparsity of 40.5%
     (ideal for 35-50% target) and MSE within 1.7% of the global minimum.
   - Laplace Distribution W ~ Laplace(0, b):
     E[|W|] = b. Memoryless property gives alpha(Delta) = Delta + b.
     Setting Delta = (2/3) E[|W|] yields zero sparsity of 1 - exp(-2/3) ~= 48.7%.
   - Empirical Tensors:
     Evaluating Delta in [0.5 * E[|W|], 0.9 * E[|W|]] via golden-section search
     or vectorized grid search captures the exact empirical minimum while maintaining
     zero sparsity strictly in the target 35-50% range.

6. Strata Native TQ1_0 (GGML Type 34) Bit-Packing Specification:
   - Block Size: 256 weights.
   - Block Memory Layout: 54 bytes (1.6875 bits per weight):
     * qs: 48 bytes (240 trits packed 5-per-byte in Radix-3)
       - Bytes 0..31: 160 trits (5 trits per byte across 32 columns)
       - Bytes 32..47: 80 trits (5 trits per byte across 16 columns)
     * qh: 4 bytes (16 trits packed 4-per-byte in Radix-3)
     * d: 2 bytes (FP16 scale stored as little-endian uint16)
   - Radix-3 Packing:
     Trits in {-1, 0, +1} are biased by +1 into {0, 1, 2}.
     For 5 trits (t0, t1, t2, t3, t4):
       q = t0*81 + t1*27 + t2*9 + t3*3 + t4  in [0, 242]
       byte = floor((q * 256 + 242) / 243)
     For 4 trits in qh (t0, t1, t2, t3):
       q = (t0*27 + t1*9 + t2*3 + t3) * 3    in [0, 240]
       byte = floor((q * 256 + 242) / 243)
   - Strata CUDA Unpack Logic (without division):
       POW3_PACKED = 0xF3511B090301ULL
       p3 = (POW3_PACKED >> (t * 8)) & 0xFF  (= 3^t)
       q = (qbyte * p3) & 0xFF
       trit = ((q * 3) >> 8) - 1
"""

from __future__ import annotations

import math
import struct
from typing import Any, NamedTuple, Optional, Tuple, Union

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==============================================================================
# Constants & Specifications
# ==============================================================================

BLOCK_SIZE: int = 256
TQ1_0_BLOCK_BYTES: int = 54
QS_BYTES: int = 48
QH_BYTES: int = 4
SCALE_BYTES: int = 2

# Strata CUDA constant for division-free trit unpacking
POW3_PACKED: int = 0xF3511B090301

# Radix-3 multipliers
RADIX_5_WEIGHTS = np.array([81, 27, 9, 3, 1], dtype=np.uint16)
RADIX_4_SHIFTED = np.array([81, 27, 9, 3], dtype=np.uint16)

# Unpack powers of 3
POW3_5 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)
POW3_4 = np.array([1, 3, 9, 27], dtype=np.uint16)


# ==============================================================================
# TritQuantResult Container
# ==============================================================================

class TritQuantResult(NamedTuple):
    """Result of ternary quantization."""
    trits: Any        # int8 array/tensor in {-1, 0, +1}
    scales: Any       # float16 array/tensor of scale per 256-element block
    sparsity: float   # Percentage of zero trits (target: 35.0 - 50.0%)
    mse: float        # Mean squared reconstruction error


# ==============================================================================
# Low-Level Radix-3 Scalar Kernels
# ==============================================================================

def tq1_0_trit(qbyte: int, t: int) -> int:
    """Extract trit at index t (0..4) from packed byte using Strata CUDA logic.

    Matches __device__ int tq1_0_trit(uint32_t qbyte, uint32_t t) in iq_kernels.cu.
    Returns:
        int in {-1, 0, +1}
    """
    p3 = (POW3_PACKED >> (t * 8)) & 0xFF
    q = (int(qbyte) * p3) & 0xFF
    return ((q * 3) >> 8) - 1


def pack_5_trits(trits: Union[Tuple[int, ...], list[int], np.ndarray]) -> int:
    """Pack 5 trits in {-1, 0, +1} into a single byte using Radix-3.

    Args:
        trits: 5 integers in {-1, 0, +1}.

    Returns:
        int in [0, 255].
    """
    q = 0
    for t in trits:
        xi = int(t) + 1  # Map {-1, 0, 1} -> {0, 1, 2}
        q = q * 3 + xi
    # Fixed-point ceiling division mapping [0, 242] into [0, 255]
    return (q * 256 + 242) // 243


def unpack_5_trits(qbyte: int) -> Tuple[int, int, int, int, int]:
    """Unpack 5 trits from a single byte using Strata CUDA division-free logic.

    Returns:
        5-tuple of ints in {-1, 0, +1}.
    """
    return tuple(tq1_0_trit(qbyte, t) for t in range(5))


def pack_4_trits_qh(trits: Union[Tuple[int, ...], list[int], np.ndarray]) -> int:
    """Pack 4 trits in {-1, 0, +1} into a high byte (qh) in Radix-3.

    Args:
        trits: 4 integers in {-1, 0, +1}.

    Returns:
        int in [0, 255].
    """
    q = 0
    for t in trits:
        xi = int(t) + 1
        q = q * 3 + xi
    q *= 3  # Shift to most significant trit
    return (q * 256 + 242) // 243


def unpack_4_trits_qh(qbyte: int) -> Tuple[int, int, int, int]:
    """Unpack 4 trits from a high byte (qh) using Strata CUDA logic.

    Returns:
        4-tuple of ints in {-1, 0, +1}.
    """
    return tuple(tq1_0_trit(qbyte, t) for t in range(4))


# ==============================================================================
# Threshold Optimization & Scale Derivation
# ==============================================================================

def optimal_scale(weights: np.ndarray, threshold: float) -> float:
    """Compute optimal scale alpha* minimizing MSE for a fixed threshold Delta.

    alpha* = (sum_{|w_i| > Delta} |w_i|) / N_active
    """
    abs_w = np.abs(weights)
    active = abs_w > threshold
    n_active = np.count_nonzero(active)
    if n_active == 0:
        return 0.0
    return float(np.sum(abs_w[active]) / n_active)


def reconstruction_mse(
    weights: np.ndarray,
    threshold: float,
    scale: Optional[float] = None,
) -> float:
    """Compute reconstruction MSE ||W - alpha * trit(W, Delta)||^2 / N."""
    if scale is None:
        scale = optimal_scale(weights, threshold)
    trits = np.zeros_like(weights, dtype=np.int8)
    trits[weights > threshold] = 1
    trits[weights < -threshold] = -1
    recon = scale * trits
    return float(np.mean((weights - recon) ** 2))


def golden_section_search_threshold(
    weights_block: np.ndarray,
    bounds_factor: Tuple[float, float] = (0.5, 0.9),
    tol: float = 1e-4,
    max_iter: int = 25,
) -> Tuple[float, float, float]:
    """Find optimal threshold Delta minimizing MSE via 1D Golden Section search.

    Args:
        weights_block: 1D array of weights (typically 256 elements).
        bounds_factor: Search interval [low * E[|W|], high * E[|W|]].
        tol: Tolerance for convergence.
        max_iter: Maximum iterations.

    Returns:
        (optimal_threshold, optimal_scale, min_mse)
    """
    mean_abs = float(np.mean(np.abs(weights_block)))
    if mean_abs < 1e-12:
        return 0.0, 0.0, 0.0

    a = bounds_factor[0] * mean_abs
    b = bounds_factor[1] * mean_abs

    invphi = (math.sqrt(5.0) - 1.0) / 2.0  # 0.6180339887...
    invphi2 = (3.0 - math.sqrt(5.0)) / 2.0  # 0.3819660113...

    c = a + invphi2 * (b - a)
    d = a + invphi * (b - a)

    fc = reconstruction_mse(weights_block, c)
    fd = reconstruction_mse(weights_block, d)

    for _ in range(max_iter):
        if (b - a) < tol * mean_abs:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = a + invphi2 * (b - a)
            fc = reconstruction_mse(weights_block, c)
        else:
            a = c
            c = d
            fc = fd
            d = a + invphi * (b - a)
            fd = reconstruction_mse(weights_block, d)

    best_delta = (a + b) / 2.0
    best_scale = optimal_scale(weights_block, best_delta)
    best_mse = reconstruction_mse(weights_block, best_delta, best_scale)
    return best_delta, best_scale, best_mse


def grid_search_threshold(
    weights_block: np.ndarray,
    bounds_factor: Tuple[float, float] = (0.5, 0.9),
    num_points: int = 33,
) -> Tuple[float, float, float]:
    """Find optimal threshold Delta minimizing MSE via dense grid search.

    Args:
        weights_block: 1D array of weights.
        bounds_factor: Search interval [low * E[|W|], high * E[|W|]].
        num_points: Number of evaluation candidates.

    Returns:
        (optimal_threshold, optimal_scale, min_mse)
    """
    mean_abs = float(np.mean(np.abs(weights_block)))
    if mean_abs < 1e-12:
        return 0.0, 0.0, 0.0

    candidates = np.linspace(
        bounds_factor[0] * mean_abs,
        bounds_factor[1] * mean_abs,
        num_points,
    )
    best_delta = candidates[0]
    best_mse = float("inf")
    best_scale = 0.0

    abs_w = np.abs(weights_block)
    w_sq_sum = float(np.sum(weights_block ** 2))
    n = len(weights_block)

    for delta in candidates:
        active = abs_w > delta
        n_act = int(np.count_nonzero(active))
        if n_act == 0:
            mse = w_sq_sum / n
            scale = 0.0
        else:
            sum_act = float(np.sum(abs_w[active]))
            scale = sum_act / n_act
            # MSE = (||W||^2 - n_act * scale^2) / n
            mse = (w_sq_sum - n_act * (scale ** 2)) / n

        if mse < best_mse:
            best_mse = mse
            best_delta = delta
            best_scale = scale

    return float(best_delta), float(best_scale), float(best_mse)


# ==============================================================================
# Vectorized Quantization Engine
# ==============================================================================

def quantize_to_trits(
    tensor: Union[np.ndarray, "torch.Tensor"],
    block_size: int = BLOCK_SIZE,
    search_threshold: bool = True,
    search_method: str = "grid",
    bounds_factor: Tuple[float, float] = (0.5, 0.9),
    grid_points: int = 33,
    target_sparsity: Optional[Tuple[float, float]] = (35.0, 50.0),
) -> TritQuantResult:
    """Quantize tensor to 1.58-bit ternary representation {-1, 0, +1} per block.

    Args:
        tensor: Input tensor (NumPy ndarray or PyTorch Tensor).
        block_size: Number of elements per quantization block (default: 256).
        search_threshold: If True, search optimal Delta in [0.5, 0.9] * E[|W|].
                          If False, use analytical Delta* = (2/3) * E[|W|].
        search_method: 'grid' (fast vectorized) or 'golden_section'.
        bounds_factor: (low, high) multipliers of E[|W|].
        grid_points: Number of candidates for grid search.
        target_sparsity: Optional (low_pct, high_pct) zero-trit sparsity target.
                         Defaults to (35.0, 50.0) for high execution efficiency.

    Returns:
        TritQuantResult containing:
        - trits: int8 array/tensor matching input shape, elements in {-1, 0, +1}.
        - scales: float16 array/tensor with scale per block.
        - sparsity: percentage of zero trits (target: 35.0 - 50.0%).
        - mse: reconstruction mean squared error.
    """
    is_torch = HAS_TORCH and isinstance(tensor, torch.Tensor)
    if is_torch:
        orig_device = tensor.device
        orig_dtype = tensor.dtype
        orig_shape = tensor.shape
        w_np = tensor.detach().cpu().to(torch.float32).numpy()
    else:
        orig_shape = tensor.shape
        w_np = np.asarray(tensor, dtype=np.float32)

    total_elems = w_np.size
    if total_elems % block_size != 0:
        raise ValueError(
            f"Tensor size ({total_elems}) must be a multiple of block_size ({block_size})"
        )

    num_blocks = total_elems // block_size
    blocks = w_np.reshape(num_blocks, block_size)
    abs_blocks = np.abs(blocks)
    mean_abs = np.mean(abs_blocks, axis=1, keepdims=True)  # (num_blocks, 1)

    if not search_threshold:
        # Analytical optimal threshold: Delta* = (2/3) * E[|W|]
        deltas = (2.0 / 3.0) * mean_abs
    elif search_method == "grid":
        # Vectorized dense grid search over candidates
        factors = np.linspace(bounds_factor[0], bounds_factor[1], grid_points)  # (K,)
        cand_deltas = mean_abs * factors.reshape(1, -1)  # (num_blocks, K)

        # Broadcast compare: abs_blocks is (B, 256, 1), cand_deltas is (B, 1, K)
        abs_exp = abs_blocks[:, :, None]
        active = abs_exp > cand_deltas[:, None, :]  # (B, 256, K)

        sum_active = np.sum(abs_exp * active, axis=1)  # (B, K)
        n_active = np.sum(active, axis=1)  # (B, K)

        # Maximize J = (sum_active)^2 / N_active
        n_safe = np.maximum(n_active, 1)
        j_obj = np.where(n_active > 0, (sum_active ** 2) / n_safe, 0.0)

        if target_sparsity is not None:
            low_sp, high_sp = target_sparsity
            zero_ratio = (block_size - n_active) / block_size * 100.0  # (B, K)
            valid_sp = (zero_ratio >= low_sp) & (zero_ratio <= high_sp)
            j_obj_constrained = np.where(valid_sp, j_obj, -1.0)
            has_valid = np.any(valid_sp, axis=1)
            best_indices = np.where(
                has_valid,
                np.argmax(j_obj_constrained, axis=1),
                np.argmax(j_obj, axis=1),
            )
        else:
            best_indices = np.argmax(j_obj, axis=1)  # (B,)

        deltas = np.take_along_axis(cand_deltas, best_indices[:, None], axis=1)
    elif search_method == "golden_section":
        deltas = np.zeros((num_blocks, 1), dtype=np.float32)
        for b in range(num_blocks):
            d_opt, _, _ = golden_section_search_threshold(
                blocks[b], bounds_factor=bounds_factor
            )
            deltas[b, 0] = d_opt
    else:
        raise ValueError(f"Unknown search_method: {search_method}")

    # Generate trits
    trits_blocks = np.zeros_like(blocks, dtype=np.int8)
    trits_blocks[blocks > deltas] = 1
    trits_blocks[blocks < -deltas] = -1

    # Optimal scale per block: alpha* = (sum_{|w| > Delta} |w|) / N_active
    active_mask = (trits_blocks != 0)
    n_act = np.sum(active_mask, axis=1)
    sum_act = np.sum(abs_blocks * active_mask, axis=1)

    scales = np.zeros(num_blocks, dtype=np.float32)
    valid = n_act > 0
    scales[valid] = sum_act[valid] / n_act[valid]

    # Convert to standard FP16 as stored in TQ1_0 format
    scales_fp16 = scales.astype(np.float16)

    # Compute exact reconstruction MSE using FP16 dequantized scale
    recon_blocks = scales_fp16.astype(np.float32)[:, None] * trits_blocks.astype(np.float32)
    mse = float(np.mean((blocks - recon_blocks) ** 2))
    zero_count = int(np.count_nonzero(trits_blocks == 0))
    sparsity = float((zero_count / total_elems) * 100.0)

    trits_out = trits_blocks.reshape(orig_shape)

    if is_torch:
        trits_final = torch.from_numpy(trits_out).to(device=orig_device)
        scales_final = torch.from_numpy(scales_fp16).to(device=orig_device)
    else:
        trits_final = trits_out
        scales_final = scales_fp16

    return TritQuantResult(
        trits=trits_final,
        scales=scales_final,
        sparsity=sparsity,
        mse=mse,
    )


# ==============================================================================
# Bit-Packing to Strata TQ1_0 Specification (GGML Type 34)
# ==============================================================================

def pack_tq1_0_block(
    trits: np.ndarray,
    scale: Union[float, np.float16],
) -> bytes:
    """Pack a single 256-element block into Strata TQ1_0 54-byte format.

    Args:
        trits: 1D array of 256 int8 values in {-1, 0, +1}.
        scale: FP16 scale for this block.

    Returns:
        54 bytes representing the TQ1_0 block.
    """
    if len(trits) != BLOCK_SIZE:
        raise ValueError(f"Expected 256 trits, got {len(trits)}")

    buf = bytearray(TQ1_0_BLOCK_BYTES)

    # 1. First 32 bytes of qs: weights 0..159 (5 trits per byte across 32 columns)
    for m in range(32):
        q = 0
        for n in range(5):
            xi = int(trits[m + n * 32]) + 1
            q = q * 3 + xi
        buf[m] = (q * 256 + 242) // 243

    # 2. Next 16 bytes of qs: weights 160..239 (5 trits per byte across 16 columns)
    base = 160
    for m in range(16):
        q = 0
        for n in range(5):
            xi = int(trits[base + m + n * 16]) + 1
            q = q * 3 + xi
        buf[32 + m] = (q * 256 + 242) // 243

    # 3. 4 bytes of qh: weights 240..255 (4 trits per byte)
    base = 240
    for j in range(4):
        q = 0
        for m in range(4):
            xi = int(trits[base + j + m * 4]) + 1
            q = q * 3 + xi
        q *= 3  # Shift to most significant trit
        buf[48 + j] = (q * 256 + 242) // 243

    # 4. 2 bytes FP16 scale (little-endian uint16)
    scale_u16 = int(np.float16(scale).view(np.uint16))
    buf[52] = scale_u16 & 0xFF
    buf[53] = (scale_u16 >> 8) & 0xFF

    return bytes(buf)


def pack_tq1_0(
    trits: Union[np.ndarray, "torch.Tensor"],
    scales: Union[np.ndarray, "torch.Tensor"],
) -> bytes:
    """Pack arbitrary tensor of trits and scales into Strata TQ1_0 byte format.

    Args:
        trits: Array/tensor of int8 trits in {-1, 0, +1}, total elements a multiple of 256.
        scales: Array/tensor of float16 scales, length = total_elements // 256.

    Returns:
        bytes of length (num_blocks * 54).
    """
    if HAS_TORCH and isinstance(trits, torch.Tensor):
        trits_np = trits.detach().cpu().numpy().astype(np.int8)
    else:
        trits_np = np.asarray(trits, dtype=np.int8)

    if HAS_TORCH and isinstance(scales, torch.Tensor):
        scales_np = scales.detach().cpu().numpy().astype(np.float16)
    else:
        scales_np = np.asarray(scales, dtype=np.float16)

    total_elems = trits_np.size
    if total_elems % BLOCK_SIZE != 0:
        raise ValueError(
            f"Trits length ({total_elems}) must be a multiple of {BLOCK_SIZE}"
        )

    num_blocks = total_elems // BLOCK_SIZE
    trits_flat = trits_np.reshape(num_blocks, BLOCK_SIZE)
    scales_flat = scales_np.reshape(num_blocks)

    # Vectorized block packing
    out = np.zeros((num_blocks, TQ1_0_BLOCK_BYTES), dtype=np.uint8)

    # 1. First 32 bytes of qs: weights 0..159
    w1 = (trits_flat[:, :160].reshape(num_blocks, 5, 32).astype(np.uint16) + 1)
    q1 = np.einsum("i,bim->bm", RADIX_5_WEIGHTS, w1)
    out[:, :32] = ((q1 * 256 + 242) // 243).astype(np.uint8)

    # 2. Next 16 bytes of qs: weights 160..239
    w2 = (trits_flat[:, 160:240].reshape(num_blocks, 5, 16).astype(np.uint16) + 1)
    q2 = np.einsum("i,bim->bm", RADIX_5_WEIGHTS, w2)
    out[:, 32:48] = ((q2 * 256 + 242) // 243).astype(np.uint8)

    # 3. 4 bytes of qh: weights 240..255
    w3 = (trits_flat[:, 240:256].reshape(num_blocks, 4, 4).astype(np.uint16) + 1)
    q3 = np.einsum("i,bim->bm", RADIX_4_SHIFTED, w3)
    out[:, 48:52] = ((q3 * 256 + 242) // 243).astype(np.uint8)

    # 4. 2 bytes scale FP16: bytes 52..53
    scales_u16 = scales_flat.view(np.uint16)
    out[:, 52:54] = scales_u16[:, None].view(np.uint8).reshape(num_blocks, 2)

    return out.tobytes()


def unpack_tq1_0(
    data: Union[bytes, bytearray, np.ndarray],
    shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unpack Strata TQ1_0 binary data to int8 trits and float16 scales.

    Uses exact division-free CUDA unpack table logic (POW3_PACKED).

    Args:
        data: Raw TQ1_0 packed bytes.
        shape: Optional target shape for trits array.

    Returns:
        (trits, scales):
        - trits: int8 ndarray in {-1, 0, +1}.
        - scales: float16 ndarray.
    """
    if isinstance(data, (bytes, bytearray)):
        raw_bytes = bytes(data)
    elif isinstance(data, np.ndarray):
        raw_bytes = data.tobytes()
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    if len(raw_bytes) % TQ1_0_BLOCK_BYTES != 0:
        raise ValueError(
            f"Data length ({len(raw_bytes)}) must be a multiple of {TQ1_0_BLOCK_BYTES}"
        )

    num_blocks = len(raw_bytes) // TQ1_0_BLOCK_BYTES
    packed = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(num_blocks, TQ1_0_BLOCK_BYTES)

    trits = np.zeros((num_blocks, BLOCK_SIZE), dtype=np.int8)

    # 1. First 32 bytes of qs -> 160 trits
    qs1 = packed[:, :32].astype(np.uint16)[:, None, :]  # (B, 1, 32)
    q1 = (qs1 * POW3_5[None, :, None]) & 0xFF
    t1 = (((q1 * 3) >> 8) - 1).astype(np.int8)  # (B, 5, 32)
    trits[:, :160] = t1.reshape(num_blocks, 160)

    # 2. Next 16 bytes of qs -> 80 trits
    qs2 = packed[:, 32:48].astype(np.uint16)[:, None, :]  # (B, 1, 16)
    q2 = (qs2 * POW3_5[None, :, None]) & 0xFF
    t2 = (((q2 * 3) >> 8) - 1).astype(np.int8)  # (B, 5, 16)
    trits[:, 160:240] = t2.reshape(num_blocks, 80)

    # 3. 4 bytes of qh -> 16 trits
    qh = packed[:, 48:52].astype(np.uint16)[:, None, :]  # (B, 1, 4)
    q3 = (qh * POW3_4[None, :, None]) & 0xFF
    t3 = (((q3 * 3) >> 8) - 1).astype(np.int8)  # (B, 4, 4)
    trits[:, 240:256] = t3.reshape(num_blocks, 16)

    # 4. FP16 scale from bytes 52..53
    scale_bytes = packed[:, 52:54].tobytes()
    scales = np.frombuffer(scale_bytes, dtype=np.float16).copy()

    if shape is not None:
        trits = trits.reshape(shape)

    return trits, scales

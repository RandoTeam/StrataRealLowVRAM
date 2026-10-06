"""Fast Walsh-Hadamard Transform (FWHT) & Randomized Hadamard Transform (RHT).

Mathematical Foundation for Outlier Suppression in Ternary Quantization:
- Qwen 3.8 Flash Next (125B MoE, 512 experts)
- Qwen 3.6 35B A3B (~3B active params, 256 experts)
- Ornith 1.5 35B A3B (~3B active params, 256 experts)

Theoretical Background:
1. Walsh-Hadamard Transform:
   H_1 = [1]
   H_{2N} = [[H_N,  H_N],
             [H_N, -H_N]]
   Normalized orthonormal matrix H_norm = (1 / sqrt(N)) * H_N.
   Orthogonality: H_norm @ H_norm.T = I_N.
   Isometry (Parseval / norm preservation): ||H_norm * x||_2 = ||x||_2.

2. Fast Butterfly Algorithm (FWHT):
   Computes H * x in O(N log2 N) operations using log2(N) recursive
   butterfly stages instead of naive O(N^2) dense GEMV.

3. Outlier Suppression Bound:
   By Holder's inequality and properties of Hadamard entries (|H_{ij}| = 1):
     max_i |(H_norm * x)_i| <= (1 / sqrt(N)) * sum_j |x_j| = (1 / sqrt(N)) * ||x||_1
   For heavy-tailed outliers concentrated in a few channels (magnitude M):
     ||x||_1 ~= M + background
     max |H_norm * x| <= M / sqrt(N) + background_term
   For N = 512:  reduction factor is up to ~22.6x (sqrt(512)).
   For N = 2048: reduction factor is up to ~45.3x (sqrt(2048)).
   In practice on Cauchy / heavy-tailed activations: 3x - 20x outlier suppression.

4. Randomized Sign Flipping (RHT):
   H_RHT = H_norm @ diag(s), where s_i in {-1, +1} are i.i.d. Rademacher signs.
   Breaks worst-case coherent alignment with Hadamard basis vectors,
   guaranteeing uniform sub-Gaussian dispersion across all coordinates.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


def is_power_of_two(n: int) -> bool:
    """Return True if n is a positive power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def decompose_dimension(dim: int, block_size: Optional[int] = None) -> List[int]:
    """Decompose dimension into power-of-2 block sizes.

    Args:
        dim: Dimension to decompose.
        block_size: Optional fixed maximum or exact power-of-2 block size.

    Returns:
        List of power-of-2 block sizes summing exactly to dim.
    """
    if dim <= 0:
        return []
    if block_size is not None and not is_power_of_two(block_size):
        raise ValueError(f"block_size must be a power of 2, got {block_size}")

    if block_size is not None and dim % block_size == 0:
        return [block_size] * (dim // block_size)

    blocks: List[int] = []
    rem = dim
    while rem > 0:
        p = 1 << (rem.bit_length() - 1)
        if block_size is not None and p > block_size:
            p = block_size
        blocks.append(p)
        rem -= p
    return blocks


def generate_random_signs(
    dim: int,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    return_torch: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """Generate Rademacher random signs s in {-1, +1}^dim.

    Args:
        dim: Number of dimensions.
        seed: Random seed for reproducibility.
        device: PyTorch device (if return_torch=True).
        dtype: PyTorch dtype (defaults to torch.float32).
        return_torch: If True return torch.Tensor, else np.ndarray.

    Returns:
        1D tensor or ndarray of random +/- 1 values.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
        raw_signs = rng.choice([-1.0, 1.0], size=dim).astype(np.float32)
    else:
        raw_signs = np.random.choice([-1.0, 1.0], size=dim).astype(np.float32)

    if not return_torch:
        return raw_signs

    target_dtype = dtype if dtype is not None else torch.float32
    tensor_signs = torch.from_numpy(raw_signs).to(dtype=target_dtype)
    if device is not None:
        tensor_signs = tensor_signs.to(device=device)
    return tensor_signs


def _fwht_torch(
    x: torch.Tensor,
    normalize: bool = True,
    inplace: bool = False,
) -> torch.Tensor:
    """Core tensorized Fast Walsh-Hadamard Transform in PyTorch.

    Computes H * x in O(N log N) time along the last dimension.
    N must be a power of 2.
    """
    orig_shape = x.shape
    N = orig_shape[-1]
    if not is_power_of_two(N):
        raise ValueError(f"Last dimension must be a power of 2, got {N}")

    if N == 1:
        return x if inplace else x.clone()

    # Upcast half-precision to float32 for numerical stability across log2(N) butterfly stages
    needs_downcast = False
    orig_dtype = x.dtype
    if x.dtype in (torch.float16, torch.bfloat16):
        work_x = x.to(torch.float32)
        needs_downcast = True
    else:
        work_x = x if inplace else x.clone()

    if not work_x.is_contiguous():
        work_x = work_x.contiguous()

    # Fast butterfly stages: O(N log N) additions / subtractions
    m = 1
    while m < N:
        work_view = work_x.view(-1, N // (2 * m), 2, m)
        # Temporary view of second half
        v = work_view[:, :, 1, :].clone()
        work_view[:, :, 1, :] = work_view[:, :, 0, :] - v
        work_view[:, :, 0, :].add_(v)
        m *= 2

    work_x = work_x.view(orig_shape)

    if normalize:
        work_x.mul_(1.0 / math.sqrt(N))

    if needs_downcast:
        res = work_x.to(orig_dtype)
        if inplace:
            x.copy_(res)
            return x
        return res

    return work_x


def _fwht_numpy(
    x: np.ndarray,
    normalize: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """Core tensorized Fast Walsh-Hadamard Transform in NumPy.

    Computes H * x in O(N log N) time along the last dimension.
    N must be a power of 2.
    """
    orig_shape = x.shape
    N = orig_shape[-1]
    if not is_power_of_two(N):
        raise ValueError(f"Last dimension must be a power of 2, got {N}")

    if N == 1:
        return x if inplace else np.array(x, copy=True)

    y = x if inplace else np.array(x, copy=True)
    if not y.flags.c_contiguous:
        y = np.ascontiguousarray(y)

    m = 1
    while m < N:
        y_view = y.reshape(-1, N // (2 * m), 2, m)
        v = y_view[:, :, 1, :].copy()
        y_view[:, :, 1, :] = y_view[:, :, 0, :] - v
        y_view[:, :, 0, :] += v
        m *= 2

    y = y.reshape(orig_shape)

    if normalize:
        y *= (1.0 / math.sqrt(N))

    return y


def fwht(
    x: Union[torch.Tensor, np.ndarray],
    normalize: bool = True,
    inplace: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """Fast Walsh-Hadamard Transform (FWHT) along the last dimension.

    Complexity: O(N log N) operations where N = x.shape[-1].
    N must be a power of 2. For non-power-of-2, see `block_hadamard_transform`
    or `padded_hadamard_transform`.

    Args:
        x: Input tensor (PyTorch or NumPy) of shape (..., N).
        normalize: If True, scale by 1 / sqrt(N) to make H orthonormal (H @ H.T = I).
        inplace: If True, perform transform in-place when possible.

    Returns:
        Transformed tensor of the same shape and type as x.
    """
    if isinstance(x, torch.Tensor):
        return _fwht_torch(x, normalize=normalize, inplace=inplace)
    elif isinstance(x, np.ndarray):
        return _fwht_numpy(x, normalize=normalize, inplace=inplace)
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")


def ifwht(
    x: Union[torch.Tensor, np.ndarray],
    normalize: bool = True,
    inplace: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """Inverse Fast Walsh-Hadamard Transform.

    Since H is symmetric and orthonormal (when normalize=True), H^{-1} = H.
    If normalize=False, H^{-1} = (1 / N) * H.
    """
    if normalize:
        return fwht(x, normalize=True, inplace=inplace)
    else:
        N = x.shape[-1]
        y = fwht(x, normalize=False, inplace=inplace)
        if isinstance(y, torch.Tensor):
            return y.div(N)
        else:
            return y / N


def randomized_hadamard_transform(
    x: Union[torch.Tensor, np.ndarray],
    signs: Optional[Union[torch.Tensor, np.ndarray]] = None,
    seed: Optional[int] = None,
    normalize: bool = True,
    inplace: bool = False,
    return_signs: bool = False,
) -> Union[Union[torch.Tensor, np.ndarray], Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]]:
    """Randomized Hadamard Transform (RHT): y = H @ diag(s) @ x.

    Applies random sign flipping s in {-1, +1}^N prior to Walsh-Hadamard transform.
    This guarantees prevention of coherence collapse and uniform sub-Gaussian dispersion.

    Args:
        x: Input tensor of shape (..., N) where N is a power of 2.
        signs: Optional predefined sign vector of shape (N,).
        seed: Optional seed for reproducible sign generation.
        normalize: If True, preserves L2 norm strictly (1 / sqrt(N)).
        inplace: If True, transforms in-place.
        return_signs: If True, returns (transformed_x, signs).

    Returns:
        Transformed tensor (or tuple of (tensor, signs) if return_signs=True).
    """
    N = x.shape[-1]
    is_torch = isinstance(x, torch.Tensor)

    if signs is None:
        if is_torch:
            signs = generate_random_signs(
                dim=N,
                seed=seed,
                device=x.device,
                dtype=x.dtype,
                return_torch=True,
            )
        else:
            signs = generate_random_signs(
                dim=N,
                seed=seed,
                return_torch=False,
            )

    # Sign flipping: R * x
    if is_torch:
        if not isinstance(signs, torch.Tensor):
            signs = torch.as_tensor(signs, device=x.device, dtype=x.dtype)
        else:
            signs = signs.to(device=x.device, dtype=x.dtype)
        if inplace:
            x.mul_(signs)
            y = _fwht_torch(x, normalize=normalize, inplace=True)
        else:
            x_signed = x * signs
            y = _fwht_torch(x_signed, normalize=normalize, inplace=False)
    else:
        if isinstance(signs, torch.Tensor):
            signs = signs.cpu().numpy()
        signs = np.asarray(signs, dtype=x.dtype)
        if inplace:
            x *= signs
            y = _fwht_numpy(x, normalize=normalize, inplace=True)
        else:
            x_signed = x * signs
            y = _fwht_numpy(x_signed, normalize=normalize, inplace=False)

    if return_signs:
        return y, signs
    return y


# Convenient alias
rht = randomized_hadamard_transform


def inverse_randomized_hadamard_transform(
    y: Union[torch.Tensor, np.ndarray],
    signs: Union[torch.Tensor, np.ndarray],
    normalize: bool = True,
    inplace: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """Inverse Randomized Hadamard Transform: x = diag(s) @ H^{-1} @ y.

    Since H is symmetric/orthonormal and s_i in {-1, +1},
    (H @ diag(s))^{-1} = diag(s) @ H^T = diag(s) @ H.
    """
    is_torch = isinstance(y, torch.Tensor)
    # Step 1: Inverse FWHT
    x_rec = ifwht(y, normalize=normalize, inplace=inplace)

    # Step 2: Invert signs (diag(s)^{-1} = diag(s))
    if is_torch:
        if not isinstance(signs, torch.Tensor):
            signs = torch.as_tensor(signs, device=y.device, dtype=y.dtype)
        else:
            signs = signs.to(device=y.device, dtype=y.dtype)
        if inplace:
            x_rec.mul_(signs)
        else:
            x_rec = x_rec * signs
    else:
        if isinstance(signs, torch.Tensor):
            signs = signs.cpu().numpy()
        signs = np.asarray(signs, dtype=y.dtype)
        if inplace:
            x_rec *= signs
        else:
            x_rec = x_rec * signs

    return x_rec


# Convenient alias
irht = inverse_randomized_hadamard_transform


def block_hadamard_transform(
    x: Union[torch.Tensor, np.ndarray],
    block_size: Optional[int] = None,
    signs: Optional[Union[torch.Tensor, np.ndarray]] = None,
    normalize: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """Block-diagonal Hadamard Transform for arbitrary dimensions D.

    Guarantees strict orthogonality (H @ H.T = I_D) and norm preservation
    (||H * x||_2 = ||x||_2) for ANY dimension D, whether power-of-2 or not.

    Decomposition strategy:
    1. If D is power of 2 and block_size is None: executes standard FWHT in O(D log D).
    2. If D is divisible by block_size: reshapes to (..., D // block_size, block_size)
       and executes batched FWHT in O(D log B).
    3. If general D: decomposes D into binary power-of-2 blocks D = sum 2^{p_i}
       (e.g., intermediate size 5632 = 4096 + 1024 + 512), transforms each block,
       and concatenates.

    Args:
        x: Input tensor of shape (..., D).
        block_size: Optional power-of-2 block size (e.g. 512, 1024, 2048).
        signs: Optional sign vector of shape (D,) for randomized sign flipping.
        normalize: If True, normalizes each block by 1 / sqrt(B_i).

    Returns:
        Transformed tensor of shape (..., D).
    """
    D = x.shape[-1]
    is_torch = isinstance(x, torch.Tensor)

    # Optional randomized sign flipping across the full dimension
    if signs is not None:
        if is_torch:
            if not isinstance(signs, torch.Tensor):
                signs = torch.as_tensor(signs, device=x.device, dtype=x.dtype)
            else:
                signs = signs.to(device=x.device, dtype=x.dtype)
            x_in = x * signs
        else:
            if isinstance(signs, torch.Tensor):
                signs = signs.cpu().numpy()
            signs = np.asarray(signs, dtype=x.dtype)
            x_in = x * signs
    else:
        x_in = x

    # Case 1: D is already a power of 2 and no custom block_size requested
    if is_power_of_two(D) and block_size is None:
        return fwht(x_in, normalize=normalize)

    # Case 2: D is divisible by power-of-2 block_size
    if block_size is not None and is_power_of_two(block_size) and D % block_size == 0:
        orig_shape = x_in.shape
        num_blocks = D // block_size
        if is_torch:
            x_reshaped = x_in.view(*orig_shape[:-1], num_blocks, block_size)
            y_reshaped = _fwht_torch(x_reshaped, normalize=normalize, inplace=False)
            return y_reshaped.view(orig_shape)
        else:
            x_reshaped = x_in.reshape(*orig_shape[:-1], num_blocks, block_size)
            y_reshaped = _fwht_numpy(x_reshaped, normalize=normalize, inplace=False)
            return y_reshaped.reshape(orig_shape)

    # Case 3: Arbitrary dimension -> greedy binary power-of-2 decomposition
    blocks = decompose_dimension(D, block_size=block_size)
    if is_torch:
        parts: List[torch.Tensor] = []
        offset = 0
        for b in blocks:
            slice_b = x_in[..., offset : offset + b]
            parts.append(_fwht_torch(slice_b, normalize=normalize, inplace=False))
            offset += b
        return torch.cat(parts, dim=-1)
    else:
        parts_np: List[np.ndarray] = []
        offset = 0
        for b in blocks:
            slice_b = x_in[..., offset : offset + b]
            parts_np.append(_fwht_numpy(slice_b, normalize=normalize, inplace=False))
            offset += b
        return np.concatenate(parts_np, axis=-1)


def inverse_block_hadamard_transform(
    y: Union[torch.Tensor, np.ndarray],
    block_size: Optional[int] = None,
    signs: Optional[Union[torch.Tensor, np.ndarray]] = None,
    normalize: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """Exact inverse of block_hadamard_transform."""
    # Since each block is symmetric and orthogonal: H_{block}^{-1} = H_{block}
    x_rec = block_hadamard_transform(y, block_size=block_size, signs=None, normalize=normalize)
    if signs is not None:
        if isinstance(x_rec, torch.Tensor):
            if not isinstance(signs, torch.Tensor):
                signs = torch.as_tensor(signs, device=x_rec.device, dtype=x_rec.dtype)
            else:
                signs = signs.to(device=x_rec.device, dtype=x_rec.dtype)
            x_rec = x_rec * signs
        else:
            if isinstance(signs, torch.Tensor):
                signs = signs.cpu().numpy()
            signs = np.asarray(signs, dtype=x_rec.dtype)
            x_rec = x_rec * signs
    return x_rec


def padded_hadamard_transform(
    x: Union[torch.Tensor, np.ndarray],
    normalize: bool = True,
) -> Tuple[Union[torch.Tensor, np.ndarray], int]:
    """Zero-pad dimension D to next power of 2, then apply FWHT.

    Args:
        x: Input tensor of shape (..., D).
        normalize: If True, orthonormal factor 1 / sqrt(N_pad).

    Returns:
        Tuple of (transformed_padded_tensor, original_dim).
    """
    orig_dim = x.shape[-1]
    if is_power_of_two(orig_dim):
        return fwht(x, normalize=normalize), orig_dim

    pad_dim = next_power_of_two(orig_dim)
    diff = pad_dim - orig_dim

    if isinstance(x, torch.Tensor):
        pad_tuple = (0, diff)
        x_pad = torch.nn.functional.pad(x, pad_tuple, mode="constant", value=0.0)
        y_pad = _fwht_torch(x_pad, normalize=normalize)
        return y_pad, orig_dim
    elif isinstance(x, np.ndarray):
        pad_width = [(0, 0)] * (x.ndim - 1) + [(0, diff)]
        x_pad = np.pad(x, pad_width, mode="constant", constant_values=0.0)
        y_pad = _fwht_numpy(x_pad, normalize=normalize)
        return y_pad, orig_dim
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")


def padded_hadamard_inverse(
    y_padded: Union[torch.Tensor, np.ndarray],
    orig_dim: int,
    normalize: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """Inverse of padded_hadamard_transform by applying FWHT and unpadding."""
    x_rec = ifwht(y_padded, normalize=normalize)
    return x_rec[..., :orig_dim]


def get_hadamard_matrix(
    n: int,
    normalize: bool = True,
    signs: Optional[Union[torch.Tensor, np.ndarray]] = None,
    block_size: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    return_torch: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """Construct explicit n x n orthogonal Hadamard matrix.

    Generated efficiently by transforming the identity matrix I_n.

    Args:
        n: Dimension of the matrix.
        normalize: If True, matrix satisfies H @ H.T = I_n.
        signs: Optional sign vector for RHT.
        block_size: Block size for block-diagonal Hadamard if non-power-of-2.
        device: PyTorch device.
        dtype: PyTorch dtype.
        return_torch: If True return torch.Tensor, else np.ndarray.

    Returns:
        n x n orthogonal matrix.
    """
    target_dtype = dtype if dtype is not None else torch.float32
    I = torch.eye(n, device=device, dtype=target_dtype)
    H = block_hadamard_transform(I, block_size=block_size, signs=signs, normalize=normalize)

    if not return_torch:
        return H.cpu().numpy()
    return H


def apply_hadamard_weight_rotation(
    weight: torch.Tensor,
    signs: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
    rotate_dim: int = 0,
) -> torch.Tensor:
    """Rotate weight matrix offline for ternary quantization: W_rot = H @ R @ W.

    Args:
        weight: Weight tensor (e.g. out_features, in_features).
        signs: Sign vector for RHT.
        block_size: Optional block size for non-power-of-2 dimensions.
        rotate_dim: Dimension to rotate (0 for out_features, 1 or -1 for in_features).

    Returns:
        Rotated weight matrix with dispersed outlier coordinates.
    """
    if rotate_dim not in (0, 1, -1):
        raise ValueError(f"rotate_dim must be 0 or 1/-1, got {rotate_dim}")

    if rotate_dim == 0:
        # Rotate rows (out_features)
        # H @ (R * W) -> we can transpose, apply along last dim, transpose back
        w_t = weight.transpose(0, -1)
        w_rot_t = block_hadamard_transform(w_t, block_size=block_size, signs=signs, normalize=True)
        return w_rot_t.transpose(0, -1)
    else:
        # Rotate columns (in_features, along last dimension)
        return block_hadamard_transform(weight, block_size=block_size, signs=signs, normalize=True)


def apply_hadamard_activation_rotation(
    activation: torch.Tensor,
    signs: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    """Rotate activations online: X_rot = X @ R @ H.

    Args:
        activation: Activation tensor of shape (..., hidden_dim).
        signs: Sign vector for RHT.
        block_size: Optional block size for non-power-of-2 dimensions.

    Returns:
        Rotated activations with suppressed outlier spikes.
    """
    return block_hadamard_transform(activation, block_size=block_size, signs=signs, normalize=True)


def outlier_suppression_bound(x: Union[torch.Tensor, np.ndarray]) -> float:
    """Compute theoretical upper bound: max |H * x| <= (1 / sqrt(N)) * ||x||_1."""
    N = x.shape[-1]
    if isinstance(x, torch.Tensor):
        l1_norm = torch.sum(torch.abs(x), dim=-1).item() if x.ndim == 1 else torch.max(torch.sum(torch.abs(x), dim=-1)).item()
    else:
        l1_norm = float(np.sum(np.abs(x), axis=-1) if x.ndim == 1 else np.max(np.sum(np.abs(x), axis=-1)))
    return (1.0 / math.sqrt(N)) * l1_norm


def measure_outlier_ratio(
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray],
) -> float:
    """Measure empirical outlier suppression ratio: max |x| / max |y|."""
    if isinstance(x, torch.Tensor):
        max_x = torch.max(torch.abs(x)).item()
        max_y = torch.max(torch.abs(y)).item()
    else:
        max_x = float(np.max(np.abs(x)))
        max_y = float(np.max(np.abs(y)))
    if max_y == 0.0:
        return float("inf") if max_x > 0 else 1.0
    return max_x / max_y

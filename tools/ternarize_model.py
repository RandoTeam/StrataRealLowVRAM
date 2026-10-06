"""End-to-End GGUF Model Ternarizer for Strata Engine.

Produces valid TQ1_0 (GGML Type 34) GGUF v3 models executable by Strata.

Mathematical Foundation & Specification:
=========================================
1. Strata TQ1_0 (GGML Type 34) Encoding:
   - 256 weights per block packed into 54 bytes (1.6875 bits/weight).
   - qs (48 bytes): 240 trits in Radix-3 (5 trits per byte across 32+16 columns).
   - qh (4 bytes): 16 trits in Radix-3 (4 trits per byte).
   - d (2 bytes): FP16 scale stored as little-endian uint16.
   - Strata CUDA / C++ dequantizer: division-free unpack via POW3_PACKED (0xF3511B090301).

2. Selective Quantization Policy:
   - PROTECTED (never ternarize, preserved in FP16 / original format):
     * token_embd.weight (embeddings)
     * output.weight (LM head)
     * *norm*.weight (RMSNorm layers: attn_norm, post_attention_norm, attn_q_norm, attn_k_norm, etc.)
     * *gate_inp* (router weights: ffn_gate_inp, ffn_gate_inp_shexp)
     * Any bias / 1D tensors (e.g. ssm_a, ssm_dt.bias)
     * Any tensor whose row length or total elements is not divisible by 256
   - TERNARIZED (quantized to TQ1_0):
     * MoE routed experts: *ffn_gate_exps*, *ffn_up_exps*, *ffn_down_exps*
     * Attention projections: *attn_q*, *attn_k*, *attn_v*, *attn_output*
     * (Optional): Shared experts (*ffn_gate_shexp*, *ffn_up_shexp*, *ffn_down_shexp*)

3. Target Models:
   - Qwen 3.8 Flash Next (125B MoE, 512 experts, 8 routed)
   - Qwen 3.6 35B A3B (35B MoE, 256 experts, 8 routed)
   - Ornith 1.5 35B A3B (35B MoE, 256 experts, 8 routed)

4. Optional Transformations:
   - Pre-rotation via Fast Walsh-Hadamard Transform (FWHT / RHT):
     Disperses outlier coordinates across the feature dimension, suppressing activation
     and weight spikes by up to sqrt(N) while strictly preserving L2 norm and GEMM output.
   - Calibration (--calib):
     Optimizes quantization threshold Delta* per block via dense grid search to minimize
     reconstruction MSE while keeping zero-trit sparsity strictly within 35.0% - 50.0%.

CLI Usage:
----------
  python tools/ternarize_model.py --input <src.gguf> --output <dest.gguf> [options]

Options:
  --hadamard                  Apply FWHT pre-rotation prior to ternarization
  --calib                     Enable optimal threshold search / calibration
  --layers <range>            Restrict ternarization to specific layer range (e.g. '0-3', '2-')
  --keep-f32                  Preserve protected tensors in F32 instead of converting to FP16
  --ternarize-shared-experts  Also ternarize shared expert projections
  --verbose                   Print detailed tensor conversion logs
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

# Ensure project root and tools directory are in sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import gguf

from tools.ternary.hadamard import (
    block_hadamard_transform,
    is_power_of_two,
)
from tools.ternary.trit_sampler import (
    BLOCK_SIZE,
    TQ1_0_BLOCK_BYTES,
    TritQuantResult,
    pack_tq1_0,
    quantize_to_trits,
    unpack_tq1_0,
)

# ==============================================================================
# Constants & Tensor Policy Specifications
# ==============================================================================

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
DEFAULT_ALIGNMENT = 32

# Regex patterns for protected tensors (highest priority, NEVER ternarize)
PROTECTED_PATTERNS = [
    re.compile(r"(^|\.)token_embd(\.weight)?$"),
    re.compile(r"(^|\.)output(\.weight)?$"),
    re.compile(r".*norm.*\.weight$"),
    re.compile(r".*gate_inp.*"),
    re.compile(r".*ssm_a.*"),
    re.compile(r".*ssm_dt.*"),
    re.compile(r".*conv1d.*"),
]

# Regex patterns for candidate ternarized tensors
TERNARIZED_PATTERNS = [
    re.compile(r".*ffn_gate_exps.*"),
    re.compile(r".*ffn_up_exps.*"),
    re.compile(r".*ffn_down_exps.*"),
    re.compile(r".*attn_q\.weight$"),
    re.compile(r".*attn_k\.weight$"),
    re.compile(r".*attn_v\.weight$"),
    re.compile(r".*attn_output\.weight$"),
]

# Shared expert patterns (optional ternarization)
SHARED_EXPERT_PATTERNS = [
    re.compile(r".*ffn_gate_shexp.*"),
    re.compile(r".*ffn_up_shexp.*"),
    re.compile(r".*ffn_down_shexp.*"),
]

# Linear attention projections (optional ternarization)
LINEAR_ATTN_PATTERNS = [
    re.compile(r".*attn_qkv\.weight$"),
    re.compile(r".*attn_gate\.weight$"),
    re.compile(r".*ssm_alpha\.weight$"),
    re.compile(r".*ssm_beta\.weight$"),
    re.compile(r".*ssm_out\.weight$"),
]


# ==============================================================================
# Helper Functions: Layer Parsing & Classification
# ==============================================================================

def parse_layer_range(range_str: Optional[str], max_layers: Optional[int] = None) -> Optional[Set[int]]:
    """Parses a layer range string into a set of integer layer indices.

    Supported formats:
        - "0-3"    -> {0, 1, 2, 3}
        - "0,1,4"  -> {0, 1, 4}
        - "2-"     -> {2, 3, ... max_layers-1} (or all >= 2 if max_layers is None)
        - "-4"     -> {0, 1, 2, 3, 4}
        - "3"      -> {3}
        - None     -> None (signifies all layers)
    """
    if not range_str or not range_str.strip():
        return None

    layers: Set[int] = set()
    parts = range_str.split(",")
    for part in parts:
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            sub = p.split("-", 1)
            start_str, end_str = sub[0].strip(), sub[1].strip()
            if not start_str and end_str:
                # "-N" -> 0..N
                end = int(end_str)
                layers.update(range(0, end + 1))
            elif start_str and not end_str:
                # "N-" -> N..max_layers-1
                start = int(start_str)
                limit = max_layers if max_layers is not None else 1000
                layers.update(range(start, limit))
            else:
                # "A-B" -> A..B
                start = int(start_str)
                end = int(end_str)
                layers.update(range(start, end + 1))
        else:
            layers.add(int(p))
    return layers


def get_tensor_layer_index(name: str) -> Optional[int]:
    """Extracts layer index from tensor name (e.g. 'blk.5.attn_q.weight' -> 5)."""
    m = re.search(r"\bblk\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    m = re.search(r"\blayers\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    return None


def classify_tensor(
    name: str,
    shape: Sequence[int],
    layer_range: Optional[Set[int]] = None,
    ternarize_shared_experts: bool = False,
    ternarize_linear_attn: bool = False,
) -> str:
    """Classifies a tensor according to the selective quantization policy.

    Returns:
        'PROTECTED': Must remain in FP16 / original format.
        'TERNARIZED': Will be quantized into TQ1_0.
        'UNMODIFIED': Stays in original format.
    """
    # 1. PROTECTED check (highest priority: norms, router, embeddings, lm_head)
    for pat in PROTECTED_PATTERNS:
        if pat.search(name):
            return "PROTECTED"

    # 2. Layer range check if tensor belongs to a block
    layer_idx = get_tensor_layer_index(name)
    if layer_idx is not None and layer_range is not None:
        if layer_idx not in layer_range:
            return "UNMODIFIED"

    # 3. Shape divisibility check: fastest-varying dimension must be divisible by 256
    # In numpy C-order, shape[-1] is the fastest-varying dimension (ne0 in GGUF).
    total_elements = 1
    for d in shape:
        total_elements *= d

    if len(shape) == 0 or total_elements % BLOCK_SIZE != 0 or shape[-1] % BLOCK_SIZE != 0:
        return "UNMODIFIED"

    # 4. TERNARIZED candidate check
    for pat in TERNARIZED_PATTERNS:
        if pat.search(name):
            return "TERNARIZED"

    # 5. Optional shared experts check
    if ternarize_shared_experts:
        for pat in SHARED_EXPERT_PATTERNS:
            if pat.search(name):
                return "TERNARIZED"

    # 6. Optional linear attention check
    if ternarize_linear_attn:
        for pat in LINEAR_ATTN_PATTERNS:
            if pat.search(name):
                return "TERNARIZED"

    return "UNMODIFIED"


# ==============================================================================
# Tensor Conversion & Quantization Engine
# ==============================================================================

def extract_tensor_f32(t: gguf.ReaderTensor) -> np.ndarray:
    """Extracts tensor elements as a contiguous float32 numpy array."""
    qtype = t.tensor_type
    logical_shape = tuple(int(x) for x in reversed(t.shape))

    if qtype == gguf.GGMLQuantizationType.F32:
        return np.asarray(t.data, dtype=np.float32).reshape(logical_shape)
    elif qtype == gguf.GGMLQuantizationType.F16:
        raw_f16 = np.frombuffer(t.data.tobytes(), dtype=np.float16)
        return raw_f16.astype(np.float32).reshape(logical_shape)
    elif qtype == gguf.GGMLQuantizationType.BF16:
        raw_u16 = np.frombuffer(t.data.tobytes(), dtype=np.uint16)
        f32_flat = (raw_u16.astype(np.uint32) << 16).view(np.float32)
        return f32_flat.reshape(logical_shape)
    elif qtype == gguf.GGMLQuantizationType.TQ1_0:
        trits, scales = unpack_tq1_0(t.data.tobytes())
        recon = scales.astype(np.float32)[:, None] * trits.astype(np.float32)
        return recon.reshape(logical_shape)
    else:
        # Generic dequantize via gguf library
        try:
            return gguf.dequantize(t.data, qtype).astype(np.float32).reshape(logical_shape)
        except Exception as e:
            raise RuntimeError(f"Unable to dequantize tensor '{t.name}' of type {qtype}: {e}")


def ternarize_tensor_to_tq1_0(
    tensor_data: np.ndarray,
    hadamard: bool = False,
    calib: bool = False,
    target_sparsity: Tuple[float, float] = (35.0, 50.0),
) -> Tuple[np.ndarray, float, float]:
    """Ternarizes a weight tensor to Strata TQ1_0 format.

    Args:
        tensor_data: Input float32 numpy array.
        hadamard: If True, applies FWHT pre-rotation prior to quantization.
        calib: If True, performs grid search for optimal MSE and target sparsity.
        target_sparsity: (min_pct, max_pct) zero-trit sparsity target.

    Returns:
        (packed_uint8_array, zero_sparsity_pct, reconstruction_mse)
    """
    work_arr = tensor_data
    if hadamard:
        # Pre-rotate last dimension via orthonormal FWHT
        work_arr = block_hadamard_transform(work_arr, normalize=True)

    # Perform optimal 1.58-bit ternary quantization
    q_res: TritQuantResult = quantize_to_trits(
        work_arr,
        block_size=BLOCK_SIZE,
        search_threshold=calib,
        search_method="grid",
        target_sparsity=target_sparsity,
    )

    # Bit-pack into TQ1_0 byte format (54 bytes per 256 weights)
    packed_bytes = pack_tq1_0(q_res.trits, q_res.scales)

    # Reshape packed bytes into byte geometry for GGUF writer:
    # shape[-1] is transformed from N to (N // 256 * 54)
    orig_shape = tensor_data.shape
    byte_last_dim = orig_shape[-1] // BLOCK_SIZE * TQ1_0_BLOCK_BYTES
    byte_shape = (*orig_shape[:-1], byte_last_dim)
    packed_arr = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(byte_shape)

    return packed_arr, q_res.sparsity, q_res.mse


# ==============================================================================
# Model Ternarizer Main Pipeline
# ==============================================================================

def ternarize_model(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    hadamard: bool = False,
    calib: bool = False,
    layers: Optional[str] = None,
    keep_f32: bool = False,
    ternarize_shared_experts: bool = False,
    ternarize_linear_attn: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Ternarizes a GGUF model into valid TQ1_0 format executable by Strata.

    Preserves all model metadata, tokenizers, and alignment strictly.
    Applies selective quantization policy protecting sensitive parameters.

    Args:
        input_path: Source GGUF file path.
        output_path: Destination GGUF file path.
        hadamard: If True, applies FWHT pre-rotation to ternarized tensors.
        calib: If True, enables optimal threshold search for high fidelity.
        layers: Optional layer range string (e.g. '0-3', '2-').
        keep_f32: If True, retains protected tensors in F32 instead of FP16.
        ternarize_shared_experts: If True, quantizes shared experts to TQ1_0.
        ternarize_linear_attn: If True, quantizes linear attention projections.
        verbose: If True, logs per-tensor progress.

    Returns:
        Dict with comprehensive conversion statistics and metrics.
    """
    src_file = Path(input_path).resolve()
    dest_file = Path(output_path).resolve()

    if not src_file.is_file():
        raise FileNotFoundError(f"Input GGUF file not found: {src_file}")

    dest_file.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    input_size = src_file.stat().st_size

    if verbose:
        print(f"[Strata-Ternarizer] Reading source GGUF: {src_file} ({input_size / (1024**2):.1f} MB)")

    reader = gguf.GGUFReader(str(src_file))

    # Architecture discovery from metadata
    arch = "unknown"
    if "general.architecture" in reader.fields:
        arch = str(reader.fields["general.architecture"].contents())

    # Alignment discovery
    alignment = DEFAULT_ALIGNMENT
    if "general.alignment" in reader.fields:
        val = reader.fields["general.alignment"].contents()
        if isinstance(val, int) and val > 0:
            alignment = val

    # Block count discovery
    max_layers: Optional[int] = None
    for k in (f"{arch}.block_count", "general.block_count"):
        if k in reader.fields:
            cnt = reader.fields[k].contents()
            if isinstance(cnt, int):
                max_layers = cnt
                break

    layer_set = parse_layer_range(layers, max_layers=max_layers)

    if verbose:
        print(f"[Strata-Ternarizer] Model architecture: '{arch}', alignment: {alignment} bytes")
        if layer_set is not None:
            print(f"[Strata-Ternarizer] Restricting ternarization to layers: {sorted(layer_set)}")
        print(f"[Strata-Ternarizer] Pre-rotation FWHT: {'ON' if hadamard else 'OFF'}")
        print(f"[Strata-Ternarizer] Optimal Calibration: {'ON' if calib else 'OFF'}")

    # Initialize GGUF writer
    writer = gguf.GGUFWriter(str(dest_file), arch=arch, use_temp_file=True)
    writer.data_alignment = alignment

    # 1. Copy all metadata unchanged
    for k, field in reader.fields.items():
        if k.startswith("GGUF.") or k == "general.architecture" or k == "general.quantization_version":
            continue
        val = field.contents()
        if field.types[0] == gguf.GGUFValueType.ARRAY:
            writer.add_key_value(k, val, gguf.GGUFValueType.ARRAY, sub_type=field.types[-1])
        else:
            writer.add_key_value(k, val, field.types[0])

    # Record quantization metadata
    writer.add_string("general.quantization_version", "Strata-TQ1_0")
    writer.add_string("strata.ternarizer_hadamard", "true" if hadamard else "false")
    writer.add_string("strata.ternarizer_calib", "true" if calib else "false")

    # Statistics tracking
    tensors_total = len(reader.tensors)
    tensors_ternarized = 0
    tensors_protected = 0
    tensors_unmodified = 0

    sparsity_list: List[float] = []
    mse_list: List[float] = []
    orig_ternarized_bytes = 0
    new_ternarized_bytes = 0

    # 2. Process tensors according to selective policy
    for idx, t in enumerate(reader.tensors):
        t_name = t.name
        t_shape = t.shape
        t_type = t.tensor_type
        numpy_shape = t.data.shape

        category = classify_tensor(
            t_name,
            numpy_shape,
            layer_range=layer_set,
            ternarize_shared_experts=ternarize_shared_experts,
            ternarize_linear_attn=ternarize_linear_attn,
        )

        if category == "PROTECTED":
            tensors_protected += 1
            if t_type == gguf.GGMLQuantizationType.F32 and not keep_f32:
                # Convert F32 protected tensors to standard FP16
                arr_fp16 = t.data.astype(np.float16)
                writer.add_tensor(t_name, arr_fp16)
                if verbose:
                    print(f"  [{idx+1}/{tensors_total}] PROTECTED (F32->FP16): {t_name} shape={t_shape}")
            elif t_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F64):
                writer.add_tensor(t_name, t.data)
                if verbose:
                    print(f"  [{idx+1}/{tensors_total}] PROTECTED ({t_type.name}): {t_name} shape={t_shape}")
            else:
                writer.add_tensor(t_name, t.data, raw_dtype=t_type)
                if verbose:
                    print(f"  [{idx+1}/{tensors_total}] PROTECTED ({t_type.name}): {t_name} shape={t_shape}")

        elif category == "TERNARIZED":
            tensors_ternarized += 1
            f32_arr = extract_tensor_f32(t)
            orig_bytes = f32_arr.size * 2  # compared against FP16 baseline
            orig_ternarized_bytes += orig_bytes

            packed_arr, sparsity, mse = ternarize_tensor_to_tq1_0(
                f32_arr,
                hadamard=hadamard,
                calib=calib,
            )

            new_bytes = packed_arr.nbytes
            new_ternarized_bytes += new_bytes
            sparsity_list.append(sparsity)
            mse_list.append(mse)

            writer.add_tensor(t_name, packed_arr, raw_dtype=gguf.GGMLQuantizationType.TQ1_0)

            if verbose:
                ratio = (new_bytes / orig_bytes) * 100.0 if orig_bytes > 0 else 0.0
                print(
                    f"  [{idx+1}/{tensors_total}] TERNARIZED -> TQ1_0: {t_name} shape={t_shape} "
                    f"({orig_bytes / 1024:.1f} KB -> {new_bytes / 1024:.1f} KB, "
                    f"{ratio:.1f}%, zero={sparsity:.1f}%, MSE={mse:.2e})"
                )

        else:  # UNMODIFIED
            tensors_unmodified += 1
            if t.data.dtype == np.uint8 or t_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F64):
                writer.add_tensor(t_name, t.data, raw_dtype=t_type)
            else:
                writer.add_tensor(t_name, t.data)
            if verbose:
                print(f"  [{idx+1}/{tensors_total}] UNMODIFIED ({t_type.name}): {t_name} shape={t_shape}")

    # 3. Finalize and write GGUF file
    if verbose:
        print("[Strata-Ternarizer] Flushing header, metadata, and tensors to disk...")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Clean up reader handles to ensure file locks are released on Windows
    del reader
    gc.collect()

    t_elapsed = time.time() - t_start
    output_size = dest_file.stat().st_size
    overall_ratio = (output_size / input_size) * 100.0 if input_size > 0 else 0.0

    avg_sparsity = float(np.mean(sparsity_list)) if sparsity_list else 0.0
    avg_mse = float(np.mean(mse_list)) if mse_list else 0.0

    summary = {
        "status": "success",
        "input_path": str(src_file),
        "output_path": str(dest_file),
        "architecture": arch,
        "input_bytes": input_size,
        "output_bytes": output_size,
        "compression_percentage": overall_ratio,
        "compression_ratio": (input_size / output_size) if output_size > 0 else 1.0,
        "tensors_total": tensors_total,
        "tensors_ternarized": tensors_ternarized,
        "tensors_protected": tensors_protected,
        "tensors_unmodified": tensors_unmodified,
        "avg_zero_sparsity": avg_sparsity,
        "avg_reconstruction_mse": avg_mse,
        "elapsed_seconds": t_elapsed,
        "hadamard": hadamard,
        "calib": calib,
    }

    if verbose or not False:
        print("\n" + "=" * 70)
        print("  STRATA TQ1_0 TERNARIZATION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"  Input Model:          {src_file.name} ({input_size / (1024**2):.2f} MB)")
        print(f"  Output Model:         {dest_file.name} ({output_size / (1024**2):.2f} MB)")
        print(f"  Overall Ratio:        {overall_ratio:.2f}% of original ({summary['compression_ratio']:.2f}x compression)")
        print(f"  Tensors Processed:    {tensors_total} total ({tensors_ternarized} TQ1_0, {tensors_protected} protected, {tensors_unmodified} unmodified)")
        if tensors_ternarized > 0:
            print(f"  Zero-Trit Sparsity:   {avg_sparsity:.2f}% (Target: 35.0% - 50.0%)")
            print(f"  Reconstruction MSE:   {avg_mse:.4e}")
        print(f"  Elapsed Time:         {t_elapsed:.2f} seconds")
        print("=" * 70)

    return summary


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strata GGUF End-to-End Model Ternarizer (Produces valid TQ1_0 GGML Type 34 models)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard ternarization with selective policy:
  python tools/ternarize_model.py --input models/Qwen3.6-35B-A3B.gguf --output models/Qwen3.6-35B-A3B-TQ1_0.gguf

  # With FWHT pre-rotation to disperse heavy-tailed outliers:
  python tools/ternarize_model.py --input models/qwen38.gguf --output models/qwen38-tq1.gguf --hadamard

  # With optimal threshold search / calibration enabled:
  python tools/ternarize_model.py --input models/ornith.gguf --output models/ornith-tq1.gguf --calib

  # Ternarize specific layer range (e.g. layers 2 to 31):
  python tools/ternarize_model.py --input models/model.gguf --output models/model-tq1.gguf --layers 2-31
""",
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to source GGUF file to ternarize",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to destination GGUF file to write",
    )
    parser.add_argument(
        "--hadamard",
        action="store_true",
        help="Apply Fast Walsh-Hadamard Transform (FWHT) pre-rotation to suppress outliers",
    )
    parser.add_argument(
        "--calib",
        action="store_true",
        help="Enable dense grid search for optimal reconstruction threshold and target sparsity",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Layer range to ternarize (e.g. '0-3', '2-', '1,4,7'). Omit to ternarize all layers.",
    )
    parser.add_argument(
        "--keep-f32",
        action="store_true",
        help="Keep protected tensors in F32 instead of converting to standard FP16",
    )
    parser.add_argument(
        "--ternarize-shared-experts",
        action="store_true",
        help="Also quantize shared expert projections (*ffn_*_shexp*) to TQ1_0",
    )
    parser.add_argument(
        "--ternarize-linear-attn",
        action="store_true",
        help="Also quantize linear attention / GDN projections (*attn_qkv*, *ssm_out*) to TQ1_0",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed per-tensor quantization progress",
    )

    args = parser.parse_args()

    try:
        ternarize_model(
            input_path=args.input,
            output_path=args.output,
            hadamard=args.hadamard,
            calib=args.calib,
            layers=args.layers,
            keep_f32=args.keep_f32,
            ternarize_shared_experts=args.ternarize_shared_experts,
            ternarize_linear_attn=args.ternarize_linear_attn,
            verbose=args.verbose,
        )
        return 0
    except Exception as e:
        print(f"Error during model ternarization: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

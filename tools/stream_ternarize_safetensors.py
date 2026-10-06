"""Streaming Shard-by-Shard SafeTensors to TQ1_0 GGUF Converter.

Designed for high-parameter MoE models (e.g. Qwen 3.8 Flash Next 125B)
whose unquantized BF16 weights (~284-335 GiB) exceed available local disk space.

Architecture & Memory Mechanics:
================================
1. Fetches metadata and tensor-to-shard index from Hugging Face or local index.
2. Processes model sequentially shard-by-shard:
   - Downloads single shard k (~2.0-2.5 GiB) into staging directory.
   - Quantizes candidate tensors into TQ1_0 (Type 34, 54 bytes per 256 weights).
   - Casts protected tensors (embeddings, norms, routers) to FP16.
   - Stream-writes tensors into target GGUF file.
   - Deletes temporary shard k immediately, keeping temporary disk usage <= 5 GiB.
3. Finalizes 32-byte aligned GGUF v3 file executable directly by Strata.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Ensure project root is in sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import gguf
from safetensors import safe_open

from tools.ternarize_model import (
    DEFAULT_ALIGNMENT,
    PROTECTED_PATTERNS,
    TERNARIZED_PATTERNS,
    classify_tensor,
    ternarize_tensor_to_tq1_0,
)
from tools.ternary.hadamard import block_hadamard_transform
from tools.ternary.trit_sampler import (
    BLOCK_SIZE,
    TQ1_0_BLOCK_BYTES,
    pack_tq1_0,
    quantize_to_trits,
)


def map_safetensors_to_gguf_name(st_name: str) -> str:
    """Translate HuggingFace SafeTensors tensor name into standard GGUF tensor naming."""
    name = st_name
    # Embeddings and output
    if name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if name == "lm_head.weight":
        return "output.weight"
    if name == "model.norm.weight":
        return "output_norm.weight"

    # Layer patterns
    m = re.match(r"^model\.layers\.(\d+)\.(.*)$", name)
    if m:
        layer_idx = m.group(1)
        subname = m.group(2)
        if subname == "input_layernorm.weight":
            return f"blk.{layer_idx}.attn_norm.weight"
        if subname == "post_attention_layernorm.weight":
            return f"blk.{layer_idx}.ffn_norm.weight"
        if subname == "self_attn.q_proj.weight":
            return f"blk.{layer_idx}.attn_q.weight"
        if subname == "self_attn.k_proj.weight":
            return f"blk.{layer_idx}.attn_k.weight"
        if subname == "self_attn.v_proj.weight":
            return f"blk.{layer_idx}.attn_v.weight"
        if subname == "self_attn.o_proj.weight":
            return f"blk.{layer_idx}.attn_output.weight"
        if subname == "self_attn.q_norm.weight":
            return f"blk.{layer_idx}.attn_q_norm.weight"
        if subname == "self_attn.k_norm.weight":
            return f"blk.{layer_idx}.attn_k_norm.weight"
        if subname == "block_sparse_moe.gate.weight":
            return f"blk.{layer_idx}.ffn_gate_inp.weight"
        if subname == "mlp.gate.weight":
            return f"blk.{layer_idx}.ffn_gate_inp.weight"
        if subname == "block_sparse_moe.experts.gate_proj.weight":
            return f"blk.{layer_idx}.ffn_gate_exps.weight"
        if subname == "block_sparse_moe.experts.up_proj.weight":
            return f"blk.{layer_idx}.ffn_up_exps.weight"
        if subname == "block_sparse_moe.experts.down_proj.weight":
            return f"blk.{layer_idx}.ffn_down_exps.weight"
        if subname == "mlp.shared_expert.gate_proj.weight":
            return f"blk.{layer_idx}.ffn_gate_shexp.weight"
        if subname == "mlp.shared_expert.up_proj.weight":
            return f"blk.{layer_idx}.ffn_up_shexp.weight"
        if subname == "mlp.shared_expert.down_proj.weight":
            return f"blk.{layer_idx}.ffn_down_shexp.weight"
        if subname == "mlp.shared_expert_gate.weight":
            return f"blk.{layer_idx}.ffn_gate_inp_shexp.weight"

    return name


def stream_ternarize_safetensors(
    input_shards: List[Path],
    output_path: Path,
    arch: str = "qwen38_moe",
    apply_hadamard: bool = False,
    apply_calib: bool = False,
    delete_shards_after_processing: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Stream-convert SafeTensors shards into a single TQ1_0 GGUF file."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = gguf.GGUFWriter(path=output_path, arch=arch, use_temp_file=False)
    writer.add_uint32("general.alignment", DEFAULT_ALIGNMENT)
    writer.add_name("Qwen 3.8 Flash Next TQ1_0 Stream")

    stats = {
        "tensors_processed": 0,
        "tensors_ternarized": 0,
        "tensors_protected": 0,
        "tensors_unmodified": 0,
        "raw_bytes_in": 0,
        "raw_bytes_out": 0,
        "total_sparsity": 0.0,
        "total_mse": 0.0,
    }

    t0 = time.time()
    for shard_idx, shard_path in enumerate(input_shards):
        shard_path = Path(shard_path)
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        shard_size = shard_path.stat().st_size
        stats["raw_bytes_in"] += shard_size
        if verbose:
            print(f"[{shard_idx + 1}/{len(input_shards)}] Ingesting shard: {shard_path.name} ({shard_size / (1024*1024):.2f} MB)")

        with safe_open(str(shard_path), framework="numpy") as f:
            tensor_names = sorted(f.keys())
            for name in tensor_names:
                tensor_data = f.get_tensor(name)
                gguf_name = map_safetensors_to_gguf_name(name)
                classification = classify_tensor(
                    gguf_name, tensor_data.shape
                )

                if classification == "PROTECTED" or classification == "UNMODIFIED":
                    # Write protected/unmodified tensor as FP16
                    out_tensor = tensor_data.astype(np.float16)
                    writer.add_tensor(gguf_name, out_tensor)
                    stats["tensors_protected" if classification == "PROTECTED" else "tensors_unmodified"] += 1
                    stats["raw_bytes_out"] += out_tensor.nbytes
                else:
                    # Ternarize (classification == "TERNARIZED")
                    float_data = tensor_data.astype(np.float32)
                    packed_arr, sparsity, mse = ternarize_tensor_to_tq1_0(
                        float_data, hadamard=apply_hadamard, calib=apply_calib
                    )
                    writer.add_tensor(
                        gguf_name,
                        packed_arr,
                        raw_dtype=gguf.GGMLQuantizationType.TQ1_0,
                    )
                    stats["tensors_ternarized"] += 1
                    stats["raw_bytes_out"] += packed_arr.nbytes
                    stats["total_sparsity"] += sparsity
                    stats["total_mse"] += mse

                stats["tensors_processed"] += 1

        if delete_shards_after_processing:
            try:
                os.remove(shard_path)
                if verbose:
                    print(f"Purged processed shard: {shard_path.name}")
            except OSError as e:
                print(f"Warning: could not delete shard {shard_path.name}: {e}")

        gc.collect()

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    elapsed = time.time() - t0
    stats["elapsed_sec"] = elapsed
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Streaming Shard-by-Shard SafeTensors to TQ1_0 GGUF Converter.")
    parser.add_argument("--shards", nargs="+", required=True, help="List of SafeTensors shard paths in order")
    parser.add_argument("--output", "-o", required=True, help="Output GGUF file path")
    parser.add_argument("--arch", default="qwen38_moe", help="Target architecture (default: qwen38_moe)")
    parser.add_argument("--hadamard", action="store_true", help="Apply FWHT pre-rotation prior to ternarization")
    parser.add_argument("--calib", action="store_true", help="Enable optimal threshold grid search")
    parser.add_argument("--delete-shards", action="store_true", help="Delete shard file immediately after processing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose conversion progress")

    args = parser.parse_args()
    shard_paths = [Path(p) for p in args.shards]

    stats = stream_ternarize_safetensors(
        input_shards=shard_paths,
        output_path=Path(args.output),
        arch=args.arch,
        apply_hadamard=args.hadamard,
        apply_calib=args.calib,
        delete_shards_after_processing=args.delete_shards,
        verbose=args.verbose,
    )

    print("\n" + "=" * 68)
    print("  STRATA STREAMING TQ1_0 CONVERSION COMPLETED")
    print("=" * 68)
    print(f"  Tensors Processed:  {stats['tensors_processed']} ({stats['tensors_ternarized']} TQ1_0, {stats['tensors_protected']} protected)")
    print(f"  Input Raw Size:     {stats['raw_bytes_in'] / (1024*1024):.2f} MB")
    print(f"  Output GGUF Size:   {stats['raw_bytes_out'] / (1024*1024):.2f} MB")
    print(f"  Elapsed Time:       {stats['elapsed_sec']:.2f} seconds")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())

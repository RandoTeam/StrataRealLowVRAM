"""Converts output.weight from Q5_K in shard 1 to Q2_0 in a standalone head GGUF.
Saves ~246 MiB of VRAM so that the RTX 3050 4GB GPU has sufficient headroom for the expert cache.
"""
from __future__ import annotations

import os
import sys
import time
import struct
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile
from gguf_writer import GGUFWriter, _Tensor, quantize_q2_0
from gguf import quants, GGMLQuantizationType as Q

def quantize_q2_0_fast(w: np.ndarray) -> bytes:
    """Vectorized Q2_0 quantizer, exactly matching docs/q2_0-contract.md."""
    w = np.asarray(w, dtype=np.float32).reshape(-1, 64)
    amax = np.max(np.abs(w), axis=1, keepdims=True)
    d_fp16 = amax.astype(np.float16)
    d_f32 = d_fp16.astype(np.float32)
    denom = np.where(d_f32 == 0, 1.0, d_f32)
    code = np.where(d_f32 == 0, 0, np.clip(np.rint(w / denom) + 1, 0, 3)).astype(np.uint8)
    code4 = code.reshape(-1, 16, 4)
    packed = (code4[:, :, 0] | (code4[:, :, 1] << 2) | (code4[:, :, 2] << 4) | (code4[:, :, 3] << 6))
    
    out = np.empty((w.shape[0], 18), dtype=np.uint8)
    out[:, :2] = d_fp16.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = packed
    return out.tobytes()

def make_head_q2_0(src_shard1: str, dst_path: str):
    print(f"Reading source GGUF: {src_shard1}")
    src_file = pathlib.Path(src_shard1)
    gf = GGUFFile(src_shard1)
    
    t_out = None
    for t in gf.tensors:
        if t.name == "output.weight":
            t_out = t
            break
            
    if not t_out:
        raise RuntimeError("output.weight not found in shard 1!")
        
    print(f"Source output.weight: shape={t_out.shape}, type={t_out.type_name}, offset={t_out.offset}")
    shape = list(t_out.shape) # [2560, 248320]
    total_elements = shape[0] * shape[1]
    n_blocks_q5k = total_elements // 256
    bytes_per_block_q5k = 176
    total_q5k_bytes = n_blocks_q5k * bytes_per_block_q5k
    
    print(f"Total elements: {total_elements:,}, Q5_K bytes: {total_q5k_bytes:,} ({total_q5k_bytes/(1024**2):.2f} MiB)")
    
    # Initialize GGUFWriter with architecture metadata from shard 1
    writer = GGUFWriter()
    # Required architecture metadata for check_architecture
    copy_keys = [
        "general.architecture",
        "general.type",
        "general.name",
        "qwen4exp.block_count",
        "qwen4exp.context_length",
        "qwen4exp.embedding_length",
        "qwen4exp.attention.head_count",
        "qwen4exp.attention.head_count_kv",
        "qwen4exp.expert_count",
        "qwen4exp.expert_used_count",
    ]
    for k in copy_keys:
        if k in gf.metadata:
            val = gf.metadata[k]
            # Infer or preserve type
            if isinstance(val, int):
                writer.add(k, val, "u32" if "count" in k or "length" in k else "i64")
            elif isinstance(val, str):
                writer.add(k, val, "string")
            elif isinstance(val, float):
                writer.add(k, val, "f32")
            elif isinstance(val, list):
                writer.add(k, val)
    
    print("Reading and converting output.weight chunks...")
    t0 = time.time()
    chunk_blocks = 50000 # 50,000 * 256 elements = 12.8M elements = 8.8 MB Q5_K -> 3.6 MB Q2_0
    q2_parts = []
    
    with src_file.open("rb") as f:
        f.seek(gf.data_start + t_out.offset)
        blocks_done = 0
        while blocks_done < n_blocks_q5k:
            cur_blocks = min(chunk_blocks, n_blocks_q5k - blocks_done)
            raw_q5k = f.read(cur_blocks * bytes_per_block_q5k)
            buf = np.frombuffer(raw_q5k, dtype=np.uint8)
            f32 = quants.dequantize(buf, Q.Q5_K)
            q2_bytes = quantize_q2_0_fast(f32)
            q2_parts.append(q2_bytes)
            blocks_done += cur_blocks
            print(f"  Processed {blocks_done:,} / {n_blocks_q5k:,} blocks ({(blocks_done/n_blocks_q5k)*100:.1f}%)", end="\r")
            
    print()
    all_q2 = b"".join(q2_parts)
    t1 = time.time()
    print(f"Converted in {t1-t0:.2f}s! Q2_0 bytes: {len(all_q2):,} ({len(all_q2)/(1024**2):.2f} MiB)")
    
    # Add tensor to writer
    writer.tensors.append(_Tensor("output.weight", shape, "Q2_0", all_q2))
    
    # Write destination file
    dst_p = pathlib.Path(dst_path)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {dst_p}...")
    writer.write(dst_p)
    print(f"Saved: {dst_p} ({dst_p.stat().st_size/(1024**2):.2f} MiB)")

if __name__ == "__main__":
    src = r"C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata-data\models\Q2_0\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
    dst = r"C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata-data\models\Q2_0\output_q2_0.gguf"
    make_head_q2_0(src, dst)

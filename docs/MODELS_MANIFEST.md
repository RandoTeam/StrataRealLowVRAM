# Strata Models Manifest and Architectural Specification

## 1. Overview and Executive Summary

This document establishes the official weights manifest, structural architecture, parameter topology, and memory footprints for all large language models deployed and supported within the **Strata** inference ecosystem. 

Strata implements an asymmetric dual-engine architecture:
1. **`engine/strata.exe`**: Optimized for **Qwen3.8-Flash-Next** (125B MoE) featuring 4-stream hyper-connections, QSA attention with indexers, Prefetch Lookup Engine (PLE), and Native CUDA Ternary TQ1_0 kernels.
2. **`engine/strata-qwen36.exe`**: Optimized for the **Qwen3.6-35B-A3B** and **Ornith-1.5-35B-A3B** model families featuring hybrid Gated Delta Networks (GDN), standard pre-RMSNorm residuals, GQA attention, and grouped sparse MoE routing.

Both engines share a unified stdin/stdout IPC protocol (`--serve`, `GEN`, `PP`, `T`, `DONE`, `STOP`), enabling seamless integration with the Strata Python server (`serve/server.py`), Web UI, REST endpoints, and Model Context Protocol (MCP) server without code divergence.

---

## 2. Model Inventory Matrix

| Model Identifier | Upstream Repository (Hugging Face) | Architecture Family | Quantization Format | Total Params | Active Params | File Size | Native Context | Target Engine |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`Ornith-1.5-35B-A3B-GSQ-RCO-3.5bit.gguf`** | `ornith-ai/Ornith-1.5-35B-A3B` | `qwen35moe` | GSQ/RCO 3.5-bit (Q2_K/Q3_K/Q4_K) | 35.0B | ~3.0B | 15.16 GB (14.12 GiB) | 262,144 | `strata-qwen36.exe` |
| **`Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf`** | `ornith-ai/Ornith-1.5-35B-A3B` | `qwen35moe` | Activation-Decoupled Q4_K / IQ4_XS | 35.0B | ~3.0B | 20.12 GB (18.74 GiB) | 262,144 | `strata-qwen36.exe` |
| **`Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf`** | `unsloth/Qwen3.6-35B-A3B-GGUF` | `qwen35moe` | Q4_K_XL + MTP Speculative MoE | 35.0B + MTP | ~3.0B | 22.18 GB (20.66 GiB) | 262,144 | `strata-qwen36.exe` |
| **`Qwen 3.8 Flash Next (125B MoE)`** | `Qwen/Qwen3.8-Flash-Next` | `qwen38moe` | GSQ/RCO Coder IQ1_M, Q2_0, TQ1_0 | 125.0B | ~14.0B | 20–38 GB (split packs) | 131,072 | `strata.exe` |

---

## 3. Deep Architectural Specifications

### 3.1. Qwen3.6-35B-A3B & Ornith-1.5-35B-A3B Architecture

The Qwen3.6 and Ornith-1.5 35B series represent a major architectural evolution over traditional pure-attention Transformers, combining linear-time recurrent layers with sparse multi-head self-attention.

```
                    ┌─────────────────────────┐
                    │      Token Input        │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────▼───────────────┐
                 │       Embedding (2048)        │
                 └───────────────┬───────────────┘
                                 │
     ┌───────────────────────────▼───────────────────────────┐
     │  Layer Cycle (Repeated 10 times = 40 Layers Total)    │
     │  ┌─────────────────────────────────────────────────┐  │
     │  │ Layer 4k+0 : GDN Block + Sparse MoE (256/8+1)    │  │
     │  │ Layer 4k+1 : GDN Block + Sparse MoE (256/8+1)    │  │
     │  │ Layer 4k+2 : GDN Block + Sparse MoE (256/8+1)    │  │
     │  │ Layer 4k+3 : Attention Block + Sparse MoE        │  │
     │  └─────────────────────────────────────────────────┘  │
     └───────────────────────────┬───────────────────────────┘
                                 │
                 ┌───────────────▼───────────────┐
                 │       Final RMSNorm (2048)    │
                 └───────────────┬───────────────┘
                                 │
                 ┌───────────────▼───────────────┐
                 │     LM Head (Vocab 248,320)   │
                 └───────────────────────────────┘
```

#### Layer Topology & Periodic Layout
- **Total Block Count**: 40 blocks (41 blocks for MTP-augmented checkpoints).
- **Macro-Layer Periodicity**: `[3 × GDN + 1 × Attention]` repeated 10 times:
  - **GDN Layers**: 30 layers (Layers 0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38).
  - **Full Attention Layers**: 10 layers (Layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39).
- **Residual Framework**: Standard Pre-RMSNorm additive residual paths:
  $$\mathbf{x}_{\text{mid}} = \mathbf{x} + \text{Mixer}(\text{RMSNorm}(\mathbf{x}))$$
  $$\mathbf{x}_{\text{out}} = \mathbf{x}_{\text{mid}} + \text{MoE}(\text{RMSNorm}(\mathbf{x}_{\text{mid}}))$$

#### Gated Delta Network (GDN) Mathematical Formulation
The GDN layer operates as a linear recurrence with hardware-accelerated causal convolution and associative delta updates:
1. **Input Projections**:
   $$\mathbf{q}, \mathbf{k}, \mathbf{v} = \mathbf{W}_{qkv} \mathbf{x}, \quad \mathbf{z} = \mathbf{W}_{\text{gate}} \mathbf{x}$$
   Where $d_q = 2048$, $d_k = 2048$, $d_v = 4096$, $d_z = 2048$.
2. **Causal 1D Convolution**: Kernel size $k=4$ per channel with $\text{SiLU}$ activation.
3. **Head Configuration**: 16 Key/Query heads, 32 Value heads, state size $S = 128$.
4. **Recurrent Delta Rule Update**:
   $$\mathbf{S}_t = \mathbf{S}_{t-1} \left(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^T\right) + \mathbf{v}_t \mathbf{k}_t^T$$
5. **Output Normalization**: RMSNorm gated by $\text{SiLU}(\mathbf{z})$:
   $$\mathbf{y}_{\text{gdn}} = \text{RMSNorm}(\mathbf{o}) \odot \text{SiLU}(\mathbf{z})$$

#### Grouped-Query Attention (GQA) Formulation
- **Head Count**: 16 Query heads, 2 Key/Value heads ($8:1$ GQA ratio).
- **Head Dimension**: $d_h = 256$ (Query dimension = $16 \times 256 = 4096$, KV dimension = $2 \times 256 = 512$).
- **Gating**: Query projections output interleaved $[\mathbf{q} \parallel \mathbf{gate}]$.
- **RoPE**: NeoX-style partial rotary embedding on the first 64 dimensions with frequency base $\theta = 10,000,000$.
- **Attention Output**: Standard scaled dot-product attention scaled by $\text{sigmoid}(\mathbf{gate})$:
  $$\mathbf{y}_{\text{attn}} = \text{Attention}(\mathbf{q}_{\text{norm}}, \mathbf{k}_{\text{norm}}, \mathbf{v}) \odot \sigma(\mathbf{gate})$$

#### Sparse Mixture of Experts (MoE) Structure
- **Expert Pool**: 256 routed experts per layer, plus 1 shared expert.
- **Feed-Forward Dimension**: Hidden dimension $d_{ff} = 512$ per expert.
- **Routing Top-K**: Router computes softmax over 256 logits, selects top-8 experts, and **re-normalizes** the top-8 weights:
  $$\mathbf{w}_{\text{routed}} = \text{Softmax}(\text{Top8}(\mathbf{W}_r \mathbf{x}))$$
- **Shared Expert**: Dedicated persistent expert computed on all tokens, gated by:
  $$\mathbf{g}_{\text{shared}} = \sigma\left(\mathbf{w}_{\text{shared\_gate}} \cdot \mathbf{x}\right)$$
- **Active Parameter Footprint**:
  - Routed experts active per token: $8 \times 512 = 4096$ hidden units.
  - Shared expert active per token: $1 \times 512 = 512$ hidden units.
  - Total active expert hidden width: $4608$ units per layer.
  - Base dense parameters (embeddings, norms, GDN states, attention heads, LM head): ~1.8B.
  - Active MoE parameters: ~1.2B.
  - **Total Active Parameters**: **~3.0B parameters** active during each forward decode step (~8.5% of the 35B parameter space).

---

### 3.2. Qwen 3.8 Flash Next (125B MoE) Architecture

The 125-billion parameter **Qwen3.8-Flash-Next** engine utilizes an extreme-scale sparse MoE topology:
- **Total Parameters**: 125 Billion.
- **Routed Experts**: 24,576 fine-grained micro-experts partitioned across layers.
- **Active Experts per Token**: 10 experts per layer.
- **Hyper-Connections**: 4-stream parallel residual highways for low-latency layer transitions.
- **Indexer Attention (QSA)**: Dedicated indexing heads prune KV cache access before full attention computation.
- **Native CUDA Ternary TQ1_0 (Type 34, 1.58-bit)**:
  - 256 weights packed into 54 bytes (1.6875 bits per weight).
  - Fast bitwise decoding via `POW3_PACKED` lookup table without division instructions.
  - Hardware integer accumulation via `STRATA_DP4A` instruction on NVIDIA Turing/Ampere/Ada architectures.
  - Compresses the active expert working set to **~2.8 GB VRAM**, enabling full expert residency in 4 GB VRAM mobile GPUs.

---

## 4. Detailed Model Profiles

### Model 1: `Ornith-1.5-35B-A3B-GSQ-RCO-3.5bit.gguf`
- **File Name**: `Ornith-1.5-35B-A3B-GSQ-RCO-3.5bit.gguf`
- **Hugging Face Repository**: [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
- **Exact File Size**: 15,163,013,536 bytes (14.12 GiB / 15.16 GB)
- **Tensor Count**: 733 tensors (44 metadata entries)
- **Quantization Distribution**:
  - `Q8_0`: 310 tensors (norms, gating vectors, router weights)
  - `F32`: 301 tensors (biases, scalar parameters, RMSNorm scales)
  - `Q2_K`: 52 tensors (expert down-projection sub-blocks)
  - `Q3_K`: 52 tensors (expert up-projection sub-blocks)
  - `Q4_K`: 18 tensors (attention projection tensors)
- **Memory Requirements**:
  - System Host RAM: 16.0 GB (fits comfortably in 24 GB / 32 GB systems).
  - VRAM Consumption: ~1.25 GB base context + 512-slot expert cache (~1.15 GB) = ~2.4 GB total VRAM.
- **Primary Use Case**: Maximum speed on 16 GB–32 GB RAM systems where DDR4 bandwidth conservation is paramount.

---

### Model 2: `Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf`
- **File Name**: `Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf`
- **Hugging Face Repository**: [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
- **Exact File Size**: 20,125,923,520 bytes (18.74 GiB / 20.13 GB)
- **Tensor Count**: 733 tensors (47 metadata entries)
- **Quantization Distribution**:
  - `Q8_0`: 312 tensors (router and normalization layers)
  - `F32`: 301 tensors (biases and scaling factors)
  - `IQ4_XS`: 80 tensors (MoE feed-forward experts utilizing non-linear importance matrices)
  - `Q4_K`: 40 tensors (GDN projections and attention projections)
- **Memory Requirements**:
  - System Host RAM: 20.5 GB resident working set.
  - VRAM Consumption: ~1.4 GB base + 512-slot expert cache (~1.3 GB) = ~2.7 GB total VRAM.
- **Primary Use Case**: High-precision reasoning and algorithmic synthesis without degradation in multi-turn conversational quality.

---

### Model 3: `Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf`
- **File Name**: `Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf`
- **Hugging Face Repository**: [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
- **Exact File Size**: 22,187,497,792 bytes (20.66 GiB / 22.19 GB)
- **Tensor Count**: 753 tensors (55 metadata entries)
- **Quantization Distribution**:
  - `Q4_K`: 341 tensors (dense layers and expert projections)
  - `F32`: 308 tensors (norms, embedding biases, RoPE constants)
  - `Q6_K`: 88 tensors (critical down-projections and router heads)
  - `Q8_0`: 14 tensors (shared projections)
  - `Type_30`: 2 tensors (MTP speculative prediction heads)
- **Block Count**: 41 blocks (Blocks 0–39 standard GDN/Attention + Block 40 Multi-Token Prediction unit).
- **Memory Requirements**:
  - System Host RAM: 22.5 GB resident working set.
  - VRAM Consumption: ~1.5 GB base + 512-slot expert cache (~1.4 GB) = ~2.9 GB total VRAM.
- **Primary Use Case**: High-throughput speculative decoding via MTP draft heads, reaching 85–115 tok/s under Ampere/Ada mobile architectures.

---

### Model 4: `Qwen 3.8 Flash Next (125B MoE)`
- **File Name**: `Qwen3.8-Flash-Next-GSQ-RCO-*` / `tq1_0` packs
- **Hugging Face Repository**: [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- **Packaging Topology**: Split format with separate dense pack (`dense.bin`), router index (`index.txt`), PLE row cache, and expert shards.
- **Target Quantization**:
  - Native CUDA Ternary `TQ1_0` (GGML Type 34, 1.58-bit / 1.6875 bpw).
  - Coder `IQ1_M` (1.75 bpw) and `Q2_0` (2.0 bpw).
- **Memory Requirements**:
  - System Host RAM: 16.0–22.0 GiB locked via `VirtualLock`.
  - VRAM Consumption: 426–700 GPU expert slots occupying 920 MiB–1.4 GiB VRAM on 4 GB GPUs.
- **Primary Use Case**: Production coding assistance, 65,536-token context reasoning, and multi-turn agent interaction.

---

## 5. Storage Directory Configuration and Zero-Copy Setup

To eliminate hardcoded local paths (such as `C:\VietnAi\qwen3.6\`) and maintain portability across different developer workstations, Strata uses a relative `models/` directory anchored to the project root.

### Automated Setup Script: `tools/setup_models_directory.ps1`

The repository provides an automated PowerShell utility that configures the `models/` directory using zero-copy NTFS junctions or file-level hard links:

```powershell
# Run from repository root to configure relative models directory:
powershell -ExecutionPolicy Bypass -File tools/setup_models_directory.ps1
```

#### Script Features
- **Zero-Copy Architecture**: Uses NTFS Reparse Points (Directory Junctions) or NTFS Hard Links. No duplicate gigabytes are written to disk.
- **Cross-Volume Fallback**: Automatically attempts HardLink first (same drive, 0 privilege required), falling back to SymbolicLink if targeting secondary volumes.
- **Automatic Discovery**: Automatically checks standard paths (`C:\VietnAi\qwen3.6\models`, `..\Strata-data\models`, or custom directories via `-SourceDirs`).

#### Syntax & Parameters
```powershell
param(
    [string]$TargetDir = "<RepoRoot>\models",
    [string[]]$SourceDirs = @("C:\VietnAi\qwen3.6\models", "..\Strata-data\models"),
    [ValidateSet("Auto", "Junction", "HardLink", "SymLink")][string]$Mode = "Auto",
    [switch]$Force
)
```

### Config File Verification

All runtime configuration profiles in `configs/` reference the unified relative `models/` directory:
- `configs/strata-ornith.json`:
  ```json
  "args": [
    "--native",
    "models/Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf",
    "--max-context", "131072",
    "--kv", "int8",
    "--kv-unified",
    "--spec-k", "6"
  ]
  ```
- `configs/strata-qwen36.json`:
  ```json
  "args": [
    "--native",
    "models/Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf",
    "--max-context", "131072",
    "--kv", "int8",
    "--kv-unified",
    "--spec-k", "6"
  ]
  ```
- `configs/strata-tq1_0.json`:
  ```json
  "args": [
    "--pack", "../Strata-data/packs/tq1_0",
    "--native", "../Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
  ]
  ```

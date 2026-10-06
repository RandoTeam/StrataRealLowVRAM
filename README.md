# Strata: High-Performance GPU-Centric Hybrid Inference Engine

Strata is a specialized inference engine designed for large sparse Mixture-of-Experts (MoE) architectures on resource-constrained hardware. By leveraging an asymmetric dual-engine design, native CUDA quantization kernels (including 1.58-bit ternary precision), dynamic VRAM expert caching, and operating system scheduling optimizations, Strata executes 35-billion and 125-billion parameter models on consumer systems with 4 GB to 16 GB of VRAM.

---

## 1. Architectural Paradigm: GPU-Centric Execution vs. CPU Offloading

### 1.1 The Memory Bus Bottleneck in Traditional Engines

Conventional inference engines (such as standard `llama.cpp` builds) offload unmapped MoE experts to the host CPU via AVX2/AVX-512 kernels when total model size exceeds physical VRAM. Under this execution model, token generation is strictly bounded by the host system's memory bandwidth.

For sparse MoE models with ~3.0 billion active parameters per token (e.g., Qwen3.6-35B-A3B or Ornith-1.5-35B-A3B):
- The model must read approximately 566 MB of active expert weights per token.
- On standard dual-channel DDR4-3200 memory systems, sustainable random gather bandwidth is approximately 26 GB/s ($26 \times 10^9$ bytes/sec).

The theoretical memory read latency per token on DDR4 is given by:

$$\tau_{\text{DDR4}} = \frac{M_{\text{active}}}{\text{BW}_{\text{DDR4}}} = \frac{566 \times 10^6 \text{ bytes}}{26 \times 10^9 \text{ bytes/s}} \approx 21.77 \text{ ms/token}$$

This imposes a hard physical ceiling of:

$$\text{Throughput}_{\text{max, DDR4}} = \frac{1}{\tau_{\text{DDR4}}} \approx 45.9 \text{ tokens/sec}$$

Under real-world CPU cache thrashing and AVX core frequency throttling, sustained CPU decode rates typically degrade to 18–25 tokens/sec with 100% CPU thread saturation.

### 1.2 Strata GPU-Centric Paradigm

Strata eliminates iterative CPU compute from the hot inference loop entirely:

1. **Hardware Expert Routing**: Router gate logits are computed directly on the GPU, with top-$k$ expert selection resolved via warp-level ballot instructions (`__ballot_sync`).
2. **VRAM Dynamic LRU Expert Cache**: A dedicated segment of physical VRAM (426 to 1,620 slots, depending on configuration) acts as a high-locality expert tier.
3. **GDDR6 Bandwidth Utilization**: With an empirical cache hit rate of 75%–85%, expert weights residing in the VRAM tier are evaluated over the local GPU memory bus (155 GB/s on an RTX 3050 Laptop GPU, and >500 GB/s on desktop cards):

$$\tau_{\text{GDDR6}} = \frac{M_{\text{active}}}{\text{BW}_{\text{GDDR6}}} = \frac{566 \times 10^6 \text{ bytes}}{155 \times 10^9 \text{ bytes/s}} \approx 3.65 \text{ ms/token}$$

4. **Accelerated CUDA Execution**: All dense projections, attention layers, Gated Delta Network recurrences, and MoE matrix multiplications (GEMM/GEMV) are executed exclusively on CUDA cores and Tensor Cores.
5. **Speculative Multi-Token Prediction (MTP)**: Utilizing speculative verification heads, generation latency is amortized across accepted token candidates, achieving sustained decoding speeds of 50–58 tokens/sec (pure decode) and 85–115+ tokens/sec (with speculative draft acceptance).

---

## 2. Universal Quantization and Format Matrix

Strata provides native runtime execution for multiple quantization representations without requiring separate binary recompilations.

| Format Identifier | Type ID | Effective Bits / Weight | Block Structure | CUDA Kernel Implementation | Target Hardware / Architecture |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **TQ1_0** | 34 | 1.6875 bpw (1.58-bit) | 54 bytes per 256 weights | Integer `STRATA_DP4A` accumulation, branchless `POW3_PACKED` decode | Ampere, Ada Lovelace, Turing (sm_75+) |
| **IQ1_M** | 24 | 1.75 bpw | Non-linear codebook | Vectorized nibble gather + scale unpacking | Turing, Ampere, Ada |
| **Q2_0** | 42 | 2.00 bpw | Standard 2-bit linear | `native_mmvq` 2-bit dot product | All CUDA architectures |
| **IQ2_XXS** | 16 | 2.06 bpw | 8-weight sub-vector | `iq_kernels.cu` grouped dispatch | Turing, Ampere, Ada |
| **IQ2_XS** | 17 | 2.31 bpw | Non-linear codebook | `iq_kernels.cu` grouped dispatch | Turing, Ampere, Ada |
| **IQ2_S** | 22 | 2.50 bpw | Non-linear codebook | `iq_kernels.cu` grouped dispatch | Turing, Ampere, Ada |
| **IQ3_XXS** | 18 | 3.06 bpw | 8-weight sub-vector | `iq_kernels.cu` grouped dispatch | Turing, Ampere, Ada |
| **IQ3_S** | 21 | 3.44 bpw | Non-linear codebook | `iq_kernels.cu` grouped dispatch | Turing, Ampere, Ada |
| **Q4_K** | 12 | 4.50 bpw | Super-block 256 (6-bit scales) | `vec_dot_q4_K_q8_1` / unified dispatch | All CUDA architectures |
| **IQ4_XS** | 23 | 4.25 bpw | Non-linear codebook | `vec_dot_iq4_xs_q8_1` dispatch | Turing, Ampere, Ada |
| **Q6_K** | 14 | 6.56 bpw | Super-block 256 (8-bit scales) | `vec_dot_q6_K_q8_1` / unified dispatch | All CUDA architectures |

### Native CUDA Ternary TQ1_0 (Type 34, 1.58-bit)

The TQ1_0 kernel implements native ternary weight execution ($\{-1, 0, +1\}$):
- **Packing Density**: 5 trits are packed into 1 byte ($3^5 = 243 \le 256$). A block of 256 weights requires 54 bytes total, yielding an effective bit rate of 1.6875 bits per weight.
- **Zero-Division Unpacking**: Unpacking is performed using precomputed table shifts (`POW3_PACKED`) in GPU constant memory, eliminating integer divisions and modulo operations.
- **Hardware DP4A Accumulation**: Four signed 8-bit integers are multiplied and accumulated into a 32-bit register in a single clock cycle via the `__dp4a` intrinsic. Floating-point operations in the inner loop are reduced to a single scale multiplication per block.
- **Cache Persistence**: Memory footprint reduction permits 100% VRAM residence of active experts for 125B models within 4 GB mobile hardware.

---

## 3. Windows WDDM Memory Guardian & Adaptive Quota Management

Windows Display Driver Model (WDDM) presents unique paging constraints for compute engines operating near physical VRAM limits. Strata integrates OS-level memory supervisors to guarantee stable operation.

### 3.1 Resolving the WDDM Zero-Byte Clamp

Under Windows WDDM, initial GPU allocations that approach physical memory thresholds cause `cudaMemGetInfo` to report zero bytes of available memory due to OS virtual paging semantics. Vanilla engines interpret this as out-of-memory and collapse expert cache allocations.

Strata bypasses this constraint via explicit configuration override (`0001-wddm-expert-cache-4gb.patch`), enabling non-auto static allocation of the GPU expert cache and preserving the resident VRAM tier.

### 3.2 Adaptive `VirtualLock` Working Set Expansion

To prevent Windows page faults and hard disk paging during MoE expert gather, weights resident in system RAM are locked into physical memory using the Win32 `VirtualLock` API.

However, default Windows process quotas restrict the maximum locked working set, triggering `ERROR_NOT_ENOUGH_QUOTA` (Win32 Error 1453). Strata's **WDDM Memory Guardian** (`src/core/expert_cache.cpp`) resolves this systematically:
1. Queries physical memory metrics dynamically via `GlobalMemoryStatusEx`.
2. Proactively expands the process working set minimum and maximum quotas via `SetProcessWorkingSetSize` prior to locking:
   ```cpp
   SIZE_T min_ws = resident_bytes + headroom_bytes;
   SIZE_T max_ws = resident_bytes + headroom_bytes + (1ULL << 30);
   SetProcessWorkingSetSize(GetCurrentProcess(), min_ws, max_ws);
   ```
3. Performs `VirtualLock` on the aligned weight arena, securing deterministic memory access without OS paging stutter.

### 3.3 Operating System Latency Optimization

To eliminate thread scheduling jitter and CPU power-state wake-up latency on mobile processors:
- **Multimedia High-Resolution Timer**: Invokes `timeBeginPeriod(1)` to enforce a 1.0 ms thread quantum (replacing the default 15.6 ms Windows scheduler slice).
- **Job Object Process Class**: Executes CPU worker pools under `HIGH_PRIORITY_CLASS` inside Windows Job Objects (`serve/winjob.py`).
- **Worker Spin-Wait (`STRATA_POOL_SPIN_US=50000`)**: Keeps worker threads active for 50 milliseconds post-dispatch, avoiding Zen 2/3 C6 deep sleep state entry and eliminating 20–30 μs wake-up penalties per MoE layer.

---

## 4. Dual-Engine Architecture Matrix

Strata deploys two distinct CUDA binaries to match target model architectures without cross-model performance regressions:

```
                               ┌────────────────────────────────┐
                               │     Strata Server Runtime      │
                               │        (serve/server.py)       │
                               └───────────────┬────────────────┘
                                               │
                       Auto-detect Architecture via GGUF Metadata
                                               │
                        ┌──────────────────────┴──────────────────────┐
                        ▼                                             ▼
          ┌───────────────────────────┐                 ┌───────────────────────────┐
          │     engine/strata.exe     │                 │ engine/strata-qwen36.exe  │
          ├───────────────────────────┤                 ├───────────────────────────┤
          │ Target: Qwen3.8-Flash-Next│                 │ Target: Qwen3.6 / Ornith  │
          │ Total Params: 125B MoE    │                 │ Total Params: 35B MoE     │
          │ 4-Stream Hyper-Connection │                 │ 3x GDN + 1x GQA Attention │
          │ QSA Attention + PLE       │                 │ Pre-RMSNorm Residuals     │
          │ Native TQ1_0 / IQ1_M      │                 │ 256/8+1 MoE + MTP Block   │
          └───────────────────────────┘                 └───────────────────────────┘
```

For complete architectural breakdowns, layer counts, active parameter calculations, and memory footprints, consult the official [`docs/MODELS_MANIFEST.md`](docs/MODELS_MANIFEST.md).

---

## 5. Quickstart

### 5.1 Prerequisites

- **Operating System**: Windows 10/11 64-bit or Linux (x86_64).
- **GPU**: NVIDIA GPU with Compute Capability $\ge 7.5$ (Turing, Ampere, Ada Lovelace, Hopper, Blackwell). Minimum 4 GB VRAM.
- **Host Memory**: 16 GB to 32 GB RAM (DDR4 or DDR5).
- **Runtime Dependencies**: Python 3.10+, CUDA Toolkit 12.x, Visual Studio 2022 C++ Build Tools (for local compilation).

### 5.2 Zero-Copy Storage Setup

Strata eliminates hardcoded external paths via relative NTFS junctions or hard links:

```powershell
# Link existing weights from standard directories into models/ without copying data:
powershell -ExecutionPolicy Bypass -File tools/setup_models_directory.ps1
```

### 5.3 Launching the Engine

Launch Strata using the unified Python server with the appropriate configuration profile:

```powershell
# Launch Ornith-1.5-35B-A3B:
.venv\Scripts\python.exe serve/server.py --config configs/strata-ornith.json --port 8080

# Launch Qwen3.6-35B-A3B (UDT with MTP):
.venv\Scripts\python.exe serve/server.py --config configs/strata-qwen36.json --port 8080

# Launch Qwen3.8-Flash-Next (125B MoE Ternary TQ1_0):
.venv\Scripts\python.exe serve/server.py --config configs/strata-tq1_0.json --port 8080
```

---

## 6. Configuration Specification

Configuration profiles are structured in JSON format. Below is an annotated specification based on `configs/strata-ornith.json`:

```json
{
  "exe": "engine/strata-qwen36.exe",
  "args": [
    "--native", "models/Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf",
    "--max-context", "131072",
    "--kv", "int8",
    "--kv-unified",
    "--spec-k", "6"
  ],
  "cwd": ".",
  "tokenizer": "../Strata-data/packs/coder-iq1_m/tokenizer",
  "model_name": "ornith-1.5-35b-a3b",
  "aliases": [
    "default",
    "ornith-1.5-35b",
    "ornith-1.5-35b-a3b"
  ],
  "log": "logs/strata-ornith.log",
  "port": 8080,
  "gpu": 0,
  "sampling": {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.05,
    "presence_penalty": 0.2,
    "repetition_penalty": 1.05,
    "penalty_last_n": 256
  },
  "env": {
    "STRATA_ARENA_LOCK": "1",
    "STRATA_RESIDENT_HEADROOM_GIB": "2.0",
    "Q36_EXPERT_CACHE": "512"
  }
}
```

### Key Parameter Definitions
- `exe`: Relative path to target engine binary (`engine/strata.exe` or `engine/strata-qwen36.exe`).
- `args`: Command-line parameters forwarded to the engine core (`--native`, `--max-context`, `--kv`, `--spec-k`).
- `STRATA_ARENA_LOCK`: Enables Windows `VirtualLock` on system RAM allocations to prevent page faults.
- `STRATA_RESIDENT_HEADROOM_GIB`: Safety margin reserved for OS operations and graphics driver structures.
- `Q36_EXPERT_CACHE`: Number of dynamic expert cache slots allocated in physical VRAM (default: 512).

---

## 7. REST API Reference

Strata exposes an OpenAI-compatible HTTP interface on port 8080 (default).

### 7.1 Chat Completions (`POST /v1/chat/completions`)

#### Request
```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "system", "content": "You are a systems programming specialist."},
      {"role": "user", "content": "Explain lock-free ring buffers in C++."}
    ],
    "temperature": 0.6,
    "max_tokens": 1024,
    "stream": false
  }'
```

#### Response
```json
{
  "id": "chatcmpl-strata-001",
  "object": "chat.completion",
  "created": 1728212400,
  "model": "ornith-1.5-35b-a3b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "A lock-free ring buffer relies on atomic sequence counters...",
        "reasoning_content": "The user is inquiring about concurrent data structures..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 28,
    "completion_tokens": 412,
    "total_tokens": 440
  }
}
```

### 7.2 Anthropic Messages Protocol (`POST /v1/messages`)

Strata natively supports the Anthropic Messages protocol for agentic frameworks (such as Claude Code):

```bash
curl -X POST http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Generate a CMake configuration."}]
  }'
```

### 7.3 Telemetry and Management Endpoints

- **`GET /health`**: Returns engine status, resident model, and memory state.
- **`GET /metrics`**: Exposes Prometheus-compatible performance telemetry (decode rate, prefill rate, cache hit ratio, MTP acceptance rate).
- **`POST /load`**: Dynamically switches the active model configuration without restarting the HTTP listener.
- **`POST /unload`**: Releases VRAM and unlocked RAM resources back to the operating system.

---

## 8. Model Context Protocol (MCP) Server

Strata integrates a native Model Context Protocol (MCP) server (`tools/strata_mcp.py`) supporting stdio transport. This enables AI agent environments (such as Antigravity, Claude Code, and Cursor) to monitor telemetry, execute benchmarks, and manage model lifecycle autonomously.

### Client Configuration (`mcp_config.json`)
```json
{
  "mcpServers": {
    "strata": {
      "command": "python",
      "args": [
        "c:/Users/Ilia V/Documents/antigravity/calm-noether/Strata/tools/strata_mcp.py"
      ],
      "env": {
        "STRATA_PORT": "8080"
      }
    }
  }
}
```

---

## 9. Verification and Repository Hygiene

To verify model loading and end-to-end inference integrity:

```powershell
# Run the automated end-to-end verification harness:
.venv\Scripts\python.exe tools/verify_model.py configs/strata-ornith.json
```

---

## 10. License and Citations

Strata core engine is distributed under the MIT License. Model weights are subject to their respective upstream licenses:
- **Ornith-1.5-35B-A3B**: MIT License ([ornith-ai](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B))
- **Qwen3.6-35B-A3B**: Apache 2.0 / Qwen License ([Qwen](https://huggingface.co/Qwen))
- **Qwen3.8-Flash-Next**: Apache 2.0 / Qwen License ([Qwen](https://huggingface.co/Qwen))

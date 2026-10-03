# StrataRealLowVRAM v0.1.36 — Extreme Low-VRAM & Mobile GPU Edition

> **Empirically verified on NVIDIA GeForce RTX 3050 Laptop (4 GB GDDR6) + AMD Ryzen 5 5500U + 32 GB DDR4 on Windows 11.**  
> Full **65,536 context** (`--max-context 65536`) with zero memory eviction or WDDM driver clamps.

---

## ⚡ Highlights of Release v0.1.36

This release combines official **Upstream Engine v0.1.36** with custom **StrataRealLowVRAM** low-VRAM optimizations, pre-tuned model configurations, and developer tools.

### 🚀 Performance Gains & Hardware Benchmarks (RTX 3050 Laptop 4GB)

| Metric / Scenario | Vanilla Upstream / Previous | StrataRealLowVRAM v0.1.36 | Improvement |
| :--- | :---: | :---: | :---: |
| **Prefill Standalone Fused MoE (Q2_0)** | 42.2 ms / layer | **15.3 ms / layer** | 🚀 **2.75x faster** |
| **Prefill Standalone Fused MoE (IQ packs)** | 22.0 – 35.0 ms | **16.0 – 21.0 ms** | 🚀 **1.2x – 1.6x faster** |
| **Chat Turn TTFT (Time to First Token)** | 20.0s – 50.0s (disk spill) | **0.04s (40 ms!)** | ⚡ **Instant response** |
| **Qwen3.8 Full Q2_0 Decode** | 3.30 tok/s | **4.80 – 5.00 tok/s** (peak 5.20) | 🚀 **+51.5%** |
| **Qwen3.8 Coder IQ1_M Decode** | 3.48 tok/s | **4.03 – 4.20 tok/s** | 🚀 **+20.7%** |
| **VRAM Expert Cache Residency** | 0 slots (evicted by WDDM) | **700 slots (Q2_0) / 600 slots (Coder)** | 🛡️ **Stable WDDM Residency** |

---

## 🛠️ Upstream v0.1.36 Core Upgrades Included

1. **Fused Int8 Tensor-Core Prefill Kernels (`moe_fused.cu`, `moe_fused_iq.cu`):**
   - Implements hardware-accelerated `mma.sync.aligned.m16n8k32.s8.s8` on SM 8.0+ (Ampere / Ada / Hopper / Blackwell).
   - In-register fused gate & up projection dequantization + SwiGLU activation + down projection without round-tripping intermediate activations through global VRAM.
   - GPU-side prefix-sum token grouping eliminates host CPU dispatch overhead.
   - Streamed ring buffer drastically shrinks MoE scratch buffers (~100 KB/token).
   - Enable via `STRATA_PF_FUSED=1` (activated by default in our configs).

2. **Decode Optimizations & Cluster Parity:**
   - Thread-Block Clusters in decode for `sm_90+` architectures with seamless fallback to high-speed native kernels on SM 8.6 (RTX 3050).
   - Expert Cache Persistence (`--expert-profile-save`, #477): saves learned cache routing into `expert-profile-learned.bin`, eliminating cold-start latency across restarts.
   - Speculative draft vocabulary pruning (`--draft-vocab cyrillic` / `en`) saving ~110 MiB VRAM for draft heads on low-VRAM GPUs.

---

## 🛡️ StrataRealLowVRAM Custom Optimizations

1. **WDDM 4GB Expert Cache Patch (`patched: true`):**
   - Resolves the Windows WDDM driver clamp where transient memory allocations falsely zeroed out GPU cache slots.
   - Preserves 700 active GPU slots (Q2_0) and 600 active GPU slots (Coder) on 4GB VRAM cards.
2. **Optimal RAM Resident Budgets:**
   - Configured `--resident-budget-gib 20` for Coder IQ1_M and Q2_0, avoiding heavy SSD I/O stalls during batched prefill.
3. **Adaptive Short-Read Routing (`--short-read 64`):**
   - Automatically processes interactive chat messages inside the fast verify window, slashing TTFT from 40s to 40ms.
4. **Antigravity Model Context Protocol (MCP) Server:**
   - Pre-packaged stdio MCP server (`tools/strata_mcp.py`) and CLI bridge (`tools/mcp_cli.py`) for AI assistants (Antigravity, Claude Code, Cursor).

---

## 📦 Release Assets

- **`StrataRealLowVRAM-v0.1.36-full-windows-x64.zip`**: Complete ready-to-run distribution bundle. Extract and double-click `START-HERE.bat` or `run-q2_0.bat`.
- **`StrataRealLowVRAM-v0.1.36-engine-windows-x64.zip`**: Precompiled, WDDM-patched `strata.exe` and `BUILD.json` drop-in replacement for existing installations.

---

## 🏃 Quickstart

1. Download and extract **`StrataRealLowVRAM-v0.1.36-full-windows-x64.zip`**.
2. Run `START-HERE.bat` to verify your environment.
3. Start high-speed inference:
   - For Full Q2_0: run `run-q2_0.bat` (or `.venv\Scripts\python.exe serve/server.py --config strata-q2_0.json`)
   - For Coder IQ1_M: run `run-coder-iq1_m.bat` (or `.venv\Scripts\python.exe serve/server.py --config strata-coder-iq1_m.json`)
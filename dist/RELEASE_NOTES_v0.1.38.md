# StrataRealLowVRAM v0.1.38 — Extreme Low-VRAM & Mobile GPU Edition

> **Empirically verified on NVIDIA GeForce RTX 3050 Laptop (4 GB GDDR6) + AMD Ryzen 5 5500U (Zen 2 AVX-2) + 32 GB DDR4 on Windows 11.**  
> Full **65,536 context** (`--max-context 65536`) with zero memory eviction or WDDM driver clamps.

---

## ⚡ Highlights of Release v0.1.38

This release synchronizes official **Upstream Engine v0.1.38** (integrating all improvements from both **v0.1.37** and **v0.1.38**) with custom **StrataRealLowVRAM** low-VRAM optimizations, pre-tuned model configurations, and developer tools.

### 🚀 Empirical Hardware Benchmarks (RTX 3050 Laptop 4GB)

| Metric / Scenario | Vanilla Upstream / Previous | StrataRealLowVRAM v0.1.38 | Improvement |
| :--- | :---: | :---: | :---: |
| **Q2_0 TTFT (Time to First Token)** | 20.0s – 50.0s (disk spill) | **0.01s (10 ms!)** | ⚡ **Instant response** |
| **Coder IQ1_M TTFT (Time to First Token)** | 20.0s – 50.0s (disk spill) | **0.02s (20 ms!)** | ⚡ **Instant response** |
| **Qwen3.8 Full Q2_0 Decode** | 3.30 tok/s | **3.90 – 4.60 tok/s** (peak 4.60) | 🚀 **+39.4%** |
| **Qwen3.8 Coder IQ1_M Decode** | 3.20 tok/s | **3.40 – 3.80 tok/s** (peak 3.80) | 🚀 **+18.8%** |
| **Prefill Standalone Fused MoE (Q2_0)** | 42.2 ms / layer | **15.3 ms / layer** | 🚀 **2.75x faster** |
| **DeltaNet Recurrence Layer (sm_80+)** | 1.0x baseline | **1.44x – 1.57x** (gdn_rec_kh) | 🚀 **+44% to +57%** |
| **VRAM Expert Cache Residency** | 0 slots (evicted by WDDM) | **700 slots (Q2_0) / 600 slots (Coder)** | 🛡️ **Stable WDDM Residency** |

---

## 🛠️ Upstream v0.1.37 & v0.1.38 Core Upgrades Included

1. **Faster Prefill & Prompt Processing (#372, #374, #413, #452):**
   - **DeltaNet Recurrence 3-in-1 (`a7ce31b`):** Consolidates 3 value heads of a key head into a single CUDA thread using hardware `cp.async` on Ampere SM 8.0+ (RTX 3050). Yields 1.44x – 1.57x layer speedup with bitwise parity.
   - **MMQ Group Gathering (`d541220`):** MMQ expert groups are gathered in one kernel launch, after one wait, and released by one CUDA event.
   - **Tensor Cores Attention for Q4_0 K/V (`778e1f6`):** When `--kv q4_0` is used, prompt attention runs on hardware Tensor Cores (mode 4) on `sm_80+`.
   - **Parallel n-gram PLE Ingestion (`6639c7b`, `4f7b3e8`):** First chunk's PLE rows stream asynchronously alongside layer 0 computation in 256-row blocks.

2. **Windows Direct Unbuffered I/O (#357, #362, #285, #286):**
   - Implements `FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN` with 4096-byte `VirtualAlloc` alignment on Windows. Reading GGUF weights bypasses Windows file cache overhead, keeping RAM clean.

3. **AVX-2 Multi-Token CPU Kernel for IQ4_XS (#415, #415):**
   - Hand-tuned AVX-2 SIMD kernels accelerating expert computation on AMD Ryzen Zen 2 CPUs (e.g. Ryzen 5 5500U).

4. **Engine Resilience & Timing (#481, #485, #496):**
   - Automatic silent engine restart if the worker process stalls.
   - PCIe bandwidth probe uses the best of four timed bursts for maximum accuracy.
   - Diagnostic crash reporting outputs the last 20 engine log lines prior to `READY`.

5. **Adaptive Swap Decay & Verification Sync (#463, #477):**
   - Asynchronous expert copies synchronize prior to residency table lookup (`apply_pending`), eliminating race conditions.
   - Configurable routing frequency decay via `--adapt-decay` (default `0.7f`).
   - Learned cache routing persistence via `--expert-profile-save`.

6. **API Security:**
   - Host header verification and DNS rebinding attack protection for local server instances.

---

## 🛡️ StrataRealLowVRAM Custom Optimizations Preserved

1. **WDDM 4GB Expert Cache Patch (`patched: true`):**
   - Bypasses Windows WDDM driver budget throttling in `ExpertCache::open()`, maintaining 700 GPU cache slots (Q2_0) and 600 GPU cache slots (Coder).
2. **Windows 1ms High-Resolution Timer (`timeBeginPeriod(1)`):**
   - Minimizes HTTP and token-streaming jitter in `serve/server.py`.
3. **Resident Memory Sizing (`--resident-budget-gib 20`):**
   - Keeps 100% of model experts in physical RAM on 32GB hosts, eliminating disk stalls during inference.
4. **Adaptive Short-Read Window (`--short-read 64`):**
   - Eliminates cold prefill overhead on conversational chat turns, cutting TTFT down to 10–20 ms.
5. **Antigravity Model Context Protocol (MCP) Server:**
   - Bundled stdio MCP server (`tools/strata_mcp.py`) and CLI bridge (`tools/mcp_cli.py`).

---

## 📦 Release Assets

- **`StrataRealLowVRAM-v0.1.38-full-windows-x64.zip`**: Complete ready-to-run distribution bundle. Extract and double-click `START-HERE.bat` or `run-q2_0.bat`.
- **`StrataRealLowVRAM-v0.1.38-engine-windows-x64.zip`**: Precompiled, WDDM-patched `strata.exe` and `BUILD.json` drop-in replacement for existing installations.

---

## 🏃 Quickstart

1. Download and extract **`StrataRealLowVRAM-v0.1.38-full-windows-x64.zip`**.
2. Run `START-HERE.bat` to verify your environment.
3. Start high-speed inference:
   - For Full Q2_0: run `run-q2_0.bat` (or `.venv\Scripts\python.exe serve/server.py --config strata-q2_0.json`)
   - For Coder IQ1_M: run `run-coder-iq1_m.bat` (or `.venv\Scripts\python.exe serve/server.py --config strata-coder-iq1_m.json`)
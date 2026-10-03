# StrataRealLowVRAM v0.1.35 — Extreme Low-VRAM & Mobile GPU Edition

> **Empirically verified on NVIDIA GeForce RTX 3050 Laptop (4 GB GDDR6) + AMD Ryzen 5 5500U + 32 GB DDR4 on Windows 11.**  
> Full **65,536 context** (`--max-context 65536`) with zero memory eviction or WDDM driver clamps.

---

## ⚡ Highlights of This Release

This release synchronizes official **Upstream Engine v0.1.35** with our custom **StrataRealLowVRAM** low-VRAM optimizations, pre-tuned model configurations, and developer tools.

### 🚀 Empirical Hardware Benchmarks (RTX 3050 Laptop 4GB)

| Model | Vanilla Upstream | StrataRealLowVRAM Champion | Speedup | VRAM Cache Slots | RAM Usage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen3.8 Full Q2_0** | 3.30 tok/s | **4.80 – 5.00 tok/s** (peak 5.20) | 🚀 **+51.5%** | **700 slots** (922 MiB) | 16.0 GiB (Frees 14 GB) |
| **Qwen3.8 Coder IQ1_M** | 3.48 tok/s | **4.03 – 4.20 tok/s** | 🚀 **+20.7%** | **600 slots** (1.15 GiB) | 20.0 GiB |

---

## 🛠️ StrataRealLowVRAM Key Features in v0.1.35

1. **WDDM 4GB Expert Cache Patch (`patched: true`):**
   - Eliminates Windows WDDM driver clamp where prior memory overcommit falsely zeroes out the expert cache.
   - Restores full 700-slot (Q2_0) and 600-slot (Coder) GPU residency on 4GB cards.
2. **Upstream v0.1.35 Integration:**
   - **Fix #467 (Windows 32GB Working Set Trimming):** Restores +1.86 GiB of available physical RAM on startup via `SetProcessWorkingSetSize`, allowing the full resident cache complement to lock cleanly in RAM.
   - **Fix #448 (Multi-GPU Prompt Chunking):** Intelligent handling of asymmetric VRAM cards.
   - **Fix #460 (JSON API Resilience):** Graceful decoding of double-encoded message strings.
   - **Fix #457 (Speculative Metrics):** Cumulative `/metrics` counters for draft acceptance.
   - **Fix #459 (Atomic Configs):** Safe atomic config writes via temporary files.
3. **Pre-Tuned Champion Configurations:**
   - `strata-q2_0.json`: 700 expert cache slots, 6 pool workers, adaptive swaps 4/8, Direct I/O queue depth 256, prefill chunk 2048.
   - `strata-coder-iq1_m.json`: 600 expert cache slots, 6 pool workers, adaptive swaps 4/8, Direct I/O queue depth 256, prefill chunk 2048.
4. **Antigravity Model Context Protocol (MCP) Server:**
   - Official stdio MCP server (`tools/strata_mcp.py`) and CLI bridge (`tools/mcp_cli.py`) for AI assistants (Antigravity, Claude Code, Cursor).
5. **Engine Auto-Update Shield:**
   - `setup.py` protects custom compiled and patched local binaries from being overwritten by vanilla upstream downloads.

---

## 📦 Release Assets

- **`StrataRealLowVRAM-v0.1.35-full-windows-x64.zip`**: Complete ready-to-run installation bundle. Just extract and double-click `START-HERE.bat` or `run-q2_0.bat`.
- **`StrataRealLowVRAM-v0.1.35-engine-windows-x64.zip`**: Precompiled, WDDM-patched `strata.exe` and `BUILD.json` drop-in replacement for existing installations.

---

## 🏃 Quickstart

1. Download and extract **`StrataRealLowVRAM-v0.1.35-full-windows-x64.zip`**.
2. Run `START-HERE.bat` to verify your environment.
3. Start high-speed inference with pre-tuned configs:
   - For Full Q2_0: run `run-q2_0.bat` (or `.venv\Scripts\python.exe serve/server.py --config strata-q2_0.json`)
   - For Coder IQ1_M: run `run-coder-iq1_m.bat` (or `.venv\Scripts\python.exe serve/server.py --config strata-coder-iq1_m.json`)
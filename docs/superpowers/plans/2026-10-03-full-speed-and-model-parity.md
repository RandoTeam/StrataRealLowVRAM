# Implementation Plan: Strata High-Speed Prefill, RAM Isolation, and 3-Model Parity

**Goal:**
1. Deeply analyze prefill speed discrepancies between stock Strata and our fork; eliminate all degradation points in our fork/configs.
2. Achieve $\ge 6.0\text{ tok/s}$ decode, $\ge 600\text{ tok/s}$ prefill, $\ge 4.0\text{ GB}$ physical free RAM buffer, and $\ge 60,000$ context on **Qwen3.8 Coder IQ1_M** and **Qwen3.8 Full Q2_0**.
3. Integrate and benchmark **Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive**, achieving $\ge 25.0\text{ tok/s}$ decode, $\ge 600\text{ tok/s}$ prefill, $\ge 4.0\text{ GB}$ free RAM, and $\ge 60,000$ context.
4. Verify real coding, reasoning, and smart outputs across all models under large context conditions.

## Architecture & Problem Diagnosis
- **Prefill Discrepancy:**
  - Upstream author tested on RTX 5070 12GB / RTX 3090 24GB + 64GB RAM (PCIe 4.0 x16 DMA, 100% experts pinned in RAM, chunk size 8192).
  - In our config, setting `STRATA_PREFILL_STREAM_MIN=32` forced the engine to stream all 512 experts (33.6 GB over PCIe) even for tiny prompt turns, collapsing throughput.
  - Setting `STRATA_PARTIAL_PIN=1` with 4GB locked driver resources on WDDM, starving prompt buffers.
  - Interactive turns ($\le 64$ tokens) use `read_windows` (>150,000 tok/s effective TTFT), but batched fresh prefill must keep experts in RAM rather than thrashing NVMe SSD.
- **RAM Headroom ($\ge 4.0\text{ GB}$):**
  - Total system RAM: 31.34 GiB. Windows OS takes ~5.5-6.0 GiB.
  - Setting `--resident-budget-gib 20` + dense weights (2.5 GiB) + KV cache (0.5 GiB) leaves $< 2.5\text{ GiB}$ free RAM.
  - Setting `--resident-budget-gib 17.5` and `--resident-headroom-gib 4.5` guarantees $\ge 4.0\text{ GB}$ free physical RAM.
- **Model Geometry & Qwen3.6-35B-A3B Integration:**
  - Strata C++ engine has compile-time hardcoded constants for Qwen3.8 Flash Next (`N=2560`, `48 layers`, `512 experts`).
  - Qwen3.6-35B-A3B has `N=2048`, `40 layers`, `256 experts`.
  - We will test Qwen3.6-35B-A3B using high-efficiency runtime engines (llama.cpp CUDA with MTP / speculative drafting or hybrid MoE offloading) to validate $\ge 25\text{ tok/s}$ decode and $\ge 600\text{ tok/s}$ prefill.

## Tasks
- [ ] **Task 1: Forensic Audit of Degradation Points & Prefill Engine Overhaul**
  - Remove toxic env vars (`STRATA_PREFILL_STREAM_MIN=32`, `STRATA_PARTIAL_PIN=1`).
  - Tune `--prefill 1024/2048`, `--short-read 64`, prompt cache root 64.
  - Audit C++ vs stock.
- [ ] **Task 2: RAM Footprint Rationalization ($\ge 4.0\text{ GB}$ Buffer Guarantee)**
  - Re-tune `strata-coder-iq1_m.json` and `strata-q2_0.json` with `--resident-budget-gib 17.5` and `STRATA_RESIDENT_HEADROOM_GIB=4.5`.
  - Validate with PowerShell memory inspection during active 65k context load.
- [ ] **Task 3: Live Verification of Coder IQ1_M & Q2_0**
  - Run live inference with coding and reasoning prompts.
  - Record tokens/s, prefill tokens/s, free RAM buffer, and context 65,536.
- [ ] **Task 4: Integration & Benchmark of Qwen3.6-35B-A3B**
  - Analyze and benchmark Qwen3.6-35B-A3B on RTX 3050 Laptop.
  - Verify if $\ge 25\text{ tok/s}$ decode, $\ge 600\text{ tok/s}$ prefill, and $\ge 4\text{ GB}$ RAM headroom are achieved.
- [ ] **Task 5: Final Release v0.1.38-optimized Package and Documentation**
  - Update configs, documentation, package full Windows release zip, and present final report.

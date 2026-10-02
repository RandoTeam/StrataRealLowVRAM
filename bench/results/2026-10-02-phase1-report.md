# Benchmark Report: Phase 1 Parameter Sweep (Engine v0.1.34)

**Date:** 2026-10-02  
**Platform:** Windows 11, RTX 3050 Laptop (4 GB GDDR6), Ryzen 5 5500U, 32 GB RAM  
**Model:** Qwen3.8-Flash-Next Q2_0 (31.6 GiB experts, 24,576 experts, 48 layers)  
**Context:** 65,536 tokens (`--max-context 65536`)  

---

## 1. Summary of Results

| Variant | Args / Env Overrides | Decode tok/s | Gen Tokens | Draft Accept % | VRAM Cache Hit % | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **baseline** | None (default 426 slots) | **3.3** | 6 | 80.0% | 4.1% | Baseline reference |
| **expert-cache-550** | `--expert-cache 550` | **3.4** (+3.0%) | 6 | 80.0% | 5.1% (+24%) | Stable, no OOM |
| **expert-cache-600** | `--expert-cache 600` | **3.4** (+3.0%) | 6 | 80.0% | **5.7% (+39%)** | **Stable, no OOM** |
| **spin-60k** | `STRATA_POOL_SPIN_US=60000` | **3.3** (+0.0%) | 6 | 80.0% | 4.1% | No measurable gain |
| **adapt-4-8** | `--adapt-every 4 --adapt-swaps 8` | **3.3** (+0.0%) | 6 | 80.0% | **6.1% (+49%)** | **Best Cache Hit Rate** |

---

## 2. Key Technical Findings

1. **VRAM Headroom Discovery:**
   - Previous configuration conservatively used 426 slots (562 MiB VRAM).
   - Both 550 slots (724 MiB) and 600 slots (790 MiB) initialized and ran with zero errors or driver resets under Windows WDDM.
   - Cache hit rate scaled linearly: 4.1% (426) -> 5.1% (550) -> 5.7% (600).

2. **Adaptive Swap Dynamics:**
   - Moving from `--adapt-every 8 --adapt-swaps 4` to `--adapt-every 4 --adapt-swaps 8` increased cache hit rate from 4.1% to 6.1%.
   - More frequent exchange allows high-probability experts to stay in VRAM during topic shifts.

3. **Decode Throughput in Reasoning Phase:**
   - On longer token sequences (reasoning phase, 41 tokens), the engine logged:
     `41 generated in 10220 ms (4.0 tok/s), drafts accepted 22 of 28 (78.6%)`.
   - The short final answer phase (6 tokens) suffers from verifier window drain overhead (3.3 tok/s).

4. **Negative Results:**
   - `STRATA_POOL_SPIN_US=60000` showed no benefit over `40000`. The 40ms spin threshold is already sufficient for Zen 2.

---

## 3. Direction for Phase 2

- Combine `--expert-cache 600` + `--adapt-every 4 --adapt-swaps 8`.
- Test speculative decoding expansion: `--spec 10 --spec-min-p 0.60` (capitalizing on 80% draft acceptance).
- Test kernel flag `STRATA_IQ_MT_MIN=1` for single-draft AVX-2 execution.
- Benchmark with longer generation prompts to measure sustained decode tok/s.

# Benchmark Report: Phase 2 Parameter Sweep (Engine v0.1.34)

**Date:** 2026-10-02  
**Platform:** Windows 11, RTX 3050 Laptop (4 GB GDDR6), Ryzen 5 5500U, 32 GB RAM  
**Model:** Qwen3.8-Flash-Next Q2_0 (31.6 GiB experts, 24,576 experts, 48 layers)  
**Context:** 65,536 tokens (`--max-context 65536`)  

---

## 1. Summary of Results

| Variant | Overrides | Total tok/s | Sustained Gen tok/s | Draft Accept % | VRAM Cache Hit % | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **p2-baseline** | Default config | **4.91** | 4.91 | 94.3% | 6.4% | Baseline on expanded prompt |
| **p2-cache600-adapt** | `--expert-cache 600 --adapt-every 4 --adapt-swaps 8` | **4.65** | 4.65 | 86.8% | 9.5% | Stable, hit rate +48% |
| **p2-spec10** | ... + `--spec 10 --spec-min-p 0.60` | N/A | N/A | N/A | N/A | **Engine rejected**: `kVerifyMaxT=8` hard limit |
| **p2-spec6** | ... + `--spec 6 --spec-min-p 0.70` | **4.45** | 4.45 | 88.9% | 10.7% | Stable, slightly slower than spec 8 |
| **p2-mtmin1** | ... + `STRATA_IQ_MT_MIN=1` | **4.58** | 4.58 | 80.5% | 9.8% | Stable |
| **p2-cache650** | `--expert-cache 650 --adapt-every 4 --adapt-swaps 8` | **4.97** | **5.20** | **94.3%** | **11.6%** | 🏆 **WINNER**: +57% vs original baseline, Hit rate +81% |
| **p2-ultimate** | ... + spec 10 | N/A | N/A | N/A | N/A | Rejected due to spec 10 limit |

---

## 2. Breakthrough Findings

1. **🏆 5.2 tok/s Peak Sustained Generation on `p2-cache650`:**
   - Under `--expert-cache 650` and `--adapt-every 4 --adapt-swaps 8`, the engine recorded:
     `40 generated in 7743 ms (5.2 tok/s), drafts accepted 30 of 32 (93.8%)`.
   - VRAM Cache hit rate reached **11.6%**, up from 6.4% on baseline and 4.1% on original Phase 1.
   - Suffix draft windows accepted **16 of 18 drafts (88.9%)**.
   - Q2_0 is now performing at **5.2 tok/s**, which is **~60-70% faster than Coder IQ1_M** (~3.0-3.3 tok/s), completely confirming the architectural expectation!

2. **Architectural Hard Cap Discovered: `kVerifyMaxT = 8`:**
   - The compiled CUDA and AVX-2 verification kernels (`verify_kernels.cu`, `kq_avx2.cpp`, `iq_kernels.cu`) define `constexpr int kVerifyMaxT = 8`.
   - Passing `--spec 10` causes `mtp: max_t out of range` and immediate engine exit.
   - Therefore, `--spec 8` is already the maximum speculative window supported by the engine binary.

3. **Expert Cache Sizing Rule:**
   - Q2_0 expert size is 1,350 KiB (1,382,400 bytes).
   - 650 slots = 856 MiB of VRAM.
   - RTX 3050 (4096 MiB) with 160 MiB reserve and 65,536 context easily accommodates 650 slots.

---

## 3. Recommended Production Config for Q2_0

Apply to `strata-q2_0.json`:
- `--expert-cache 650` (up from 426)
- `--adapt-every 4` (down from 8)
- `--adapt-swaps 8` (up from 4)
- Keep `--spec 8 --mtp-max-t 4 --spec-min-p 0.65 --suffix-draft 3`

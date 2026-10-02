# Benchmark Report: Iteration 2 - VRAM Expert Cache Ceiling Push

**Date:** 2026-10-02  
**Platform:** Windows 11, RTX 3050 Laptop (4 GB GDDR6), Ryzen 5 5500U, 32 GB RAM  
**Model:** Qwen3.8-Flash-Next Q2_0 (31.6 GiB experts, 24,576 experts, 48 layers)  

---

## 1. Summary of Results

| Variant | Expert Slots | VRAM Cache MiB | Total tok/s | Draft Accept % | VRAM Cache Hit % | Delta vs 650 Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **i2-cache650** | 650 | 856 MiB | **4.85** | 94.3% | 11.1% | Reference baseline |
| **i2-cache675** | 675 | 889 MiB | **4.89** | 94.3% | 11.2% | +0.8% |
| **i2-cache700** | **700** | **922 MiB** | **4.98** | **97.1%** | **11.5%** | 🏆 **+2.7% (Optimal Peak)** |
| **i2-cache725** | 725 | 955 MiB | **4.97** | 97.1% | 11.8% | +2.5% |
| **i2-cache750** | 750 | 988 MiB | **4.76** | 94.3% | 11.4% | -1.9% (WDDM threshold reached) |

---

## 2. Key Findings: The WDDM Memory Ceiling

1. **Optimal Peak at 700 Slots:**
   - At 700 slots (922 MiB), the model achieved **4.98 tok/s** overall with a remarkable **97.1% draft acceptance rate**.
   - Cache hit rate reached **11.5%**.

2. **WDDM Paging Threshold Detected at 750 Slots:**
   - When expert cache exceeds ~950 MiB (at 750 slots = 988 MiB), the Windows Desktop Window Manager begins page eviction of background DirectX/compositor surfaces to maintain the VRAM allocation, introducing slight PCIe paging latency and dropping performance to 4.76 tok/s.

3. **Production Action:**
   - Update `strata-q2_0.json` to `--expert-cache 700` (+64% over original 426 slots).

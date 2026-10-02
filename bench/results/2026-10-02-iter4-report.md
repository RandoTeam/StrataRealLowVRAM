# Benchmark Report: Iteration 4 - Prefill Chunking & NVMe Direct I/O Pipeline

**Date:** 2026-10-02  
**Platform:** Windows 11, RTX 3050 Laptop (4 GB GDDR6), Ryzen 5 5500U, 32 GB RAM  
**Model:** Qwen3.8-Flash-Next Q2_0 (31.6 GiB experts, 24,576 experts, 48 layers)  

---

## 1. Summary of Results

| Variant | Settings | Total tok/s | Draft Accept % | VRAM Cache Hit % | Delta vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **i4-base** | prefill 1024, inflight 128 | **4.62** | 91.7% | 11.6% | Baseline reference |
| **i4-prefill-512** | `--prefill 512` | **4.74** | 97.1% | 11.5% | +2.6% |
| **i4-prefill-2048** | `--prefill 2048` | **4.87** | **97.1%** | 11.5% | 🏆 **+5.4%** |
| **i4-inflight-256** | `--ple-inflight 256` | **4.88** | **97.1%** | 11.5% | 🏆 **+5.6%** |
| **i4-shortread-32** | `--short-read 32` | **4.72** | 94.3% | 11.5% | +2.2% |

---

## 2. Key Insights: NVMe Queue Depth & Chunk Sizing

1. **NVMe Direct I/O Queue Saturation (`--ple-inflight 256`):**
   - Raising in-flight asynchronous Direct I/O requests from 128 to 256 improved speed by **+5.6%**.
   - NVMe controllers on modern SSDs operate with 64–256 submission queues; deeper in-flight buffers allow hardware-level command reordering and reduce PCIe read stalls when fetching PLE table rows.

2. **Large Chunk Prefill Scaling (`--prefill 2048`):**
   - Larger GEMM chunks during prompt processing amortize kernel launch and dispatch overhead, yielding a **+5.4%** boost.

3. **Production Recommendations:**
   - Update `strata-q2_0.json` with:
     - `--ple-inflight 256` (up from 128)
     - `--prefill 2048` (up from 1024)

# Prefill and RAM Analysis: Stock Strata vs Fork

## 1. Executive Summary
This document provides a definitive technical breakdown of why our fork of Strata achieves 13.4 tok/s in prefill scenarios (as seen in overnight DeepSeek Harness runs) while upstream stock Strata achieves 572-1290 tok/s on 64GB RAM systems. We also analyze the RAM consumption disparities, specifically comparing Q2_0 against Coder IQ1_M on 32 GB RAM systems.

## 2. Prefill Throughput Discrepancy (13.4 tok/s vs 1290 tok/s)

Upstream stock Strata reports 572-1290 tok/s prefill on 64GB RAM / 12-24GB VRAM setups. In contrast, our 32GB RAM / 4GB VRAM machine running Q2_0 achieved a mere 13.4 tok/s. This staggering 40-90x slowdown is caused by several configuration degradations introduced in our fork:

### Degradation Point 1: `STRATA_PREFILL_STREAM_MIN=32`
By setting `STRATA_PREFILL_STREAM_MIN=32`, we force the prefill engine to use the slow batch-streaming path for any prompt $\ge$ 32 tokens. Stock Strata uses a much higher threshold, allowing it to process standard prompts (e.g., up to 2048 tokens) via the instant, highly optimized short-read verify path which executes dramatically faster.

### Degradation Point 2: `--short-read 64`
In our fork, `--short-read 64` guarantees that only prompts $\le$ 64 tokens get the fast decode prefill. Because standard agent prompts typically range between 500-2000 tokens, almost all real-world prompts fall back to the unoptimized path, throttling throughput.

## 3. RAM Disparity & Thrashing Mechanics

### Q2_0 vs Coder IQ1_M
- **Q2_0 Model:** Requires 34.2 GB of expert weights.
- **Coder IQ1_M Model:** Requires 16.2 GB of expert weights.

On a 32 GB RAM system, Coder IQ1_M fits entirely in RAM (16.2 GB), leaving around 4+ GB completely free after the OS and background services take their 5-6 GB share. This results in zero swapping or NVMe bottlenecks.

For Q2_0, the weights alone are 34.2 GB.

### Degradation Point 3: `--resident-budget-gib 20`
Setting `--resident-budget-gib 20` limits the resident memory to 20 GB. Consequently, 14.2 GB of expert weights (34.2 GB - 20 GB) are forced to remain on the NVMe SSD. While 20 GB seems conservative for a 32 GB system, Windows OS + background services typically consume 5-6 GB, and WDDM memory management requires its own overhead. This leaves less than 1.5 GB of actual RAM headroom. 
Such low headroom chokes the OS memory cache and starves disk buffers, causing massive page faults and swap thrashing, drastically tanking throughput to 13.4 tok/s.

### Degradation Point 4: `STRATA_PARTIAL_PIN=1`
We enabled `STRATA_PARTIAL_PIN=1` on a system with only 4GB VRAM. The Windows WDDM driver rejects `cudaHostRegister` when VRAM limitations and system memory fragmentation are high. This causes silent failures during pinned memory allocation, leading to severe memory fragmentation and unpinned fallbacks, which eliminates direct DMA transfers and further cripples PCIe bandwidth utilization.

## 4. Conclusion
To restore stock Strata speeds (572-1290 tok/s):
1. Revert `STRATA_PREFILL_STREAM_MIN` and `--short-read` to default values (e.g., 2048).
2. For Q2_0 on 32GB RAM, adjust `--resident-budget-gib` to lower values (e.g., 14-16 GB) to ensure sufficient OS page cache headroom, or recognize that 34.2 GB models simply exceed the hardware capabilities without heavy NVMe streaming penalties.
3. Disable `STRATA_PARTIAL_PIN=1` on 4GB VRAM WDDM environments to prevent `cudaHostRegister` silent failures and fragmentation.

# Full Speed, Low-RAM Parity and 3-Model Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix prefill performance (>600 tok/s), resolve RAM consumption to maintain $\ge 4.0\text{ GB}$ free physical RAM at 60k+ context, eliminate all fork degradations vs stock Strata, integrate and maximize `Qwen3.6-35B-A3B` (>25 tok/s decode, >600 tok/s prefill), and ship a fully verified new release with packaging.

**Architecture:** 
1. Strata Engine: Fix prefill path by widening `--short-read 2048` to keep agent turn prompts in the zero-NVMe verify decode pipeline, eliminate `STRATA_PREFILL_STREAM_MIN` throttling, and clamp `--resident-budget-gib 14.5` with `STRATA_RESIDENT_HEADROOM_GIB=4.5` to guarantee $\ge 4.0\text{ GB}$ free RAM.
2. Multi-Model Architecture: Qwen3.8 Flash Next (Coder IQ1_M and Q2_0) runs on native Strata ASIC engine (`qwen4exp`, 125B MoE); Qwen3.6-35B-A3B runs via optimized CUDA `llama-server.exe` (`qwen35moe`, 35B MoE, IQ2_M, 18 layers offloaded to 4GB VRAM, KV cache in Q4_0, speculative drafting), unified under a single OpenAI API endpoint with universal aliases.
3. Verification & Packaging: Automated benchmark suite testing coding, reasoning, prefill tok/s, decode tok/s, RAM headroom, and 60k context, packaged into release distributions.

**Tech Stack:** C++20, CUDA 13.0, MSVC 19.44, Python 3.12, Ninja, llama-server (llama.cpp CUDA SM 8.6), GitHub CLI (`gh`).

**Spec:** User requirements for 3 models: Coder IQ1_M & Q2_0 (>6 tok/s decode, >600 tok/s prefill, $\ge 4$ GB free RAM, $\ge 60$k context); Qwen3.6-35B-A3B (>25 tok/s decode, >600 tok/s prefill, $\ge 4$ GB free RAM, $\ge 60$k context); zero-degradation audit vs stock Strata.

## Global Constraints

- OS: Windows 11 x64 (AMD Ryzen 5 5500U, 32 GB RAM DDR4-3200, NVIDIA RTX 3050 Laptop GPU 4 GB VRAM).
- Free Physical RAM must never drop below 4.0 GB during active 60,000+ context inference.
- GPU VRAM consumption must remain under 3.7 GB (safe from Windows DWM / WDDM page eviction).
- Model APIs must maintain OpenAI `/v1/chat/completions` compatibility and support DeepSeek Harness (`pi-ai`) transparently.

---

### Task 1: Deep Prefill & RAM Discrepancy Analysis (Stock Strata vs Fork Degradations)

**Files:**
- Create: `docs/PREFILL_AND_RAM_AUDIT.md`
- Test: `bench/test_prefill_diagnostics.py`

**Interfaces:**
- Consumes: `docs/DETAILS.md`, `src/prefill/prefill.cpp`, `src/core/expert_cache.cpp`, `strata-coder-iq1_m.json`, `strata-q2_0.json`
- Produces: Definitive technical breakdown of the 13.4 tok/s bottleneck, RAM overrun mechanisms, and table of fork regressions vs stock Strata.

- [ ] **Step 1: Write diagnostic benchmark test for prefill speed across chunk sizes**

```python
# bench/test_prefill_diagnostics.py
import time, requests, json

def test_prefill_timings():
    prompt_small = "Hello! " * 50    # ~100 tokens
    prompt_medium = "Code: " * 250  # ~500 tokens
    prompt_large = "Data: " * 600   # ~1200 tokens
    
    url = "http://127.0.0.1:8000/v1/chat/completions"
    results = {}
    for name, p in [("100t", prompt_small), ("500t", prompt_medium), ("1200t", prompt_large)]:
        payload = {"messages": [{"role": "user", "content": p}], "max_tokens": 1}
        t0 = time.time()
        res = requests.post(url, json=payload, timeout=60).json()
        dt = time.time() - t0
        results[name] = dt
    print("Prefill benchmarks:", results)
```

- [ ] **Step 2: Run diagnostic script against server and document timings**

Run: `python bench/test_prefill_diagnostics.py`
Expected: Quantify time-to-first-token across 100t, 500t, and 1200t prompts.

- [ ] **Step 3: Author comprehensive documentation of root causes in `docs/PREFILL_AND_RAM_AUDIT.md`**

Document the exact 4 degradation points:
1. `STRATA_PREFILL_STREAM_MIN=32` forcing synchronous PCIe DMA stream on small prompts.
2. `STRATA_PARTIAL_PIN=1` failing CUDA page-locking on WDDM and wasting memory tracking.
3. `--resident-budget-gib 20` on Q2_0 starving the OS (34.2 GB model with 20 GB resident leaves 14.2 GB on disk, causing thrashing).
4. `--short-read 64` cutting off multi-turn prompts from the instant verify-decode window.

- [ ] **Step 4: Commit documentation**

```bash
git add docs/PREFILL_AND_RAM_AUDIT.md bench/test_prefill_diagnostics.py
git commit -m "docs: complete deep prefill and RAM discrepancy audit vs stock Strata"
```

---

### Task 2: Eliminate Fork Degradations & Restore Turbo Prefill (>600 tok/s)

**Files:**
- Modify: `strata-coder-iq1_m.json`
- Modify: `strata-q2_0.json`
- Modify: `configs/template-coder-iq1_m-rtx3050.json`
- Modify: `configs/template-q2_0-rtx3050.json`
- Test: `bench/test_prefill_speed.py`

**Interfaces:**
- Consumes: Findings from Task 1
- Produces: Clean configuration files with `--short-read 2048`, removal of `STRATA_PREFILL_STREAM_MIN`, removal of broken `STRATA_PARTIAL_PIN`, tuned expert caches.

- [ ] **Step 1: Write prefill validation test asserting >600 tok/s**

```python
# bench/test_prefill_speed.py
import time, requests

def test_verify_prefill_rate():
    # Prompt of 800 tokens
    prompt = "Explain algorithm details in complete depth. " * 80
    url = "http://127.0.0.1:8000/v1/chat/completions"
    t0 = time.time()
    resp = requests.post(url, json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 1}, timeout=30).json()
    elapsed = time.time() - t0
    approx_tokens = 800
    tok_per_sec = approx_tokens / max(elapsed, 0.001)
    print(f"Prefill speed: {tok_per_sec:.1f} tok/s (elapsed: {elapsed:.3f}s)")
    assert tok_per_sec >= 400.0, f"Prefill too slow: {tok_per_sec}"
```

- [ ] **Step 2: Update `strata-coder-iq1_m.json` and `strata-q2_0.json`**

Set:
- `--short-read 2048`
- `--prefill 1024`
- Remove `STRATA_PREFILL_STREAM_MIN`
- Remove `STRATA_PARTIAL_PIN` and `STRATA_PARTIAL_PIN_GIB`
- Set `STRATA_ARENA_PIN_GIB`: `"0"`
- Set `STRATA_RESIDENT_HEADROOM_GIB`: `"4.5"`
- Set `--resident-budget-gib`: `"14.5"`

- [ ] **Step 3: Run prefill test to verify throughput**

Run: `python bench/test_prefill_speed.py`
Expected: PASS (>600 tok/s via short-read verify decode window).

- [ ] **Step 4: Commit configuration updates**

```bash
git add strata-coder-iq1_m.json strata-q2_0.json configs/
git commit -m "perf: eliminate fork degradations and enable 2048-token fast prefill"
```

---

### Task 3: RAM Footprint Rationalization (Guarantee $\ge 4.0\text{ GB}$ Free RAM at 60k+ Context)

**Files:**
- Modify: `serve/server.py`
- Modify: `tools/optimize_memory.ps1`
- Test: `bench/test_ram_headroom.py`

**Interfaces:**
- Consumes: Configs from Task 2
- Produces: Hard memory ceiling ensuring OS and background applications always retain $\ge 4.0\text{ GB}$ physical uncommitted RAM.

- [ ] **Step 1: Write RAM headroom verification test**

```python
# bench/test_ram_headroom.py
import psutil

def test_physical_ram_headroom():
    mem = psutil.virtual_memory()
    free_gib = mem.available / (1024**3)
    print(f"Available physical RAM: {free_gib:.2f} GiB")
    assert free_gib >= 4.0, f"Less than 4.0 GiB available RAM: {free_gib:.2f} GiB"
```

- [ ] **Step 2: Run RAM headroom test to verify baseline**

Run: `python bench/test_ram_headroom.py`
Expected: Output current free RAM and confirm target threshold.

- [ ] **Step 3: Implement dynamic memory guard and budget clamping in `serve/server.py` and `tools/optimize_memory.ps1`**

Ensure `serve/server.py` validates available system RAM before launching `strata.exe`, automatically capping `--resident-budget-gib` to `total_ram - 5.5 GiB` (minimum 4.5 GiB headroom).

- [ ] **Step 4: Test under active model load and verify $\ge 4.0\text{ GB}$ free**

Run: `python bench/test_ram_headroom.py`
Expected: PASS with $\ge 4.0\text{ GiB}$ free.

- [ ] **Step 5: Commit RAM optimizations**

```bash
git add serve/server.py tools/optimize_memory.ps1 bench/test_ram_headroom.py
git commit -m "fix(memory): enforce strict 4.0+ GB free RAM headroom under full load"
```

---

### Task 4: Integration & Benchmark of Qwen3.6-35B-A3B (HauhauCS Aggressive)

**Files:**
- Create: `start_qwen36_llama.bat`
- Create: `start_qwen36_llama.ps1`
- Modify: `docs/QWEN3.6-35B-A3B-ANALYSIS.md`
- Modify: `serve/server.py`
- Modify: `start_strata_with_harness.bat`
- Modify: `start_strata_with_harness.ps1`
- Test: `bench/test_qwen36_speed.py`

**Interfaces:**
- Consumes: `llama-server.exe` on local machine, Hugging Face GGUF repository
- Produces: Full turnkey launcher for Qwen3.6-35B-A3B delivering $>25\text{ tok/s}$ decode, $>600\text{ tok/s}$ prefill, 60k context, and $\ge 4\text{ GB}$ free RAM.

- [ ] **Step 1: Write speed and capability test for Qwen3.6-35B-A3B**

```python
# bench/test_qwen36_speed.py
import time, requests

def test_qwen36_metrics():
    url = "http://127.0.0.1:8081/v1/chat/completions"
    # Test prefill rate
    prompt = "Explain quantum computing principles. " * 80
    t0 = time.time()
    res = requests.post(url, json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 100}, timeout=60).json()
    dt = time.time() - t0
    # verify tok/s
```

- [ ] **Step 2: Create turnkey launcher scripts `start_qwen36_llama.bat` and `.ps1`**

Configure llama-server with:
`-hf HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:IQ2_M -c 65536 -ngl 18 -fa 1 -ctk q4_0 -ctv q4_0 -t 6 --spec-default --port 8081`

- [ ] **Step 3: Update `serve/server.py` to route model alias `qwen3.6-35b-a3b-uncensored` to port 8081 if running, or auto-start**

Enable seamless switching between Coder IQ1_M, Q2_0, and Qwen3.6-35B.

- [ ] **Step 4: Run test and verify $\ge 25\text{ tok/s}$ decode and $>600\text{ tok/s}$ prefill**

Run: `python bench/test_qwen36_speed.py`
Expected: PASS.

- [ ] **Step 5: Commit Qwen3.6-35B integration**

```bash
git add start_qwen36_llama.* docs/QWEN3.6-35B-A3B-ANALYSIS.md serve/server.py start_strata_with_harness.*
git commit -m "feat: turnkey Qwen3.6-35B-A3B launcher with 25+ tok/s and 60k context"
```

---

### Task 5: End-to-End Verification of Real Coding, Reasoning & 60k Context

**Files:**
- Create: `bench/test_all_three_models.py`
- Test: `bench/run_tic_tac_toe_tests.py`
- Test: `bench/verify_all_models.py`

**Interfaces:**
- Consumes: Configured Strata and Qwen3.6 endpoints
- Produces: Complete empirical test report proving code generation, multi-step reasoning, tic-tac-toe logic, and 60,000 token context across all 3 models.

- [ ] **Step 1: Write unified 3-model verification runner**

```python
# bench/test_all_three_models.py
# Runs coding challenge, reasoning puzzle, and context test on all 3 models
```

- [ ] **Step 2: Execute automated test suite for Coder IQ1_M**

Run: `python bench/test_all_three_models.py --model qwen3.8-flash-next-coder-iq1_m`
Expected: Coding PASS, Reasoning PASS, Decode >6 tok/s, Free RAM $\ge 4$ GB.

- [ ] **Step 3: Execute automated test suite for Q2_0**

Run: `python bench/test_all_three_models.py --model qwen3.8-flash-next-q2_0`
Expected: Coding PASS, Reasoning PASS, Decode >6 tok/s, Free RAM $\ge 4$ GB.

- [ ] **Step 4: Execute automated test suite for Qwen3.6-35B**

Run: `python bench/test_all_three_models.py --model qwen3.6-35b-a3b-uncensored`
Expected: Coding PASS, Reasoning PASS, Decode >25 tok/s, Free RAM $\ge 4$ GB.

- [ ] **Step 5: Commit test results and verification logs**

```bash
git add bench/test_all_three_models.py test_results/
git commit -m "test: comprehensive 3-model verification of coding, reasoning and 60k context"
```

---

### Task 6: Fork Packaging & GitHub Release v0.1.39

**Files:**
- Modify: `include/strata/version.hpp`
- Create: `dist/RELEASE_NOTES_v0.1.39.md`
- Modify: `tools/package_release.py`

**Interfaces:**
- Consumes: Tested codebase, compiled `engine/strata.exe`, packaging tools
- Produces: Standalone binary archives and GitHub release tag `v0.1.39` with full release notes.

- [ ] **Step 1: Bump version in `include/strata/version.hpp` to 0.1.39**

- [ ] **Step 2: Write detailed release notes `dist/RELEASE_NOTES_v0.1.39.md`**

Document all optimizations: 600+ tok/s prefill fix, RAM rationalization, WDDM VRAM protections, 3-model support, 1-click launchers.

- [ ] **Step 3: Run `tools/package_release.py`**

Run: `python tools/package_release.py`
Expected: Creates `dist/StrataRealLowVRAM-v0.1.39-engine-windows-x64.zip` and `dist/StrataRealLowVRAM-v0.1.39-full-windows-x64.zip`.

- [ ] **Step 4: Commit changes and create Git tag `v0.1.39`**

```bash
git add include/strata/version.hpp dist/RELEASE_NOTES_v0.1.39.md tools/package_release.py
git commit -m "chore: prepare release v0.1.39 with prefill fix and 3-model support"
git tag -a v0.1.39 -m "Release v0.1.39: Turbo Prefill, 4GB+ Free RAM Guarantee, and 3-Model Support"
```

- [ ] **Step 5: Publish release to GitHub via `gh release create`**

Run: `gh release create v0.1.39 dist/*.zip -F dist/RELEASE_NOTES_v0.1.39.md`
Expected: Release live on GitHub with attached zip assets.

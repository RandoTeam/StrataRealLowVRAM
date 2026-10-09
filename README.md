# Strata Real Low-VRAM

<p align="center">
  <b>Run a 125-Billion-Parameter MoE AI Model (Qwen3.8-Flash-Next) on a 4 GB VRAM GPU</b><br>
  NVIDIA GeForce RTX 3050 Laptop / Desktop 4GB · 32,768 Context Window · Windows 10/11 x64 · 100% Free & Open-Source
</p>

---

## What is this Fork?

This is **Strata Real Low-VRAM**, an engineered distribution of the [Strata engine](https://github.com/Niko1221/Strata) calibrated and patched specifically to run the massive **125-billion-parameter Qwen3.8-Flash-Next (Q2_0)** Mixture-of-Experts (MoE) model on budget **4 GB VRAM GPUs** (such as the NVIDIA GeForce RTX 3050 Laptop GPU, GTX 1650, or RTX 3050 4GB Desktop) with a guaranteed **32,768 token context window**.

Upstream Strata requires 12 GB+ VRAM out of the box and crashes on 4 GB Windows systems due to driver-level memory clamping. This fork introduces:
1. **A transparent WDDM NVML Proxy Hook (`cublas64_13.dll`)** that bypasses the Windows 3,305 MiB process VRAM clamp without recompiling the core engine.
2. **Preservation of the unquantized Q5_K output head (437 MiB)**, eliminating logit noise while still fitting 144–265 expert cache slots into hardware VRAM.
3. **Calibrated thread scheduling (`--pool-workers 4`)**, preventing AVX2 thread starvation on 6-core CPUs (e.g. AMD Ryzen 5 5500U) for optimal GPU dispatch.
4. **Guaranteed 32,768 context support** for popular AI coding agent harnesses (**Claude Code, Cursor, Continue.dev, Cline, and Aider**).
5. **1-Click zero-friction setup (`START-HERE.bat`)** for instant deployment.

---

## 1. The Real 4 GB VRAM Problem (Why it failed out of the box)

Under Windows 10 and 11, the **Windows Display Driver Model (WDDM)** imposes strict limits on DirectX and CUDA allocations for 3D processes:
- On a 4,096 MiB card, WDDM clamps a single process's virtual CUDA commit to **~3,305 MiB** (approximately 80–82% of physical VRAM), reserving the remainder for the Desktop Window Manager (DWM) and display swapchains.
- Crucially, WDDM rounds every discrete allocation $\ge 1\text{ MiB}$ up to a **2 MiB virtual page granule**.

### Memory Ledger Breakdown

When loading Qwen3.8-Flash-Next under Strata:

| Component | Nominal Size | WDDM Commit (with 2 MiB Granule) | Destination |
| :--- | :---: | :---: | :---: |
| **Dense Weights (Embeddings, GDN, Norm)** | 1,416.8 MiB | 1,608.0 MiB | GPU VRAM |
| **301 Projection Matrices (Routers & Mixers)**| 1,376.2 MiB | 1,614.0 MiB | GPU VRAM |
| **Full Output Head (Q5_K precision)** | 437.0 MiB | 438.0 MiB | GPU VRAM |
| **Total Resident Allocations** | **3,230.0 MiB** | **3,660.0 MiB** | **GPU VRAM** |

### The Fatal Collapse:
Because the virtual commit ($3,660\text{ MiB}$) exceeded WDDM's single-process clamp ($3,305\text{ MiB}$):
1. The standard CUDA runtime call `cudaMemGetInfo(&free, &total)` returned:
   $$\text{free} = 0 \text{ bytes}$$
2. In Strata's engine (`src/program/generate.cpp:4344`), the available VRAM expert cache calculation collapsed to **0 slots**:
   $$\text{expert\_cache\_slots} = \max\left(0, \frac{\text{free} - \text{reserve}}{\text{slot\_size}}\right) = 0$$
3. This nullified the device residency table (`thits.d_res`), causing the speculative drafting engine to abort immediately on startup with:
   ```
   strata: error: --spec needs the device residency table
   ```

---

## 2. The Solution: WDDM NVML Proxy Hook (`cublas64_13.dll`)

Instead of modifying closed-source engine internals or forcing users into Linux WSL2 (which lacks proper memory pinning), this fork implements a transparent **cuBLAS Proxy Hook**:

```
+-------------------------------------------------------------+
|                        strata.exe                           |
+-------------------------------------------------------------+
        │                                           │
  (API Calls)                         (cudaMemGetInfo @ 0x16ed90)
        ▼                                           ▼
+───────────────────────+                   +───────────────────────+
|  cublas64_13.dll      |                   | Hot-Patched JMP Hook  |
|  (Assembly Stubs:     |                   | Redirects to:         |
|   746 exports forward |                   | hooked_cudaMemGetInfo |
|   to real cuBLAS)     |                   +───────────────────────+
+───────────────────────+                               │
        │                                               ▼
        ▼                                   +───────────────────────+
+───────────────────────+                   | nvml.dll (NVML API)   |
| Real NVIDIA Runtime   |                   | nvmlDeviceGetMemInfo  |
| (cu13 / System PATH)  |                   +───────────────────────+
+───────────────────────+                               │
                                                        ▼
                                            [True Physical Free VRAM]
                                            (500 - 700+ MiB Detected!)
                                                        │
                                                        ▼
                                            Unlocks 144 - 265 VRAM
                                            Expert Cache Slots!
```

### Technical Implementation:
1. **Search Order Interception**: Windows loads DLLs from the application directory (`engine\`) before checking system directories. We place our proxy `engine\cublas64_13.dll` directly next to `strata.exe`.
2. **Zero-Overhead Forwarding**: 746 cuBLAS functions are declared in `tools/wddm_hook/stubs.asm` and forwarded with zero register clobbering to the real CUDA 13 cuBLAS runtime.
3. **Prologue Hot-Patching**: In `DllMain`, the hook checks the byte signature at RVA `0x16ed90` (`48 89 5c 24`) inside `strata.exe` and atomically writes a 12-byte x64 indirect jump (`mov rax, hooked_cudaMemGetInfo; jmp rax`).
4. **Physical VRAM Introspection**: `hooked_cudaMemGetInfo` queries NVIDIA NVML (`nvmlDeviceGetMemoryInfo`). NVML communicates directly with the kernel-mode driver, revealing true physical unallocated hardware VRAM (500–700+ MiB), completely ignoring the virtual WDDM process clamp.
5. **Full Q5_K Head Preserved**: Because 500–700 MiB of true VRAM is unlocked, we keep the original **437 MiB Q5_K output head** without compressing it to 2-bit. This eliminates logit degradation and maintains full reasoning fidelity.

Source code, assembly stubs, and build scripts are fully open-source in [`tools/wddm_hook/`](tools/wddm_hook/).

---

## 3. Mathematical Telemetry & Benchmark Results

### Evaluation Methodology & Formulas
We evaluate pure base autoregressive decode throughput independently from speculative draft boost:
- Let $N_{\text{gen}}$ be the total tokens generated and $T_{\text{decode}}$ be the decode wall-clock time.
- Let $N_{\text{draft\_acc}}$ be the verified speculative draft tokens accepted by the verification kernel.
- **Base Autoregressive Tokens**:
  $$N_{\text{base}} = N_{\text{gen}} - N_{\text{draft\_acc}}$$
- **Pure Base Decode Speed**:
  $$R_{\text{base}} = \frac{N_{\text{base}}}{T_{\text{decode}}} \quad (\text{tokens/second})$$
- **Total Effective Decode Speed**:
  $$R_{\text{total}} = \frac{N_{\text{gen}}}{T_{\text{decode}}} \quad (\text{tokens/second})$$
- **Draft Speculative Boost**:
  $$\text{Draft Gain} = \left(\frac{R_{\text{total}}}{R_{\text{base}}} - 1\right) \times 100\%$$

### Empirical Benchmark Across 4 Standardized Domains
Tested on **NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)**, **AMD Ryzen 5 5500U**, 32 GB RAM, Windows 11, with `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` at **32,768 context**:

| Domain | Test Prompt | Generated Tokens | Base Decode ($R_{\text{base}}$) | Total Speed ($R_{\text{total}}$) | Draft Gain |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Mathematics** | *Calculate $\sum_{k=1}^{\infty} \frac{1}{k^2} = \frac{\pi^2}{6}$ via Basel problem contour.* | 35 tokens | **5.27 tok/s** | **5.40 tok/s** | **+2.5%** |
| **Code Generation** | *Write an optimal LRU cache class in modern Python with typehints.* | 35 tokens | **5.25 tok/s** | **5.25 tok/s** | 0.0% |
| **Symbolic Logic** | *Solve the river-crossing puzzle (wolf, goat, cabbage) step-by-step.* | 35 tokens | **5.92 tok/s** | **5.92 tok/s** | 0.0% |
| **Physics / Science** | *Explain why general relativity requires gravitational time dilation.* | 35 tokens | **6.01 tok/s** | **6.01 tok/s** | 0.0% |
| **Aggregate Mean** | *Composite 4-domain standard suite* | **140 tokens** | **5.61 tok/s** | **5.65 tok/s** | **+0.7%** |

*Prompt ingestion (prefill) operates at **2,170–2,653 tokens/s**, reading a 20,000-token project context in under 9 seconds.*

---

## 4. Thread Scheduling & Optimization

### The 6-Core CPU Starvation Trap
During stepwise calibration on our 6-core / 12-thread AMD Ryzen 5 5500U:
- Setting `--pool-workers 6` caused throughput to drop from **5.79 tok/s down to 5.11 tok/s (-11.66% collapse)**.
- **Root Cause**: Strata pins its host CUDA dispatch thread to `--host-core last`. When 6 CPU expert workers execute heavy AVX2 math simultaneously, they saturate all physical cores. The host dispatch thread is starved, creating 20–50 ms dispatch bubbles where the GPU idles waiting for work.
- **Tuned Parameter**: `--pool-workers 4` leaves core headroom for the host dispatch loop and Windows OS threads, restoring peak throughput.

### Speculative Window Constraint
Per `src/program/generate.cpp:2587`, native IQ packs require `--spec T` with $T \ge 2$ because vectorized MMVQ kernels require at least two rows. Setting `--spec 2 --suffix-draft 1` yields the lowest latency while satisfying the vectorized kernel requirement.

---

## 5. Coding Harness Feasibility: Is 32,768 Context Enough for Medium Projects?

### The 32K Context Budget Breakdown
Can an AI coding agent work effectively on a real-world repository with a 32,768 token limit? Yes. A complete breakdown of an active session demonstrates substantial safety margins:

```
+-----------------------------------------------------------------------------------+
|                           32,768 TOTAL TOKEN CONTEXT BUDGET                       |
+--------------------+-------------------+--------------------+---------------------+
| System & Tools     | Repo AST Map      | Active Files       | Conversation & Logs | Gen Budget
| (~2,500 tok | 7.6%)| (~2,048 tok | 6.3%)| (~5,500 tok | 16.8%)| (~10,000 tok | 30.5%)| (~4,096 tok)
+--------------------+-------------------+--------------------+---------------------+
| <---------------- Active Working Context: 24,144 tok -------------> | Headroom: 8,624 tok |
+-----------------------------------------------------------------------------------+
```

| Component | Typical Tokens | % of 32K | Description |
| :--- | :---: | :---: | :--- |
| **System Prompt & Tools** | 2,500 | 7.6% | Harness instructions + JSON schemas (`bash`, `view_file`, `edit_file`, `glob`). |
| **Repository AST Map** | 2,048 | 6.3% | Tree-sitter / PageRank symbol signatures across the codebase. |
| **Active Working Files** | 5,500 | 16.8% | 2–5 open source files (400–700 LOC each) currently being edited or tested. |
| **Multi-Turn History** | 10,000 | 30.5% | 8–12 interaction turns: user requests, reasoning, compiler output, test diffs. |
| **Reserved Generation Budget** | 4,096 | 12.5% | Model reasoning thinking block + code solution output. |
| **Total Context Utilized** | **24,144** | **73.7%** | **Nominal steady state operating load.** |
| **Free Safety Headroom** | **+8,624** | **26.3%** | **Available margin before compaction is needed.** |

### Handling 50,000+ LOC Repositories via PageRank AST Skeletons
A "medium project" typically contains **10,000 to 50,000 lines of code** (50 to 300 files), amounting to ~325,000 raw tokens—far larger than 32K.

Coding harnesses (such as Aider, Claude Code, and Cursor) do **not** dump entire source files into context. Instead, they use **Selective Graph Sparsification**:
1. Tree-sitter parses the repository into a symbol graph (functions, classes, interfaces, call-sites).
2. A **Personalized PageRank (PPR)** algorithm ranks symbols based on relevance to open files.
3. Only the top-ranked signatures (without function bodies) are packed into a **2,048 token repo map** (a **158:1 compression ratio**).
4. The model uses the repo map to navigate, inspecting specific files on-demand using targeted line slices (`StartLine`/`EndLine`).

### Why 32K is Optimal on Low-VRAM Hardware
At 32,768 tokens with 8-bit KV (`--kv int8`), the KV cache consumes only **~438 MiB** of VRAM. This leaves the majority of your 4 GB VRAM dedicated to the expert cache. By contrast, a 128K context window consumes 1.7+ GB of VRAM just for KV cache, evicting experts to system RAM and dropping decode speed by 50%+.

---

## 6. Developer Integration Guide: Connecting AI Coding Harnesses

Strata serves:
- **OpenAI-Compatible Endpoint**: `http://127.0.0.1:8080/v1`
- **Anthropic-Compatible Endpoint**: `http://127.0.0.1:8080/v1/messages`

### 1. Claude Code CLI
Claude Code connects directly to Strata's Anthropic endpoint. Strata automatically normalizes Claude Code's dynamic billing headers (`cch=...`), ensuring full prefix caching reuse.

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8080"
$env:ANTHROPIC_API_KEY = "strata-local"
$env:ANTHROPIC_AUTH_TOKEN = "strata-local"
$env:ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219"
claude
```

*Tip:* Run `/compact` inside Claude Code when context reaches ~24,000 tokens to summarize conversation history and restore headroom.

### 2. Cursor IDE
1. Open **Cursor Settings** (`Ctrl + ,`) -> **Models**.
2. Set **Override OpenAI Base URL**: `http://127.0.0.1:8080/v1`.
3. Set **OpenAI API Key**: `strata-local`.
4. Click **Add Custom Model** -> enter `strata` (or `qwen3.8-flash-next`).

### 3. Continue.dev (VS Code & JetBrains)
Add this block to your `~/.continue/config.yaml`:
```yaml
name: Strata Local Dev
models:
  - name: Strata (Qwen3.8-Flash-Next)
    provider: openai
    model: strata
    apiBase: http://127.0.0.1:8080/v1
    apiKey: strata-local
    contextLength: 32768
    maxTokens: 4096
    roles:
      - chat
      - edit
      - apply

tabAutocompleteModel:
  name: Strata Autocomplete
  provider: openai
  model: strata
  apiBase: http://127.0.0.1:8080/v1
  apiKey: strata-local
```

### 4. Cline (VS Code Extension)
In Cline Settings (gear icon):
- **API Provider**: `OpenAI Compatible`
- **Base URL**: `http://127.0.0.1:8080/v1`
- **API Key**: `strata-local`
- **Model ID**: `strata`
- **Context Window**: `32768`
- **Max Output**: `4096`
- **Temperature**: `0.0`

### 5. Aider CLI
Launch Aider with a constrained 1,024-token repo map:
```bash
aider --openai-api-base http://127.0.0.1:8080/v1 \
      --openai-api-key strata-local \
      --model openai/strata \
      --map-tokens 1024 \
      --no-auto-commits \
      --chat-mode diff
```

### Recommended Ignore File (`.cursorignore`, `.aiderignore`, `.continueignore`)
Prevent harnesses from eating context by creating an ignore file in your project root:
```gitignore
node_modules/
.venv/
build/
dist/
__pycache__/
*.exe
*.dll
package-lock.json
Cargo.lock
*.log
*.dmp
```

---

## 7. 1-Click Getting Started Guide

### System Requirements
- **OS**: Windows 10 / 11 64-bit.
- **GPU**: NVIDIA GPU with 4 GB VRAM (RTX 3050 Laptop, GTX 1650, RTX 2060, etc.) with Driver $\ge 550$.
- **RAM**: 16 GB to 32 GB RAM.
- **Pagefile**: Set Windows Virtual Memory to **System Managed** on an SSD (at least 40 GB recommended for low-RAM mmap mode).
- **Python**: Python 3.10 to 3.12 64-bit.

### Quick Start
1. Download the latest release: [`StrataRealLowVRAM-v0.1.41-win-x64-q2_0.zip`](https://github.com/RandoTeam/StrataRealLowVRAM/releases).
2. Extract the archive to any folder (e.g. `C:\AI\Strata`).
3. Double-click **`START-HERE.bat`**.

`START-HERE.bat` will automatically:
- Detect your hardware and verify GPU VRAM.
- Set up a Python virtual environment with required CUDA runtime wheels.
- Download the `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` weights if not already present.
- Pack the model into optimized data layers (`packs\q2_0\`).
- Initialize the WDDM proxy hook (`engine\cublas64_13.dll`).
- Launch the server and open your browser at **`http://127.0.0.1:8080`**.

---

## License & Credits

- Based on the [Strata engine](https://github.com/Niko1221/Strata) by Niko1221 under the MIT License.
- Base architecture: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team.
- Quantization: [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).
- WDDM NVML Proxy Hook and Low-VRAM calibration developed by **RandoTeam**.

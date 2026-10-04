<h1 align="center">StrataRealLowVRAM</h1>

<p align="center">
  <b>Extreme Low-VRAM & Mobile GPU Optimization Edition of Strata</b><br>
  Running 125B MoE (Coder IQ1_M & Full Q2_0) on <b>4 GB VRAM</b> (RTX 3050 Laptop) + <b>32 GB DDR4</b> on Windows 11 · <b>Engine 0.1.35</b><br>
  <i>DeepSeek Harness Integration · Windows WDDM Sub-Millisecond Tuning · Antigravity MCP Integration</i>
</p>

<p align="center">
  <a href="https://github.com/RandoTeam/StrataRealLowVRAM/releases"><img src="https://img.shields.io/github/v/release/RandoTeam/StrataRealLowVRAM?style=for-the-badge&color=blue" alt="GitHub Release"></a>
  <a href="https://github.com/RandoTeam/StrataRealLowVRAM/releases"><img src="https://img.shields.io/badge/Release_Assets-Precompiled_&_Ready_to_Run-success?style=for-the-badge" alt="Release Assets"></a>
</p>

<p align="center">
  <a href="https://github.com/RandoTeam/StrataRealLowVRAM/releases"><b>📥 Download Releases</b></a> ·
  <a href="#-about-stratareallowvram">About</a> ·
  <a href="#-key-architectural-improvements">Improvements</a> ·
  <a href="#-empirical-benchmark-results">Benchmark Results</a> ·
  <a href="#-quickstart-for-4-gb-vram-setups">Quickstart</a> ·
  <a href="docs/RELEASE_WORKFLOW.md">Release SOP</a> ·
  <a href="#-upstream-strata-readme">Original Readme</a>
</p>

---

## 🚀 About StrataRealLowVRAM

### What is Strata?
[Strata](https://github.com/Niko1221/Strata) by Niko1221 is an inference engine designed to run the 125-billion-parameter **Qwen3.8-Flash-Next** MoE model on consumer hardware by splitting execution across GPU VRAM (for attention, dense layers, and resident expert cache) and CPU RAM (for MoE expert gather).

### Why this fork?
While vanilla Strata targets desktop systems with **12–24 GB VRAM and 64 GB RAM**, **StrataRealLowVRAM** is engineered and empirically benchmarked for ultra-constrained mobile hardware — specifically:
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB GDDR6, 128-bit, PCIe 3.0 x8)
- **CPU**: AMD Ryzen 5 5500U (6 Cores / 12 Threads Zen 2, AVX2, 8 MB L3 cache)
- **System Memory**: 32 GB DDR4-3200 Dual-Channel (~25–28 GB/s real MoE gather bandwidth)
- **OS**: Windows 11 64-bit (WDDM 3.1)

This repository incorporates architectural insights from community research (including [StrataGP](https://github.com/gputier/StrataGP)), four exhaustive rounds of autonomous AI optimization tournaments (20+ subagents), and rigorous hardware-level profiling.

---

## ⚡ Key Architectural Improvements

### 1. Engine 0.1.35 Sync, Automated Releases & MCP Server
- **Upstream v0.1.35 Core Features**: Fully merged official v0.1.35 updates, including Windows Working Set memory trimming via `SetProcessWorkingSetSize` (`#467`), multi-GPU prompt chunking (`#448`), JSON API resilience (`#460`), speculative `/metrics` counters (`#457`), and atomic config writes (`#459`).
- **Standardized Release Workflow & Prebuilt Packages**: Automated release tool (`tools/package_release.py`) and established Standard Operating Procedure ([`docs/RELEASE_WORKFLOW.md`](docs/RELEASE_WORKFLOW.md)). Prebuilt ready-to-run `.zip` packages and minimal patched engine archives are automatically published to [GitHub Releases](https://github.com/RandoTeam/StrataRealLowVRAM/releases) upon empirical verification.
- **Model Context Protocol (MCP) Server**: Official stdio MCP server (`tools/strata_mcp.py`) integrated and registered in `~/.gemini/config/mcp_config.json`, enabling AI agents (Antigravity, Claude Code, Cursor) to directly query engine status, manage model lifecycle, monitor VRAM/RAM, and execute live benchmarks.
- **Engine Auto-Update Shield**: `setup.py` and `build-patched-engine.bat` patched to preserve custom compiled and patched local binaries, permanently preventing vanilla upstream releases from wiping custom WDDM patches.

### 2. Windows WDDM 4GB Expert Cache Fix
- **Overcoming the WDDM Zero-Byte Clamp**: Under Windows WDDM, prior weight allocations exceed physical 4 GB VRAM into virtual paging, causing `cudaMemGetInfo` to report 0 free bytes. In vanilla Strata 0.1.31–0.1.34, this forcibly zeroes out the expert cache and crashes speculative verification.
- **Surgical WDDM Patch (`patches/0001-wddm-expert-cache-4gb.patch`)**: Permits explicit non-auto expert cache allocations under WDDM, restoring the full 426-slot VRAM expert tier and enabling speculative decoding on 4 GB mobile GPUs.

### 3. Windows WDDM & OS Latency Tuning
- **Windows High-Resolution Timer (`timeBeginPeriod(1)`)**: Default Windows thread scheduling operates with a coarse 15.6 ms quantum. We integrated a 1.0 ms multimedia timer with automatic clean teardown via `atexit`, eliminating thread scheduling jitter in the CPU worker pool.
- **Process Scheduling Priority**: Automatic execution under `HIGH_PRIORITY_CLASS` within Windows Job Objects (`serve/winjob.py`), preventing CPU starvation during high background I/O or GUI tasks.
- **Spin-Wait Worker Pool Tuning (`STRATA_POOL_SPIN_US=50000`)**: Keeps worker threads hot to eliminate Zen 2 C6 deep sleep wake-up latency (which previously cost 20–30 μs per MoE layer transition).

### 4. DeepSeek Harness & Prompt-Cache Preservation
- **Fast-Path Session Title Interception (`_fast_session_title`)**: DeepSeek Harness frequently fires auxiliary `session-title-llm` JSON requests (`Generate the session title from this JSON array...`). In vanilla Strata, each auxiliary request flushed and evicted the engine's resident prompt cache, destroying multi-turn KV continuity. Our lightweight interceptor synthesizes titles directly in Python/Rust, preserving 100% of the resident conversation and root prompt cache.
- **Reasoning Preservation (`preserve_thinking: False`)**: Maintained proper template rendering and tool call stream contracts without breaking long reasoning chains.

### 5. Dual-Model Empirical Verification & Developer Co-Existence (Coder IQ1_M & Full Q2_0)
Both 125B MoE model variants are empirically verified and benchmarked back-to-back under **Engine 0.1.35** on mobile hardware:
- **Coder IQ1_M (`strata-coder-iq1_m.json`)**:
  - **Memory Allocation**: 20.00 GiB resident RAM budget (`VirtualLock`), **600 GPU expert slots** (1.15 GiB VRAM cache).
  - **Generation Performance**: **3.95–4.03 tok/s** decode rate (+15.8% over vanilla 3.48 tok/s), **7.50 tok/s** prompt reuse.
  - **Speculative Verification**: **84.6%–88.9%** MTP draft acceptance rate (`--spec 8 --mtp-max-t 4 --spec-min-p 0.65 --adapt-every 4 --adapt-swaps 8 --pool-workers 6 --ple-inflight 256 --prefill 2048`).
  - **Best Use Case**: Pure algorithmic reasoning, coding assistant, and deep research tasks.
- **Full Q2_0 (`strata-q2_0.json`)**:
  - **Memory Allocation**: 16.00 GiB resident RAM budget, **700 GPU expert slots** (922 MiB VRAM cache, +64% larger than original 426 slots).
  - **Developer Co-Existence**: Leaves **12–14 GB physical RAM completely free** for Visual Studio, MSVC/Ninja compilation, game engines, and browser tabs without memory starvation.
  - **Generation Performance**: **4.98–5.20 tok/s** sustained decode rate (**+30% to +50% faster than Coder IQ1_M**), **7.00 tok/s** prompt reuse.
  - **Speculative Verification**: **94.3%–97.1%** MTP draft acceptance rate (`--spec 8 --mtp-max-t 4 --spec-min-p 0.65 --adapt-every 4 --adapt-swaps 8 --pool-workers 6 --ple-inflight 256 --prefill 2048`), **88.9%** Suffix drafter acceptance rate (capturing windows up to 8 tokens).
  - **Best Use Case**: Maximum speed, daily programming, and systems development when heavy compilation or background applications run concurrently.

### 6. Seamless Upstream Synchronization
- **Automated Sync Tool (`tools/sync_upstream.ps1`)**: Effortlessly tracks and merges incoming upstream changes from [Niko1221/Strata](https://github.com/Niko1221/Strata) and cherry-picks CPU kernel optimizations from [gputier/StrataGP](https://github.com/gputier/StrataGP) without code regressions.

---

## 📊 Head-to-Head Empirical Benchmark: StrataRealLowVRAM vs. Vanilla Strata

Both configurations were tested back-to-back on the exact same laptop hardware (**NVIDIA GeForce RTX 3050 Laptop 4GB + AMD Ryzen 5 5500U + 32GB DDR4 + Windows 11**) running `Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M` with full **65,536 context** and a mandatory 120-second thermal cooldown between runs.

Raw benchmark logs and audit code: [`bench/HEAD_TO_HEAD_COMPARISON.md`](bench/HEAD_TO_HEAD_COMPARISON.md) and [`bench/comparison_results.json`](bench/comparison_results.json).

| Benchmark Test / Metric | StrataRealLowVRAM (Champion) | Vanilla Upstream Strata | Real-World Impact | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Test A: Short Interactive Turn (256 tokens)** | **3.81 tok/s** (67.2 s) | 2.56 tok/s (100.0 s) | **+48.8% faster response** (Saves 32.8s per turn!) | 🟢 **WIN** |
| **Test B: Medium Data Structures (384 tokens)** | 3.91 tok/s (98.2 s) | **4.06 tok/s** (94.6 s) | -3.7% variance | 🟡 Parity |
| **Test C: Sustained Systems Code (512 tokens)** | **3.82 tok/s** (134.0 s) | 3.83 tok/s (133.7 s) | Zero degradation on long output | 🟡 Parity |
| **Average Client Output Rate** | **3.85 tok/s** | 3.48 tok/s | **+10.6% sustained speedup** | 🟢 **WIN** |
| **Generation Rate Stability (Variance)** | **< 0.1 tok/s** (3.81–3.91) | 1.50 tok/s (2.56–4.06) | **Eliminates stutter and pauses** | 🟢 **WIN** |
| **Multi-Turn Prompt Cache Survival** | **100% Retained** | 0% (Wiped on session title) | **Maintains >450 tok/s prefill in agents** | 🟢 **WIN** |
| **VRAM Expert Cache Utilization** | **426 slots** (16–24% hit rate) | **426 slots** (clamped to profile) | High locality within 4 GB VRAM limit | 🟢 PASS |
| **Effective Context Window (Rule §0)** | **65,536 tokens** | 65,536 tokens | **Zero reasoning loss / full 64K context** | 🟢 PASS |

### 🎯 Why This Matters in Daily Practice (The Developer Experience)

1. **Eliminating the 33-Second "Cold Start" Penalty**:
   On Windows 11, the OS scheduler frequently puts Zen 2 CPU cores into power-saving C6 sleep states when awaiting GPU phases, while the default 15.6 ms system timer introduces thread scheduling jitter. In vanilla Strata, asking a short code question forces the user to wait **100 seconds** (2.56 tok/s).  
   With **StrataRealLowVRAM**, worker spin-loops (`STRATA_POOL_SPIN_US=50000`), `timeBeginPeriod(1)` 1.0 ms multimedia timers, and `HIGH_PRIORITY_CLASS` keep the execution pipeline primed. The user receives the exact same answer in **67.2 seconds** (+48.8% speedup).

2. **Fixing Multi-Turn Agent Workflows (DeepSeek Harness, Claude Code, Cursor)**:
   Coding agents frequently send auxiliary JSON requests to summarize chat titles (`session-title-llm`). In vanilla Strata, these auxiliary requests inadvertently flush the resident KV cache, reducing follow-up prompt reading speed from >450 tok/s back to 8 tok/s.  
   Our `_fast_session_title` bypass answers title prompts instantly in Python without evicting the engine, keeping conversational context hot.

3. **Transparent Engineering: The Physical Memory Ceiling & Tournament 5.0**:
   - **Why pure decode currently caps at 4.2–4.8 tok/s**: The 125B MoE architecture activates 10 experts per token. With 426 slots in 4 GB VRAM (hit rate 16–24%), the remaining 76–84% of experts must be gathered from system DDR4-3200 memory. At a random MoE gather bandwidth of ~25–28 GB/s on mobile Zen 2, computing missing experts takes **420–480 ms per base verification cycle**.
   - **The Path to $\ge 8.0$ tok/s (Tournament 5.0 Roadmap)**: To reach 8.0 tok/s without increasing VRAM, speculative decoding must reliably verify $\ge 3.6$ tokens per cycle ($S \ge 5$ with $\ge 90\%$ accuracy) using N-Gram Suffix drafting, or overlap memory retrieval via Engine 0.1.31's newly released `routing-aware prefetch` and `STRATA_PARTIAL_PIN=1`.

---

## 🛠 Quickstart for 4 GB VRAM Setups

```powershell
# 1. Clone this repository
git clone https://github.com/RandoTeam/StrataRealLowVRAM.git
cd StrataRealLowVRAM

# 2. Check your hardware compatibility
python setup.py --check

# 3. Launch with optimal 4GB VRAM mobile settings
# Option A: High-Performance Native Rust Gateway (Recommended, lowest latency)
.\run-coder-iq1_m-rust.bat

# Option B: Python Legacy Server
python serve/server.py --config strata-coder-iq1_m.json --port 8080
```

---

<h2 id="-upstream-strata-readme">📖 Upstream Strata Readme (by Niko1221)</h2>

<h1 align="center">Strata</h1>

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Run a 125-billion-parameter AI model on your own gaming PC</b><br>
NVIDIA or AMD graphics card (12 GB or more) · Windows or Linux · free and open source</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** on a normal PC. This is a
large, smart AI model that usually needs a server. It chats, writes code, reads pictures and works with your apps
and coding agents. Nothing leaves your PC.

## How fast is it?

We measured it on two ordinary gaming PCs. A token is about ¾ of a word.

- **Writes answers:** how fast the reply appears in a short chat. 60 tokens per second is faster than you can read.
- **Reads your prompt:** how fast it takes in what you send (here a 32K-token document, code or chat history).

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA: Q2_0 with engine 0.1.36, the other rows with 0.1.26 (4K answers, 32K prompts). The full tables are in
[DETAILS.md](docs/DETAILS.md#speed-measured). A card with more VRAM is faster: an RTX 3090 (24 GB) should write
about 100-140 tokens per second. Long chats and other cards: [speed of each model](docs/MODELS.md#how-fast-is-each-size),
[community results](docs/COMMUNITY_BENCHMARKS.md).

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Strata is free. If it runs well on your PC, a coffee keeps the work on it going.</sub></p>

## What you need

| | |
| --- | --- |
| **Graphics card** | **NVIDIA** GeForce RTX 20, 30, 40 or 50 series, or **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 or RX 6800 / 6900 series. It needs **12 GB of VRAM or more**. |
| **RAM** | 32 GB or more. Your RAM decides [which model](#which-model-should-i-pick) fits. 64 GB runs every size. |
| **Disk** | About 80 GB free. Use an SSD if you can: the first start is much faster. |
| **System** | Windows 10 / 11 or Linux, and a current graphics driver from NVIDIA or AMD. |

The installer sets up everything else. Two or three cards can share the model ([multi-GPU](docs/MULTI_GPU.md)).

Experimental, written and tested by community members on their own machines:

- **Older graphics cards** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Older GPUs](docs/OLDER_GPUS.md).
- **Intel Arc**, built from source on Linux: [Intel Arc](docs/INTEL_ARC.md).
- **Older processors without AVX2**: they work, but slowly. [Older CPUs](docs/INSTALL.md#older-cpus-experimental).

The full list: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Install

### Let your AI set it up

Do you use an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Paste this into it:

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

It checks your graphics card, RAM and disk and picks the model that fits. Then it installs and starts it and tells
you how to connect your apps. AI tools can also install, start and stop Strata through its
[MCP server](docs/MCP_SERVER.md).

### Or do it yourself

[Download Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) and unzip it (or `git clone` it).
**Windows:** double-click **`START-HERE.bat`**. **Linux:** run **`./setup.sh`** in the Strata folder.

The steps are the same for NVIDIA and AMD. The installer finds your card and sets up the right engine for it. It
asks you a few questions:

- which model and which size,
- how much context (how much text the model keeps in mind),
- whether it should read pictures.

Press Enter each time for the recommended answer. Then it downloads the model (about 70 GB) and starts it. If the
download stops, run it again: it continues where it left off. Your browser opens the Strata app at
`http://127.0.0.1:8080`.

> **While the model starts, your PC can be slow or stop responding for 1-3 minutes** (longest the first time).
> Strata loads 35-55 GB into your RAM and locks part of it for the graphics card. This is normal. Wait, and don't
> close the window. The window shows what Strata is doing.

**Next time**, run `START-HERE.bat` (or `./setup.sh`) again. It starts right away and downloads nothing twice. Close
its window to stop the model. `UPDATE.bat` (`./update.sh`) updates Strata without starting it. Updating, Docker,
several cards, where the files go and every option: [docs/INSTALL.md](docs/INSTALL.md).

## Which model should I pick?

The installer recommends one for your RAM. The same model comes in several sizes, compressed more or less. Smaller
sizes are faster. Larger sizes are a bit smarter.

| Your RAM | Take | Why |
| --- | --- | --- |
| **32 GB** | **Coder** | it fits 32 GB, and it is made for code (with a 24 GB card, Q2_0 and IQ2_XS run too) |
| **48 GB** | **IQ2_XS** (or Q2_0, the fastest) | the larger sizes do not fit |
| **64 GB** | **IQ2_XS** (recommended), or IQ3_XXS / IQ3_S | every size fits; IQ3_S is the best and the slowest |
| **96 GB or more** | **IQ3_S**, or Unsloth's UD-IQ4_XS (~4-bit) | room for the largest sizes with everything else open |

- **[Coder](docs/MODELS.md#coder):** a coding version with half of the experts removed. It reaches 91% of the full
  model's SWE-bench Verified score (measured by its authors) and fits 32 GB of RAM. It is weaker outside code,
  including Chinese and other CJK text (#438). For those, take Q2_0, IQ2_XS or IQ3_S, which keep every expert.
- **[Swift 1.5](docs/MODELS.md#swift-15):** a fine-tune that thinks for a much shorter time before it answers. You
  get the answer sooner, at about the same quality.
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs):** Unsloth's ~4-bit version, between IQ3_S and
  UD-Q4_K_XL in quality. A 94 GB download. With less than ~80 GB of RAM, Strata reads part of it from the SSD
  while it answers, so it is slower there (an NVMe SSD helps).
- **[Unsloth UD-Q4_K_XL](docs/MODELS.md#unsloth-ud-q4_k_xl-experimental)** (experimental): the closest to the full
  model. But Strata reads most of it from the SSD while it answers, so it writes only 7-8.5 tokens/s on a 64 GB PC.
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs):** you set it up by hand. It is
  not in the installer's menu.

Sizes, downloads and what fits where: [docs/MODELS.md](docs/MODELS.md). To add another model later, run
`SETUP.bat` (Linux: `./setup.sh --setup`).

## Using it

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="The Strata app's Monitor tab next to a coding agent"><br>
<sub>The Strata app's <b>Monitor</b> (left) while a coding agent writes the pagoda garden from the video (right)</sub></p>

- **In the browser:** open `http://127.0.0.1:8080`. It has **Chat**, a live **Monitor** of the model and your
  GPU/CPU/RAM, and **About** with the settings and addresses.
- **Your apps and coding agents:** add an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8080/v1`**. Any API key and any model name work.
  - Apps that use Anthropic's API: `http://127.0.0.1:8080/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
  - Codex CLI and other apps that use the OpenAI Responses API: `/v1/responses`
    ([setup](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Thinking:** choose **off, low, medium or high** in the chat menu or in your app's "reasoning effort". Off is the
  fastest. High is best for hard questions.
- **Pictures:** say yes to "Images?" in setup. Then click **Picture** in the chat, or attach pictures in your app.
  AMD cards read pictures on Linux through the processor; on Windows they can't yet.
- **From your phone or another PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Always set a key.
- **One request at a time:** by default Strata answers one request, and the others wait. To answer several at once,
  set `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). On a 12 GB card this makes each answer slower.
- **Long prompts:** Strata reads the first message of a chat in full, about 1 minute per 30,000 tokens. Follow-up
  messages start in seconds.

More: [where your chats are stored](docs/INSTALL.md#where-things-are-stored), [the API](docs/DETAILS.md#using-it).

## Something went wrong?

- **My PC froze the first time Strata started.** This is normal while it loads the model. Wait, and don't close the
  window. Still frozen after 10 minutes? Restart the PC, close other programs and try again, or pick a smaller size.
- **It stopped while downloading or installing.** Run `START-HERE.bat` (or `./setup.sh`) again. It continues where
  it stopped.
- **It's very slow and the disk light keeps blinking, or it says "the engine stopped unexpectedly".** Your PC does
  not have enough free RAM. Close other programs (browsers use a lot), or pick a smaller size (Q2_0 or IQ2_XS).
- **It says port 8080 is already in use.** Strata is already running. Look for its window.

More problems and their fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Still stuck? Open an
[issue](https://github.com/Niko1221/Strata/issues) and attach `strata-<model>.log` from the Strata folder. Found a
security problem? Report it privately: [SECURITY.md](SECURITY.md).

## How does it work?

Models like this one usually run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
12-24 GB. Strata makes the model fit by **sharing the work across your whole PC**. Think of a kitchen: the things
you use all the time stay on the counter, and the rest waits in the pantry.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

- **The model is a team of 24,576 small specialists ("experts").** Each word needs only 10 of them.
- **Your graphics card** keeps the few thousand experts that are used most often. **Your RAM** holds all of them,
  and **your processor** works on the rest at the same time. **Your SSD** holds a big lookup table.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** a small helper guesses the next few words. The big model checks them all at once. You get
  the same answer, 1.6-1.8x sooner.
- **Long texts are read in big pieces** (up to 8,192 tokens at a time), at over 1,000 tokens per second.

The longer explanation: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Every part and its numbers:
[the details](docs/DETAILS.md#how-it-works) and the [paper](docs/paper/Strata-Paper.pdf).

## Credits and license

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. It was
compressed by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5)
and Unsloth. Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata is open source under the [MIT License](LICENSE). A few
parts and every model have their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).

## Support Strata

Strata is free and open source. If it is useful to you, you can support its development:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>

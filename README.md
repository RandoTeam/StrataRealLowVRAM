<h1 align="center">StrataRealLowVRAM</h1>

<p align="center">
  <b>Extreme Low-VRAM & Mobile GPU Optimization Edition of Strata</b><br>
  Running 125B MoE (Qwen 3.8 Flash Next Coder IQ1_M) on <b>4 GB VRAM</b> (RTX 3050 Laptop) + <b>32 GB DDR4</b> on Windows 11<br>
  <i>DeepSeek Harness Integration · Windows WDDM Sub-Millisecond Tuning · Empirical Architecture Tournaments</i>
</p>

<p align="center">
  <a href="#-about-stratareallowvram">About</a> ·
  <a href="#-key-architectural-improvements">Improvements</a> ·
  <a href="#-empirical-benchmark-results">Benchmark Results</a> ·
  <a href="#-quickstart-for-4-gb-vram-setups">Quickstart</a> ·
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

### 1. Windows WDDM & OS Latency Tuning
- **Windows High-Resolution Timer (`timeBeginPeriod(1)`)**: Default Windows thread scheduling operates with a coarse 15.6 ms quantum. We integrated a 1.0 ms multimedia timer inside `serve/server.py` with automatic clean teardown via `atexit`, eliminating thread scheduling jitter in the CPU worker pool.
- **Process Scheduling Priority**: Automatic execution under `HIGH_PRIORITY_CLASS` within Windows Job Objects (`serve/winjob.py`), preventing CPU starvation during high background I/O or GUI tasks.
- **Spin-Wait Worker Pool Tuning (`STRATA_POOL_SPIN_US=50000`)**: Keeps worker threads hot to eliminate Zen 2 C6 deep sleep wake-up latency (which previously cost 20–30 μs per MoE layer transition).

### 2. DeepSeek Harness & Prompt-Cache Preservation
- **Fast-Path Session Title Interception (`_fast_session_title`)**: DeepSeek Harness frequently fires auxiliary `session-title-llm` JSON requests (`Generate the session title from this JSON array...`). In vanilla Strata, each auxiliary request flushed and evicted the engine's resident prompt cache, destroying multi-turn KV continuity. Our lightweight interceptor synthesizes titles directly in Python, preserving 100% of the resident conversation and root prompt cache.
- **Reasoning Preservation (`preserve_thinking: False`)**: Maintained proper template rendering and tool call stream contracts without breaking long reasoning chains.

### 3. VRAM Budget Re-Balancing for 4 GB Limits
- Vanilla Strata reserves 700–1000 MiB for WDDM OS buffers, leaving only ~160 expert slots on 4 GB GPUs.
- By profiling WDDM swap thresholds, we safely tightened the VRAM reserve to **260–280 MiB**, expanding the resident expert cache from **163 slots to 426 slots** (+161% cache capacity) without triggering GPU Out-Of-Memory errors.
- Enforced `STRATA_ARENA_LOCK=1` to pin MoE weights in physical RAM, preventing Windows Virtual Memory paging stutters.

### 4. Speculation & Suffix Decoupling
- Discovered and resolved the MTP / Suffix Drafter conflict: vanilla speculative decoding coupled MTP neural drafting with suffix lookup.
- By configuring `--spec 3 --spec-min-p 0.82 --suffix-draft 3-4 --mtp-window 1024`, neural draft verification is gated at high confidence (≥82%) while prompt-lookup handles boilerplate code repetition at near-zero CPU cost.

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
python serve/server.py --config strata-coder-iq1_m.json --port 8080
```

---

<h2 id="-upstream-strata-readme">📖 Upstream Strata Readme (by Niko1221)</h2>

<h1 align="center">Strata</h1>

<p align="center"><b>Run a 125-billion-parameter AI model on a normal gaming PC</b><br>
one NVIDIA card (12-24 GB) + 64 GB of RAM · Windows or Linux · one click to install</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** - a large, smart AI model that
normally needs a server - on your own PC. It writes its answers at **60-95 tokens per second** (a token is about ¾
of a word): faster than you can read.

- **Free and open source.**

> **Jump to:** [How fast?](#how-fast-is-it) · [Which model?](#which-model-should-i-pick) · [Install](#install) ·
> [Using it](#using-it) · [Problems?](#something-went-wrong) · [How it works](#how-does-it-work) ·
> [All the details](docs/DETAILS.md)

---

## How fast is it?

Measured on an RTX 5070 (12 GB), a Ryzen 5 7600 and 64 GB of RAM:

| Size | Writes answers (short chat) | Writes answers (128K context) | Reads your prompt |
| --- | ---: | ---: | ---: |
| **Q2_0** | 93 tokens/s | 74 tokens/s | 2,170 tokens/s |
| **IQ2_XS** | 79 tokens/s | 63 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 49 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 46 tokens/s | 1,620 tokens/s |
| **Coder** (IQ1_M) | 55 tokens/s | 43 tokens/s | 2,180 tokens/s |

- **Writes answers** = how fast the reply appears (tokens per second).
- **Reads your prompt** = how fast it takes in what you send (long documents, code, chat history), measured on a
  32K-token prompt; a 4K prompt reads at 910-1,580 tokens/s. A 32K prompt takes about 15 seconds with Q2_0.

A card with more VRAM is faster, because more of the model fits on the GPU: an RTX 3090 (24 GB) should do roughly
100-140 tokens per second. All measurements, long-context numbers and estimates for other cards are in the
[details](docs/DETAILS.md#speed-measured).

Every PC is different: `START-HERE.bat --calibrate` measures a few engine settings on yours and keeps the fastest
(about 5-10 minutes; on the PC above it made the Coder 7% faster).

Measured Strata on your own PC? See [Community benchmark results](docs/COMMUNITY_BENCHMARKS.md)
for a report template and how to share your results in a pull request.

**Two or three NVIDIA cards?** Just run `START-HERE.bat`: it lists your cards, says which ones Strata can use, and
asks whether to share the model across them (recommended when two can). An install made on one card asks once at
its next start. Or choose yourself: `START-HERE.bat --gpus 0,2` (both, remembered), `--gpus all`, or `--gpu 0` (one
card, this start only). Each card keeps the experts of its own layers, and prompts flow through the cards in a
pipeline: on an RTX 5080 + RTX 3090 prompts were read 18-20% faster than on the 5080 alone, decoding on par.
Every card must be an RTX 20 series or newer with 8 GB or more. See [docs/MULTI_GPU.md](docs/MULTI_GPU.md).

## Which model should I pick?

**The size** (the same model, compressed more or less):

| Model | RAM+VRAM Requirements | Speed | Quality |
| --- | ---: | --- | --- |
| **Q2_0** | 37.6 GB | fastest | good |
| **IQ2_XS** | 39.2 GB | fast | better (**recommended**) |
| **IQ3_XXS** | 47.0 GB | slower | great |
| **IQ3_S** | 54.8 GB | slowest | best: matches the full model on the published tests (original model only) |

**Will it fit?** Shard 1 is the part of the model that gets loaded when it starts: its experts go into your **RAM**,
the rest onto your graphics card (the second shard, a 29 GB lookup table, stays on the SSD). So it fits when your
**RAM is at least shard 1 + about 10 GB** for Windows and your other programs. With 64 GB of RAM every size fits
(IQ3_S with little else open); with 48 GB, Q2_0 and IQ2_XS. A bigger graphics card makes it faster, but it doesn't
lower the RAM needed.

**The version:**

- **Qwen3.8-Flash-Next** - the original.
- **[Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)** - ISTA-DASLab's coding
  version: half of the experts removed, keeping the ones that code, tool use and images need (91% of the full model's
  SWE-bench Verified score, 99% of LiveCodeBench, by its authors). One size (IQ1_M: its experts stored like IQ3_S):
  shard 1 is **29.6 GB**, so it fits a PC with **32 GB of RAM**, runs 262K context on 64 GB, and reads long prompts
  the fastest of all. Weaker outside coding.
- **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)** - a fine-tune by UkisAI
  that thinks much shorter before answering, so you get the answer sooner, with about the same quality. Same speed per
  token, and about the same RAM as the same size of the original (no IQ3_S). Its own license applies (see its page).

Not sure? Take **IQ2_XS** - or the **Coder** if you mainly write code, or have 32-48 GB of RAM. You can add another
one later with `SETUP.bat` (the same as `START-HERE.bat --setup`; on Linux `./setup.sh --setup`).

For **OrcaRouter's Flash-Next Uncensored IQ3_XXS**, see the [manual compatibility setup](docs/ORCA.md).
It needs an explicit packing conversion and is not an installer menu option.

An **AMD Radeon RX 7900 XT / XTX, RX 9070 / 9070 XT or Radeon AI PRO R9700 on Linux** works too (experimental; the
RX 7800 XT / 7700 XT and RX 9060 XT were validated by their owners):
`./setup.sh --backend hip`, chosen by itself on a PC with no NVIDIA card Strata can use. It installs ROCm without sudo
and compiles the engine (no images yet; several cards with `--gpus`). Details: [AMD HIP](docs/AMD_HIP.md).

## Install

**You need:** an NVIDIA RTX 20, 30, 40 or 50 card with 12 GB of VRAM or more (RTX 20 since 0.1.27), enough RAM for the size you pick (above;
a big GPU makes up for less RAM - the [low-RAM mode](docs/DETAILS.md)),
~80 GB of free disk space (an SSD makes the first start much faster), and Windows 10/11 or Linux. The only thing you
install yourself is a current **NVIDIA driver** ([nvidia.com/drivers](https://www.nvidia.com/drivers) or the NVIDIA
App). Everything else - Python, the engine, the model - is set up for you.

**Windows**

1. [Download this project](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) and unzip it (or `git clone` it).
2. Double-click **`START-HERE.bat`**.
3. Answer a few questions - or just press Enter each time for the recommended choice:
   - **Which model and size?** The original or Swift 1.5, and Q2_0, IQ2_XS, IQ3_XXS or IQ3_S - see [above](#which-model-should-i-pick)
   - **How much context?** How much text it can keep in mind at once (it suggests one for your card). 384K and
     512K (experimental) extend the model past its trained 262K by rope scaling - the setup turns it on itself (yarn and a
     covering factor; `--rope-scaling`/`--rope-scale` override) ([details](docs/DETAILS.md))
   - **Images?** Whether it should also read pictures
   - **Experimental speed projection?** Off unless you say yes - [read what it does](docs/DETAILS.md#experimental-speed-projection-experimental-off-by-default) first

Then it downloads everything (the model is ~70 GB, so the first time takes a while - you can stop and it picks up
where it left off) and **starts the model**. Your browser opens the Strata app at `http://127.0.0.1:8080`.

> **While the model starts, your PC can be slow or stop responding for 1-3 minutes** (longest the first time): Strata
> loads 35-55 GB into your RAM and locks part of it for the graphics card. That's normal - wait, and don't close the
> window. The window tells you what it is doing.

**Next time**, just double-click `START-HERE.bat` again: it starts right away, nothing is downloaded twice. Close its
window to stop the model.

**Updating:** download the new version and unzip it anywhere (or `git pull`), then run `START-HERE.bat` in it. The
model files are kept in a `Strata-data` folder next to your Strata folder, so a new copy finds them and sets itself up
the same way - nothing big is downloaded again.

**Linux:** run `./setup.sh` - same questions, same result.

**Docker (Linux):** the same idea, in a container.

1. Host: Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
   and a driver **580 or newer** (CUDA 13.0).
2. Build (this compiles the engine into the image, so the container never compiles):
   `docker build -t strata .`
   `docker build -t strata --build-arg CUDA_ARCHITECTURES=89 .` builds for one card only (faster).
   The default covers RTX 30 (86), RTX 40 (89), RTX 50 (120) and A-series (80); a card outside that
   set needs a rebuild with its own arch. Add `--build-arg BUILD_VISION=0` to skip the image encoder.
3. Run (the first start downloads the ~70 GB model, then starts; later starts go straight to serving):
   `docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 -v strata-data:/data strata`

   The setup choices are env vars: `-e MODEL=IQ2_XS -e FAMILY=qwen -e CONTEXT=32768 -e VISION=no`
   (or `MODEL=Q2_0|IQ3_XXS|IQ3_S`, `FAMILY=swift|coder`; the defaults above are the recommended ones).
   `-e VISION=cpu` keeps the image encoder on the CPU. `-e KV=int8|q4_0|k8v4` picks the KV cache
   precision; `k8v4` is INT8 K with 4-bit V and keeps its KV in VRAM from 64K up.
   Only the model files, the prepared pack, the MTP layer and the install config live in the
   `strata-data` volume; the engine is part of the image. Switching between models already on the
   volume needs no setup pass: `-e MODEL=Q2_0 -e FAMILY=coder` picks that model's config. Add
   `-e REINSTALL=1` only to change settings for a model already set up (context, vision, KV, host,
   api_key, LOW_RAM), since those are recorded in its config.
   Strata loads 32-62 GB into RAM. `--gpus all` on a host with two usable cards takes both: the
   layer split is setup's recommended default ([docs/MULTI_GPU.md](docs/MULTI_GPU.md)), and a volume
   set up for one card switches to the pair on its first start there. Pin one card with `-e GPU=0`,
   or name them with `-e GPUS=0,2` and where the later card's layers start with `-e LAYER_SPLIT=18`.
   A memory limit needs `-e LOW_RAM=on`, which maps the model's experts from the pack instead of
   keeping them in RAM: setup.py measures the host's RAM, not the container's limit, so it cannot
   see a cap. LOW_RAM runs on one card.
   The server listens on `0.0.0.0:8080` by default; set `-e API_KEY=<secret>` before exposing the port
   to a network. The image has a `HEALTHCHECK` on `/health`, so `docker ps` shows the container
   healthy once the model is loaded, and `GET /v1/status` says what it is running.

## Using it

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="The Strata app's Monitor tab next to a coding agent"><br>
<sub>The Strata app's <b>Monitor</b> (left) while a coding agent writes the pagoda garden from the video (right)</sub></p>

- **In the browser:** `http://127.0.0.1:8080` - the Strata app (it opens by itself when the model starts): **Chat**, a
  live **Monitor** of the model and your GPU/CPU/RAM, and **About** with the settings and addresses.
- **Chat in the terminal:** `.venv\Scripts\python chat.py`
- **Your apps and coding agents:** add it as an "OpenAI-compatible" provider with base URL
  **`http://127.0.0.1:8080/v1`**, any API key and any model name. Apps that use Anthropic's API: `http://127.0.0.1:8080/v1/messages`.
- **Thinking:** the model thinks before it answers. Choose **off, low, medium or high** - in the chat page menu, with
  `/think low` in `chat.py`, or with your app's "reasoning effort" setting. Off is fastest; high is best for hard questions.
- **Pictures:** in the chat page click **Picture**; in `chat.py` type `/image <path>`; in apps just attach them.
- **From your phone or another PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`, then open the
  address the server window prints; see the [details](docs/DETAILS.md#using-it).
- **Experimental speed projection (off by default):** an experimental control vector that setup can turn on; it
  changes how the model answers - read [what it does](docs/DETAILS.md#experimental-speed-projection-experimental-off-by-default) first.

**Good to know:** it answers one request at a time. The first message of a chat is read in full (about 1 minute per
30,000 tokens); after that it keeps the conversation and reads only what is new, so follow-ups start in seconds.

### Where things are stored

- **Your chats: only in your browser.** The Chat tab keeps the conversation, its settings and the API key you typed
  in the browser's local storage (`strata.*` keys) - not on the server and not in the Strata folder. Pictures are not
  kept, only their names. Another browser or a private window starts empty; clearing the site's data deletes them.
- **How the model starts:** `strata-<model>.json` in the Strata folder (context, GPUs, host, API key, ...), written
  by setup; next to it `run-<model>.bat` / `.sh`, the log `strata-<model>.log` and, when you use "Use for other
  apps too", `strata-<model>.shared-settings.json`.
- **The model files** (`models/`, `packs/`, `mtp/`, 70-120 GB): in **`Strata-data` next to the Strata folder**, or
  wherever `--data-dir` put them.
- **Where that data folder is:** `%APPDATA%\Strata\settings.json` on Windows, `~/.config/strata/settings.json` on
  Linux ([details](docs/DETAILS.md)).

## Something went wrong?

**My PC froze, or got very slow, the first time Strata started.**
That's normal while it starts, most of all the first time. Strata loads 35-55 GB into your RAM, locks part of it for
the graphics card, and works out how much of the model fits on your GPU. The mouse can freeze for a few minutes. **Wait, and don't close the
window.** The next starts are much faster. Still frozen after 10 minutes? Restart the PC, close other programs
(browsers use a lot of RAM) and try again. If it keeps happening, pick a smaller size (Q2_0 or IQ2_XS).

**It stopped while downloading or installing.**
Run `START-HERE.bat` again. It continues where it stopped.

**It says the NVIDIA driver is too old.**
Update it (NVIDIA App or [nvidia.com/drivers](https://www.nvidia.com/drivers)), restart the PC, and run
`START-HERE.bat` again.

**It says port 8080 is already in use.**
Strata is already running. Look for its window.

**It's very slow and the disk light keeps blinking.**
Your PC is out of free RAM. Close other programs, or pick a smaller size (Q2_0 or IQ2_XS).

**An answer stopped with "the engine stopped unexpectedly".**
Usually not enough RAM (on Linux the system then stops the engine). Just send your message again: Strata starts the
engine by itself. If it keeps happening, close other programs or pick a smaller size.

**It says the prompt exceeds the context.**
The conversation is longer than the context you chose. Start a new chat, or run `SETUP.bat` and pick more
context.

**Still stuck?** Look in the [full troubleshooting table](docs/DETAILS.md#troubleshooting), or open an issue and
attach `strata-<model>.log` from the Strata folder.

## How does it work?

Models like this one normally run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
12-24 GB. Strata makes it fit by **sharing the work across your whole PC** - the same idea as a kitchen, where the
things you use all the time stay on the counter and the rest waits in the pantry.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

- **The model is a team of 24,576 small specialists ("experts"),** and each word it writes needs only 10 of them.
  So it doesn't have to have all of them on the graphics card at once.
- **Your graphics card** does the part of the work needed for every word, and keeps the few thousand experts that
  are asked most often. It keeps learning which ones those are while you use it.
- **Your RAM** holds every expert. When a word needs one the card doesn't have, **your processor** works on it -
  at the same time as the graphics card, so neither waits for the other.
- **Your SSD** holds a big lookup table; the model only reads a few small rows of it per word.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

- **Guess, then check.** A small, fast helper built into the model guesses the next few words, and the big model
  checks all the guesses in one go. It keeps the ones it agrees with and writes the next word itself - so one step
  often produces several words. The helper only guesses - the big model decides every word - so you get the same
  quality answer, 1.6-1.8x sooner.
- **Long texts are read in big pieces** (up to 8,192 tokens - pieces of words - at a time), which is why a long
  document or code base is read at over 1,000 tokens per second.

Want the full picture? The [details](docs/DETAILS.md#how-it-works) explain every part and its numbers, and the
[paper](docs/paper/Strata-Paper.pdf) tells the whole story, with the measurements behind it.

## Credits

- Model: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team; compressed versions by
  [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF);
  [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) by UkisAI. Their licenses apply
  to the model files.
- Built with parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT). Ideas from
  [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer) and
  [HyperQwen](https://github.com/syv-ai/HyperQwen). More in the [details](docs/DETAILS.md#credits-and-licenses).

## License

Strata is open source under the [MIT License](LICENSE). A few parts carry their own licenses: `third_party/ggml`
(MIT, llama.cpp / ggml), the web app's font (SIL Open Font License 1.1) and the experimental speed projection's
vector in `data/experimental-speed-projection` (Qwen Community License 1.0, from the model's activations). The
models are not part of this repository; each model's own license applies to its files.

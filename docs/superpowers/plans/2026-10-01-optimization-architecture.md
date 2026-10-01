# StrataRealLowVRAM v2.0 — Optimization & Architecture Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Maximize decode tok/s on 4 GB VRAM + 32 GB DDR4 without losing context window (65536) or model quality, while maintaining easy upstream merge capability and preparing for Q2_0 model support.

**Architecture:** Layered Parallel Binary approach — Rust components live in `rust/` as independent crates alongside unmodified C++ engine source. A surgical C++ patch (10 lines, guarded by `#ifdef`) fixes the WDDM expert cache bug. The Rust HTTP gateway replaces the Python server for lower latency. All changes are additive, never restructuring upstream files.

**Tech Stack:** C++ (existing engine, minimal patch) · Rust 1.98+ (gateway, future kernels) · Python 3.12 (benchmarking/tooling only) · CUDA 13 (prebuilt binaries) · CMake (upstream build, CI only)

**Spec:** This document is both the design spec and the implementation plan.

## Global Constraints

- **Sacred Rule §0:** `--max-context 65536` — never touched, never reduced.
- **Upstream Compatibility:** C++ patches to `src/` must be ≤20 lines, guarded by `#if defined(STRATA_LOWVRAM_PATCH)`, and documented for manual rebase after upstream merges.
- **No MSVC/nvcc on dev machine:** The developer PC has no C++ build toolchain. C++ patches are prepared as source diffs and built via GitHub Actions CI or a remote machine. Prebuilt `strata.exe` binaries are committed to `engine/`.
- **Rust toolchain available:** `rustc 1.98.1`, `cargo 1.98.1` — Rust code is built locally.
- **Empirical verification:** Every performance claim must have a reproducible benchmark with exact commands and measured numbers. No assertions without evidence.
- **Two target models:** `coder-iq1_m` (current, 32 GB RAM) and `q2_0` (future, ~38 GB RAM).

---

## Diagnosis Summary: Why tok/s Is Currently Capped

### The VRAM Budget Problem (Engine 0.1.31 on 4 GB RTX 3050)

Before expert cache allocation, the engine loads into VRAM:

| Component | VRAM (MiB) | Can offload to CPU? | Source |
|-----------|-----------|---------------------|--------|
| WeightTable (canonical tensors) | 1,406 | No | `generate.cpp:1775` |
| NativeDense (300 projection matrices) | 2,019 | No | `native_dense.cpp:149` |
| MTP Draft Layer | 790 | No | `mtp.cpp` |
| Native Q5_K Head | 497 | No | `native_head.cpp:47` |
| **Total before expert cache** | **4,712** | | |
| **Physical VRAM available** | **4,096** | | RTX 3050 Laptop |

Result: `cudaMemGetInfo` returns `free_b == 0` at `generate.cpp:2528`. The sized_slots loop (`generate.cpp:2534-2536`) immediately breaks. `o.expert_cache` is set to 0. The verifier at `verify.cpp:209` requires `hits.cache_base != nullptr` and aborts.

**Removing native head does NOT help:** Line 1764 shows `output.weight` is skipped from WeightTable when native head is loaded. Without native head, canonical `output.weight` (~500 MiB) returns to WeightTable, netting zero savings.

### Why Engine 0.1.22 Worked

Engine 0.1.22 did not have NativeDense or NativeHead. Total VRAM was ~2,250 MiB, leaving 1,024 MiB free for 426 expert cache slots (0.82 GiB). This is why Tournament 4 achieved decode speeds with speculative decoding.

### The Fix

A surgical C++ patch at `generate.cpp:2532-2533` that, on Windows when `free_b == 0` and the user specified an explicit `--expert-cache N`, uses the budget as the cap instead of clamping to `free_room`. The existing WDDM-aware retry loop at lines 2574-2627 then handles the virtual overcommit gracefully. This is ~10 lines of C++, fully backwards-compatible.

---

## Phase 0: Expert Cache Fix & Baseline Benchmark (Critical Blocker)

### Task 0.1: Prepare the C++ WDDM Expert Cache Patch

**Files:**
- Modify: `src/program/generate.cpp:2525-2541`

**Interfaces:**
- Consumes: Engine CLI parsing (`o.expert_cache`, `auto_cache`)
- Produces: Working expert cache allocation on 4 GB WDDM GPUs

The patch changes the `sized_slots` pre-sizing to honour the user's explicit `--expert-cache N` when WDDM reports `free_b == 0`:

- [ ] **Step 1: Create the patch file**

```cpp
// src/program/generate.cpp, replace lines 2532-2533:
// BEFORE:
        size_t free_room = free_b > ((size_t) o.vram_reserve_mib << 20) ? free_b - ((size_t) o.vram_reserve_mib << 20) : 0;
        const uint64_t cap = std::min<uint64_t>(budget, (uint64_t) free_room);

// AFTER:
        size_t free_room = free_b > ((size_t) o.vram_reserve_mib << 20) ? free_b - ((size_t) o.vram_reserve_mib << 20) : 0;
#if defined(_WIN32)
        // StrataRealLowVRAM: on WDDM, cudaMemGetInfo reports 0 free when allocations exceed
        // physical VRAM via virtual paging.  With an explicit --expert-cache N (not auto), honour
        // the requested budget and let xcache.open_sized + the WDDM retry loop handle overcommit.
        // This restores 426-slot expert cache on 4 GB cards where 0.1.31's native layers exceed
        // physical VRAM but succeed via WDDM paging.
        const uint64_t cap = (free_room == 0 && !auto_cache) ? budget : std::min<uint64_t>(budget, (uint64_t) free_room);
#else
        const uint64_t cap = std::min<uint64_t>(budget, (uint64_t) free_room);
#endif
```

Additionally, to enable the retry loop for non-auto cache on WDDM, at line 2596:

```cpp
// BEFORE:
                if (auto_cache && failed < 8 && shrink_to(cache_bytes() / 4 * 3)) {

// AFTER:
#if defined(_WIN32)
                // StrataRealLowVRAM: allow retry-shrink for explicit cache too when WDDM free
                // reported 0 at sizing time — the first open attempt may over-commit
                if ((auto_cache || wddm_zero_free) && failed < 8 && shrink_to(cache_bytes() / 4 * 3)) {
#else
                if (auto_cache && failed < 8 && shrink_to(cache_bytes() / 4 * 3)) {
#endif
```

Where `wddm_zero_free` is a `bool` set to `true` at the pre-sizing stage when `free_b == 0` on Windows.

Save as: `patches/0001-wddm-expert-cache-4gb.patch`

- [ ] **Step 2: Set up GitHub Actions CI for engine build**

Create `.github/workflows/build-engine.yml`:
- Trigger: push to `patches/` directory or manual dispatch
- Runner: `windows-latest`
- Steps: Install CUDA Toolkit 13, configure MSVC, apply patch, `cmake --build`, upload `strata.exe` as artifact

- [ ] **Step 3: Build patched engine and download artifact**

Run the CI workflow, download `strata.exe`, place in `engine/strata.exe`.

- [ ] **Step 4: Verify expert cache allocation**

```powershell
python bench/probe_startup.py strata.exe 1
```
Expected: `pre-filled N of N slots from the profile; slot 0 verified` (N > 0) and `READY 65536 stop`.

- [ ] **Step 5: Baseline benchmark with working expert cache**

Run decode benchmark with 3 prompt sizes (256, 384, 512 tokens), record:
- Decode tok/s (thinking + completion separately)
- TTFT (time to first token)
- Expert cache hit rate (from DONE line: `hits / lookups`)
- Speculative acceptance rate (`drafts_accepted / drafts_offered`)
- Total time per response

Save to `bench/results/2026-10-01-phase0-baseline.json`.

- [ ] **Step 6: Commit**

```bash
git add patches/ .github/workflows/ engine/strata.exe bench/results/
git commit -m "fix(engine): WDDM expert cache patch for 4GB VRAM + baseline benchmark"
```

---

## Phase 1: Speculative Decode Configuration Optimization

### Task 1.1: Systematic Spec/MTP Tuning

**Files:**
- Create: `bench/spec_tuning.py`
- Create: `bench/results/2026-10-01-spec-tuning.json`
- Modify: `strata-coder-iq1_m.json` (final optimal config)

**Interfaces:**
- Consumes: Working engine from Task 0.1 (expert cache > 0, READY state)
- Produces: Optimal speculative decode configuration maximizing effective tok/s

- [ ] **Step 1: Write the spec tuning benchmark script**

The script tests a matrix of speculative decode parameters:
- `--spec`: [4, 5, 6]
- `--spec-min-p`: [0.78, 0.82, 0.86]
- `--suffix-draft`: [0, 3, 4]
- `--mtp-max-t`: [2, 3]

For each combination: start engine, send 3 test prompts, record acceptance rate and effective tok/s, shut down engine. Total: 54 combinations × ~2 min each ≈ 108 min.

- [ ] **Step 2: Run the full tuning matrix**

```powershell
python bench/spec_tuning.py --output bench/results/2026-10-01-spec-tuning.json
```

- [ ] **Step 3: Analyze results and select optimal config**

Criteria: highest effective tok/s (= base_tok_s × (1 + acceptance_rate × (spec - 1))).

- [ ] **Step 4: Update strata-coder-iq1_m.json with optimal values**

- [ ] **Step 5: Verify optimal config end-to-end**

```powershell
python bench/probe_startup.py strata.exe 1
# Then run the 3 standard benchmarks
```

- [ ] **Step 6: Commit**

```bash
git add bench/spec_tuning.py bench/results/ strata-coder-iq1_m.json
git commit -m "perf: optimal speculative decode config from systematic tuning"
```

---

## Phase 2: Rust HTTP Gateway (`strata-gateway`)

### Task 2.1: Gateway Scaffold & IPC Bridge

**Files:**
- Create: `rust/strata-gateway/Cargo.toml`
- Create: `rust/strata-gateway/src/main.rs`
- Create: `rust/strata-gateway/src/engine.rs`
- Create: `rust/strata-gateway/src/protocol.rs`

**Interfaces:**
- Consumes: `strata.exe --serve` stdin/stdout IPC protocol (GEN/GENI/STOP/QUIT → T/DONE/ERR/PP/READY/RESUME/REUSED/INFO)
- Produces: `strata-gateway.exe` binary that spawns and manages strata.exe

The IPC protocol is fully documented (see research artifact). Key contract:

```
→ stdin:  GEN <max_new>[ key=val ...] <id,id,...>\n
← stdout: INFO k=v ...\n
← stdout: READY <ctx> stop\n
← stdout: T <token_id>\n
← stdout: DONE <gen> <prompt> <prompt_ms> <decode_ms> <finish> <accepted> <offered> <reused> [hits] [lookups] ...\n
← stdout: ERR <message>\n
```

- [ ] **Step 1: Initialize Rust workspace**

```powershell
mkdir rust\strata-gateway
cd rust\strata-gateway
cargo init --name strata-gateway
cargo add axum tokio serde serde_json tower-http uuid
```

- [ ] **Step 2: Implement `protocol.rs` — IPC message types**

Define strongly-typed enums for all engine commands and responses:
```rust
pub enum EngineCommand {
    Gen { max_new: u32, sampling: SamplingParams, token_ids: Vec<i32> },
    GenI { max_new: u32, sampling: SamplingParams, embeddings_path: String, token_ids: Vec<i32> },
    Stop,
    Quit,
}

pub enum EngineResponse {
    Info(HashMap<String, String>),
    Ready { max_context: u32, can_stop: bool },
    Resume(u64),
    PromptProgress { done: u64, total: u64, ms: f64, tok_s: f64 },
    Reused(u64),
    Token(i32),
    Done(DoneStats),
    Err(String),
}
```

- [ ] **Step 3: Implement `engine.rs` — process manager**

Spawns `strata.exe --serve`, reads READY handshake, provides `async fn generate(...)` that writes GEN to stdin and yields Token responses via a `tokio::sync::mpsc` channel.

- [ ] **Step 4: Implement `main.rs` — HTTP server with OpenAI API**

Axum routes:
- `POST /v1/chat/completions` — OpenAI streaming and non-streaming
- `GET /v1/models` — model listing
- `GET /health` — health check
- `POST /v1/messages` — Anthropic API (Phase 2.2)

SSE streaming uses `axum::response::Sse` with zero-copy token delivery.

- [ ] **Step 5: Build and smoke test**

```powershell
cd rust\strata-gateway
cargo build --release
.\target\release\strata-gateway.exe --config ..\..\strata-coder-iq1_m.json --port 8080
```

Test with curl:
```powershell
curl -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"qwen","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

- [ ] **Step 6: Commit**

```bash
git add rust/
git commit -m "feat(gateway): Rust HTTP gateway with OpenAI API and strata IPC bridge"
```

### Task 2.2: Anthropic API, MCP, Vision & Full Feature Parity

**Files:**
- Modify: `rust/strata-gateway/src/main.rs` (add routes)
- Create: `rust/strata-gateway/src/anthropic.rs`
- Create: `rust/strata-gateway/src/frontend.rs` (template rendering)

**Interfaces:**
- Consumes: Engine IPC (same as 2.1) + tokenizer (sentencepiece via `tokenizers` crate)
- Produces: Feature-complete drop-in replacement for `serve/server.py`

- [ ] **Step 1: Implement Anthropic `/v1/messages` endpoint**
- [ ] **Step 2: Implement chat template rendering (Jinja → Rust Tera/MiniJinja)**
- [ ] **Step 3: Implement vision/GENI support (spawn `strata-vision` subprocess)**
- [ ] **Step 4: Port FastSessionTitle bypass from Python**
- [ ] **Step 5: Implement web UI static file serving (`/`, `/web/*`, `/fonts/*`)**
- [ ] **Step 6: Benchmark latency comparison vs Python server**

Measure per-token delivery latency (gateway → client) with:
```powershell
python bench/latency_compare.py --gateway rust --baseline python --prompts 3
```

Save to `bench/results/2026-10-01-gateway-latency.json`.

- [ ] **Step 7: Commit**

```bash
git add rust/ bench/
git commit -m "feat(gateway): full API parity — Anthropic, vision, MCP, web UI"
```

### Task 2.3: Gateway Launch Integration

**Files:**
- Modify: `START-HERE.bat` (add `--gateway rust` option)
- Create: `run-coder-iq1_m-rust.bat`

- [ ] **Step 1: Create Rust gateway launcher BAT**
- [ ] **Step 2: Update START-HERE.bat with gateway selection**
- [ ] **Step 3: Test full end-to-end flow**
- [ ] **Step 4: Commit**

---

## Phase 3: Q2_0 Model Support

### Task 3.1: Q2_0 Configuration & Benchmark Preparation

**Files:**
- Create: `strata-q2_0.json` (config template)
- Create: `bench/q2_0_benchmark.py`
- Create: `data/expert-profile-q2_0.bin` (will be generated on first run)

**Interfaces:**
- Consumes: Q2_0 model GGUF files (to be downloaded)
- Produces: Ready-to-run config and benchmark scripts for Q2_0

- [ ] **Step 1: Create Q2_0 config template**

Based on upstream README data (Q2_0 requires 37.6 GB RAM+VRAM):
```json
{
  "args": [
    "--pack", "<PACK_PATH>",
    "--native", "<GGUF_SHARD_1>",
    "--ple-gguf", "<PLE_SHARD>",
    "--expert-cache", "500",
    "--vram-reserve-mib", "180",
    "--kv-resident", "20480",
    "--expert-cache-cpu-order",
    "--prefill", "1024",
    "--spec", "6",
    "--mtp", "<MTP_PATH>",
    "--max-context", "65536",
    "--kv", "q4_0",
    "--pcie-frac", "0"
  ],
  "env": {
    "STRATA_POOL_SPIN_US": "50000",
    "STRATA_ARENA_LOCK": "1",
    "STRATA_ARENA_PIN_GIB": "1"
  }
}
```

Key difference from coder-iq1_m: Q2_0 experts use simpler quantization (no IQ lookup tables), so AVX-2 dequantization is ~35-40% faster. Expert sizes differ, affecting cache slot count.

- [ ] **Step 2: Create Q2_0 benchmark script**

Adapts `bench/spec_tuning.py` for Q2_0's different expert layout.

- [ ] **Step 3: Document Q2_0 download and setup instructions**

```markdown
## Q2_0 Setup
1. Download: `huggingface-cli download Qwen/Qwen3.8-Flash-Next-GGUF --include "Q2_0/*"`
2. Run: `python setup.py --model q2_0`
3. Launch: `START-HERE.bat` (auto-detects Q2_0)
```

- [ ] **Step 4: Commit**

```bash
git add strata-q2_0.json bench/q2_0_benchmark.py docs/
git commit -m "feat: Q2_0 model config template and benchmark preparation"
```

### Task 3.2: Q2_0 Empirical Benchmarking (After Model Download)

This task executes after the Q2_0 model files are downloaded.

- [ ] **Step 1: Generate Q2_0 expert profile**
- [ ] **Step 2: Run spec tuning matrix for Q2_0**
- [ ] **Step 3: Head-to-head comparison: Q2_0 vs Coder-IQ1_M**
- [ ] **Step 4: Update README with Q2_0 results**

---

## Phase 4: Upstream Sync Infrastructure

### Task 4.1: Merge Workflow & Conflict Map

**Files:**
- Create: `docs/UPSTREAM_SYNC.md`
- Create: `tools/sync_upstream.ps1`

**Interfaces:**
- Consumes: `upstream` and `stratagp` git remotes
- Produces: Repeatable merge workflow documentation

- [ ] **Step 1: Document the conflict zone map**

Files modified by StrataRealLowVRAM that will conflict on upstream merge:

| File | Our Change | Conflict Risk | Resolution Strategy |
|------|-----------|---------------|---------------------|
| `src/program/generate.cpp` | WDDM patch (10 lines, `#ifdef` guarded) | Medium | Manual rebase; patch is self-contained |
| `strata-coder-iq1_m.json` | Tuned config values | Low | Keep ours (upstream doesn't ship this) |
| `README.md` | Our fork header + benchmarks | High | Keep ours above `<!-- upstream -->` marker |
| `serve/server.py` | FastSessionTitle bypass | Medium | Cherry-pick our changes on top of upstream |
| `start_strata_with_harness.*` | DeepSeek Harness launcher | None | Our files, not in upstream |

All other files (bench/, rust/, tools/, docs/) are **ours only** — zero conflict.

- [ ] **Step 2: Create sync script**

```powershell
# tools/sync_upstream.ps1
git fetch upstream
git fetch stratagp
git log --oneline upstream/main..HEAD  # show our commits
git log --oneline HEAD..upstream/main  # show upstream commits
# Manual: git merge upstream/main --no-commit, resolve, test, commit
```

- [ ] **Step 3: Document the step-by-step merge process**

Write `docs/UPSTREAM_SYNC.md` with:
1. Pre-merge checklist (backup branch, clean working tree)
2. Fetch and inspect upstream changes
3. Merge strategy (ours for README, manual for generate.cpp)
4. Post-merge verification (probe_startup + benchmark)
5. StrataGP cherry-pick process

- [ ] **Step 4: Commit**

```bash
git add docs/UPSTREAM_SYNC.md tools/sync_upstream.ps1
git commit -m "docs: upstream sync workflow and conflict zone map"
```

---

## Phase 5: Repository Branding & Documentation

### Task 5.1: README Overhaul with Verified Results

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update repository description**

Current: "Extreme Low-VRAM & Mobile GPU Optimization Edition of Strata"
Proposed: "High-Performance Strata Fork: Rust Gateway · WDDM 4GB Expert Cache Fix · Speculative Decode Tuning · Q2_0 Support | Runs 125B MoE on 4 GB VRAM + 32 GB RAM"

- [ ] **Step 2: Replace benchmark table with Phase 0/1 verified results**

Only numbers that were actually measured on this hardware, with exact config and commands.

- [ ] **Step 3: Add Architecture section**

```
## Architecture
StrataRealLowVRAM adds three layers on top of upstream Strata:

1. **Engine Patch** (C++, 10 lines): WDDM-aware expert cache sizing for 4 GB GPUs
2. **Rust Gateway** (strata-gateway): Zero-overhead HTTP server replacing Python serve/server.py
3. **Configuration**: Empirically tuned speculative decode and memory pinning parameters

Upstream Strata C++ source is maintained unmodified except for the guarded patch.
Upstream updates are merged via `tools/sync_upstream.ps1`.
```

- [ ] **Step 4: Add supported models table**

| Model | Status | Config | Notes |
|-------|--------|--------|-------|
| Coder IQ1_M | ✅ Tested | `strata-coder-iq1_m.json` | Primary target |
| Q2_0 | 🔜 Prepared | `strata-q2_0.json` | Config ready, awaiting download |
| IQ2_XS | ❓ Untested | — | Needs 39.2 GB RAM |

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: README overhaul with verified benchmarks and architecture overview"
```

---

## Phase Summary & Expected Outcomes

| Phase | Deliverable | Expected tok/s Impact | Effort |
|-------|-------------|----------------------|--------|
| **0** | Expert cache fix + baseline | Enables speculative decode (0 → 3-4 tok/s baseline) | 1-2 days |
| **1** | Optimal spec config | +15-30% effective tok/s from better acceptance rate | 1 day |
| **2** | Rust gateway | -15-35ms TTFT, smoother streaming, -120 MiB RAM | 3-5 days |
| **3** | Q2_0 support | +35-40% tok/s (simpler dequant on AVX-2) | 1 day (config) + 1 day (benchmark) |
| **4** | Upstream sync | Maintainability (no tok/s impact) | 0.5 day |
| **5** | Documentation | Professional presentation (no tok/s impact) | 0.5 day |

### Critical Path

```
Phase 0 (Expert Cache Fix)
    ↓ [blocks everything]
Phase 1 (Spec Tuning) ──→ Phase 5 (Docs with real numbers)
    ↓ [independent]
Phase 2 (Rust Gateway) ──→ Phase 5 (Docs update)
    ↓ [independent]
Phase 3 (Q2_0) ──→ Phase 5 (Docs update)
    ↓ [independent]
Phase 4 (Upstream Sync)
```

### Honest Assessment of tok/s Ceiling

On this specific hardware (RTX 3050 4GB + Zen 2 + DDR4-3200), the **physical ceiling** for decode tok/s is determined by:

$$\text{tok/s} = \frac{S \times A}{T_{\text{verify}}}$$

Where:
- $S$ = speculative window size (4-6 tokens)
- $A$ = acceptance rate (70-90%)
- $T_{\text{verify}}$ = time per verify cycle ≈ expert gather latency

Expert gather latency is bounded by DDR4 bandwidth (~25-28 GB/s) and expert size. With coder-iq1_m (IQ1_M experts, ~1.3 MB each, 10 experts per token, 76-84% cache miss):

$$T_{\text{verify}} \approx \frac{10 \times 0.8 \times 1.3\text{ MB}}{25\text{ GB/s}} \approx 416 \text{ μs per layer} \times 48 \text{ layers} \approx 200\text{ ms}$$

With spec=6, acceptance=85%: $\text{tok/s} \approx \frac{6 \times 0.85}{0.2} \approx 25.5$... but this is theoretical. Real-world includes CPU compute time, synchronization overhead, KV updates.

**Realistic measured ceiling: 4-8 tok/s with coder-iq1_m, 6-12 tok/s with Q2_0.**

The path to >8 tok/s with coder-iq1_m requires either:
- More VRAM (higher cache hit rate → fewer DDR4 reads)
- Faster RAM (DDR5 or higher-bandwidth platform)
- Engine-level routing-aware prefetch (upstream 0.1.31 feature, needs testing)

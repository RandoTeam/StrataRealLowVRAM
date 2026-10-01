# Strata Dual-Model Stabilization and Empirical Verification Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the WDDM expert cache allocation crash on 4GB VRAM, eliminate 100% CPU spin-loop burn, eliminate 100% RAM VirtualLock exhaustion, and execute a fully transparent, rigorous benchmark of both models (`Coder IQ1_M` and `Full Q2_0`) with complete generated thinking/code output and exact hardware telemetry (CPU, GPU, RAM, VRAM).

**Architecture:** 
1. Fix the `generate.cpp` WDDM overcommit condition so explicit `--expert-cache N` is unconditionally honored without zero-clamping or collapsing to 0 slots.
2. Rebuild `engine/strata.exe` with MSVC and CUDA 13.
3. Configure `strata-coder-our.json` and `strata-q2_0-our.json` with safe memory parameters: `STRATA_ARENA_LOCK=0`, `STRATA_ARENA_PIN_GIB=2`, `--resident-budget-gib 16`, `STRATA_RESIDENT_HEADROOM_GIB 3.0`, and `STRATA_POOL_SPIN_US=0`.
4. Sequentially run and verify both models on a real programming benchmark, streaming tokens, logging thoughts `<think>...</think>`, verifying Python code logic, and measuring exact hardware telemetry.

**Tech Stack:** C++17, CUDA 13.0, CMake, Ninja, Python 3.12, Strata MoE Engine, Win32 VirtualAlloc.

---

## Global Constraints
- Maximum context window: 65,536 tokens preserved strictly (Holy Rule §0).
- Physical RAM capacity: 31.34 GB; at least 3.0–6.0 GB RAM MUST remain free/available for Windows OS.
- VRAM capacity: 4,096 MB GDDR6 (RTX 3050 Laptop).
- Zero placeholders in implementation.

---

### Phase 1: C++ WDDM Expert Cache Allocation Fix
**Files:**
- Modify: `src/program/generate.cpp:2700-2795`
- Target: `build-patched-engine.bat` -> `engine/strata.exe`

- [ ] **Step 1.1: Fix the budget clamping condition in generate.cpp**
Ensure that when `!auto_cache` is true (explicit `--expert-cache N`), `cap` is ALWAYS set to `budget`, ignoring near-zero `free_room` under WDDM.
- [ ] **Step 1.2: Protect explicit cache from auto-shrink loop**
Ensure that when `!auto_cache` is true, line 2788 immediately breaks out of the `shrink_to` retry loop so an explicit cache is never shrunk to 0 slots.
- [ ] **Step 1.3: Rebuild and deploy engine/strata.exe**
Run `build-patched-engine.bat` and verify that `engine/strata.exe` is updated.

---

### Phase 2: Configuration Stabilization for Both Models
**Files:**
- Modify: `strata-coder-our.json`
- Modify: `strata-q2_0-our.json`

- [ ] **Step 2.1: Tune strata-coder-our.json**
  - Set `--expert-cache 426` (explicit fixed cache).
  - Set `--resident-budget-gib 16`.
  - Set `STRATA_ARENA_LOCK 0` (prevent VirtualLock).
  - Set `STRATA_ARENA_PIN_GIB 2`.
  - Set `STRATA_RESIDENT_HEADROOM_GIB 3.0`.
  - Set `STRATA_POOL_SPIN_US 0` (disable active spin-loop, sleep CPU threads on events).
  - Set `STRATA_MMVQ_FAST 1`.
  - Keep `--max-context 65536`.
- [ ] **Step 2.2: Tune strata-q2_0-our.json**
  - Apply identical safe memory and CPU settings: `--expert-cache 426`, `--resident-budget-gib 16`, `STRATA_ARENA_LOCK 0`, `STRATA_ARENA_PIN_GIB 2`, `STRATA_RESIDENT_HEADROOM_GIB 3.0`, `STRATA_POOL_SPIN_US 0`, `STRATA_MMVQ_FAST 1`, `--max-context 65536`.

---

### Phase 3: Transparent Test & Verification - Model 1 (`Coder IQ1_M`)
**Files:**
- Script: `bench/test_single_model.py`
- Target Config: `strata-coder-our.json`

- [ ] **Step 3.1: Launch Coder IQ1_M server**
  Start `server.py --engine strata --config strata-coder-our.json --port 8080`.
- [ ] **Step 3.2: Verify startup logs**
  Verify that `expert cache 426 slots` is successfully allocated on GPU, and `READY 65536` is emitted.
- [ ] **Step 3.3: Execute algorithmic coding prompt via streaming**
  Send prompt: "Реши задачу на Python: дан массив целых чисел, найди длину наибольшей строго возрастающей подпоследовательности за время O(n log n). Напиши эффективный код с подробным объяснением алгоритма."
- [ ] **Step 3.4: Capture & print full output and hardware telemetry**
  - Print full `<think>...</think>` reasoning.
  - Print full generated Python code.
  - Record TTFT, Thinking tok/s, Decode tok/s, Overall tok/s.
  - Record peak RAM, free RAM, VRAM, CPU utilization.
- [ ] **Step 3.5: Cleanly terminate server and cooldown**

---

### Phase 4: Transparent Test & Verification - Model 2 (`Full Q2_0`)
**Files:**
- Script: `bench/test_single_model.py`
- Target Config: `strata-q2_0-our.json`

- [ ] **Step 4.1: Launch Full Q2_0 server**
  Start `server.py --engine strata --config strata-q2_0-our.json --port 8080`.
- [ ] **Step 4.2: Verify startup logs**
  Verify that `expert cache 426 slots` is allocated in VRAM, 16 GiB resident budget is honored, and `READY 65536` is emitted.
- [ ] **Step 4.3: Execute identical coding prompt via streaming**
- [ ] **Step 4.4: Capture & print full output and hardware telemetry**
  - Print full `<think>...</think>` reasoning.
  - Print full generated Python code.
  - Record TTFT, Thinking tok/s, Decode tok/s, Overall tok/s.
  - Record peak RAM, free RAM, VRAM, CPU utilization.
- [ ] **Step 4.5: Cleanly terminate server and cooldown**

---

### Phase 5: Comparative Evaluation & Goal Finalization
**Files:**
- Artifact: `docs/superpowers/plans/2026-10-01-benchmark-report.md`

- [ ] **Step 5.1: Compile comparative table**
  Compare Coder vs Q2_0 across all metrics: TTFT, Thinking tok/s, Decode tok/s, Free RAM, VRAM, CPU load.
- [ ] **Step 5.2: Verify algorithmic accuracy and reasoning preservation**
  Confirm both solutions are mathematically sound and valid Python code.
- [ ] **Step 5.3: Output comprehensive final report with GOAL_COMPLETE marker**

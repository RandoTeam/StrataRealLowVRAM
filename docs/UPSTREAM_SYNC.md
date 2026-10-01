# Upstream Synchronization & Conflict Management Guide

This document defines the exact workflow for keeping **StrataRealLowVRAM** synchronized with upstream [Niko1221/Strata](https://github.com/Niko1221/Strata) and incorporating innovations from community forks such as [gputier/StrataGP](https://github.com/gputier/StrataGP).

---

## 1. Architectural Philosophy: The Surgical Principle

To ensure updates from upstream can be merged with zero friction:
1. **Never restructure upstream directories or rename existing C++ files.**
2. **All core engine C++ changes are isolated to minimal, `#ifdef`-guarded patches** stored in `patches/` and recorded in Git.
3. **New functionality (such as `strata-gateway`) lives in independent subdirectories** (`rust/`, `tools/`, `bench/`), which have zero upstream collision surface.

---

## 2. Conflict Zone Map

When merging `upstream/main`, only the following files require attention:

| File | Nature of StrataRealLowVRAM Changes | Upstream Risk | Merge Strategy |
| :--- | :--- | :--- | :--- |
| `src/program/generate.cpp` | WDDM 4GB expert cache fix (`#if defined(_WIN32)`) | Medium | Check `patches/0001-wddm-expert-cache-4gb.patch`. Reapply if lines shift. |
| `strata-coder-iq1_m.json` | Tuned mobile parameters (`--vram-reserve-mib 180`, `STRATA_ARENA_PIN_GIB=1`) | Zero | Upstream does not ship this file; keep ours. |
| `strata-q2_0.json` | Optimized Q2_0 template | Zero | Upstream does not ship this file; keep ours. |
| `serve/server.py` | `_fast_session_title` DeepSeek harness interceptor | Low | Cherry-pick or preserve our method; gateway replaces this. |
| `README.md` | Fork header, mobile benchmarks, and architecture overview | High | Keep our fork header and benchmarks; merge upstream release notes below marker. |

---

## 3. Step-by-Step Sync Workflow

### Step 1: Check Upstream Status
Run the automated sync tool:
```powershell
.\tools\sync_upstream.ps1 -CheckOnly
```

### Step 2: Merge Upstream Main
```bash
git fetch upstream
git merge upstream/main --no-commit
```

### Step 3: Verify and Re-apply Patches
If `src/program/generate.cpp` had conflicts or was overwritten:
```bash
git apply --check patches/0001-wddm-expert-cache-4gb.patch
git apply patches/0001-wddm-expert-cache-4gb.patch
```

### Step 4: Verification Gate
Run the startup probe and verify `READY` is achieved:
```powershell
python bench/probe_startup.py strata.exe 1
```

### Step 5: Incorporating StrataGP CPU Optimizations
To cherry-pick specific CPU/AVX kernel optimizations from StrataGP:
```bash
git fetch stratagp
git cherry-pick <commit-hash>
```
Key candidate commits from `stratagp/perf/cpu`:
- Prefetch enhancements (`4ff9e0a`, `7e1cb22`)
- Native activation spacing (`32fce1a`)

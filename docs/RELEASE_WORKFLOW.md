# StrataRealLowVRAM Release Workflow & Policy

> **Standard Operating Procedure (SOP) for Release Packaging, Verification, and Publishing.**

---

## 🎯 1. Release Philosophy

1. **Version Parity with Upstream**:  
   We mirror the upstream engine release versioning (e.g. `v0.1.35`), but publish our releases under the title **`StrataRealLowVRAM v<VERSION> (Extreme Low-VRAM Edition)`**.
2. **Empirical Verification Gate**:  
   No release is ever published without passing our empirical benchmark suite on real 4 GB VRAM hardware:
   - Full **65,536 tokens context** (`--max-context 65536`).
   - Zero Windows WDDM driver clamps or OOM errors.
   - Dual-model verification on both **Qwen3.8 Full Q2_0** and **Coder IQ1_M**.
3. **Batteries-Included Dual Packaging**:  
   Every release provides two distinct archive assets:
   - **`StrataRealLowVRAM-v<VERSION>-full-windows-x64.zip`**: Complete portable distribution containing the patched engine, pre-tuned configurations (`strata-q2_0.json`, `strata-coder-iq1_m.json`), launch scripts (`run-q2_0.bat`, `run-coder-iq1_m.bat`, `START-HERE.bat`), tools, and documentation.
   - **`StrataRealLowVRAM-v<VERSION>-engine-windows-x64.zip`**: Minimal drop-in replacement containing only `engine/strata.exe` and `engine/BUILD.json` for existing installations.
4. **Transparent Comparative Release Notes**:  
   Every release note explicitly highlights:
   - Upstream updates incorporated in this version.
   - Custom low-VRAM patches, kernel optimizations, and tools added.
   - Side-by-side empirical benchmark numbers (tok/s, VRAM cache hit rate, draft accept rate).

---

## 📋 2. Release Steps (Checklist)

Whenever a new upstream release is detected:

```
[1] Sync Upstream   ──> git fetch upstream && git merge upstream/main
[2] Compile Engine  ──> cmd /c build-patched-engine.bat (MSVC + CUDA 13.0 sm_86)
[3] Verify Stamp    ──> engine/BUILD.json has "source": "local", "patched": true
[4] Benchmark Gate  ──> python bench/bench_sweep.py on Q2_0 and Coder IQ1_M
[5] Package Assets  ──> python tools/package_release.py
[6] Publish Release ──> python tools/package_release.py --publish
[7] Commit & Push   ──> git add -A && git commit -m "release: v<VERSION>" && git push
```

---

## 🛠️ 3. Automation Tooling

The release process is fully automated via `tools/package_release.py`:

```bash
# Package archives and generate markdown release notes into dist/
python tools/package_release.py

# Package and publish directly to GitHub Releases via GitHub CLI (gh)
python tools/package_release.py --publish
```

The script automatically detects the version from `engine/BUILD.json`, builds the ZIP archives, formats the changelog, and uploads the assets using authenticated `gh release create`.

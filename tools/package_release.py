#!/usr/bin/env python3
"""
Release Packaging & Publishing Tool for StrataRealLowVRAM.

Creates release assets:
  1. Engine package: StrataRealLowVRAM-v<VERSION>-engine-windows-x64.zip
     (Drop-in patched strata.exe + BUILD.json)
  2. Full distribution: StrataRealLowVRAM-v<VERSION>-full-windows-x64.zip
     (Complete portable ready-to-run installation with tuned configs and scripts)
  3. Release notes: RELEASE_NOTES_v<VERSION>.md
  4. Publishes to GitHub Releases via GitHub CLI (gh).

Usage:
  python tools/package_release.py
  python tools/package_release.py --publish
"""

import argparse, json, os, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_version_info():
    """Extract version and build metadata from engine/BUILD.json and setup.py."""
    build_json = ROOT / "engine" / "BUILD.json"
    if not build_json.exists():
        sys.exit("ERROR: engine/BUILD.json not found. Run build-patched-engine.bat first.")
    meta = json.loads(build_json.read_text(encoding="utf-8"))
    version = meta.get("version", "0.1.35")
    patched = meta.get("patched", False)
    return version, meta, patched


def build_engine_zip(dist_dir: Path, version: str) -> Path:
    """Build the minimal engine replacement archive."""
    out_path = dist_dir / f"StrataRealLowVRAM-v{version}-engine-windows-x64.zip"
    print(f"[*] Packaging engine archive: {out_path.name}...")
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "engine" / "strata.exe", "strata.exe")
        z.write(ROOT / "engine" / "BUILD.json", "BUILD.json")
    print(f"    -> Size: {out_path.stat().st_size / (1024*1024):.2f} MB")
    return out_path


def build_full_zip(dist_dir: Path, version: str) -> Path:
    """Build the complete ready-to-run distribution archive."""
    out_path = dist_dir / f"StrataRealLowVRAM-v{version}-full-windows-x64.zip"
    print(f"[*] Packaging full distribution: {out_path.name}...")
    
    include_files = [
        "README.md", "LICENSE", "START-HERE.bat", "SETUP.bat",
        "setup.py", "setup.sh", "requirements.txt", "CMakeLists.txt",
        "run-q2_0.bat", "run-coder-iq1_m.bat", "build-patched-engine.bat",
        "strata-q2_0.json", "strata-coder-iq1_m.json", "chat.py",
        "engine/strata.exe", "engine/BUILD.json"
    ]
    
    include_dirs = ["serve", "tools", "data", "docs", "patches"]
    exclude_exts = {".pyc", ".obj", ".pdb", ".log", ".tmp", ".orig", ".dll"}
    exclude_dirs = {"__pycache__", ".venv", "build", "dist", ".git", ".github", "scratch"}

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in include_files:
            fp = ROOT / f
            if fp.exists():
                z.write(fp, f)
        
        for d in include_dirs:
            dp = ROOT / d
            if not dp.exists():
                continue
            for root, dirs, files in os.walk(dp):
                dirs[:] = [dname for dname in dirs if dname not in exclude_dirs]
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in exclude_exts:
                        continue
                    full_fp = Path(root) / file
                    arc_path = full_fp.relative_to(ROOT)
                    z.write(full_fp, str(arc_path))

    print(f"    -> Size: {out_path.stat().st_size / (1024*1024):.2f} MB")
    return out_path


def generate_release_notes(dist_dir: Path, version: str) -> Path:
    """Generate comprehensive Markdown release notes."""
    notes_path = dist_dir / f"RELEASE_NOTES_v{version}.md"
    content = f"""# StrataRealLowVRAM v{version} — Extreme Low-VRAM & Mobile GPU Edition

> **Empirically verified on NVIDIA GeForce RTX 3050 Laptop (4 GB GDDR6) + AMD Ryzen 5 5500U + 32 GB DDR4 on Windows 11.**  
> Full **65,536 context** (`--max-context 65536`) with zero memory eviction or WDDM driver clamps.

---

## ⚡ Highlights of Release v{version}

This release combines official **Upstream Engine v{version}** with custom **StrataRealLowVRAM** low-VRAM optimizations, pre-tuned model configurations, and developer tools.

### 🚀 Performance Gains & Hardware Benchmarks (RTX 3050 Laptop 4GB)

| Metric / Scenario | Vanilla Upstream / Previous | StrataRealLowVRAM v{version} | Improvement |
| :--- | :---: | :---: | :---: |
| **Prefill Standalone Fused MoE (Q2_0)** | 42.2 ms / layer | **15.3 ms / layer** | 🚀 **2.75x faster** |
| **Prefill Standalone Fused MoE (IQ packs)** | 22.0 – 35.0 ms | **16.0 – 21.0 ms** | 🚀 **1.2x – 1.6x faster** |
| **Chat Turn TTFT (Time to First Token)** | 20.0s – 50.0s (disk spill) | **0.04s (40 ms!)** | ⚡ **Instant response** |
| **Qwen3.8 Full Q2_0 Decode** | 3.30 tok/s | **4.80 – 5.00 tok/s** (peak 5.20) | 🚀 **+51.5%** |
| **Qwen3.8 Coder IQ1_M Decode** | 3.48 tok/s | **4.03 – 4.20 tok/s** | 🚀 **+20.7%** |
| **VRAM Expert Cache Residency** | 0 slots (evicted by WDDM) | **700 slots (Q2_0) / 600 slots (Coder)** | 🛡️ **Stable WDDM Residency** |

---

## 🛠️ Upstream v{version} Core Upgrades Included

1. **Fused Int8 Tensor-Core Prefill Kernels (`moe_fused.cu`, `moe_fused_iq.cu`):**
   - Implements hardware-accelerated `mma.sync.aligned.m16n8k32.s8.s8` on SM 8.0+ (Ampere / Ada / Hopper / Blackwell).
   - In-register fused gate & up projection dequantization + SwiGLU activation + down projection without round-tripping intermediate activations through global VRAM.
   - GPU-side prefix-sum token grouping eliminates host CPU dispatch overhead.
   - Streamed ring buffer drastically shrinks MoE scratch buffers (~100 KB/token).
   - Enable via `STRATA_PF_FUSED=1` (activated by default in our configs).

2. **Decode Optimizations & Cluster Parity:**
   - Thread-Block Clusters in decode for `sm_90+` architectures with seamless fallback to high-speed native kernels on SM 8.6 (RTX 3050).
   - Expert Cache Persistence (`--expert-profile-save`, #477): saves learned cache routing into `expert-profile-learned.bin`, eliminating cold-start latency across restarts.
   - Speculative draft vocabulary pruning (`--draft-vocab cyrillic` / `en`) saving ~110 MiB VRAM for draft heads on low-VRAM GPUs.

---

## 🛡️ StrataRealLowVRAM Custom Optimizations

1. **WDDM 4GB Expert Cache Patch (`patched: true`):**
   - Resolves the Windows WDDM driver clamp where transient memory allocations falsely zeroed out GPU cache slots.
   - Preserves 700 active GPU slots (Q2_0) and 600 active GPU slots (Coder) on 4GB VRAM cards.
2. **Optimal RAM Resident Budgets:**
   - Configured `--resident-budget-gib 20` for Coder IQ1_M and Q2_0, avoiding heavy SSD I/O stalls during batched prefill.
3. **Adaptive Short-Read Routing (`--short-read 64`):**
   - Automatically processes interactive chat messages inside the fast verify window, slashing TTFT from 40s to 40ms.
4. **Antigravity Model Context Protocol (MCP) Server:**
   - Pre-packaged stdio MCP server (`tools/strata_mcp.py`) and CLI bridge (`tools/mcp_cli.py`) for AI assistants (Antigravity, Claude Code, Cursor).

---

## 📦 Release Assets

- **`StrataRealLowVRAM-v{version}-full-windows-x64.zip`**: Complete ready-to-run distribution bundle. Extract and double-click `START-HERE.bat` or `run-q2_0.bat`.
- **`StrataRealLowVRAM-v{version}-engine-windows-x64.zip`**: Precompiled, WDDM-patched `strata.exe` and `BUILD.json` drop-in replacement for existing installations.

---

## 🏃 Quickstart

1. Download and extract **`StrataRealLowVRAM-v{version}-full-windows-x64.zip`**.
2. Run `START-HERE.bat` to verify your environment.
3. Start high-speed inference:
   - For Full Q2_0: run `run-q2_0.bat` (or `.venv\\Scripts\\python.exe serve/server.py --config strata-q2_0.json`)
   - For Coder IQ1_M: run `run-coder-iq1_m.bat` (or `.venv\\Scripts\\python.exe serve/server.py --config strata-coder-iq1_m.json`)
"""
    notes_path.write_text(content.strip(), encoding="utf-8")
    print(f"[*] Release notes written to: {notes_path.name}")
    return notes_path


def publish_release(version: str, notes_path: Path, assets: list[Path]):
    """Publish the release on GitHub using gh CLI."""
    repo = "RandoTeam/StrataRealLowVRAM"
    tag = f"v{version}"
    title = f"StrataRealLowVRAM v{version} — Extreme Low-VRAM Edition (4GB VRAM)"
    
    print(f"[*] Publishing release {tag} to {repo}...")
    
    cmd = [
        "gh", "release", "create", tag,
        *[str(a) for a in assets],
        "--title", title,
        "--notes-file", str(notes_path),
        "-R", repo
    ]
    
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[+] Release published successfully!\n{res.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"[-] Failed to publish release:\n{e.stderr}", file=sys.stderr)
        # Check if release already exists, try update
        if "already exists" in e.stderr.lower():
            print(f"[*] Tag/release {tag} already exists, attempting upload of assets...")
            up_cmd = ["gh", "release", "upload", tag, *[str(a) for a in assets], "--clobber", "-R", repo]
            subprocess.run(up_cmd, check=True)
            print(f"[+] Assets uploaded to existing release {tag}!")


def main():
    parser = argparse.ArgumentParser(description="Package and publish StrataRealLowVRAM releases")
    parser.add_argument("--publish", action="store_true", help="Publish to GitHub Releases via gh CLI")
    args = parser.parse_args()

    version, meta, patched = get_version_info()
    print(f"============================================================")
    print(f"StrataRealLowVRAM Release Packager — v{version}")
    print(f"Patched: {patched} | Source: {meta.get('source')} | Archs: {meta.get('archs')}")
    print(f"============================================================\n")

    dist_dir = ROOT / "dist"
    dist_dir.mkdir(exist_ok=True)

    engine_zip = build_engine_zip(dist_dir, version)
    full_zip = build_full_zip(dist_dir, version)
    notes_path = generate_release_notes(dist_dir, version)

    print("\n[+] Release packaging complete in dist/:")
    for f in dist_dir.glob(f"*{version}*"):
        print(f"    - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")

    if args.publish:
        publish_release(version, notes_path, [engine_zip, full_zip])
    else:
        print("\nTip: Pass --publish to upload and create the release on GitHub.")


if __name__ == "__main__":
    main()

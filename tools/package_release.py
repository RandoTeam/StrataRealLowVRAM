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

## ⚡ Highlights of This Release

This release synchronizes official **Upstream Engine v{version}** with our custom **StrataRealLowVRAM** low-VRAM optimizations, pre-tuned model configurations, and developer tools.

### 🚀 Empirical Hardware Benchmarks (RTX 3050 Laptop 4GB)

| Model | Vanilla Upstream | StrataRealLowVRAM Champion | Speedup | VRAM Cache Slots | RAM Usage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen3.8 Full Q2_0** | 3.30 tok/s | **4.80 – 5.00 tok/s** (peak 5.20) | 🚀 **+51.5%** | **700 slots** (922 MiB) | 16.0 GiB (Frees 14 GB) |
| **Qwen3.8 Coder IQ1_M** | 3.48 tok/s | **4.03 – 4.20 tok/s** | 🚀 **+20.7%** | **600 slots** (1.15 GiB) | 20.0 GiB |

---

## 🛠️ StrataRealLowVRAM Key Features in v{version}

1. **WDDM 4GB Expert Cache Patch (`patched: true`):**
   - Eliminates Windows WDDM driver clamp where prior memory overcommit falsely zeroes out the expert cache.
   - Restores full 700-slot (Q2_0) and 600-slot (Coder) GPU residency on 4GB cards.
2. **Upstream v{version} Integration:**
   - **Fix #467 (Windows 32GB Working Set Trimming):** Restores +1.86 GiB of available physical RAM on startup via `SetProcessWorkingSetSize`, allowing the full resident cache complement to lock cleanly in RAM.
   - **Fix #448 (Multi-GPU Prompt Chunking):** Intelligent handling of asymmetric VRAM cards.
   - **Fix #460 (JSON API Resilience):** Graceful decoding of double-encoded message strings.
   - **Fix #457 (Speculative Metrics):** Cumulative `/metrics` counters for draft acceptance.
   - **Fix #459 (Atomic Configs):** Safe atomic config writes via temporary files.
3. **Pre-Tuned Champion Configurations:**
   - `strata-q2_0.json`: 700 expert cache slots, 6 pool workers, adaptive swaps 4/8, Direct I/O queue depth 256, prefill chunk 2048.
   - `strata-coder-iq1_m.json`: 600 expert cache slots, 6 pool workers, adaptive swaps 4/8, Direct I/O queue depth 256, prefill chunk 2048.
4. **Antigravity Model Context Protocol (MCP) Server:**
   - Official stdio MCP server (`tools/strata_mcp.py`) and CLI bridge (`tools/mcp_cli.py`) for AI assistants (Antigravity, Claude Code, Cursor).
5. **Engine Auto-Update Shield:**
   - `setup.py` protects custom compiled and patched local binaries from being overwritten by vanilla upstream downloads.

---

## 📦 Release Assets

- **`StrataRealLowVRAM-v{version}-full-windows-x64.zip`**: Complete ready-to-run installation bundle. Just extract and double-click `START-HERE.bat` or `run-q2_0.bat`.
- **`StrataRealLowVRAM-v{version}-engine-windows-x64.zip`**: Precompiled, WDDM-patched `strata.exe` and `BUILD.json` drop-in replacement for existing installations.

---

## 🏃 Quickstart

1. Download and extract **`StrataRealLowVRAM-v{version}-full-windows-x64.zip`**.
2. Run `START-HERE.bat` to verify your environment.
3. Start high-speed inference with pre-tuned configs:
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

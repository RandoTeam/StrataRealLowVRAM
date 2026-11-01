"""Build the Windows release zip of 4gbVramStrata with the ready-made 131K engine.

The zip is what a user downloads: START-HERE.bat, setup.py, the server and web app, the tools setup runs, the
documents a user reads, engine-fourgbvram/ (the built strata-fourgbvram.exe with its BUILD.json), and the engine's
sources.  The sources are there because setup.source_hash() walks them: without them the shipped BUILD.json names a
source the zip does not have, and setup takes the engine for stale and tries to compile one.  Tests, benchmark
records, the paper, the local build folders and the quarantined working files (_attic/) stay out.

    python tools/package_release_131k.py             # build dist/4gbVramStrata-v<version>-131k-win-x64.zip
    python tools/package_release_131k.py --check     # list what is missing or would be left out, write nothing
    python tools/package_release_131k.py --engine DIR  # take the engine from DIR instead of engine-fourgbvram/

Every required path is checked before anything is copied, so a release is never built from a half-cleaned tree.
The version comes from CMakeLists.txt (project(strata VERSION x.y.z)).
"""
import argparse
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Files a user runs or setup needs, relative to ROOT.  A missing one stops the build.
FILES = [
    "START-HERE.bat", "SETUP.bat", "UPDATE.bat",
    "setup.py", "fourgbvramstrata.py", "chat.py",
    "requirements.txt", "LICENSE", "README.md",
]
# Whole trees, with the filter below applied to every file.
TREES = ["serve", "tools", "data", "docs"]
# The engine's sources, copied verbatim (no filter: setup.source_hash() covers every file, test sources included).
SOURCES = ["CMakeLists.txt", "cmake", "src", "include", "third_party/ggml", "tools/vision"]
# Dropped everywhere: test sources, caches, and the paper (it has its own release).
SKIP_DIRS = {"__pycache__", ".pytest_cache", "paper", "superpowers"}
SKIP_NAMES = {"test_setup_golden.json"}
SKIP_PREFIXES = ("test_",)
SKIP_SUFFIXES = ("_test.py", ".pyc", ".pyo", ".tmp", ".obj", ".exp", ".lib", ".pdb")


def version() -> str:
    text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    m = re.search(r"project\(\s*strata\s+VERSION\s+([0-9.]+)", text)
    if not m:
        raise SystemExit("CMakeLists.txt has no project(strata VERSION x.y.z)")
    return m.group(1)


def wanted(rel: Path) -> bool:
    if any(part in SKIP_DIRS for part in rel.parts):
        return False
    name = rel.name
    return not (name in SKIP_NAMES or name.startswith(SKIP_PREFIXES) or name.endswith(SKIP_SUFFIXES))


def tree_files(tree: str) -> list:
    base = ROOT / tree
    out = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            rel = Path(dirpath, name).relative_to(ROOT)
            if wanted(rel):
                out.append(rel)
    return out


def source_files() -> list:
    """Every file under SOURCES, unfiltered: the engine's fingerprint covers all of them."""
    out = []
    for part in SOURCES:
        base = ROOT / part
        if base.is_file():
            out.append(Path(part))
        elif base.is_dir():
            out += [p.relative_to(ROOT) for p in sorted(base.rglob("*")) if p.is_file()]
    return out


def engine_files(engine: Path) -> list:
    if not engine.is_dir():
        return []
    return [Path("engine-fourgbvram", p.name) for p in sorted(engine.iterdir())
            if p.is_file() and (p.suffix.lower() in (".exe", ".dll") or p.name == "BUILD.json")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="list problems, write nothing")
    ap.add_argument("--engine", default=str(ROOT / "engine-fourgbvram"),
                    help="folder with strata-fourgbvram.exe and BUILD.json (default: engine-fourgbvram/)")
    a = ap.parse_args()

    ver = version()
    release_name = f"4gbVramStrata-v{ver}-131k-win-x64"
    dist = ROOT / "dist"
    stage = dist / release_name
    zip_path = dist / f"{release_name}.zip"

    missing = [f for f in FILES if not (ROOT / f).is_file()]
    plan = [Path(f) for f in FILES if (ROOT / f).is_file()]
    for tree in TREES:
        if not (ROOT / tree).is_dir():
            missing.append(tree + "/")
        plan += tree_files(tree)
    missing += [s + ("/" if (ROOT / s).is_dir() else "") for s in SOURCES if not (ROOT / s).exists()]
    plan += source_files()
    plan = list(dict.fromkeys(plan))                 # tools/vision arrives from TREES and SOURCES alike
    eng = engine_files(Path(a.engine))
    if not eng:
        missing.append(f"{Path(a.engine).name}/strata-fourgbvram.exe")

    print(f"release: {release_name}")
    print(f"files:   {len(plan)} from the tree (the engine's sources included), {len(eng)} from {Path(a.engine).name}/")
    if missing:
        print("missing:")
        for m in missing:
            print(f"  ! {m}")
        if a.check:
            return 1
        raise SystemExit("a required file is missing: fix the tree (or pass --engine DIR) and run again")
    if a.check:
        print("ok: every required file is present")
        return 0

    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for rel in plan + eng:
        src = ROOT / rel if not rel.parts[0] == "engine-fourgbvram" else Path(a.engine) / rel.name
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(dist))
    print(f"ready: {zip_path} ({zip_path.stat().st_size / 2**20:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

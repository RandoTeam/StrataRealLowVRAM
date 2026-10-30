#!/usr/bin/env python3
"""Write this fork's name, 4gbVramStrata, where the text taken from PolyStrata names PolyStrata.

4gbVramStrata is built on PolyStrata (https://github.com/VecSzn/PolyStrata), which is built on
Strata (https://github.com/Niko1221/Strata), and takes their changes. Run this after taking changes:

    python tools/brand_names.py            # rewrite the files
    python tools/brand_names.py --check    # list what it would change, change nothing

It replaces the word PolyStrata in the files a user reads or runs (documents, setup, the server, the web app,
the tools and their tests) and nothing else; a second run changes nothing. It leaves alone the engines' sources
and build files (src/, include/, sycl/, metal/, engine targets), the benchmark records, the paper, docs/FAQ.md
(it is about several projects), docs/COMMUNITY_BENCHMARKS.md (community records), and the places where PolyStrata
means the donor project this fork is built on (KEEP). To name that project in new text, link it,
[PolyStrata](https://github.com/VecSzn/PolyStrata), or write "the PolyStrata project".

Code identifiers stay as they are on purpose: the entry script polystrata.py (imported by tools/test_benchmark_prompt.py,
and a name starting with a digit is not a valid Python module), the src/poly/ tree and the strata::poly namespace,
the strata-poly CMake target and binary, and engine-poly/ (build output, git-ignored).
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = ("src/", "include/", "sycl/", "metal/", "tests/", "third_party/", "bench/", "docs/paper/",
        "docs/COMMUNITY_BENCHMARKS.md", "docs/FAQ.md", "tools/brand_names.py", "tools/polystrata_names.py")
KINDS = (".md", ".py", ".html", ".js", ".css", ".bat", ".sh", ".yml", ".yaml", ".jinja", ".json")
NAMES = ("Dockerfile",)
KEEP = "|".join((
    r"\[PolyStrata\]\(https://github\.com/VecSzn/PolyStrata",  # a Markdown link to it
    r"VecSzn/PolyStrata\">PolyStrata",                         # an HTML link to it
    r"(?:VecSzn)/PolyStrata",                                  # their repository, in an address or by name
    r"[Tt]he\s+PolyStrata project",
    r"polystrata\.py",                                         # the kept entry-script file name
    r"polystrata_names",                                       # the kept upstream brand tool
    r"strata-poly",                                            # the kept engine target and binary
    r"engine-poly",                                            # the kept build-output directory
    r"src/poly",                                               # the kept engine source tree
    r"strata::poly",                                           # the kept C++ namespace
    r"strata_poly",                                            # the kept CMake library name
    r"strata-cfg-polystrata",                                  # benchmark record file names
    r"community-\w*-polystrata",                               # benchmark record directory names
    r"NVFP4(?:-GPTQ)?-Strata",                                 # the NVFP4 packs' Hugging Face repositories
))
WORD = re.compile(f"{KEEP}|(?<![A-Za-z0-9_])(PolyStrata)(?![A-Za-z0-9_])")


def rename(text: str) -> str:
    return WORD.sub(lambda m: "4gbVramStrata" if m.group(1) else m.group(0), text)


def files() -> list:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines()
            if not f.startswith(SKIP) and (f.endswith(KINDS) or f in NAMES)]


def main() -> int:
    check = "--check" in sys.argv[1:]
    changed = 0
    for f in files():
        path = ROOT / f
        try:
            with open(path, encoding="utf-8", newline="") as fh:
                old = fh.read()
        except (OSError, UnicodeDecodeError) as e:
            print(f"skip {f}: {e}")
            continue
        new = rename(old)
        if new != old:
            changed += 1
            if check:
                print(f"would change {f}")
            else:
                with open(path, "w", encoding="utf-8", newline="") as fh:
                    fh.write(new)
                print(f"changed {f}")
    print(f"{changed} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

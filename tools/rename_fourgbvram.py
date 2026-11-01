#!/usr/bin/env python3
"""One-shot migration: rename the code codename `poly` to `fourgbvram`.

Brand strings (PolyStrata) were renamed by tools/brand_names.py; this moves the identifiers the brand tool
leaves alone on purpose: directories, the entry script, CMake targets, the C++ namespace, setup constants,
the engine value, and user-facing commands. `4gbvram` itself cannot be used: identifiers may not start with
a digit (C++, Python imports), so the code token is `fourgbvram`.

Kept on purpose (documented, not oversight):
- tools/polystrata_names.py + tools/test_polystrata_names.py: the donor's brand tool, needed for merges.
- bench/, docs/paper/: benchmark evidence and the paper.
- docs/FAQ.md, docs/COMMUNITY_BENCHMARKS.md: shared/community records. Their prose name was rewritten by hand
  (the heading "The models this project's own engine adds" and the anchors to it), and what stays is what must:
  the VecSzn/PolyStrata links that lead to where people posted their reports, and the versions they ran
  (0.1.45-poly.1, 0.1.44.1-poly.6).
- Donor links (VecSzn/PolyStrata, Niko1221/Strata), record file names, the old remote name old-StrataRealLowVRAM,
  strata-glm/engine-glm history.
- serve's engine-stem check keeps the old stems after the new one (upgraders' configs).

Run:  python tools/rename_fourgbvram.py --check    (lists, changes nothing)
        python tools/rename_fourgbvram.py            (applies; git mv first, then content)
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = ("bench/", "docs/paper/", "tools/polystrata_names.py", "tools/test_polystrata_names.py",
        "docs/FAQ.md", "docs/COMMUNITY_BENCHMARKS.md", "tools/rename_fourgbvram.py")
MOVES = [
    ("src/poly", "src/fourgbvram"),
    ("include/strata/poly", "include/strata/fourgbvram"),
    ("polystrata.py", "fourgbvramstrata.py"),
    ("tools/make_poly_release.py", "tools/make_fourgbvram_release.py"),
    ("tools/test_poly_prebuilt.py", "tools/test_fourgbvram_prebuilt.py"),
]
KEEP = "|".join((
    r"VecSzn/PolyStrata",
    r"Niko1221/Strata",
    r"old-StrataRealLowVRAM",
    r"STRATA_POLY_PREBUILT_URL",  # legacy env fallback, kept on purpose (see setup.py)
    r'"strata-poly", "strata-glm"',  # serve compat stems for upgraders, kept on purpose
    r"strata-cfg-polystrata",               # record file names (bench/ is skipped anyway)
    r"polystrata-setup",
    r"polystrata_names",                    # donor tool (file is skipped anyway)
    r"test_polystrata_names",
    r"strata-glm",
    r"engine-glm",
    r"PolyStrata",                          # brand strings: brand_names.py owns those files
    r"POLYSTRATA",
    r"poly(?=\.\d)",                        # versions: 0.1.44.1-poly.6, poly.5
))
RULES = [
    (r"polystrata", "fourgbvramstrata"),
    (r"strata-poly", "strata-fourgbvram"),
    (r"strata_poly", "strata_fourgbvram"),
    (r"STRATA_POLY", "STRATA_FOURGBVRAM"),
    (r"POLY_", "FOURGBVRAM_"),
    (r"engine-poly", "engine-fourgbvram"),
    (r"src/poly", "src/fourgbvram"),
    (r"strata/poly", "strata/fourgbvram"),
    (r"(?<![A-Za-z0-9])poly(?![A-Za-z0-9])(?!\.\d)", "fourgbvram"),
]
WORD = re.compile(f"({KEEP})|{'|'.join(f'({r})' for r in [r[0] for r in RULES])}")


def files() -> list:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if not f.startswith(SKIP)]


def rename(text: str) -> str:
    def sub(m: re.Match) -> str:
        if m.group(1) is not None:
            return m.group(0)
        for i, (pat, rep) in enumerate(RULES, start=2):
            if m.group(i) is not None:
                return rep
        return m.group(0)
    return WORD.sub(sub, text)


def main() -> int:
    check = "--check" in sys.argv[1:]
    if not check:
        for old, new in MOVES:
            if (ROOT / old).exists() and not (ROOT / new).exists():
                subprocess.run(["git", "mv", old, new], cwd=ROOT, check=True)
                print(f"moved {old} -> {new}")
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

"""The files tools/brand_names.py reads say 4gbVramStrata where they mean this project.  A text that means
a donor project names it by a link or as "the PolyStrata project" / "the Strata project": the tool leaves those
alone, as it does the engines' sources and build files, the benchmark records, and the kept code identifiers
(polystrata.py, src/poly, strata::poly, strata-poly, engine-poly).

    python -m unittest tools.test_brand_names
"""
from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import brand_names as N  # noqa: E402


class Names(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(N.rename("PolyStrata runs models"), "4gbVramStrata runs models")
        self.assertEqual(N.rename("[PolyStrata](https://github.com/VecSzn/PolyStrata)"),
                         "[PolyStrata](https://github.com/VecSzn/PolyStrata)")
        self.assertEqual(N.rename("the PolyStrata project"), "the PolyStrata project")
        self.assertEqual(N.rename("python polystrata.py --model x"), "python polystrata.py --model x")
        self.assertEqual(N.rename("target strata-poly"), "target strata-poly")
        self.assertEqual(N.rename("src/poly/engine.cpp"), "src/poly/engine.cpp")

    def test_nothing_is_left_to_rename(self):
        files = io.StringIO()
        try:
            with mock.patch.object(sys, "argv", ["brand_names.py", "--check"]), \
                    contextlib.redirect_stdout(files), contextlib.redirect_stderr(io.StringIO()):
                left = N.main()
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("the tool lists the files with git: this is no checkout")
        self.assertEqual(left, 0, "python tools/brand_names.py --check lists:\n" + files.getvalue())


if __name__ == "__main__":
    unittest.main()

"""Tests for setup.py's LFM2.5-24B-A2B (docs/BACKENDS.md): its five pinned files, two repositories, the estimates
for an 8 GB card, and the install by GLM's steps with no draft model and no image encoder.  Mocked like
tools/test_setup_glm.py, whose harness this uses.

    python -m unittest tools.test_setup_lfm
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools import test_setup_glm as glm_tests  # noqa: E402

FAM = setup.FAMILIES["lfm25"]
# setup's order: the size taken with the RAM for it, the one with the least RAM, then the others
SIZES = [("UD-Q4_K_XL", 14876901632), ("UD-Q3_K_XL", 10947100928), ("QAD-Q4_0", 13578190880),
         ("UD-Q6_K_XL", 21861204224), ("Q8_0", 25493733344)]


class Pins(unittest.TestCase):
    def test_the_files(self):
        self.assertEqual(list(setup.family_models("lfm25")), [m for m, _ in SIZES])
        self.assertRegex(FAM["hf"], r"^https://huggingface.co/unsloth/LFM2.5-24B-A2B-GGUF/resolve/[0-9a-f]{40}/$")
        self.assertEqual(list(FAM["sha256"]), [f"LFM2.5-24B-A2B-{m}.gguf" for m, _ in SIZES])
        for size, n in SIZES:
            file = f"LFM2.5-24B-A2B-{size}.gguf"
            d = FAM["models"][size]
            self.assertEqual((setup.model_file(FAM, size, 1), setup.model_shards(FAM, size)), (file, 1))
            self.assertEqual(FAM["sha256"][file][0], n)
            self.assertRegex(FAM["sha256"][file][1], r"^[0-9a-f]{64}$")
            self.assertAlmostEqual(n / 1e9, d["download_gb"], delta=0.05)
            self.assertLess(d["arena_gb"], d["download_gb"])
            self.assertEqual(d["ram_gb"], round(setup.fourgbvram_ram_need(d, 8.0)), size)
        # Liquid AI's own file is in its repository, the others in Unsloth's
        self.assertRegex(FAM["models"]["QAD-Q4_0"]["hf"],
                         r"^https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF/resolve/[0-9a-f]{40}/$")
        self.assertEqual([m for m, d in FAM["models"].items() if "hf" in d], ["QAD-Q4_0"])

    def test_the_about_line_says_the_first_size(self):
        n = setup.fourgbvram_ram_need(FAM["models"]["UD-Q4_K_XL"], 8.0)
        self.assertIn(f"~{n:.0f} GB of RAM beside an 8 GB card", FAM["about"])


class Batch(unittest.TestCase):
    def test_where_the_card_holds_every_expert_beside_it(self):
        sizes = FAM["models"]
        # LFM2.5 on any card from 8 GB; Qwen3.6 where the card holds every expert of the size
        for vram, want in ((6.0, []), (7.6, list(sizes)), (31.8, list(sizes))):
            with self.subTest(vram=vram):
                self.assertEqual([m for m, d in sizes.items() if setup.fourgbvram_batch(FAM, d, vram, 32768)], want)
        qwen = setup.FAMILIES["qwen36"]
        for vram, ctx, want in ((8.0, 32768, []), (23.6, 32768, ["UD-Q3_K_M", "UD-Q2_K_XL"]),
                                (23.6, 131072, ["UD-Q3_K_M", "UD-Q2_K_XL"]), (31.8, 131072, list(qwen["models"]))):
            with self.subTest(vram=vram, ctx=ctx):
                self.assertEqual([m for m, d in qwen["models"].items() if setup.fourgbvram_batch(qwen, d, vram, ctx)], want)
        # a family without the figure keeps the engine's batch
        self.assertFalse(setup.fourgbvram_batch(setup.FAMILIES["gemma4"], setup.FAMILIES["gemma4"]["models"]["UD-Q2_K_XL"],
                                          96.0, 32768))


class Main(unittest.TestCase):
    setUp, tearDown, main = glm_tests.Main.setUp, glm_tests.Main.tearDown, glm_tests.Main.main

    def test_install_each_size(self):
        for size, _n in SIZES:
            with self.subTest(size):
                self.downloads, self.runs, self.verified = [], [], []
                code, out, _ = self.main(family="lfm25", model=size, ram=64.0, vram=8188 / 1024)
                self.assertEqual(code, 0, out)
                cfg = json.loads((self.t / f"strata-lfm25-{size.lower()}.json").read_text())   # one config a size
                file = f"LFM2.5-24B-A2B-{size}.gguf"
                self.assertEqual(self.downloads, [FAM["models"][size].get("hf", FAM["hf"]) + file])
                self.assertEqual(self.verified, [(file, *FAM["sha256"][file])])
                tag = "lfm25-" + size
                pack = self.t / "data" / "packs" / tag.lower()
                self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])
                args = cfg["args"]
                self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / tag / file)
                self.assertEqual(Path(args[args.index("--expert-profile") + 1]), pack / "expert-profile.txt")
                for flag in ("--draft-model", "--tail-skip"):
                    self.assertNotIn(flag, args)
                self.assertEqual(args[args.index("--prompt-batch") + 1], "2048")   # on the 8 GB card too
                self.assertEqual(cfg["model_name"], f"lfm2.5-24b-a2b-{size.lower()}")
                self.assertEqual(cfg["display_name"], f"LFM2.5-24B-A2B {size}")
                self.assertNotIn("vision", cfg)

    def test_a_card_that_holds_every_expert_reads_2048_tokens_at_a_time(self):
        code, out, _ = self.main(family="lfm25", model="Q8_0", ram=64.0, vram=32607 / 1024)   # an RTX 5090
        self.assertEqual(code, 0, out)
        args = json.loads((self.t / "strata-lfm25-q8_0.json").read_text())["args"]
        self.assertEqual(args[args.index("--prompt-batch") + 1], "2048")

    def test_the_context_question_has_this_model_s_laptop_numbers(self):
        code, out, _ = self.main(family="lfm25", ram=32.0, vram=8.0)
        self.assertEqual(code, 0, out)
        for line in ("32K tokens   (recommended; 8 GB card: 69-74 tokens/s after a full prompt)",
                     "128K tokens   (2.0 GB more VRAM; 8 GB card: 35-37 tokens/s after a full prompt)"):
            self.assertIn(line, out)
        self.assertNotIn("52-58", out)   # Qwen3.6's and the others'

    def test_the_size_taken_without_a_name(self):
        for ram, size in ((32.0, "UD-Q4_K_XL"), (14.0, "UD-Q3_K_XL")):
            with self.subTest(ram=ram):
                code, out, _ = self.main(family="lfm25", ram=ram, vram=8.0)
                self.assertEqual(code, 0, out)
                self.assertTrue((self.t / f"strata-lfm25-{size.lower()}.json").is_file(), out)

    def test_windows_is_not_asked(self):
        code, out, cfg = self.main(win=True, family="lfm25", ram=32.0, vram=8.0)
        self.assertEqual(code, 0, out)
        self.assertNotIn("Windows", out)

    def test_vision_is_not_used(self):
        code, out, cfg = self.main(["--vision", "yes"], family="lfm25", ram=32.0, vram=8.0)
        self.assertEqual(code, 0, out)
        self.assertIn("not used with LFM2.5-24B-A2B", out)
        self.assertNotIn("vision", cfg)


if __name__ == "__main__":
    unittest.main()

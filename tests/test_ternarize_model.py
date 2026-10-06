"""Unit test suite for tools/ternarize_model.py.

Verifies end-to-end GGUF model ternarization to Strata TQ1_0 (GGML Type 34):
1. Selective quantization policy adherence:
   - PROTECTED: token_embd.weight, output.weight, *norm*.weight, *gate_inp* stay FP16/Q8.
   - TERNARIZED: MoE experts (*ffn_gate_exps*, *ffn_up_exps*, *ffn_down_exps*) and
     attention projections (*attn_q*, *attn_k*, *attn_v*, *attn_output*) become TQ1_0 (Type 34).
2. Data byte sizes strictly equal (N / 256) * 54 bytes.
3. Metadata preservation (general.architecture, context_length, alignment, block_count).
4. Pre-rotation via FWHT (--hadamard).
5. Calibration (--calib) with sparsity in 35.0% - 50.0% range.
6. Layer range filtering (--layers).
7. Strata C++ header reader verification (strata-gguf / tools/gguf_reader.py).
8. Target models synthetic compatibility:
   - Qwen 3.8 Flash Next
   - Qwen 3.6 35B A3B
   - Ornith 1.5 35B A3B

Usage:
    python -m unittest tests/test_ternarize_model.py -v
"""

from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Ensure project root and tools are in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import gguf

from tools.gguf_reader import GGUFFile
from tools.ternarize_model import (
    classify_tensor,
    parse_layer_range,
    ternarize_model,
)
from tools.ternary.trit_sampler import (
    BLOCK_SIZE,
    TQ1_0_BLOCK_BYTES,
    unpack_tq1_0,
)


def create_synthetic_model_gguf(
    filepath: Path,
    arch: str = "qwen35moe",
    num_layers: int = 2,
    hidden_size: int = 512,
    intermediate_size: int = 256,
    num_experts: int = 8,
    use_fp16_protected: bool = True,
) -> Path:
    """Creates a synthetic GGUF model for testing."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(filepath), arch=arch)

    # Metadata
    writer.add_uint32(f"{arch}.block_count", num_layers)
    writer.add_uint32(f"{arch}.expert_count", num_experts)
    writer.add_uint32(f"{arch}.expert_used_count", 2)
    writer.add_uint32("general.alignment", 32)
    writer.add_string("general.name", f"synthetic-{arch}-test")
    writer.add_array("tokenizer.ggml.tokens", ["<|endoftext|>", "test", "tensor"])

    rng = np.random.default_rng(42)

    prot_dtype = np.float16 if use_fp16_protected else np.float32

    # Globals
    embd = (rng.standard_normal((hidden_size, hidden_size)) * 0.02).astype(prot_dtype)
    writer.add_tensor("token_embd.weight", embd)

    lm_head = (rng.standard_normal((hidden_size, hidden_size)) * 0.02).astype(prot_dtype)
    writer.add_tensor("output.weight", lm_head)

    out_norm = (np.ones((hidden_size,))).astype(prot_dtype)
    writer.add_tensor("output_norm.weight", out_norm)

    # Per-layer tensors
    for il in range(num_layers):
        p = f"blk.{il}."

        # RMSNorms (PROTECTED)
        writer.add_tensor(p + "attn_norm.weight", np.ones(hidden_size, dtype=prot_dtype))
        writer.add_tensor(p + "post_attention_norm.weight", np.ones(hidden_size, dtype=prot_dtype))
        writer.add_tensor(p + "attn_q_norm.weight", np.ones(128, dtype=prot_dtype))
        writer.add_tensor(p + "attn_k_norm.weight", np.ones(128, dtype=prot_dtype))

        # Router weights (PROTECTED)
        router = (rng.standard_normal((num_experts, hidden_size)) * 0.02).astype(prot_dtype)
        writer.add_tensor(p + "ffn_gate_inp.weight", router)

        # Attention projections (TERNARIZED)
        # Note: in numpy C-order, shape is (out_features, in_features)
        # fastest varying dim is hidden_size (512)
        q = (rng.standard_normal((hidden_size, hidden_size)) * 0.02).astype(np.float32)
        k = (rng.standard_normal((hidden_size // 2, hidden_size)) * 0.02).astype(np.float32)
        v = (rng.standard_normal((hidden_size // 2, hidden_size)) * 0.02).astype(np.float32)
        o = (rng.standard_normal((hidden_size, hidden_size)) * 0.02).astype(np.float32)

        writer.add_tensor(p + "attn_q.weight", q)
        writer.add_tensor(p + "attn_k.weight", k)
        writer.add_tensor(p + "attn_v.weight", v)
        writer.add_tensor(p + "attn_output.weight", o)

        # MoE expert projections (TERNARIZED)
        # 3D: (num_experts, intermediate_size, hidden_size)
        gate_exps = (rng.standard_normal((num_experts, intermediate_size, hidden_size)) * 0.02).astype(np.float32)
        up_exps = (rng.standard_normal((num_experts, intermediate_size, hidden_size)) * 0.02).astype(np.float32)
        down_exps = (rng.standard_normal((num_experts, hidden_size, intermediate_size)) * 0.02).astype(np.float32)

        writer.add_tensor(p + "ffn_gate_exps.weight", gate_exps)
        writer.add_tensor(p + "ffn_up_exps.weight", up_exps)
        writer.add_tensor(p + "ffn_down_exps.weight", down_exps)

        # Shared experts (optional / unmodified)
        shexp_g = (rng.standard_normal((intermediate_size, hidden_size)) * 0.02).astype(np.float32)
        shexp_u = (rng.standard_normal((intermediate_size, hidden_size)) * 0.02).astype(np.float32)
        shexp_d = (rng.standard_normal((hidden_size, intermediate_size)) * 0.02).astype(np.float32)
        writer.add_tensor(p + "ffn_gate_shexp.weight", shexp_g)
        writer.add_tensor(p + "ffn_up_shexp.weight", shexp_u)
        writer.add_tensor(p + "ffn_down_shexp.weight", shexp_d)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return filepath


class TestTernarizeModel(unittest.TestCase):
    """Main verification suite for GGUF model ternarizer."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="strata_tern_test_"))
        self.src_gguf = self.tmp_dir / "src_model.gguf"
        self.out_gguf = self.tmp_dir / "dest_model_tq1_0.gguf"
        create_synthetic_model_gguf(self.src_gguf, arch="qwen35moe", num_layers=2)

    def tearDown(self):
        # Force garbage collection to release Windows memmap file handles
        gc.collect()
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_layer_range_parser(self):
        """Verifies parsing of various layer range patterns."""
        self.assertEqual(parse_layer_range("0-3"), {0, 1, 2, 3})
        self.assertEqual(parse_layer_range("0,2,5"), {0, 2, 5})
        self.assertEqual(parse_layer_range("2-", max_layers=5), {2, 3, 4})
        self.assertEqual(parse_layer_range("-2"), {0, 1, 2})
        self.assertEqual(parse_layer_range("4"), {4})
        self.assertIsNone(parse_layer_range(None))
        self.assertIsNone(parse_layer_range(""))

    def test_tensor_classification_policy(self):
        """Verifies that selective policy correctly categorizes all tensor types."""
        # Protected: embeddings, LM head, norms, router
        self.assertEqual(classify_tensor("token_embd.weight", (512, 512)), "PROTECTED")
        self.assertEqual(classify_tensor("output.weight", (512, 512)), "PROTECTED")
        self.assertEqual(classify_tensor("output_norm.weight", (512,)), "PROTECTED")
        self.assertEqual(classify_tensor("blk.0.attn_norm.weight", (512,)), "PROTECTED")
        self.assertEqual(classify_tensor("blk.0.post_attention_norm.weight", (512,)), "PROTECTED")
        self.assertEqual(classify_tensor("blk.0.attn_q_norm.weight", (128,)), "PROTECTED")
        self.assertEqual(classify_tensor("blk.0.attn_k_norm.weight", (128,)), "PROTECTED")
        self.assertEqual(classify_tensor("blk.0.ffn_gate_inp.weight", (8, 512)), "PROTECTED")

        # Ternarized: attention projections
        self.assertEqual(classify_tensor("blk.0.attn_q.weight", (512, 512)), "TERNARIZED")
        self.assertEqual(classify_tensor("blk.0.attn_k.weight", (256, 512)), "TERNARIZED")
        self.assertEqual(classify_tensor("blk.0.attn_v.weight", (256, 512)), "TERNARIZED")
        self.assertEqual(classify_tensor("blk.0.attn_output.weight", (512, 512)), "TERNARIZED")

        # Ternarized: MoE routed experts
        self.assertEqual(classify_tensor("blk.0.ffn_gate_exps.weight", (8, 256, 512)), "TERNARIZED")
        self.assertEqual(classify_tensor("blk.0.ffn_up_exps.weight", (8, 256, 512)), "TERNARIZED")
        self.assertEqual(classify_tensor("blk.0.ffn_down_exps.weight", (8, 512, 256)), "TERNARIZED")

        # Unmodified: shared experts (by default)
        self.assertEqual(classify_tensor("blk.0.ffn_gate_shexp.weight", (256, 512)), "UNMODIFIED")
        # Ternarized: shared experts if enabled
        self.assertEqual(
            classify_tensor("blk.0.ffn_gate_shexp.weight", (256, 512), ternarize_shared_experts=True),
            "TERNARIZED",
        )

        # Layer range filtering
        self.assertEqual(
            classify_tensor("blk.0.attn_q.weight", (512, 512), layer_range={1, 2}),
            "UNMODIFIED",
        )
        self.assertEqual(
            classify_tensor("blk.1.attn_q.weight", (512, 512), layer_range={1, 2}),
            "TERNARIZED",
        )

    def test_end_to_end_synthetic_ternarization(self):
        """Verifies end-to-end ternarization on synthetic model."""
        res = ternarize_model(
            input_path=self.src_gguf,
            output_path=self.out_gguf,
            hadamard=False,
            calib=False,
            verbose=False,
        )

        self.assertEqual(res["status"], "success")
        self.assertGreater(res["compression_ratio"], 1.5)
        self.assertGreater(res["tensors_ternarized"], 0)
        self.assertGreater(res["tensors_protected"], 0)

        # Inspect resulting GGUF with gguf.GGUFReader
        r = gguf.GGUFReader(str(self.out_gguf))

        # Check metadata preservation
        self.assertIn("general.architecture", r.fields)
        self.assertEqual(r.fields["general.architecture"].contents(), "qwen35moe")
        self.assertEqual(r.fields["qwen35moe.block_count"].contents(), 2)
        self.assertEqual(r.fields["qwen35moe.expert_count"].contents(), 8)
        self.assertEqual(r.fields["tokenizer.ggml.tokens"].contents(), ["<|endoftext|>", "test", "tensor"])

        for t in r.tensors:
            name = t.name
            t_type = t.tensor_type
            n_bytes = t.n_bytes
            total_elements = 1
            for d in t.shape:
                total_elements *= d

            # 1. Protected tensors verification
            is_protected = (
                "token_embd" in name
                or name == "output.weight"
                or name.endswith(".output.weight")
                or "norm" in name
                or "gate_inp" in name
            )
            if is_protected:
                self.assertIn(
                    t_type,
                    (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.Q8_0),
                    f"Protected tensor '{name}' had invalid type {t_type}",
                )
                self.assertNotEqual(t_type, gguf.GGMLQuantizationType.TQ1_0)

            # 2. MoE routed experts & Attention projections verification
            elif any(k in name for k in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight")):
                self.assertEqual(
                    t_type,
                    gguf.GGMLQuantizationType.TQ1_0,
                    f"Tensor '{name}' expected TQ1_0 (Type 34) but got {t_type}",
                )
                # Verify exact byte count formula: (N / 256) * 54
                expected_bytes = (total_elements // BLOCK_SIZE) * TQ1_0_BLOCK_BYTES
                self.assertEqual(
                    n_bytes,
                    expected_bytes,
                    f"Tensor '{name}' byte size mismatch: {n_bytes} != {expected_bytes}",
                )

                # Verify unpack round-trip produces valid trits in {-1, 0, 1}
                trits, scales = unpack_tq1_0(t.data.tobytes())
                self.assertTrue(np.all(np.isin(trits, [-1, 0, 1])))
                self.assertEqual(len(scales), total_elements // BLOCK_SIZE)

        del r
        gc.collect()

    def test_pre_rotation_hadamard(self):
        """Verifies ternarization with FWHT pre-rotation enabled (--hadamard)."""
        out_hadamard = self.tmp_dir / "dest_hadamard.gguf"
        res = ternarize_model(
            input_path=self.src_gguf,
            output_path=out_hadamard,
            hadamard=True,
            calib=False,
            verbose=False,
        )

        self.assertEqual(res["status"], "success")
        self.assertTrue(res["hadamard"])

        r = gguf.GGUFReader(str(out_hadamard))
        self.assertEqual(r.fields["strata.ternarizer_hadamard"].contents(), "true")

        # Verify all ternarized tensors have type 34 and valid unpacked data
        for t in r.tensors:
            if "attn_q.weight" in t.name or "ffn_gate_exps" in t.name:
                self.assertEqual(t.tensor_type, gguf.GGMLQuantizationType.TQ1_0)
                trits, _ = unpack_tq1_0(t.data.tobytes())
                self.assertTrue(np.all(np.isin(trits, [-1, 0, 1])))

        del r
        gc.collect()

    def test_optimal_calibration_sparsity(self):
        """Verifies ternarization with calibration (--calib) produces target sparsity in [35%, 50%]."""
        out_calib = self.tmp_dir / "dest_calib.gguf"
        res = ternarize_model(
            input_path=self.src_gguf,
            output_path=out_calib,
            hadamard=False,
            calib=True,
            verbose=False,
        )

        self.assertEqual(res["status"], "success")
        self.assertTrue(res["calib"])
        # Average zero-trit sparsity should be strictly within target range
        self.assertGreaterEqual(res["avg_zero_sparsity"], 35.0)
        self.assertLessEqual(res["avg_zero_sparsity"], 50.0)

    def test_layer_range_selective_ternarization(self):
        """Verifies restricting ternarization to specific layer index (--layers)."""
        out_layer1 = self.tmp_dir / "dest_layer1_only.gguf"
        res = ternarize_model(
            input_path=self.src_gguf,
            output_path=out_layer1,
            layers="1",  # Only ternarize layer 1; layer 0 remains unmodified
            verbose=False,
        )

        self.assertEqual(res["status"], "success")
        r = gguf.GGUFReader(str(out_layer1))

        # Check Layer 0: attention and MoE should be UNMODIFIED (F16 / F32)
        l0_q = next(t for t in r.tensors if t.name == "blk.0.attn_q.weight")
        self.assertNotEqual(l0_q.tensor_type, gguf.GGMLQuantizationType.TQ1_0)

        l0_moe = next(t for t in r.tensors if t.name == "blk.0.ffn_gate_exps.weight")
        self.assertNotEqual(l0_moe.tensor_type, gguf.GGMLQuantizationType.TQ1_0)

        # Check Layer 1: attention and MoE should be TQ1_0
        l1_q = next(t for t in r.tensors if t.name == "blk.1.attn_q.weight")
        self.assertEqual(l1_q.tensor_type, gguf.GGMLQuantizationType.TQ1_0)

        l1_moe = next(t for t in r.tensors if t.name == "blk.1.ffn_gate_exps.weight")
        self.assertEqual(l1_moe.tensor_type, gguf.GGMLQuantizationType.TQ1_0)

        del r
        gc.collect()

    def test_shared_experts_option(self):
        """Verifies optional ternarization of shared experts (--ternarize-shared-experts)."""
        out_shexp = self.tmp_dir / "dest_shexp.gguf"
        res = ternarize_model(
            input_path=self.src_gguf,
            output_path=out_shexp,
            ternarize_shared_experts=True,
            verbose=False,
        )

        self.assertEqual(res["status"], "success")
        r = gguf.GGUFReader(str(out_shexp))

        shexp_tensor = next(t for t in r.tensors if "ffn_gate_shexp" in t.name)
        self.assertEqual(shexp_tensor.tensor_type, gguf.GGMLQuantizationType.TQ1_0)

        del r
        gc.collect()

    def test_target_architectures_synthetic_compatibility(self):
        """Tests compatibility across the 3 target model architectures:
        1. Qwen 3.8 Flash Next (qwen38_moe, 512 experts)
        2. Qwen 3.6 35B A3B (qwen35moe, 256 experts)
        3. Ornith 1.5 35B A3B (qwen35moe, 256 experts)
        """
        targets = [
            ("qwen38_moe", 512, "qwen3.8-flash-next"),
            ("qwen35moe", 256, "qwen3.6-35b-a3b"),
            ("qwen35moe", 256, "ornith1.5-35b-a3b"),
        ]

        for arch, n_exp, model_name in targets:
            with self.subTest(arch=arch, model=model_name):
                src = self.tmp_dir / f"{model_name}_src.gguf"
                out = self.tmp_dir / f"{model_name}_tq1.gguf"

                create_synthetic_model_gguf(
                    src,
                    arch=arch,
                    num_layers=1,
                    hidden_size=256,
                    intermediate_size=256,
                    num_experts=4,  # mini count for rapid test execution
                )

                res = ternarize_model(src, out, verbose=False)
                self.assertEqual(res["status"], "success")

                r = gguf.GGUFReader(str(out))
                self.assertEqual(r.fields["general.architecture"].contents(), arch)

                # Confirm MoE experts are TQ1_0
                exp_t = next(t for t in r.tensors if "ffn_gate_exps" in t.name)
                self.assertEqual(exp_t.tensor_type, gguf.GGMLQuantizationType.TQ1_0)

                del r
                gc.collect()

    def test_strata_cpp_reader_verification(self):
        """Verifies that Strata C++ reader and Python GGUFFile parse the output without error."""
        res = ternarize_model(self.src_gguf, self.out_gguf, verbose=False)
        self.assertEqual(res["status"], "success")

        # 1. Verify with tools/gguf_reader.py reference parser
        gfile = GGUFFile(self.out_gguf)
        self.assertEqual(gfile.version, 3)
        self.assertEqual(gfile.alignment, 32)
        self.assertGreater(len(gfile.tensors), 0)

        for t in gfile.tensors:
            # expected_bytes() must match byte size in file
            exp = t.expected_bytes()
            self.assertIsNotNone(exp, f"Unknown geometry for tensor {t.name} (type {t.type_name})")
            if t.type_name == "TQ1_0":
                self.assertEqual(exp, (t.elements // BLOCK_SIZE) * TQ1_0_BLOCK_BYTES)

        # 2. Verify with compiled Strata C++ binary if available
        strata_gguf_bin = ROOT / "build" / "strata-gguf.exe"
        if strata_gguf_bin.is_file():
            cp = subprocess.run(
                [str(strata_gguf_bin), str(self.out_gguf), "--check"],
                capture_output=True,
                text=True,
            )
            # Check geometry bracket: must be 0 out of range, 0 types unknown
            self.assertIn(
                "geometry bracket: 0 out of range, 0 types unknown",
                cp.stdout,
                f"Strata C++ reader geometry check failed:\n{cp.stdout}\n{cp.stderr}",
            )

    def test_cli_execution(self):
        """Verifies execution of tools/ternarize_model.py CLI via subprocess."""
        cli_out = self.tmp_dir / "cli_output.gguf"
        cmd = [
            sys.executable,
            str(ROOT / "tools" / "ternarize_model.py"),
            "--input",
            str(self.src_gguf),
            "--output",
            str(cli_out),
            "--hadamard",
            "--calib",
            "--verbose",
        ]

        cp = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(cp.returncode, 0, f"CLI command failed:\n{cp.stderr}\n{cp.stdout}")
        self.assertTrue(cli_out.is_file())
        self.assertIn("STRATA TQ1_0 TERNARIZATION COMPLETED SUCCESSFULLY", cp.stdout)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for tools/stream_ternarize_safetensors.py."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

import gguf
from tools.stream_ternarize_safetensors import (
    map_safetensors_to_gguf_name,
    stream_ternarize_safetensors,
)


class TestStreamTernarizeSafeTensors(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="strata_stream_test_")
        self.test_dir_path = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_name_mapping(self):
        self.assertEqual(map_safetensors_to_gguf_name("model.embed_tokens.weight"), "token_embd.weight")
        self.assertEqual(map_safetensors_to_gguf_name("lm_head.weight"), "output.weight")
        self.assertEqual(map_safetensors_to_gguf_name("model.layers.3.self_attn.q_proj.weight"), "blk.3.attn_q.weight")
        self.assertEqual(map_safetensors_to_gguf_name("model.layers.0.block_sparse_moe.experts.gate_proj.weight"), "blk.0.ffn_gate_exps.weight")
        self.assertEqual(map_safetensors_to_gguf_name("model.layers.2.block_sparse_moe.gate.weight"), "blk.2.ffn_gate_inp.weight")

    def test_end_to_end_streaming_and_shard_deletion(self):
        shard1_path = self.test_dir_path / "shard-00001-of-00002.safetensors"
        shard2_path = self.test_dir_path / "shard-00002-of-00002.safetensors"
        output_gguf = self.test_dir_path / "stream_output_tq1_0.gguf"

        rng = np.random.default_rng(42)

        # Shard 1: Embed tokens (protected) + Attention Q (ternarized)
        tensors1 = {
            "model.embed_tokens.weight": rng.normal(0, 0.02, size=(256, 512)).astype(np.float32),
            "model.layers.0.self_attn.q_proj.weight": rng.normal(0, 0.02, size=(256, 512)).astype(np.float32),
        }
        save_file(tensors1, str(shard1_path))

        # Shard 2: MoE expert gate (ternarized) + Output norm (protected)
        tensors2 = {
            "model.layers.0.block_sparse_moe.experts.gate_proj.weight": rng.normal(0, 0.02, size=(256, 512)).astype(np.float32),
            "model.norm.weight": np.ones((512,), dtype=np.float32),
        }
        save_file(tensors2, str(shard2_path))

        self.assertTrue(shard1_path.exists())
        self.assertTrue(shard2_path.exists())

        stats = stream_ternarize_safetensors(
            input_shards=[shard1_path, shard2_path],
            output_path=output_gguf,
            arch="qwen38_moe",
            apply_hadamard=True,
            apply_calib=True,
            delete_shards_after_processing=True,
            verbose=False,
        )

        self.assertEqual(stats["tensors_processed"], 4)
        self.assertEqual(stats["tensors_ternarized"], 2)
        self.assertEqual(stats["tensors_protected"], 2)

        # Verify shards were purged to conserve disk space
        self.assertFalse(shard1_path.exists())
        self.assertFalse(shard2_path.exists())

        # Verify output GGUF exists
        self.assertTrue(output_gguf.exists())

        # Verify GGUF structure with reader
        reader = gguf.GGUFReader(str(output_gguf))
        tensor_types = {t.name: t.tensor_type for t in reader.tensors}

        self.assertEqual(tensor_types["token_embd.weight"], gguf.GGMLQuantizationType.F16)
        self.assertEqual(tensor_types["output_norm.weight"], gguf.GGMLQuantizationType.F16)
        self.assertEqual(tensor_types["blk.0.attn_q.weight"], gguf.GGMLQuantizationType.TQ1_0)
        self.assertEqual(tensor_types["blk.0.ffn_gate_exps.weight"], gguf.GGMLQuantizationType.TQ1_0)


if __name__ == "__main__":
    unittest.main()

"""tests/test_engine_unification.py - Verification suite for Qwen 3.8 Flash Next & Qwen 3.6 engine unification.

Tests:
1. Automatic architecture detection for Qwen 3.8 Flash Next configs
   (strata-coder-iq1_m.json, strata-q2_0.json, strata-tq1_0.json) and
   Qwen 3.6 / Ornith configs (strata-qwen36.json, strata-ornith.json).
2. Direct binary GGUF v3 header and model pack directory structure inspection.
3. Engine executable resolution for generic strata executables and explicitly pinned binaries.
4. CLI argument synthesis guaranteeing preservation of key Qwen 3.8 flags
   (--ple-gguf, --expert-profile, --spec, --mtp) and adaptation of Qwen 3.6 flags
   (--kv int8, --kv-unified, --spec-k).
5. Mock execution, handshake, exit code handling, and runtime resolution in StrataEngine.
6. Schema adherence and validation for unified configs and RTX 3050 production templates.

Usage:
    python -m unittest tests/test_engine_unification.py -v
"""
from __future__ import annotations

import io
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from tools.inspect_gguf_arch import (
    GGUF_MAGIC,
    build_engine_command,
    detect_config_architecture,
    detect_model_architecture,
    inspect_gguf_header,
    inspect_model_pack,
    is_generic_strata,
    resolve_engine_executable,
    validate_unified_config,
)
from serve.server import StrataEngine


class TestArchitectureDetection(unittest.TestCase):
    """Verifies automatic detection of Qwen 3.8 Flash Next and contrast models."""

    def test_detect_coder_iq1m_config(self):
        cfg_path = ROOT / "configs" / "strata-coder-iq1_m.json"
        self.assertTrue(cfg_path.is_file(), f"Config missing: {cfg_path}")
        arch = detect_config_architecture(cfg_path)
        self.assertEqual(arch, "qwen38_moe")

    def test_detect_q2_0_config(self):
        cfg_path = ROOT / "configs" / "strata-q2_0.json"
        self.assertTrue(cfg_path.is_file(), f"Config missing: {cfg_path}")
        arch = detect_config_architecture(cfg_path)
        self.assertEqual(arch, "qwen38_moe")

    def test_detect_tq1_0_config(self):
        cfg_path = ROOT / "configs" / "strata-tq1_0.json"
        self.assertTrue(cfg_path.is_file(), f"Config missing: {cfg_path}")
        arch = detect_config_architecture(cfg_path)
        self.assertEqual(arch, "qwen38_moe")

    def test_detect_qwen36_config_contrast(self):
        cfg_path = ROOT / "configs" / "strata-qwen36.json"
        self.assertTrue(cfg_path.is_file(), f"Config missing: {cfg_path}")
        arch = detect_config_architecture(cfg_path)
        self.assertEqual(arch, "qwen36_moe")

    def test_detect_ornith_config_contrast(self):
        cfg_path = ROOT / "configs" / "strata-ornith.json"
        self.assertTrue(cfg_path.is_file(), f"Config missing: {cfg_path}")
        arch = detect_config_architecture(cfg_path)
        self.assertEqual(arch, "qwen36_moe")

    def test_detect_gguf_metadata_qwen38(self):
        # 512 experts is characteristic of Qwen 3.8 Flash Next MoE
        meta_512 = {
            "general.architecture": "qwen2moe",
            "qwen2moe.expert_count": 512,
            "general.name": "Qwen3.8-Flash-Next-Coder",
        }
        self.assertEqual(detect_model_architecture(meta_512), "qwen38_moe")

    def test_detect_gguf_metadata_qwen36(self):
        # 256 experts is characteristic of Qwen 3.6 / Ornith MoE
        meta_256 = {
            "general.architecture": "qwen35moe",
            "qwen35moe.expert_count": 256,
            "general.name": "Qwen3.6-35B-A3B",
        }
        self.assertEqual(detect_model_architecture(meta_256), "qwen36_moe")

    def test_detect_synthetic_gguf_header(self):
        """Constructs an in-memory GGUF v3 header to verify direct binary parsing."""
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tf:
            temp_path = Path(tf.name)
            try:
                # GGUF Magic (4B), Version 3 (4B), n_tensors=0 (8B), n_kv=2 (8B)
                tf.write(struct.pack("<IIQQ", GGUF_MAGIC, 3, 0, 2))

                # KV 1: "general.architecture" -> "qwen38moe" (string type 8)
                key1 = b"general.architecture"
                tf.write(struct.pack("<Q", len(key1)))
                tf.write(key1)
                tf.write(struct.pack("<I", 8))  # string type
                val1 = b"qwen38moe"
                tf.write(struct.pack("<Q", len(val1)))
                tf.write(val1)

                # KV 2: "qwen38moe.expert_count" -> 512 (u32 type 4)
                key2 = b"qwen38moe.expert_count"
                tf.write(struct.pack("<Q", len(key2)))
                tf.write(key2)
                tf.write(struct.pack("<I", 4))  # u32 type
                tf.write(struct.pack("<I", 512))
                tf.flush()
                tf.close()

                arch = detect_model_architecture(temp_path)
                self.assertEqual(arch, "qwen38_moe")
            finally:
                if temp_path.exists():
                    temp_path.unlink()

    def test_detect_model_pack_directory_qwen38(self):
        """Verifies architecture detection from a model pack directory with manifest and profile."""
        temp_dir = Path(tempfile.mkdtemp(prefix="strata_pack_q38_"))
        try:
            manifest = {"model_name": "qwen3.8-flash-next-test", "architecture": "qwen38_moe", "has_ple": True}
            (temp_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            arch = detect_model_architecture(temp_dir)
            self.assertEqual(arch, "qwen38_moe")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_detect_model_pack_directory_qwen36(self):
        """Verifies architecture detection from a model pack directory with qwen36 manifest."""
        temp_dir = Path(tempfile.mkdtemp(prefix="strata_pack_q36_"))
        try:
            manifest = {"model_name": "ornith-1.5-35b-a3b", "architecture": "qwen35moe", "expert_count": 256}
            (temp_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            arch = detect_model_architecture(temp_dir)
            self.assertEqual(arch, "qwen36_moe")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestExecutableResolution(unittest.TestCase):
    """Verifies that generic or unset executables resolve to the proper binary, and custom bins are kept."""

    def test_generic_exe_resolves_to_qwen36_for_qwen36(self):
        self.assertEqual(resolve_engine_executable("qwen36_moe", None), "engine/strata-qwen36.exe")
        self.assertEqual(resolve_engine_executable("qwen36_moe", ""), "engine/strata-qwen36.exe")
        self.assertEqual(resolve_engine_executable("qwen36_moe", "strata"), "engine/strata-qwen36.exe")
        self.assertEqual(resolve_engine_executable("qwen36_moe", "strata.exe"), "engine/strata-qwen36.exe")
        self.assertEqual(resolve_engine_executable("qwen36_moe", "engine/strata.exe"), "engine/strata-qwen36.exe")
        self.assertEqual(resolve_engine_executable("qwen36_moe", "engine/strata-qwen36.exe"), "engine/strata-qwen36.exe")

    def test_generic_exe_resolves_to_qwen38_for_qwen38(self):
        self.assertEqual(resolve_engine_executable("qwen38_moe", None), "engine/strata.exe")
        self.assertEqual(resolve_engine_executable("qwen38_moe", ""), "engine/strata.exe")
        self.assertEqual(resolve_engine_executable("qwen38_moe", "strata"), "engine/strata.exe")
        self.assertEqual(resolve_engine_executable("qwen38_moe", "strata.exe"), "engine/strata.exe")
        self.assertEqual(resolve_engine_executable("qwen38_moe", "engine/strata.exe"), "engine/strata.exe")
        # If config pointed to strata-qwen36 for a qwen38 model, bridge switches it to strata.exe
        self.assertEqual(resolve_engine_executable("qwen38_moe", "engine/strata-qwen36.exe"), "engine/strata.exe")

    def test_custom_pinned_exe_preserved(self):
        custom_bin = "custom/bin/strata_custom.exe"
        self.assertEqual(resolve_engine_executable("qwen36_moe", custom_bin), custom_bin)
        self.assertEqual(resolve_engine_executable("qwen38_moe", custom_bin), custom_bin)


class TestCliArgumentSynthesis(unittest.TestCase):
    """Verifies that synthesized CLI invocations preserve critical parameters for both architectures."""

    def _verify_qwen38_flags(self, cmd: list[str], expected_spec: str = "6"):
        self.assertIn("--serve", cmd)
        self.assertIn("--ple-gguf", cmd, "Missing --ple-gguf in synthesized command")
        self.assertIn("--expert-profile", cmd, "Missing --expert-profile in synthesized command")
        self.assertIn("--spec", cmd, "Missing --spec in synthesized command")
        self.assertIn("--mtp", cmd, "Missing --mtp in synthesized command")

        # Verify argument-value pairings
        idx_ple = cmd.index("--ple-gguf")
        self.assertTrue(idx_ple + 1 < len(cmd) and not cmd[idx_ple + 1].startswith("--"))

        idx_prof = cmd.index("--expert-profile")
        self.assertTrue(idx_prof + 1 < len(cmd) and not cmd[idx_prof + 1].startswith("--"))

        idx_spec = cmd.index("--spec")
        self.assertTrue(idx_spec + 1 < len(cmd) and cmd[idx_spec + 1] == expected_spec)

        idx_mtp = cmd.index("--mtp")
        self.assertTrue(idx_mtp + 1 < len(cmd) and not cmd[idx_mtp + 1].startswith("--"))

    def test_coder_iq1m_cli_synthesis(self):
        cfg_path = ROOT / "configs" / "strata-coder-iq1_m.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        cmd = build_engine_command("qwen38_moe", cfg)
        self.assertTrue(cmd[0].endswith("strata.exe") or "strata" in cmd[0])
        self._verify_qwen38_flags(cmd, expected_spec="6")
        self.assertIn("--pack", cmd)
        self.assertIn("--native", cmd)

    def test_q2_0_cli_synthesis(self):
        cfg_path = ROOT / "configs" / "strata-q2_0.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        cmd = build_engine_command("qwen38_moe", cfg)
        self._verify_qwen38_flags(cmd, expected_spec="6")

    def test_tq1_0_cli_synthesis(self):
        cfg_path = ROOT / "configs" / "strata-tq1_0.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        cmd = build_engine_command("qwen38_moe", cfg)
        self._verify_qwen38_flags(cmd, expected_spec="6")

    def test_qwen36_cli_synthesis(self):
        cfg_path = ROOT / "configs" / "strata-qwen36.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        cmd = build_engine_command("qwen36_moe", cfg)
        self.assertEqual(cmd[0], "engine/strata-qwen36.exe")
        self.assertIn("--serve", cmd)
        self.assertIn("--kv", cmd)
        idx_kv = cmd.index("--kv")
        self.assertEqual(cmd[idx_kv + 1], "int8")
        self.assertIn("--kv-unified", cmd)
        self.assertIn("--spec-k", cmd)
        idx_spec = cmd.index("--spec-k")
        self.assertEqual(cmd[idx_spec + 1], "6")

    def test_ornith_cli_synthesis(self):
        cfg_path = ROOT / "configs" / "strata-ornith.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        cmd = build_engine_command("qwen36_moe", cfg)
        self.assertEqual(cmd[0], "engine/strata-qwen36.exe")
        self.assertIn("--serve", cmd)
        self.assertIn("--kv", cmd)
        idx_kv = cmd.index("--kv")
        self.assertEqual(cmd[idx_kv + 1], "int8")
        self.assertIn("--kv-unified", cmd)
        self.assertIn("--spec-k", cmd)

    def test_qwen36_default_flags_adaptation(self):
        """Verifies that a bare Qwen 3.6 config automatically gets --kv int8, --kv-unified, and mapped --spec-k."""
        cfg = {
            "exe": "strata",
            "model_name": "qwen3.6-35b-a3b",
            "args": [
                "--native", "models/qwen3.6.gguf",
                "--spec", "4",
                "--max-context", "65536"
            ]
        }
        cmd = build_engine_command("qwen36_moe", cfg)
        self.assertEqual(cmd[0], "engine/strata-qwen36.exe")
        self.assertIn("--serve", cmd)
        self.assertIn("--kv", cmd)
        self.assertEqual(cmd[cmd.index("--kv") + 1], "int8")
        self.assertIn("--kv-unified", cmd)
        self.assertIn("--spec-k", cmd)
        self.assertEqual(cmd[cmd.index("--spec-k") + 1], "4")
        self.assertNotIn("--spec", cmd)

    def test_incompatible_flags_sanitization_qwen38(self):
        """Ensures Qwen 3.6-specific flags like --kv-unified and --spec-k are stripped for Qwen 3.8."""
        cfg = {
            "exe": "engine/strata.exe",
            "model_name": "qwen3.8-test",
            "args": [
                "--native", "model.gguf",
                "--ple-gguf", "ple.gguf",
                "--expert-profile", "prof.bin",
                "--spec", "6",
                "--mtp", "mtp/rt",
                "--kv-unified",
                "--spec-k", "4"
            ]
        }
        cmd = build_engine_command("qwen38_moe", cfg)
        self.assertNotIn("--kv-unified", cmd)
        self.assertNotIn("--spec-k", cmd)
        self._verify_qwen38_flags(cmd, expected_spec="6")

    def test_incompatible_flags_sanitization_qwen36(self):
        """Ensures Qwen 3.8-specific flags like --ple-gguf and --expert-profile are stripped for Qwen 3.6."""
        cfg = {
            "exe": "engine/strata.exe",
            "model_name": "qwen3.6-35b",
            "args": [
                "--native", "model.gguf",
                "--ple-gguf", "ple.gguf",
                "--expert-profile", "prof.bin",
                "--mtp-window", "512",
                "--resident-budget-gib", "12.0",
                "--spec", "6"
            ]
        }
        cmd = build_engine_command("qwen36_moe", cfg)
        self.assertEqual(cmd[0], "engine/strata-qwen36.exe")
        self.assertNotIn("--ple-gguf", cmd)
        self.assertNotIn("--expert-profile", cmd)
        self.assertNotIn("--mtp-window", cmd)
        self.assertNotIn("--resident-budget-gib", cmd)
        self.assertIn("--kv-unified", cmd)
        self.assertIn("--kv", cmd)
        self.assertIn("--spec-k", cmd)


class MockStream:
    """Mock stdout stream that yields initial lines and remains open until closed."""
    def __init__(self, lines: list[str]):
        self.lines = list(lines)
        self.event = threading.Event()
        self.closed = False

    def __iter__(self):
        for line in self.lines:
            yield line
        self.event.wait()

    def readline(self):
        if self.lines:
            return self.lines.pop(0)
        self.event.wait()
        return ""

    def close(self):
        self.closed = True
        self.event.set()


class TestMockExecutionAndExitCodeHandling(unittest.TestCase):
    """Tests StrataEngine mock execution, handshake, and exit code handling."""

    @mock.patch("subprocess.Popen")
    @mock.patch("serve.winjob.contain")
    def test_successful_mock_handshake(self, mock_contain, mock_popen):
        """Simulates engine starting and printing READY line."""
        mock_stream = MockStream([
            "INFO version=0.1.38 arena_mib=20480\n",
            "READY 65536 stop\n",
        ])
        mock_proc = mock.MagicMock()
        mock_proc.stdout = mock_stream
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        try:
            engine = StrataEngine("engine/strata.exe", ["--serve", "--max-context", "65536"])
            self.assertEqual(engine.max_context, 65536)
            self.assertTrue(engine.can_stop)
            self.assertFalse(engine.ended)
            self.assertEqual(engine.info.get("arena_mib"), 20480)
        finally:
            mock_stream.close()

    @mock.patch("subprocess.Popen")
    @mock.patch("serve.winjob.contain")
    def test_early_exit_code_1_raises_runtime_error(self, mock_contain, mock_popen):
        """Simulates engine crashing with non-zero exit code before READY."""
        mock_proc = mock.MagicMock()
        mock_proc.stdout = io.StringIO("CUDA error: out of memory\n")
        mock_proc.returncode = 1
        mock_proc.poll.return_value = 1
        mock_popen.return_value = mock_proc

        with self.assertRaises(RuntimeError) as ctx:
            StrataEngine("engine/strata.exe", ["--serve", "--max-context", "65536"])

        self.assertIn("the engine exited before it was ready", str(ctx.exception))
        mock_proc.stdin.close.assert_called_once()
        self.assertTrue(mock_proc.stdout.closed)

    @mock.patch("subprocess.Popen")
    @mock.patch("serve.winjob.contain")
    def test_lazy_mode_does_not_spawn_immediately(self, mock_contain, mock_popen):
        """Validates that lazy=True defers process spawning."""
        engine = StrataEngine("engine/strata.exe", ["--serve"], lazy=True)
        self.assertTrue(engine.unloaded)
        mock_popen.assert_not_called()

    @mock.patch("subprocess.Popen")
    @mock.patch("serve.winjob.contain")
    def test_strata_engine_auto_resolves_qwen36(self, mock_contain, mock_popen):
        """Validates that StrataEngine automatically resolves generic exe to strata-qwen36."""
        mock_stream = MockStream([
            "READY 131072 stop\n",
        ])
        mock_proc = mock.MagicMock()
        mock_proc.stdout = mock_stream
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        try:
            engine = StrataEngine("strata", ["--native", "models/Qwen3.6-35B-A3B.gguf"])
            self.assertEqual(engine.spawn[0], "engine/strata-qwen36.exe")
            self.assertIn("--kv-unified", engine.spawn[1])
            self.assertIn("--kv", engine.spawn[1])
        finally:
            mock_stream.close()


class TestTemplateSchemaAdherence(unittest.TestCase):
    """Verifies that RTX 3050 templates adhere to the unified engine schema."""

    def test_template_coder_iq1m_schema(self):
        tmpl_path = ROOT / "configs" / "template-coder-iq1_m-rtx3050.json"
        self.assertTrue(tmpl_path.is_file(), f"Template not found: {tmpl_path}")
        cfg = json.loads(tmpl_path.read_text(encoding="utf-8-sig"))

        errors = validate_unified_config(cfg)
        self.assertEqual(errors, [], f"Template validation errors: {errors}")

        arch = detect_config_architecture(cfg)
        self.assertEqual(arch, "qwen38_moe")

        cmd = build_engine_command(arch, cfg)
        self.assertIn("--ple-gguf", cmd)
        self.assertIn("--expert-profile", cmd)
        self.assertIn("--spec", cmd)
        self.assertIn("--mtp", cmd)

        # Check RTX 3050 memory environment variables
        env = cfg.get("env", {})
        self.assertEqual(env.get("STRATA_RESIDENT_HEADROOM_GIB"), "4.5")
        self.assertEqual(env.get("STRATA_ARENA_LOCK"), "0")
        self.assertEqual(env.get("STRATA_MMVQ_FAST"), "1")
        self.assertEqual(env.get("STRATA_PF_FUSED"), "1")

    def test_template_q2_0_schema(self):
        tmpl_path = ROOT / "configs" / "template-q2_0-rtx3050.json"
        self.assertTrue(tmpl_path.is_file(), f"Template not found: {tmpl_path}")
        cfg = json.loads(tmpl_path.read_text(encoding="utf-8-sig"))

        errors = validate_unified_config(cfg)
        self.assertEqual(errors, [], f"Template validation errors: {errors}")

        arch = detect_config_architecture(cfg)
        self.assertEqual(arch, "qwen38_moe")

        cmd = build_engine_command(arch, cfg)
        self.assertIn("--ple-gguf", cmd)
        self.assertIn("--expert-profile", cmd)
        self.assertIn("--spec", cmd)
        self.assertIn("--mtp", cmd)

        env = cfg.get("env", {})
        self.assertEqual(env.get("STRATA_RESIDENT_HEADROOM_GIB"), "4.5")
        self.assertEqual(env.get("STRATA_ARENA_LOCK"), "0")


class TestConfigValidation(unittest.TestCase):
    """Verifies validation logic for both Qwen 3.8 and Qwen 3.6 configs."""

    def test_validate_qwen38_missing_flag(self):
        cfg = {
            "exe": "engine/strata.exe",
            "model_name": "qwen3.8-test",
            "args": ["--native", "model.gguf"]
        }
        errors = validate_unified_config(cfg)
        self.assertTrue(any("missing required flag" in e for e in errors))

    def test_validate_qwen36_missing_native(self):
        cfg = {
            "exe": "engine/strata-qwen36.exe",
            "model_name": "qwen3.6-test",
            "args": ["--kv", "int8"]
        }
        errors = validate_unified_config(cfg)
        self.assertTrue(any("requires '--native' or '--model'" in e for e in errors))


if __name__ == "__main__":
    unittest.main()

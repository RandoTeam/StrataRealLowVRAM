"""tools/inspect_gguf_arch.py - GGUF architecture inspection and unified engine CLI dispatching.

Identifies model architectures (e.g. "qwen38_moe" vs "qwen36_moe") from GGUF metadata,
model pack structures, or configuration files, and synthesizes unified command-line invocations
for Strata engines.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, Mapping

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian


def inspect_gguf_header(filepath: str | Path) -> dict[str, Any]:
    """Inspects GGUF file header and extracts key metadata without loading tensors."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"GGUF file not found: {path}")

    meta: dict[str, Any] = {}
    with path.open("rb") as f:
        magic_bytes = f.read(4)
        if len(magic_bytes) < 4:
            raise ValueError(f"File too short: {path}")
        magic = struct.unpack("<I", magic_bytes)[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {magic:#x} in {path.name}")

        version_bytes = f.read(4)
        if len(version_bytes) < 4:
            raise ValueError(f"File header truncated: {path}")
        version = struct.unpack("<I", version_bytes)[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version}")

        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        for _ in range(n_kv):
            try:
                key_len_bytes = f.read(8)
                if len(key_len_bytes) < 8:
                    break
                n_key = struct.unpack("<Q", key_len_bytes)[0]
                key = f.read(n_key).decode("utf-8", errors="replace")
                val_type_bytes = f.read(4)
                if len(val_type_bytes) < 4:
                    break
                val_type = struct.unpack("<I", val_type_bytes)[0]

                # Parse metadata values based on type ID
                if val_type == 0:    # u8
                    meta[key] = struct.unpack("<B", f.read(1))[0]
                elif val_type == 1:  # i8
                    meta[key] = struct.unpack("<b", f.read(1))[0]
                elif val_type == 2:  # u16
                    meta[key] = struct.unpack("<H", f.read(2))[0]
                elif val_type == 3:  # i16
                    meta[key] = struct.unpack("<h", f.read(2))[0]
                elif val_type == 4:  # u32
                    meta[key] = struct.unpack("<I", f.read(4))[0]
                elif val_type == 5:  # i32
                    meta[key] = struct.unpack("<i", f.read(4))[0]
                elif val_type == 6:  # f32
                    meta[key] = struct.unpack("<f", f.read(4))[0]
                elif val_type == 7:  # bool
                    meta[key] = bool(struct.unpack("<B", f.read(1))[0])
                elif val_type == 8:  # string
                    slen = struct.unpack("<Q", f.read(8))[0]
                    meta[key] = f.read(slen).decode("utf-8", errors="replace")
                elif val_type == 9:  # array
                    arr_type = struct.unpack("<I", f.read(4))[0]
                    arr_len = struct.unpack("<Q", f.read(8))[0]
                    if arr_type == 8:  # string array
                        arr = []
                        for _ in range(arr_len):
                            slen = struct.unpack("<Q", f.read(8))[0]
                            arr.append(f.read(slen).decode("utf-8", errors="replace"))
                        meta[key] = arr
                    else:
                        size_map = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
                        elem_size = size_map.get(arr_type, 4)
                        f.seek(elem_size * arr_len, os.SEEK_CUR)
                elif val_type in (10, 11, 12):  # u64, i64, f64
                    meta[key] = struct.unpack("<Q", f.read(8))[0]
                else:
                    break
            except Exception:
                break
    return meta


def inspect_model_pack(path: str | Path) -> dict[str, Any]:
    """Inspects a model pack directory for manifest, profiles, and model structure."""
    pack_dir = Path(path)
    if not pack_dir.is_dir():
        return {}

    info: dict[str, Any] = {"is_pack": True}

    manifest_file = pack_dir / "manifest.json"
    if manifest_file.is_file():
        try:
            info["manifest"] = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
        except Exception:
            pass

    config_file = pack_dir / "config.json"
    if config_file.is_file():
        try:
            info["config"] = json.loads(config_file.read_text(encoding="utf-8-sig"))
        except Exception:
            pass

    # Inspect expert profile header if present
    for prof_candidate in pack_dir.glob("expert-profile*.bin"):
        try:
            with prof_candidate.open("rb") as pf:
                head = pf.read(24)
                if len(head) == 24:
                    _, nl, ne, _, _ = struct.unpack("<5I", head[4:])
                    info["expert_count"] = ne
                    info["layers"] = nl
                    break
        except Exception:
            pass

    # Inspect first GGUF file if present
    for gguf_candidate in pack_dir.glob("*.gguf"):
        try:
            meta = inspect_gguf_header(gguf_candidate)
            info["gguf_meta"] = meta
            break
        except Exception:
            pass

    # Check for PLE files
    info["has_ple"] = any(pack_dir.glob("*ple*"))
    return info


def detect_model_architecture(source: str | Path | Mapping[str, Any]) -> str:
    """Detects whether model architecture is 'qwen38_moe' or 'qwen36_moe'.

    Accepts:
    - Path to GGUF file
    - Path to model pack directory
    - Dictionary of GGUF metadata or model pack manifest
    - Path or identifier string
    """
    if isinstance(source, Mapping):
        # Inspect metadata dictionary
        arch = str(source.get("general.architecture", "")).lower()
        expert_count = (
            source.get(f"{arch}.expert_count") or
            source.get("expert_count") or
            source.get(f"{arch}.expert_used_count")
        )
        if expert_count == 512 or "qwen38" in arch:
            return "qwen38_moe"
        if expert_count == 256 or "qwen36" in arch or "qwen35" in arch:
            return "qwen36_moe"
        name = str(source.get("general.name", "") or source.get("model_name", "")).lower()
        if "3.8" in name or "coder" in name:
            return "qwen38_moe"
        if "3.6" in name or "ornith" in name:
            return "qwen36_moe"
        if source.get("has_ple") or source.get("ple_gguf"):
            return "qwen38_moe"
        return "qwen38_moe"

    source_path = Path(source)
    if source_path.is_file():
        try:
            meta = inspect_gguf_header(source_path)
            return detect_model_architecture(meta)
        except Exception:
            pass

    if source_path.is_dir():
        pack_info = inspect_model_pack(source_path)
        if "gguf_meta" in pack_info:
            return detect_model_architecture(pack_info["gguf_meta"])
        if "manifest" in pack_info:
            return detect_model_architecture(pack_info["manifest"])
        if "config" in pack_info:
            return detect_model_architecture(pack_info["config"])
        if pack_info.get("expert_count") == 512:
            return "qwen38_moe"
        if pack_info.get("has_ple"):
            return "qwen38_moe"

    # Heuristic fallback on path/name string
    s = str(source).lower()
    if any(k in s for k in ("3.8", "qwen38", "coder-iq1_m", "coder", "q2_0", "tq1_0", "iq1_m")):
        return "qwen38_moe"
    if any(k in s for k in ("3.6", "qwen36", "ornith", "35b", "qwen35")):
        return "qwen36_moe"

    return "qwen38_moe"


def detect_config_architecture(cfg: dict[str, Any] | str | Path) -> str:
    """Detects architecture from configuration dictionary or JSON file path."""
    if isinstance(cfg, (str, Path)):
        cfg_path = Path(cfg)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))

    model_name = str(cfg.get("model_name", "")).lower()
    if any(k in model_name for k in ("qwen3.8", "coder", "tq1_0", "q2_0", "iq1_m")):
        return "qwen38_moe"
    if any(k in model_name for k in ("qwen3.6", "ornith", "35b-a3b")):
        return "qwen36_moe"

    args = cfg.get("args", [])

    # Check model path specified in args
    for k in ("--native", "--model", "-m", "--pack"):
        if k in args:
            idx = args.index(k)
            if idx + 1 < len(args):
                detected = detect_model_architecture(args[idx + 1])
                if detected in ("qwen38_moe", "qwen36_moe"):
                    return detected

    if any(k in args for k in ("--ple-gguf", "--pack", "--mtp", "--expert-profile")):
        return "qwen38_moe"
    if any(k in args for k in ("--kv-unified", "--spec-k")):
        return "qwen36_moe"

    exe = str(cfg.get("exe", "")).lower()
    if "strata-qwen36" in exe:
        return "qwen36_moe"
    if "strata.exe" in exe:
        return "qwen38_moe"

    return "qwen38_moe"


def is_generic_strata(exe: str | None) -> bool:
    """Checks whether executable name is a generic Strata executable."""
    if not exe:
        return True
    clean_exe = str(exe).strip().replace("\\", "/")
    base = os.path.basename(clean_exe).lower()
    return base in ("strata", "strata.exe")


def resolve_engine_executable(model_arch: str, exe: str | None) -> str:
    """Selects engine executable based on architecture if exe is generic or unset.

    Explicitly pinned custom engine binaries are preserved.
    """
    if not exe:
        return "engine/strata-qwen36.exe" if model_arch == "qwen36_moe" else "engine/strata.exe"

    clean_exe = str(exe).strip().replace("\\", "/")
    base = os.path.basename(clean_exe).lower()

    if model_arch == "qwen36_moe":
        if is_generic_strata(exe) or base in ("strata", "strata.exe"):
            return "engine/strata-qwen36.exe"
        return exe
    else:  # qwen38_moe or other
        if is_generic_strata(exe) or base in ("strata", "strata.exe", "strata-qwen36", "strata-qwen36.exe"):
            return "engine/strata.exe"
        return exe


def build_engine_command(model_arch: str, cfg: dict[str, Any]) -> list[str]:
    """Synthesizes the unified CLI execution arguments for the given model architecture.

    Ensures that Qwen 3.8 Flash Next flags (--ple-gguf, --expert-profile, --spec, --mtp)
    and Qwen 3.6 flags (--kv int8, --kv-unified, --spec-k) are properly adapted and mapped.
    """
    raw_args = list(cfg.get("args", []))

    if model_arch == "qwen38_moe":
        exe = resolve_engine_executable("qwen38_moe", cfg.get("exe"))
        filtered_args = []
        i = 0
        incompatible_q36_flags = {"--kv-unified", "--spec-k"}
        spec_k_val = None
        while i < len(raw_args):
            flag = raw_args[i]
            if flag in incompatible_q36_flags:
                if flag == "--spec-k" and i + 1 < len(raw_args) and not raw_args[i + 1].startswith("-"):
                    spec_k_val = raw_args[i + 1]
                    i += 2
                else:
                    i += 1
                continue
            filtered_args.append(flag)
            i += 1

        # If --spec was absent but --spec-k was specified, preserve it as --spec
        if "--spec" not in filtered_args:
            spec_val = spec_k_val or str(cfg.get("spec") or cfg.get("spec_k") or "")
            if spec_val:
                filtered_args += ["--spec", str(spec_val)]

        # Ensure critical flags from top-level config are included if omitted from args
        if "--ple-gguf" not in filtered_args and cfg.get("ple_gguf"):
            filtered_args += ["--ple-gguf", str(cfg["ple_gguf"])]
        if "--expert-profile" not in filtered_args and cfg.get("expert_profile"):
            filtered_args += ["--expert-profile", str(cfg["expert_profile"])]
        if "--mtp" not in filtered_args and cfg.get("mtp"):
            filtered_args += ["--mtp", str(cfg["mtp"])]

        # Memory budget flags from config
        if "--resident-budget-gib" not in filtered_args and cfg.get("resident_budget_gib"):
            filtered_args += ["--resident-budget-gib", str(cfg["resident_budget_gib"])]
        if "--vram-reserve-mib" not in filtered_args and cfg.get("vram_reserve_mib"):
            filtered_args += ["--vram-reserve-mib", str(cfg["vram_reserve_mib"])]
        if "--kv-resident" not in filtered_args and cfg.get("kv_resident"):
            filtered_args += ["--kv-resident", str(cfg["kv_resident"])]
        if "--expert-cache" not in filtered_args and cfg.get("expert_cache"):
            filtered_args += ["--expert-cache", str(cfg["expert_cache"])]
        if "--resident-experts" not in filtered_args and cfg.get("resident_experts") is True:
            filtered_args.append("--resident-experts")
        if "--expert-cache-per-layer" not in filtered_args and cfg.get("expert_cache_per_layer") is True:
            filtered_args.append("--expert-cache-per-layer")
        if "--expert-cache-cpu-order" not in filtered_args and cfg.get("expert_cache_cpu_order") is True:
            filtered_args.append("--expert-cache-cpu-order")

        # PLE defaults if --ple-gguf is present
        if "--ple-gguf" in filtered_args and "--ple-io" not in filtered_args and cfg.get("ple_io"):
            filtered_args += ["--ple-io", str(cfg["ple_io"])]

        return [exe, "--serve", *filtered_args]

    elif model_arch == "qwen36_moe":
        exe = resolve_engine_executable("qwen36_moe", cfg.get("exe"))

        # Flags specific to Qwen 3.8 that strata-qwen36 does not consume
        q38_arg_flags = {
            "--ple-gguf", "--ple-io", "--ple-row-cache", "--ple-inflight",
            "--expert-profile", "--mtp-window", "--resident-budget-gib",
            "--vram-reserve-mib", "--kv-resident", "--expert-cache",
            "--adapt-every", "--adapt-swaps", "--pool-workers",
            "--adapt-decay", "--prompt-cache", "--prompt-cache-root",
            "--prompt-cache-every", "--pcie-frac", "--short-read",
            "--mtp-max-t", "--spec-min-p", "--suffix-draft"
        }
        q38_bool_flags = {
            "--resident-experts", "--expert-cache-per-layer",
            "--expert-cache-cpu-order", "--mmap-experts"
        }

        filtered_args = []
        i = 0
        spec_val = None
        has_kv = False
        has_kv_unified = False

        while i < len(raw_args):
            flag = raw_args[i]
            if flag == "--spec":
                if i + 1 < len(raw_args) and not raw_args[i + 1].startswith("-"):
                    spec_val = raw_args[i + 1]
                    i += 2
                else:
                    i += 1
                continue
            elif flag in ("--spec-k", "--mtp-k", "--draft-k"):
                if i + 1 < len(raw_args) and not raw_args[i + 1].startswith("-"):
                    spec_val = raw_args[i + 1]
                    i += 2
                else:
                    i += 1
                continue
            elif flag == "--kv":
                has_kv = True
                if i + 1 < len(raw_args) and not raw_args[i + 1].startswith("-"):
                    kv_val = raw_args[i + 1]
                    if kv_val.lower() in ("fp16", "f16"):
                        filtered_args += ["--kv", "fp16"]
                    else:
                        filtered_args += ["--kv", "int8"]
                    i += 2
                else:
                    filtered_args += ["--kv", "int8"]
                    i += 1
                continue
            elif flag == "--kv-unified":
                has_kv_unified = True
                filtered_args.append(flag)
                i += 1
                continue
            elif flag in q38_arg_flags:
                if i + 1 < len(raw_args) and not raw_args[i + 1].startswith("-"):
                    i += 2
                else:
                    i += 1
                continue
            elif flag in q38_bool_flags:
                i += 1
                continue
            elif flag == "--pack":
                if "--native" in raw_args:
                    if i + 1 < len(raw_args) and not raw_args[i + 1].startswith("-"):
                        i += 2
                    else:
                        i += 1
                    continue
                else:
                    filtered_args.append(flag)
                    i += 1
                    continue
            else:
                filtered_args.append(flag)
                i += 1

        # Automatically adapt default CLI flags for Qwen 3.6
        if not has_kv:
            kv_default = cfg.get("kv", "int8")
            if str(kv_default).lower() in ("fp16", "f16"):
                filtered_args += ["--kv", "fp16"]
            else:
                filtered_args += ["--kv", "int8"]

        if not has_kv_unified:
            filtered_args.append("--kv-unified")

        # Map spec parameter to --spec-k
        final_spec = spec_val or str(cfg.get("spec_k") or cfg.get("spec") or "")
        if final_spec:
            filtered_args += ["--spec-k", str(final_spec)]

        return [exe, "--serve", *filtered_args]

    else:
        exe = resolve_engine_executable(model_arch, cfg.get("exe"))
        return [exe, "--serve", *raw_args]


def validate_unified_config(cfg: dict[str, Any]) -> list[str]:
    """Validates that a configuration dictionary conforms to the unified engine schema.

    Returns a list of error strings. An empty list means the config is valid.
    """
    errors: list[str] = []
    if not isinstance(cfg, dict):
        return ["Configuration must be a JSON object (dict)."]

    exe = cfg.get("exe")
    if exe is not None and not isinstance(exe, str):
        errors.append("Invalid 'exe' parameter: must be a string path or executable name.")

    model_name = cfg.get("model_name")
    if not model_name or not isinstance(model_name, str):
        errors.append("Missing or invalid 'model_name' parameter.")

    args = cfg.get("args")
    if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
        errors.append("Missing or invalid 'args' parameter: must be a list of strings.")
        return errors

    arch = detect_config_architecture(cfg)
    if arch == "qwen38_moe":
        required_flags = ["--expert-profile", "--spec", "--mtp"]
        for rf in required_flags:
            if rf not in args and not cfg.get(rf.lstrip("-").replace("-", "_")):
                errors.append(f"Qwen 3.8 Flash Next configuration missing required flag: '{rf}'")
            elif rf in args:
                idx = args.index(rf)
                if idx + 1 >= len(args) or args[idx + 1].startswith("--"):
                    errors.append(f"Flag '{rf}' missing argument value.")

        if "--pack" not in args and "--native" not in args:
            errors.append("Configuration requires at least '--pack' or '--native'.")

        if "--spec" in args:
            idx = args.index("--spec")
            if idx + 1 < len(args):
                val = args[idx + 1]
                if not val.isdigit() or int(val) <= 0:
                    errors.append(f"--spec must be a positive integer, got '{val}'")

    elif arch == "qwen36_moe":
        if not any(k in args for k in ("--native", "--model", "-m")) and not cfg.get("native") and not cfg.get("model"):
            errors.append("Qwen 3.6 / Ornith configuration requires '--native' or '--model' specifying the model file.")

    if "env" in cfg and not isinstance(cfg["env"], dict):
        errors.append("'env' must be a dictionary of environment variables if specified.")

    return errors


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect GGUF metadata and architecture.")
    parser.add_argument("source", help="Path to GGUF file, model pack directory, or config JSON file.")
    parser.add_argument("--command", action="store_true", help="Print synthesized engine command.")
    pargs = parser.parse_args()

    spath = Path(pargs.source)
    if spath.suffix.lower() == ".json":
        cfg = json.loads(spath.read_text(encoding="utf-8-sig"))
        arch = detect_config_architecture(cfg)
        print(f"Architecture: {arch}")
        errors = validate_unified_config(cfg)
        if errors:
            print(f"Validation warnings/errors: {errors}")
        if pargs.command:
            cmd = build_engine_command(arch, cfg)
            print("Command:", " ".join(cmd))
    else:
        arch = detect_model_architecture(spath)
        print(f"Architecture: {arch}")
        if spath.is_file():
            try:
                meta = inspect_gguf_header(spath)
                print(f"Metadata entries ({len(meta)}):")
                for k, v in list(meta.items())[:20]:
                    print(f"  {k}: {v}")
            except Exception as e:
                print(f"Could not read GGUF metadata: {e}")

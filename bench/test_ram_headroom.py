#!/usr/bin/env python3
"""bench/test_ram_headroom.py

Measures available physical RAM on Windows and verifies headroom >= 4.0 GB.
If a Strata model process is active, computes current RSS footprint
and projects free memory under 60k+ context load.
Saves verification report to test_results/ram_headroom_results.json.
"""

import ctypes
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def get_memory_status_windows():
    """Query memory status using Win32 GlobalMemoryStatusEx."""
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        total_gib = stat.ullTotalPhys / (1024 ** 3)
        avail_gib = stat.ullAvailPhys / (1024 ** 3)
        return total_gib, avail_gib
    return None, None


def get_system_memory():
    """Get system total and available physical RAM in GiB."""
    # 1. Windows API
    if os.name == "nt":
        try:
            total_gib, avail_gib = get_memory_status_windows()
            if total_gib is not None:
                return total_gib, avail_gib
        except Exception as e:
            print(f"[Warning] Win32 GlobalMemoryStatusEx failed: {e}", file=sys.stderr)

    # 2. psutil fallback
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            return vm.total / (1024 ** 3), vm.available / (1024 ** 3)
        except Exception as e:
            print(f"[Warning] psutil query failed: {e}", file=sys.stderr)

    # 3. Unix sysconf fallback
    try:
        pagesize = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        return (pagesize * phys_pages) / (1024 ** 3), (pagesize * avail_pages) / (1024 ** 3)
    except Exception:
        pass

    raise RuntimeError("Unable to determine physical RAM on this system.")


def find_model_processes():
    """Detect any running strata engine or server processes."""
    found = []
    if psutil is None:
        return found

    for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            name = (p.info["name"] or "").lower()
            cmdline = " ".join(p.info["cmdline"] or []).lower()
            if "strata.exe" in name or ("python" in name and "serve.server" in cmdline):
                rss_gib = p.info["memory_info"].rss / (1024 ** 3)
                found.append({
                    "pid": p.info["pid"],
                    "name": p.info["name"],
                    "rss_gib": round(rss_gib, 3),
                    "cmdline": cmdline[:120],
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def project_memory_at_60k(avail_ram_gib, processes):
    """Projects free physical RAM at 60,000+ context length.

    KV cache growth at 60k tokens for Q4_0 KV is ~2.4 GiB,
    with ~0.6 GiB scratch / working buffer growth.
    Total estimated 60k context delta: ~3.0 GiB.
    """
    total_rss_gib = sum(p["rss_gib"] for p in processes) if processes else 0.0

    # If engine is resident, projected context expansion delta
    kv_expansion_at_60k_gib = 3.0 if processes else 0.0
    projected_free = max(0.0, avail_ram_gib - kv_expansion_at_60k_gib)
    return {
        "engine_rss_gib": round(total_rss_gib, 3) if processes else None,
        "kv_expansion_estimated_gib": kv_expansion_at_60k_gib,
        "projected_free_at_60k_gib": round(projected_free, 3),
    }


def main():
    print("=" * 60)
    print("Strata Physical RAM Headroom Verification")
    print("=" * 60)

    total_ram_gib, avail_ram_gib = get_system_memory()
    total_ram_gb = round(total_ram_gib, 2)
    avail_ram_gb = round(avail_ram_gib, 2)

    req_headroom_gib = float(os.environ.get("STRATA_RESIDENT_HEADROOM_GIB", "4.0"))

    print(f"Total Physical RAM:      {total_ram_gb:.2f} GiB")
    print(f"Available Physical RAM:  {avail_ram_gb:.2f} GiB")
    print(f"Headroom Requirement:   >= {req_headroom_gib:.2f} GiB")

    processes = find_model_processes()
    if processes:
        print("\nDetected active Strata process(es):")
        for p in processes:
            print(f"  PID {p['pid']} ({p['name']}): RSS {p['rss_gib']:.2f} GiB")
    else:
        print("\nNo running Strata engine processes currently active.")

    projection = project_memory_at_60k(avail_ram_gib, processes)
    print(f"Projected Free RAM at 60k Context: {projection['projected_free_at_60k_gib']:.2f} GiB")

    passed = avail_ram_gb >= req_headroom_gib
    if processes:
        passed = passed and (projection["projected_free_at_60k_gib"] >= req_headroom_gib)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_ram_gib": total_ram_gb,
        "available_ram_gib": avail_ram_gb,
        "required_headroom_gib": req_headroom_gib,
        "headroom_met": avail_ram_gb >= req_headroom_gib,
        "processes_detected": processes,
        "projection_at_60k": projection,
        "status": "PASS" if passed else "FAIL",
    }

    out_dir = Path("test_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ram_headroom_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to: {out_file}")

    assert avail_ram_gb >= req_headroom_gib, (
        f"Available physical RAM ({avail_ram_gb:.2f} GiB) is below required {req_headroom_gib:.2f} GiB threshold!"
    )
    if processes:
        assert projection["projected_free_at_60k_gib"] >= req_headroom_gib, (
            f"Projected free RAM at 60k context ({projection['projected_free_at_60k_gib']:.2f} GiB) "
            f"is below {req_headroom_gib:.2f} GiB threshold!"
        )

    print(f"\n[SUCCESS] RAM Headroom Verification PASSED (>= {req_headroom_gib:.2f} GiB available).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

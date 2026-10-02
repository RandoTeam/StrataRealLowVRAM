import os
import subprocess
import sys
import re

MSVC_BIN = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64"
DUMPBIN = os.path.join(MSVC_BIN, "dumpbin.exe")
LIB_TOOL = os.path.join(MSVC_BIN, "lib.exe")

def generate_lib(dll_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dll_name = os.path.basename(dll_path)
    base_name = dll_name.split(".")[0]
    # For cublas64_13.dll, base target lib should be cublas.lib
    target_base = "cublas" if "cublas64" in base_name else ("cublasLt" if "cublasLt" in base_name else base_name)
    def_path = os.path.join(out_dir, f"{target_base}.def")
    lib_path = os.path.join(out_dir, f"{target_base}.lib")

    print(f"Extracting exports from {dll_path}...")
    res = subprocess.run([DUMPBIN, "/EXPORTS", dll_path], capture_output=True, text=True, check=True)

    lines = res.stdout.splitlines()
    start = False
    symbols = []

    for line in lines:
        if re.search(r"ordinal\s+hint\s+RVA\s+name", line):
            start = True
            continue
        if start:
            if not line.strip():
                continue
            if "Summary" in line:
                break
            parts = line.split()
            if len(parts) >= 4:
                # Format: ordinal hint RVA name [= forwarder]
                sym = parts[3]
                symbols.append(sym)

    print(f"Found {len(symbols)} exported symbols.")
    with open(def_path, "w", encoding="utf-8") as f:
        f.write(f"LIBRARY {dll_name}\nEXPORTS\n")
        for sym in symbols:
            f.write(f"    {sym}\n")

    print(f"Generating {lib_path}...")
    cmd = [LIB_TOOL, f"/def:{def_path}", f"/out:{lib_path}", "/machine:x64"]
    subprocess.run(cmd, check=True)
    print(f"Successfully generated {lib_path}!")

if __name__ == "__main__":
    STRATA_DIR = r"C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata"
    OUT_DIR = os.path.join(STRATA_DIR, r".venv\Lib\site-packages\nvidia\cu13\lib\x64")
    generate_lib(os.path.join(STRATA_DIR, r"engine\cublas64_13.dll"), OUT_DIR)
    generate_lib(os.path.join(STRATA_DIR, r"engine\cublasLt64_13.dll"), OUT_DIR)

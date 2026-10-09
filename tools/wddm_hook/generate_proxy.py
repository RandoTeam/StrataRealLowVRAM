import os
import sys
import pefile

script_dir = os.path.dirname(os.path.abspath(__file__))

def generate_proxy(real_cublas_path):
    print(f"Parsing exports from: {real_cublas_path}")
    pe = pefile.PE(real_cublas_path)
    raw_exports = [exp.name.decode('ascii') for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols if exp.name]
    exports = sorted(list(set(raw_exports)))
    print(f"Found {len(exports)} unique exports.")

    # 1. Generate stubs.asm
    asm_lines = [
        ".code",
        "EXTERN g_real_cublas_funcs: QWORD",
        ""
    ]
    for idx, name in enumerate(exports):
        asm_lines.append(f"PUBLIC {name}")
        asm_lines.append(f"{name} PROC")
        asm_lines.append(f"    mov rax, QWORD PTR [g_real_cublas_funcs + {idx * 8}]")
        asm_lines.append(f"    jmp rax")
        asm_lines.append(f"{name} ENDP")
        asm_lines.append("")
    asm_lines.append("END")

    with open(os.path.join(script_dir, "stubs.asm"), "w", encoding="utf-8") as f:
        f.write("\n".join(asm_lines))

    # 2. Generate exports.def
    def_lines = ["EXPORTS"]
    for name in exports:
        def_lines.append(f"    {name}")

    with open(os.path.join(script_dir, "exports.def"), "w", encoding="utf-8") as f:
        f.write("\n".join(def_lines))

    print("Generated stubs.asm and exports.def successfully.")

if __name__ == "__main__":
    cublas_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not cublas_path or not os.path.exists(cublas_path):
        candidate = os.path.join(script_dir, "..", "..", ".venv", "Lib", "site-packages", "nvidia", "cu13", "bin", "x86_64", "cublas64_13.dll")
        cublas_path = os.path.abspath(candidate)
    
    if os.path.exists(cublas_path):
        generate_proxy(cublas_path)
    else:
        print(f"Error: could not find source cublas64_13.dll at {cublas_path}")
        sys.exit(1)

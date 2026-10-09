# Strata WDDM Hook (`cublas64_13.dll`)

## Problem Overview
On Windows 10/11 with NVIDIA GPUs under the WDDM display driver model, Windows clamps the virtual CUDA commit of any single 3D application to approximately 80–82% of total physical VRAM (~3,305 MiB on 4,096 MiB cards like the RTX 3050 Laptop).
Furthermore, WDDM rounds every discrete allocation $\ge 1\text{ MiB}$ up to a 2 MiB page granule. With Strata's dense layers (1416.8 MiB), 301 projection matrices (1376.2 MiB), and Q5_K output head (437 MiB), the nominal allocations total 3,230 MiB, but the WDDM virtual commit expands to **3,660 MiB**.

Because $3,660 > 3,305\text{ MiB}$, the standard CUDA runtime function `cudaMemGetInfo` reports `0 bytes free`. This causes Strata's engine (`src/program/generate.cpp:4344`) to calculate 0 expert cache slots and fail with:
```
strata: error: --spec needs the device residency table
```

## How the Hook Works
Instead of patching the core engine binary or forcing Linux WSL2, this directory implements a transparent **cuBLAS Proxy DLL** (`engine/cublas64_13.dll`):
1. Because `strata.exe` loads `cublas64_13.dll` directly from its application directory (`engine\`), Windows DLL search order prioritizes our proxy over system/CUDA directories.
2. The proxy dynamically intercepts all 746 cuBLAS API functions via high-performance assembly jump stubs (`stubs.asm`), forwarding them with zero latency to the real NVIDIA cuBLAS runtime.
3. Upon initialization (`DllMain`), the hook locates `cudaMemGetInfo` inside `strata.exe` at RVA `0x16ed90` (verified via signature check `48 89 5c 24`) and hot-patches the function prologue to redirect calls to `hooked_cudaMemGetInfo`.
4. `hooked_cudaMemGetInfo` queries NVIDIA NVML (`nvmlDeviceGetMemoryInfo`), bypassing the virtual WDDM process clamp and reading true physical unallocated hardware VRAM (500–700+ MiB).
5. This unlocks **144–265 expert cache slots** in VRAM while keeping the full-precision 437 MiB Q5_K output head intact!

## Building from Source
Prerequisites:
- Visual Studio Build Tools 2019/2022 (with MSVC v142/v143 and ML64 assembler)
- Python 3.10+ with `pefile` (`pip install pefile`)

Steps:
```cmd
cd tools\wddm_hook
build_hook.bat
```
The compiled `cublas64_13.dll` will be automatically placed in `engine\cublas64_13.dll`.

@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo Building strata-poly-131k with MSVC + CUDA 12.8 ...
echo ===================================================

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to initialize Visual Studio Build Tools environment.
    exit /b 1
)

set CMAKE_DIR=c:\VietnAi\4gbVramStrata\.venv\Scripts
set PATH=%CMAKE_DIR%;%PATH%

if not exist build-131k mkdir build-131k

echo [1/3] Configuring CMake for strata-poly ...
cmake -B build-131k -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DSTRATA_ENABLE_CUDA=ON ^
    -DCMAKE_CUDA_ARCHITECTURES=86 ^
    -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" ^
    -DSTRATA_MMQ_KQUANTS=ON ^
    -DSTRATA_Q6K_EXPERTS=ON

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CMake configuration failed.
    exit /b 1
)

echo [2/3] Building target strata-poly ...
ninja -C build-131k strata-poly

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Ninja build failed.
    exit /b 1
)

echo [3/3] Copying output to engine-poly\strata-poly-131k.exe ...
copy /y build-131k\strata-poly.exe engine-poly\strata-poly-131k.exe

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Binary successfully created: engine-poly\strata-poly-131k.exe
) else (
    echo [ERROR] Failed to copy binary.
    exit /b 1
)

@echo off
setlocal enabledelayedexpansion
title Strata Real Low VRAM Launcher (125B MoE on 4GB VRAM)

echo ===============================================================================
echo       Strata Real Low VRAM (v0.1.41) - 125B MoE on Budget 4GB GPUs
echo       Optimized for RTX 3050 Laptop / 4GB Desktop GPUs + 32,768 Context
echo ===============================================================================
echo.

rem 1. Check Architecture
if not "%PROCESSOR_ARCHITECTURE%"=="AMD64" (
    echo [ERROR] 64-bit Windows is required.
    pause
    exit /b 1
)

rem 2. Check NVIDIA GPU
where nvidia-smi >nul 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] nvidia-smi not found in PATH. Please verify NVIDIA Drivers are installed.
) else (
    echo [OK] NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
)
echo.

rem 3. Check Python
set "PY_CMD="
if exist ".venv\Scripts\python.exe" (
    set "PY_CMD=.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set "PY_CMD=python"
    ) else (
        where py >nul 2>nul
        if %errorlevel% equ 0 (
            set "PY_CMD=py -3"
        )
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Python 3.10+ is required but not installed or not in PATH.
    echo Please install Python 3.12 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

rem 4. Create/Verify Virtual Environment
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating virtual environment (.venv)...
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [*] Installing required Python wheels...
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install nvidia-cublas-cu13 nvidia-cuda-runtime-cu13 aiohttp requests safetensors numpy pefile
)
set "PY=.venv\Scripts\python.exe"

rem 5. Check Engine and WDDM Proxy Hook
if not exist "engine\strata.exe" (
    echo [ERROR] engine\strata.exe is missing!
    echo Please ensure the release archive was extracted completely.
    pause
    exit /b 1
)

if not exist "engine\cublas64_13.dll" (
    echo [*] WDDM proxy hook not found in engine\. Attempting to compile from source...
    if exist "tools\wddm_hook\build_hook.bat" (
        call tools\wddm_hook\build_hook.bat
    ) else (
        echo [ERROR] engine\cublas64_13.dll is missing!
        pause
        exit /b 1
    )
)

rem 6. Check Model & Packs
if not exist "packs\q2_0\layers.bin" (
    echo.
    echo ===============================================================================
    echo [NOTICE] Pre-packed model layers (packs\q2_0\layers.bin) not found!
    echo ===============================================================================
    if not exist "models\Q2_0" (
        echo Models folder "models\Q2_0" is missing.
        echo Would you like to download Qwen3.8-Flash-Next-GSQ-RCO-Q2_0 now?
        set /p DOWNLOAD_CHOICE="Download model shards (~34 GB)? [Y/n]: "
        if /i "!DOWNLOAD_CHOICE!"=="n" (
            echo Aborted by user. Please place model files in models\Q2_0\
            pause
            exit /b 1
        )
        mkdir "models\Q2_0" 2>nul
        echo [*] Launching Strata automated model downloader and pack builder...
        %PY% setup.py --model Q2_0 --context 32768 --yes
    ) else (
        echo [*] Raw models folder found. Creating optimized IQ data packs (packs\q2_0)...
        mkdir "packs\q2_0" 2>nul
        %PY% tools\iq_pack.py --model models\Q2_0 --out packs\q2_0
    )
)

rem 7. Verify Configuration
if not exist "strata-q2_0.json" (
    echo [ERROR] strata-q2_0.json configuration file not found!
    pause
    exit /b 1
)

echo.
echo ===============================================================================
echo Starting Strata Low-VRAM Server (OpenAI & Anthropic Compatible)
echo Web UI / API Base URL: http://127.0.0.1:8080
echo Model: Qwen3.8-Flash-Next (125B MoE, Q2_0)
echo Context Window: 32,768 tokens
echo Speculative Decoding: 2-step verification + 1 suffix draft
echo ===============================================================================
echo.

rem Launch browser in background after 3 seconds
start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8080"

rem Start Server
%PY% serve\server.py --config strata-q2_0.json --host 127.0.0.1 --port 8080
if errorlevel 1 (
    echo.
    echo [ERROR] Strata server stopped unexpectedly.
    echo Please review log output above or check strata-q2_0.log.
    pause
)

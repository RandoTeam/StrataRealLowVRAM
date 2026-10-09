@echo off
title Huihui-Qwen3.8-27B (IQ3_XXS / RTX 3050 4GB)
cd /d "%~dp0"

set "MODEL_PATH=C:\AI\Models\Huihui-Qwen3.8-27B-IQ3_XXS\Huihui-Qwen3.8-abliterated-GSQ-RCO-IQ3_XXS-mtp.gguf"
if not exist "%MODEL_PATH%" (
    echo [ERROR] Model file not found at %MODEL_PATH%!
    pause
    exit /b 1
)

set "PATH=%~dp0bin\llama;%~dp0.venv\Lib\site-packages\nvidia\cu13\bin\x86_64;%PATH%"

echo ===============================================================================
echo Starting Huihui-Qwen3.8-27B Server (OpenAI Compatible)
echo Model: Huihui-Qwen3.8-27B IQ3_XXS
echo Offload: 22 GPU layers (VRAM: ~3.3 GiB) + 42 CPU layers (4 threads)
echo Web UI / API Base URL: http://127.0.0.1:8080
echo ===============================================================================
echo.

rem Launch browser in background after 3 seconds
start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8080"

"bin\llama\llama-server.exe" ^
    -m "%MODEL_PATH%" ^
    -ngl 22 ^
    -t 4 ^
    -c 8192 ^
    -fa ^
    --port 8080 ^
    --host 127.0.0.1

if errorlevel 1 pause

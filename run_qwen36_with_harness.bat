@echo off
chcp 65001 > nul
title Qwen3.6-35B-A3B Uncensored + DeepSeek Harness
echo ===================================================================
echo  1. Запуск llama-server с Qwen3.6-35B-A3B (CUDA + 64K контекст)
echo  2. Запуск DeepSeek Harness Web UI в браузере
echo ===================================================================
echo.

echo [1/3] Очистка и компактификация оперативной памяти...
powershell -ExecutionPolicy Bypass -File tools\optimize_memory.ps1

set MODEL_PATH="C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata-data\models\qwen3.6-35b\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf"
if not exist %MODEL_PATH% (
    set MODEL_PATH="C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata-data\models\qwen3.6-35b\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q2_K_P.gguf"
)
if not exist %MODEL_PATH% (
    set MODEL_PATH="C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata-data\models\qwen3.6-35b\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ3_M.gguf"
)

if not exist %MODEL_PATH% (
    echo.
    echo [-] ОШИБКА: Файл модели не найден!
    echo     Сначала запустите download_qwen36.bat для скачивания модели.
    echo.
    pause
    exit /b 1
)

echo.
echo [2/3] Запуск сервера в фоне (http://127.0.0.1:8080/v1)...
start "Llama Server (Qwen3.6 35B)" cmd /c "cd /d "C:\Users\Ilia V\llama-server" && llama-server.exe -m %MODEL_PATH% -ngl 14 -t 6 -c 65536 --flash-attn on --context-shift --cache-reuse 256 --alias default,qwen3.6-35b-a3b-uncensored --host 127.0.0.1 --port 8080"

echo [3/3] Ожидание готовности сервера и запуск DeepSeek Harness...
timeout /t 5 /nobreak > nul

cd /d "C:\Users\Ilia V"
npx @deepseek-ai/dsh web

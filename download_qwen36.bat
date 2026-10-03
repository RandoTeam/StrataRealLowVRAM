@echo off
chcp 65001 > nul
title Загрузка Qwen3.6-35B-A3B Uncensored
echo ===================================================================
echo  Загрузка Qwen3.6-35B-A3B Uncensored (GGUF) с Hugging Face
echo ===================================================================
echo.
echo Выберите квантование для загрузки:
echo   1. IQ2_M  [10.86 ГБ] - Рекомендуется (макс. скорость 30-35 tok/s, 16+ ГБ своб. RAM)
echo   2. Q2_K_P [13.95 ГБ] - Повышенная точность (25-28 tok/s, 14+ ГБ своб. RAM)
echo   3. IQ3_M  [14.38 ГБ] - Высокая точность
echo.
set /p choice="Введите номер варианта (по умолчанию 1): "

set FILENAME=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
if "%choice%"=="2" set FILENAME=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q2_K_P.gguf
if "%choice%"=="3" set FILENAME=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ3_M.gguf

echo.
echo Запуск скачивания: %FILENAME%
python tools\download_qwen36.py %FILENAME%

echo.
pause

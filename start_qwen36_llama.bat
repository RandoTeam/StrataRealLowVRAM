@echo off
setlocal
set "LLAMA_EXE=C:\Users\Ilia V\llama-server\llama-server.exe"
if not exist "%LLAMA_EXE%" (
    where llama-server.exe >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set "LLAMA_EXE=llama-server.exe"
    ) else (
        echo [ERROR] llama-server.exe not found at "%LLAMA_EXE%" or in PATH.
        exit /b 1
    )
)

echo Starting Qwen 3.6 35B A3B Uncensored (HauhauCS Aggressive) on port 8081...
"%LLAMA_EXE%" -hf HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:IQ2_M -c 65536 -ngl 18 -fa 1 -ctk q4_0 -ctv q4_0 -t 6 --spec-default --port 8081 %*

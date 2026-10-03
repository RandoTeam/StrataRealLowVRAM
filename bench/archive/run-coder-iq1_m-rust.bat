@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ====================================================================
echo  StrataRealLowVRAM: Starting Engine with Native Rust Gateway (Port 8080)
echo ====================================================================

set "PATH=C:\Python312\Lib\site-packages\nvidia\cu13\bin\x86_64;%~dp0engine;%PATH%"

"%~dp0engine\strata-gateway.exe" --config "%~dp0strata-coder-iq1_m.json" --port 8080
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Gateway exited with code %ERRORLEVEL%
    pause
)

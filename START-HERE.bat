@echo off
rem Strata for Windows: unified high-performance model runner with 64K context
setlocal
title Strata LLM Runner
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto run

call :findpy
if defined PY goto venv
echo.
echo  Python 3.10 or newer is not installed. Installing Python 3.12 for your user account ...
where winget >nul 2>nul
if errorlevel 1 goto pyorg
winget install -e --id Python.Python.3.12 --scope user --silent --source winget --accept-package-agreements --accept-source-agreements --disable-interactivity
call :findpy
if defined PY goto venv
:pyorg
echo  Downloading the Python installer from python.org ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest -UseBasicParsing https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe -OutFile \"$env:TEMP\strata-python-setup.exe\""
if exist "%TEMP%\strata-python-setup.exe" "%TEMP%\strata-python-setup.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0
call :findpy
if defined PY goto venv
echo.
echo  Python could not be installed automatically.
echo  Install 64-bit Python 3.12 from https://www.python.org/downloads/ ("Add python.exe to PATH"),
echo  then double-click START-HERE.bat again.
pause
exit /b 1

:venv
rem a private environment inside this folder, so nothing is installed into the system Python
%PY% -m venv .venv
if exist ".venv\Scripts\python.exe" goto run
echo  Could not create the Python environment in .venv
pause
exit /b 1

:run
if "%~1"=="--setup" goto run_setup
if "%~1"=="--help" goto run_setup
if "%~1"=="--check" goto run_setup

set "CHOICE=%~1"
if "%CHOICE%"=="1" goto start_coder
if "%CHOICE%"=="2" goto start_q20

echo.
echo ====================================================================
echo   STRATA ENGINE RUNNER - 64K CONTEXT ^& 10-15+ TOK/S ACCELERATION
echo ====================================================================
echo   Select model to run:
echo.
echo     [1] Qwen 3.8 Flash Next Coder (IQ1_M) - 64K Context (Fast Coding)
echo     [2] Qwen 3.8 Flash Next Full (Q2_0)    - 64K Context (Full Reasoning)
echo     [S] Advanced Strata Setup / Re-download
echo.
set /p "CHOICE=  Enter choice [1-2, default: 2]: "
if "%CHOICE%"=="" set "CHOICE=2"
if /i "%CHOICE%"=="s" goto run_setup
if "%CHOICE%"=="1" goto start_coder
if "%CHOICE%"=="2" goto start_q20
goto start_q20

:start_coder
echo.
echo  Starting Qwen 3.8 Flash Next Coder (IQ1_M) with 64K Context...
powershell -ExecutionPolicy Bypass -File "tools\optimize_memory.ps1"
".venv\Scripts\python.exe" "serve\server.py" --engine strata --config "strata-coder-iq1_m.json" --port 8080 --open
if errorlevel 1 pause
exit /b

:start_q20
echo.
echo  Starting Qwen 3.8 Flash Next Full (Q2_0) with 64K Context...
powershell -ExecutionPolicy Bypass -File "tools\optimize_memory.ps1"
".venv\Scripts\python.exe" "serve\server.py" --engine strata --config "strata-q2_0.json" --port 8080 --open
if errorlevel 1 pause
exit /b

:run_setup
".venv\Scripts\python.exe" setup.py %*
if errorlevel 1 pause
exit /b

:findpy
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 set "PY=py -3" & goto :eof
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 set "PY=python" & goto :eof
for %%V in (313 312 311 310) do if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"" & goto :eof
goto :eof

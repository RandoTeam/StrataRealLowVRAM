@echo off
setlocal enabledelayedexpansion

echo ====================================================
echo Building Strata WDDM Hook (cublas64_13.dll)
echo ====================================================

rem Locate Visual Studio Build Tools / MSVC
set "VCVARS="
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)

if "%VCVARS%"=="" (
    echo [ERROR] Could not find vcvars64.bat. Please run from Visual Studio Developer Command Prompt.
    exit /b 1
)

call "%VCVARS%"
cd /d "%~dp0"

echo [1/3] Assembling ML64 stubs...
ml64 /c /nologo stubs.asm
if errorlevel 1 (
    echo [ERROR] ml64 assembly failed.
    exit /b 1
)

echo [2/3] Compiling and linking cublas64_13.dll...
cl /nologo /O2 /LD init.c stubs.obj /Fe:cublas64_13.dll /link /DEF:exports.def
if errorlevel 1 (
    echo [ERROR] cl / link failed.
    exit /b 1
)

echo [3/3] Copying hook binary to engine directory...
if not exist "..\..\engine" mkdir "..\..\engine"
copy /y cublas64_13.dll "..\..\engine\cublas64_13.dll"

echo ====================================================
echo Build successful! cublas64_13.dll deployed to engine\
echo ====================================================

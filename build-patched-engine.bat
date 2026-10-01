@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ==============================================================
echo Building Strata Engine with WDDM 4GB Expert Cache Patch...
echo ==============================================================

call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

set "CUDA_PATH=%~dp0.venv\Lib\site-packages\nvidia\cu13"
set "PATH=%CUDA_PATH%\bin;%PATH%"

echo [1/2] Configuring CMake...
cmake -G Ninja -S . -B build ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DSTRATA_ENABLE_CUDA=ON ^
  -DSTRATA_BUILD_TESTS=OFF ^
  -DCMAKE_CUDA_ARCHITECTURES=86 ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" ^
  -DSTRATA_PORTABLE=ON

if %ERRORLEVEL% NEQ 0 (
  echo CMake configuration failed!
  exit /b %ERRORLEVEL%
)

echo [2/2] Compiling strata.exe...
cmake --build build --target strata -j 6

if %ERRORLEVEL% NEQ 0 (
  echo Compilation failed!
  exit /b %ERRORLEVEL%
)

echo Engine build successful!
copy /y build\strata.exe engine\strata.exe
echo Deployed to engine\strata.exe

echo Updating engine\BUILD.json stamp...
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import setup, json, pathlib; src = setup.source_hash(setup.ENGINE_SOURCES); meta = {'version': setup.source_version(), 'source': 'local', 'archs': [86], 'ptx': True, 'cuda': '13.0', 'vision': 'gpu', 'src': src, 'patched': True}; pathlib.Path('engine/BUILD.json').write_text(json.dumps(meta, indent=1))"
)


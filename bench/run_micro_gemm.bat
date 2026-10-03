@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=%~dp0..\.venv\Lib\site-packages\nvidia\cu13"
set "PATH=%CUDA_PATH%\bin;%PATH%"

"%CUDA_PATH%\bin\nvcc.exe" -std=c++17 -O3 -arch=sm_86 -I include -I src -I third_party\ggml bench\micro_gemm_test.cu src\kernels\cuda\dequant_bf16.cu src\kernels\cuda\iq_kernels.cu -L "%CUDA_PATH%\lib\x64" -lcublas -lcudart -o bench\micro_gemm_test.exe
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
set "PATH=%CUDA_PATH%\bin;%~dp0..\engine;%PATH%"
cd /d "%~dp0.."
bench\micro_gemm_test.exe

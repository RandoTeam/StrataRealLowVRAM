@echo off
title Strata Q2_0 (RTX 3050 4GB / 32K Context / Low-RAM)
cd /d "%~dp0"

set "PATH=%~dp0.venv\Lib\site-packages\nvidia\cu13\bin\x86_64;%PATH%"
set STRATA_FILE_RELEASE=1
set STRATA_STAGER_SLEEP=0
set STRATA_Q2_BITPLANE=1
set STRATA_HOST_CORE=last
set STRATA_OWNED_PRICE=exact
set STRATA_KV_GROW=1
set STRATA_KV_GROW_INIT=256
set STRATA_PREFILL_CPU_SHARE=0
set STRATA_PREFILL_STREAM_MIN=64
set STRATA_PREFILL_RING=64
set STRATA_PF_FUSED=1
set STRATA_EMB_REUSE_ACCOUNT=1

".venv\Scripts\python.exe" "serve\server.py" --engine strata --config "strata-q2_0.json" --port 8080
if errorlevel 1 pause

@echo off

setlocal

if not defined JEV_TRAIN_ROOT set "JEV_TRAIN_ROOT=%~dp0.."

if not defined PYTHON_BIN (
    if exist "%JEV_TRAIN_ROOT%\.venv\Scripts\python.exe" (
        set "PYTHON_BIN=%JEV_TRAIN_ROOT%\.venv\Scripts\python.exe"
    ) else (
        set "PYTHON_BIN=python"
    )
)

if not defined GEMMA_CHECKPOINT set "GEMMA_CHECKPOINT=%JEV_TRAIN_ROOT%\train\runs\jev_harness_v4\best.pt"



powershell -NoProfile -Command "try { if ((Invoke-WebRequest -Uri 'http://127.0.0.1:8765/api/info' -TimeoutSec 1 -UseBasicParsing -ErrorAction Stop).StatusCode -eq 200) { exit 0 } } catch { exit 1 }"

if %errorlevel% equ 0 (

    echo [Gemma-JEV] Daemon is already online and serving on port 8765. Skipping duplicate spawn.

    exit /b 0

)



if exist "%JEV_TRAIN_ROOT%" (

    cd /d "%JEV_TRAIN_ROOT%"

    "%PYTHON_BIN%" train/serve_gemma.py --ckpt "%GEMMA_CHECKPOINT%" --port 8765 --device cuda --no-open

) else (

    echo [Gemma-Prep-Core] JEV_TRAIN_ROOT directory not found: %JEV_TRAIN_ROOT%

)


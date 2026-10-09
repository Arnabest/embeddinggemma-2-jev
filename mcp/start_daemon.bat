@echo off
setlocal
if not defined JEV_TRAIN_ROOT set "JEV_TRAIN_ROOT=%USERPROFILE%\.gemma\jev-train"
if not defined PYTHON_BIN set "PYTHON_BIN=python"

if exist "%JEV_TRAIN_ROOT%" (
    cd /d "%JEV_TRAIN_ROOT%"
    "%PYTHON_BIN%" train/serve_gemma.py --port 8765 --no-open
) else (
    echo [Gemma-Prep-Core] JEV_TRAIN_ROOT directory not found: %JEV_TRAIN_ROOT%
)

@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run Setup-Training.cmd first, or follow the CPU setup instructions.
  pause
  exit /b 1
)
if "%~1"=="" (
  ".venv\Scripts\python.exe" -m starfall.cli predict-image "tests\fixtures\gameplay.png"
) else (
  ".venv\Scripts\python.exe" -m starfall.cli predict-image "%~1"
)
if errorlevel 1 (
  pause
  exit /b 1
)
start "" "reports\trained-model-test.png"
pause

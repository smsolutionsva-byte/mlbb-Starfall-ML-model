@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run Setup-Training.cmd first, or follow the CPU setup instructions.
  pause
  exit /b 1
)
echo Connect the phone, authorize USB debugging, and open Starfall.
echo This preview shows detections and proposed paths. You play manually. Press q to stop.
".venv\Scripts\python.exe" -m starfall.cli live --config configs\phone-detector-1988.yaml --seconds 0
pause

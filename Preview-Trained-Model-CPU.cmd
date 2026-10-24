@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run Setup-Training.cmd first, or follow the CPU setup instructions.
  pause
  exit /b 1
)
if not exist "weights\starfall-detector-1988\best_openvino_model\best.xml" (
  echo Preparing the trained detector for CPU preview...
  ".venv\Scripts\python.exe" -c "import torch; from ultralytics import YOLO; torch.set_num_threads(2); YOLO('weights/starfall-detector-1988/best.pt').export(format='openvino', imgsz=416, batch=1, dynamic=False)"
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
echo Connect the phone, authorize USB debugging, and open Starfall.
echo You play manually while this shows detections and proposed paths. Press q to stop.
".venv\Scripts\python.exe" -m starfall.cli live --config configs\phone-detector-1988-cpu.yaml --seconds 0
pause

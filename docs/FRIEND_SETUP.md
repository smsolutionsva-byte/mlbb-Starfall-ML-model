# Run on the training laptop

The 6 GB **dedicated NVIDIA** GPU is used for neural detector training. Intel's 8 GB shared GPU memory is borrowed system RAM; it does not increase NVIDIA's dedicated memory to 14 GB. Setup reports the exact GPU name and tests a tensor operation. Windows Task Manager's GPU 1 is normally CUDA GPU 0 when there is one NVIDIA card, because CUDA does not enumerate the Intel adapter.

## Download and install

1. Download this repository using GitHub **Code → Download ZIP** and extract it, or clone it. Keep the whole folder together.
2. Install **64-bit Python 3.11** from [python.org](https://www.python.org/downloads/windows/) and the current NVIDIA graphics driver.
3. Double-click **Setup-Training.cmd**. It creates `.venv`, installs CUDA PyTorch from its official CUDA 12.8 wheel index, installs this project, downloads the verified scrcpy bundle, and checks GPU computation. Internet is needed for setup and initial pretrained weight downloads. Phone connection is optional for setup.
4. Read the final GPU report. GPU setup must succeed before requesting `--device 0`; if CUDA is unavailable, setup fails explicitly. The laptop's exact GPU, driver, temperature and power settings determine speed. Keep it plugged in during training.

The current PC's `requirements-lock.txt` records a CPU environment. **Do not use that CPU snapshot to set up the NVIDIA laptop**, and do not copy `.venv` between PCs. For a CPU installation, run `powershell -NoProfile -ExecutionPolicy Bypass -File tools/setup_training.ps1 -Mode cpu`. If the installed driver requires a different supported build, use `-Mode gpu -CudaBuild cu126` or `cu118` after checking the [official PyTorch installer](https://pytorch.org/get-started/locally/).

## Import recorded training data

The published three-class detector trained from 1,988 reviewed images is included in Git. Full datasets, recordings, and additional private movement/control models are not. A demonstration video is available separately in the [demo release](https://github.com/smsolutionsva-byte/mlbb-Starfall-ML-model/releases/tag/demo-2026-10-24).

For a private transfer of your own collected training data, run this on the collecting PC:

```powershell
.\.venv\Scripts\python.exe tools\export_training_data.py
```

Transfer the generated `exports/starfall-training-data-*.zip` privately and extract its contents **into the repository folder** on the training laptop. It contains captured frames, labels, reviewed player splits, movement logs, and current detector/response candidates. Dataset paths in the archive are portable. Its manifest lists each file and SHA256; the archive checksum sits next to the ZIP. This command does not publish or upload anything. Keep the archive private: it may contain unpublished movement/control artifacts, and it must not be attached to a public issue or release. Stop phone control while making a fresh export if it is actively recording.

The earlier player-only prototype dataset had **31 selected positive frames: 7 train / 12 validation / 12 test**, grouped by actual attempt. This historical subset is separate from the included 1,988-image detector; see its [model card](../weights/starfall-detector-1988/README.md). It is an experimental subset. Raw game frames have many unfinished labels; they are not ready for full meteor/orb training. Importing existing models does not automatically enable them in the live controller.

## GPU detector training

Reproduce the small player candidate with automatic NVIDIA selection:

```powershell
.\.venv\Scripts\python.exe tools\train_player_candidate.py --device 0
```

For a larger manually reviewed dataset, use its split YAML:

```powershell
.\.venv\Scripts\python.exe -m starfall.cli train-detector data\game\YOUR_SPLIT\dataset.yaml --device 0 --imgsz 416 --batch 4 --epochs 40
.\.venv\Scripts\python.exe -m starfall.cli evaluate-detector models\detector-experiments\YOUR_RUN\weights\best.pt data\game\YOUR_SPLIT\dataset.yaml --device 0 --split test
```

`--device auto` prefers CUDA and otherwise uses CPU; `--device 0` requires a working CUDA GPU. Start with batch 4 for 6 GB VRAM. If memory runs out, lower the batch and then image size. Evaluation must use a separate held-out attempt. Do not deploy a candidate merely because training completed. The current player candidate has material false positives and remains disabled for live control.

## Multiple offline arenas

Double-click **Train-Offline.cmd** to train PPO in **four separate synthetic arenas** sharing one model. It uses CPU because this policy is a small network over numeric features. It needs no phone or tickets. Alternatively:

```powershell
.\.venv\Scripts\python.exe -m starfall.cli train-policy --algorithm ppo --steps 200000 --n-envs 4 --vec-backend subproc --device cpu
```

`--vec-backend dummy` runs several arenas in one process; `subproc` runs separate processes with Windows-safe spawn. For a cheap arena, process overhead can make `dummy` faster. Compare wall times recorded in each experiment before raising the arena count. `--threads 2` limits training CPU threads; the best arena count depends on the laptop's CPU. Checkpoints are saved under `models/` and requested/actual steps are recorded.

**This is an engineering sandbox, not an offline copy of MLBB.** Full shooting/damage, boss phases, orbs and ultimate mechanics are not yet represented faithfully. Synthetic rewards are not real game points. More training in this sandbox does not establish stronger real gameplay. Candidates still require paired held-out simulation checks and actual phone evaluation.

## Observe a recording or phone

Drag an original gameplay MP4 onto **Preview-Recorded-Video.cmd**. It prepares an OpenVINO export locally if needed, shows square detection boxes and suggested paths, and saves an annotated video with its original audio under `reports/`. A phone is not needed for recorded playback. This is analysis, not training or game control.

For manual live observation, connect the calibrated Android phone by USB, enable USB debugging, authorize the laptop, and open Starfall. Use **Preview-Trained-Model.cmd** on the NVIDIA laptop or **Preview-Trained-Model-CPU.cmd** on a CPU-only PC. You play manually; **q** closes the preview. These observation launchers do not send joystick, Challenge, or ultimate taps.

Current phone coordinates/templates were calibrated at **960 × 430**. Another device or UI size needs recalibration for reliable game-page/combat readings. Recorded playback scales the analysis to that reference while retaining the recording's original output resolution and timing.

The repository retains experimental baseline controller utilities. The maintainer's additional movement/control work and another model remain private, as explained in the [public-scope note](../README.md#what-is-public-and-what-stays-private). This guide focuses on recording analysis, detector training, and synthetic experiments.

## Verify installation

```powershell
.\.venv\Scripts\python.exe tools\check_training.py --require-cuda
.\.venv\Scripts\python.exe -m starfall.cli doctor
.\.venv\Scripts\python.exe -m pytest -q
```

Tests include small regression image fixtures, not the full recordings. UI-only fixtures omit account/friend information and are marked as modified test examples. They must never enter a training dataset.

References: [Ultralytics training](https://docs.ultralytics.com/modes/train/), [PyTorch installation](https://pytorch.org/get-started/locally/), [PPO CPU/vector environment guidance](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [Windows vector environment precautions](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html).

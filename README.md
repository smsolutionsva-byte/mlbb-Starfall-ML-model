# Starfall Breakout: Computer Vision and ML Research

A Starfall Breakout research project with a trained object detector, tracking and planning experiments, offline training tools, and recorded-gameplay previews.

## Recorded result: 282,793 points

The maintainer reports a **282,793-point result**. The screenshot below shows **rank #1 in the displayed weekly ranking list**, and the end of the accompanying recording shows the same highest-points total. This describes the list shown in the screenshot; it is not a claim of global rank #1.

![Weekly ranking showing first place and 282,793 points; both username labels redacted](docs/media/weekly-rank-282793-redacted.png)

[![Gameplay with the published detector's square-box overlay](docs/media/gameplay-preview.jpg)](https://github.com/smsolutionsva-byte/mlbb-Starfall-ML-model/releases/download/demo-2026-10-24/starfall-282793-preview.mp4)

**[Watch / download the full 61-second preview](https://github.com/smsolutionsva-byte/mlbb-Starfall-ML-model/releases/download/demo-2026-10-24/starfall-282793-preview.mp4)** · [Demo release and checksums](https://github.com/smsolutionsva-byte/mlbb-Starfall-ML-model/releases/tag/demo-2026-10-24) · [Evidence and limitations](docs/RESULTS.md)

The video is a phone recording with detection boxes and suggested paths added afterward. It documents the recorded gameplay and the public perception pipeline; the overlay itself does not establish who controlled each action or independently reproduce the score. The movement/control artifacts described below are separate from the published detector.

## What is public and what stays private

| Component | Availability |
| --- | --- |
| YOLO26n detector trained from 1,988 reviewed images | Included: player, meteor, and pickup classes |
| Image testing, manual live observation, and recorded-video preview | Included, with CPU/OpenVINO support |
| Experimental baseline tracking, planning, movement-response code, and synthetic training utilities | Already present in this repository; experimental research components |
| Additional movement/control work and another model described by the maintainer | Private; their additional trained artifacts are not included in this release |

**Maintainer's note:** I also have movement/control work and another model that I am keeping private. I am not publishing those additional artifacts because I want to respect MLBB's guidelines and avoid turning this project into a ready-made tool for automated farming. Use the public detector and research code as a base for your own ML experiments: improve the dataset, train further, evaluate on fresh recordings, or study tracking and decision-making in an offline environment.

MLBB's [Terms of Service](https://www.mobilelegends.com/news/articleldetail?newsid=2728920) restrict automated interaction and bots. The intended public use here is recording analysis and offline ML research. This project is not affiliated with or endorsed by Moonton.

## Preview the trained detector

1. Follow [the setup guide](docs/FRIEND_SETUP.md). The published checkpoint is already in `weights/starfall-detector-1988/`; additional private models are not required for observation.
2. Double-click **Test-Trained-Model.cmd** for an annotated screenshot, or drag your own screenshot onto it.
3. Drag an original phone recording onto **Preview-Recorded-Video.cmd**. It displays the analysis and saves an annotated MP4 under `reports/`, preserving the source timing, resolution, and audio. Press **q** to finish early.
4. For manual live observation, connect the calibrated phone, authorize USB debugging, and use **Preview-Trained-Model.cmd**. On a PC without NVIDIA graphics, use **Preview-Trained-Model-CPU.cmd**. You control the game during these previews.

The large square boxes use **green for the player, red for meteors, cyan for pickups**, and a **thick yellow suggested path**. Boxes have display padding and are not measured collision hitboxes. Recorded previews perform no training and send no phone input.

For command-line playback:

```powershell
.\.venv\Scripts\python.exe -m starfall.cli preview-video "your-recording.mp4"
```

The release video is the finished annotated output, rather than a raw training recording. For another experiment, use your own original recording. Full recordings and additional checkpoints stay outside Git; the demonstration video is hosted as a release asset.

## Train and evaluate as an ML project

The included detector recognizes **player, meteor, and pickup**. It is a perception model, not a complete movement or boss-strategy policy. See [its model card](weights/starfall-detector-1988/README.md) for the split, training settings, checksums, reported metrics, and evaluation limitations.

Useful research directions include fresh hand-labelled gameplay, capture-group-separated train/test splits, hard examples with overlapping meteors and ultimate effects, temporal tracking, and offline policy experiments. Keep recorded detections separate from human-reviewed ground truth.

The synthetic arena supports PPO/DQN and multiple parallel environments. It is an engineering sandbox with approximate physics and rewards, not a faithful offline copy of MLBB. Synthetic scores are not game points, and an offline training run does not establish improved live gameplay.

- [Training-laptop setup and offline experiments](docs/FRIEND_SETUP.md)
- [Research rationale and references](docs/RESEARCH.md)
- [Recorded results and demo provenance](docs/RESULTS.md)
- [Earlier measurements and experiment history](reports/EVIDENCE.md)

## Verification

The preview update passed **54 software regression tests**. For the published demonstration, all **3,066 video frames** retained their original presentation times, and all **2,654 audio packets** were copied without changing their encoded bytes. These are media-integrity checks, not a detector-accuracy benchmark or a measurement of live reaction time.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

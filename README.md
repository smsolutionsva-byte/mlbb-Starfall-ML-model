# Starfall Breakout local visual agent

This project observes MLBB on a real Android phone through scrcpy and runs perception and decisions locally on the PC. The first milestone is a measured, reviewable prototype. A 100,000-point autonomous run has not been demonstrated.

**Use the published detector:** double-click `Test-Trained-Model.cmd` for an annotated screenshot, or `Preview-Trained-Model.cmd` for live phone observation. See [the model guide](weights/starfall-detector-1988/README.md). The detector recognizes objects; the planner supplies movement/combat decisions. Actual gameplay validation of this checkpoint remains pending.

**New PC / NVIDIA training laptop:** follow [the transfer and setup guide](docs/FRIEND_SETUP.md). `Setup-Training.cmd` installs CUDA training support; `Train-Offline.cmd` runs four synthetic arenas. Code and essential templates are in Git; recordings and checkpoints transfer separately using `tools/export_training_data.py`. The dedicated NVIDIA memory is separate from Intel shared memory.

## Human-assisted attempts and learning

**Heart scanning is disabled, as requested. You handle low health and Return to Lobby.** The agent recognizes the Starfall ready page, taps Challenge once, plays using repeated joystick drags, and records each attempt. After you return to MLBB's lobby and reopen Starfall, it starts again. It does not press Return to Lobby. A title/button check and gameplay HUD check distinguish the ready page from gameplay during transitions.

Double-click **Start-Learning.cmd** to run. Press `q` in the preview or double-click **Stop-Learning.cmd** to stop and release input. Only one controller may run at a time. The loop waits for USB reconnection after a disconnect. A failed start is not repeatedly tapped on the same ready screen. Leave health resets and reopening the Starfall page to the human.

At the end of each attempt, the program fits/evaluates an action-conditioned movement-response model from real, timestamped gestures and player observations. A whole latest attempt is held out. Only actions with enough independent data, plausible response, and a measured reduction in held-out error can update the planner. Reports and candidate versions are saved under `models/real-response/`. This learns movement for predictive control; score-based policy learning still needs reliable reward observations. Unknown scores and health stay null. Human resets are marked as truncations, not deaths.

## Shooting, pickups and the boss

The live planner now seeks safe lanes for rightward automatic shots, prioritizes a visible Magma Wall core, and pursues reachable power-up orbs. Collision predictions remain a hard constraint. Bounded proximity costs let safe shooting/pickup routes compete with simply hiding, and earlier pickups are rewarded so replanning cannot postpone a collection indefinitely. The wall is represented as a moving rectangle rather than a circular meteor.

A separate combat reader verifies colored orb candidates against reviewed game icons, finds the wall/core, and reads the ultimate charge counter. The orange button itself glows even at `0/2`; zero or unreadable charges never trigger a tap. Positive charges can be used for an aligned boss core, a meteor cluster or an imminent threat. A dedicated touch pointer taps the bottom-right button after releasing the joystick; cooldown and observed charge changes limit repeated taps. Cast frames are excluded from movement-response targets. No heart reader is involved.

These changes have recorded-frame checks and controller regression tests. The first updated live trial is waiting for USB reconnection. Projectile speed, muzzle offset, wall extent and physical hitboxes remain approximate. Full orb appearance coverage, `2/2` recognition, actual boss damage, successful collection and a score improvement still require real gameplay validation. The existing synthetic sandbox does not yet reproduce this complete combat system and should not be treated as a faithful offline Starfall copy.

## Setup and entry points

Use PowerShell in this project folder. On a new Windows PC, run `Setup-Training.cmd` first. The scrcpy bundle and `.venv` are downloaded locally, not stored in Git.

```powershell
.\.venv\Scripts\python.exe -m starfall.cli doctor
.\.venv\Scripts\python.exe -m starfall.cli capture-benchmark --seconds 15
.\.venv\Scripts\python.exe -m starfall.cli learning-loop
```

The loop runs until stopped. `--seconds 180` gives a bounded session; `--episodes 3` stops after three manually ended attempts; `--no-preview` hides the preview. Rotation/resolution changes suspend input in the restart loop, which waits for the calibrated game view to return. Single-run control still stops on geometry changes. Repeated starts were tested on the actual phone. The updated detector and automatic response training were exercised after USB reconnection. Transition/grouping, combat, and orientation recovery have software regression tests.

```powershell
.\.venv\Scripts\python.exe -m starfall.cli live --control --collect --seconds 60
.\.venv\Scripts\python.exe -m starfall.cli train-real-response
```

Joystick geometry and drag response were measured on this phone. Initial holds produced movement followed by a plateau, so the active controller uses bounded repeated strokes. Individual action velocities and physical stroke limits are configured from real measurements. Two unmeasured diagonal actions remain disabled. The current allowed region keeps the player above the joystick overlay so it remains visible. This prototype detector has limited appearance coverage; it is not a validated detector for all game phases. The single-run command remains bounded to a 20-second pilot; the restart loop continues through human-managed attempts.

## Dataset and detector

The [three-class detector trained from 1,988 reviewed images](weights/starfall-detector-1988/README.md) is included in Git, with its checkpoint, portable metadata, checksum and evaluation comparison. It detects player, meteor and pickup. This is a detector candidate; the trained weights do not establish autonomous gameplay improvement.

Real frames are under `data/game/images`, their capture provenance is in `manifest.jsonl`, and manually reviewed YOLO boxes go under `data/game/labels`. Observed appearance labels currently include player, orange/blue/purple meteor, and Starshot Split; these are a starting set, not an exhaustive inventory of mechanics.

```powershell
.\.venv\Scripts\python.exe -m starfall.cli annotate --classes player,orange_meteor,purple_meteor,blue_meteor,starshot_split
.\.venv\Scripts\python.exe -m starfall.cli validate-data --classes player,orange_meteor,purple_meteor,blue_meteor,starshot_split
.\.venv\Scripts\python.exe -m starfall.cli split-data --classes player,orange_meteor,purple_meteor,blue_meteor,starshot_split
.\.venv\Scripts\python.exe -m starfall.cli benchmark-detectors reports\challenge-after12s.png --openvino
.\.venv\Scripts\python.exe -m starfall.cli train-detector data\game\SPLIT_DIRECTORY\dataset.yaml
.\.venv\Scripts\python.exe -m starfall.cli evaluate-detector models\YOUR_CHECKPOINT.pt data\game\SPLIT_DIRECTORY\dataset.yaml
```

Annotation keys: `b` adds a box and asks for its class; `c` clears boxes; `s` saves a manually reviewed image (including an explicit negative); `n` skips; `q` quits. Existing labels remain editable. Unreviewed images are rejected by validation. Session groups and near-duplicates stay in the same split; three independent groups are required. Generated test images never enter this dataset.

Template and YOLO detectors share an interface. `type: openvino` selects the bounded-thread native backend for an exported custom detection model. General COCO weights have no validated game-class mapping. Fine-tuning and AP/precision/recall cannot be claimed before the real dataset has sufficient labels and independent evaluation.

## Explicit learning and evaluation

The simulator is an engineering sandbox with synthetic physics, rewards and spawn distributions. Its scores are not MLBB points. The agent uses 132 numerical features, explicit unknown-value masks, and nine directions. The live planner compares 81 two-stage paths and uses continuous swept-circle and wall-rectangle collisions, uncertainty margins, boundary costs, configurable pickup value, and alignment with rightward attacks and the boss core. Existing learned feature policies do not yet encode the complete combat state.

```powershell
.\.venv\Scripts\python.exe -m starfall.cli train-policy --algorithm ppo --steps 20000
.\.venv\Scripts\python.exe -m starfall.cli train-policy --algorithm ppo --steps 200000 --n-envs 4 --vec-backend subproc --device cpu
.\.venv\Scripts\python.exe -m starfall.cli train-policy --algorithm dqn --steps 20000
.\.venv\Scripts\python.exe -m starfall.cli evaluate --policy mpc --episodes 30 --output reports\mpc.json
.\.venv\Scripts\python.exe -m starfall.cli evaluate --policy learned --model models\EXPERIMENT\candidate.zip --episodes 30 --output reports\candidate.json
.\.venv\Scripts\python.exe -m starfall.cli compare reports\candidate.json reports\mpc.json
```

Runs save configuration, action timestamps, tracks, uncertainty, latency, and explicit unknown game metrics. Simulation experiments save seeds and versioned checkpoints. Candidate policies are compared on paired held-out seeds with a bootstrap interval and collision gate; passing simulation does not automatically deploy them to MLBB. A newly trained policy must also pass real-game comparisons. `summarize-runs` summarizes scored runs without fabricating missing scores.

## Evidence and research

See [the research report](docs/RESEARCH.md) for official references and comparisons of predictive control, PPO, DQN, imitation learning, ByteTrack, TD-MPC2 and Dreamer. Local measurements and screenshots are in `reports/`. The actual phone capture reached approximately 58 fps at 960 × 430, with 6.9 ms median decode time; receive-to-dispatch measurements exclude phone capture, USB transfer and device-visible response.

PPO and DQN were both trained and evaluated in the synthetic sandbox. Both short-budget candidates underperformed predictive control and were not deployed. A small player-only YOLO26n was also fine-tuned on real reviewed frames, exported to OpenVINO, and tested on a separate attempt. It remains an offline/shadow candidate because precision is too low; `configs/phone-player-candidate.yaml` rejects joystick deployment. More attempts alone do not establish improvement. See [recorded evidence](reports/EVIDENCE.md) for the results and limits; no 100,000-point autonomous result is claimed.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

For a fresh installation, use [the setup guide](docs/FRIEND_SETUP.md). Detector training accepts `--device auto` (NVIDIA when available), `--device cpu`, or `--device 0`, along with `--batch` and `--imgsz`. Numerical policy training defaults to CPU and supports multiple arenas. The bootstrap verifies the pinned Windows scrcpy bundle SHA256. Enable USB debugging, accept authorization, and use `--serial` when multiple phones are attached. `requirements-lock.txt` is a historical CPU environment snapshot, not the GPU installer.

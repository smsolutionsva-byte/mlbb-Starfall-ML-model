# Evidence recorded on this PC

## Published gameplay demonstration — 24 October 2026

The maintainer supplied a 282,793-point ranking screenshot and phone recording. The [current results page](../docs/RESULTS.md) includes the redacted screenshot and the annotated video release. The recording's overlays were computed afterward, with no phone input. They demonstrate perception on recorded gameplay; the media alone does not verify autonomous control. Additional movement/control artifacts are described by the maintainer as private. The measurements below are historical results from the particular experiments listed, rather than evaluations of those private artifacts.

The full research rationale and primary sources are in `docs/RESEARCH.md`. Measurements below are project-specific and have limited sample coverage.

## Hardware and capture

Windows 11, Intel i5-10210U, approximately 16 GB RAM, Intel UHD graphics, Python 3.11.9. No CUDA. OpenVINO enumerates CPU and GPU. scrcpy 5.0.1 is pinned and its official portable archive SHA256 was verified.

At 960 × 430, a 15-second real-phone capture measurement decoded 878 frames, approximately 58.45 fps, without consumer-skipped frames. Median decode was 6.90 ms, p95 11.68 ms, p99 19.67 ms. Receive-to-consume median was 7.20 ms. These figures exclude phone capture/encoding, USB transport and action response. See `capture-benchmark.json`.

## Detector timing

Both models use general COCO weights, so these results establish timing only. They do not establish detection accuracy on Starfall. Ten measured calls per case followed three warmups, on one real screen frame. CPU thread counts were bounded; OpenVINO includes resizing, inference and detection-head decoding/NMS. Some initial measurements showed substantial scheduling variation; larger multi-session benchmarks are required for deployment selection.

| Model / backend | Input | Median | p95 |
|---|---:|---:|---:|
| YOLO26n / PyTorch CPU | 320 | 63.0 ms | 68.4 ms |
| YOLO26n / native OpenVINO CPU | 320 | 44.6 ms | 59.4 ms |
| YOLO26n / native OpenVINO CPU | 416 | 78.2 ms | 97.8 ms |
| YOLO11n / native OpenVINO CPU | 416 | 70.7 ms | 86.3 ms |

The wrapper's initial OpenVINO execution was interrupted after excessive resource use. A native backend with two CPU threads and one stream compiled/inferred successfully, then completed the comparison. No integrated-GPU performance has been established. See `detector-benchmark.json`.

## Explicit policy learning

PPO and DQN each trained for 8,192 requested steps on a synthetic sandbox using a compact MLP. Checkpoints and training logs exist under `models/`. Evaluation used 30 paired held-out seeds (starting at 10,000), with 20-second episode limits.

| Synthetic policy | Mean synthetic score | Mean survival | Runs with collision | Mean collections |
|---|---:|---:|---:|---:|
| Stay | 55.4 | 5.54 s | 100% | 0 |
| Nearest-hazard avoidance | 111.4 | 11.14 s | 86.7% | 0 |
| Random | 63.4 | 6.34 s | 93.3% | 0 |
| Predictive controller | 270.0 | 20.00 s | 0% | 2.80 |
| PPO, short training budget | 73.4 | 7.01 s | 96.7% | 0.13 |
| DQN, short training budget | 63.8 | 5.96 s | 96.7% | 0.17 |

Both learned candidates failed the paired comparison against predictive control and were not deployed to the phone. This does not rule out PPO/DQN after sufficient training and tuning. Synthetic physics and rewards are not validated MLBB mechanics. These scores are never represented as game points. See `offline-experiments.json` and per-policy reports.

## Control and human-managed restarts

Persistent injected holds produced an initial movement then a plateau. The constant-speed calibration correctly failed. The user confirmed that normal movement also requires finger dragging. Repeated bounded drag strokes are implemented; their visual calibration is separate from a continuous-speed fit.

The user superseded the one-heart exit requirement: **heart scanning is disabled, and the human handles returning to the lobby.** The active restart loop has no health reader and never taps Return to Lobby. The earlier optional guard and its regression tests remain historical components. Screenshots before/after earlier early returns showed 8 tickets; that supports preservation in those checked exits, not an unconditional infinite-attempt guarantee.

A three-minute real session automatically tapped Challenge three times after human resets, without heart scanning or any automatic lobby exit. It processed 6,487 frames, including lobby frames, at 36.00 frames/s overall, with 36.49 ms median and 62.02 ms p95 receive-to-dispatch latency. The latter excludes phone capture/transport/action response. The event log initially counted two transient page transitions as extra episodes; requiring the gameplay Current Points label now rejects the recorded transition frame. Three start taps correspond to three actual attempts, not five completed games. See `runs/20261012T011902Z-12c37b4f/`.

Player detection was intermittent: only 117/2,391 processed gameplay frames in the first attempt had a valid tracked player. Review identified a search-region bug clipping the sprite at arena boundaries. Padding the region recovered a player detection on 118/123 saved frames from that attempt; an additional head template recovered 119/123. These are detection-presence counts, not precision/recall or independently labelled accuracy. Lower-screen overlap remains a limitation; the active region now keeps the player above the joystick overlay.

The first three-minute run yielded three isolated response samples, which were insufficient for any model update. After USB reconnection, two further actual attempts raised the dataset to 80 isolated real movement samples across three data-bearing attempts. In the first resumed attempt, 392/453 recorded gameplay frames had a valid tracked player, and 150 gestures were sent. These are tracker-presence and action counts, not labelled detection accuracy or score gains.

The learning loop automatically fitted left/right response models between attempts and evaluated them on the latest attempt. Leftward prediction error changed from .00804 to .00763 normalized units, but its bootstrap improvement interval included zero. Rightward error worsened from .00123 to .00199. Other actions lacked sufficient train/test coverage. No learned movement estimate was promoted in that run. This establishes actual recording, training and held-out evaluation, without claiming an improved score. See `runs/20261012T014752Z-8ee0a2ae/`.

The subsequent continuous run `runs/20261012T015357Z-3bcef2c5/` confirmed three actual Challenge starts with verified physical attempt IDs. After its first human reset, the dataset contained 144 isolated samples. A down-left response candidate passed the configured prediction-error gate on 9 samples from the held-out attempt (14 training samples, 12 retained): mean normalized error .02527 to .01001. The bootstrap difference interval barely excluded zero, and the candidate was almost vertical, so this remains limited movement-prediction evidence rather than a gameplay improvement. After the next reset, 152 samples were available, but the newest attempt had only two samples for that action; the fresh report promoted no actions and the controller reverted to calibrated responses. Reports are versioned, including `models/real-response/20261012T015509452986Z.json`; `latest.json` continues changing as new attempts finish. No game-score gain has been established.

Pause/unknown pages now suspend control while preserving the same attempt until a ready/lobby page is confirmed. Physical attempt IDs survive capture reconnects within a controller process; startup halfway through a game is excluded from independent response training until an observed start. The latest running loop has heart scans disabled. Restart/pause handling, independent attempt grouping, no-heart control, response-target isolation, model improvement gates, input wire format, controller exclusivity, continuous collision checks and dataset split handling have 41 passing tests.


## Dataset and remaining limits

### Combat correction from user feedback

The user confirmed that the bottom-right ultimate fires on one tap, meteors must be destroyed for points, the map-height boss must be hit at its core, and power-up orbs should be collected. The earlier live configuration weighted shooting at only 0.1 and recognized only Starshot Split. It also used a minimum-distance pickup reward that could favor waiting now and collecting later on every replanning step.

The updated planner uses shooting weight 2.0, boss-core weight 8.0, explicit default orb value 2.5 / charge-orb value 3.0, bounded soft proximity penalties, and time-sensitive pickup rewards. Predicted collisions remain a hard rejection. Boss collision uses swept segments against a moving expanded rectangle, including crossings between sampled endpoints. Automatic rightward shots use a provisional muzzle offset and projectile lead; those numbers are not a measured damage model.

Twenty guide icons and three actual orb crops provide recognition references. Color/shape proposals are verified against these icons; joystick and ultimate UI areas are excluded from collectible detection. Three actual wall/core phases provide core references. In the 24-frame screening grid, five wall frames had a detected core, and several actual power and charge orbs were found. A top-edge orb was missed. These cases partly overlap reference sources and do not constitute independently measured precision/recall. The bounded two-thread combat reader measured median 34.78 ms / p95 50.55 ms for those 24 calls; timing varies with the scene and excludes the existing detector and input dispatch. See `combat-screening.json` and `combat-recordings-grid.jpg`.

The ultimate reader checks the glyph and tiny counter independently of the orange glow. Across 728 previously recorded frames it returned 564 zero-charge, 121 one-charge and 43 unknown readings; these are output counts without independently labelled counter accuracy. No two-charge frame was present. Policy/controller tests verify no tap at zero/unknown, alignment with a visible boss core, a single dedicated-pointer tap, cooldown, suspension on rotated frames, and no health scanning. The updated suite has **48 passing tests**. USB was disconnected when the first new live trial was launched, so actual post-change kills, boss damage, ultimate consumption, collections and score gains remain unverified. The continuous controller is waiting for reconnection.

Real frames were captured from several observation/calibration sessions. A 31-frame player-only subset now has approximate boxes visually reviewed by Codex from template proposals. One whole attempt forms each split: 7 train / 12 val / 12 test. No cross-attempt near duplicate met the grouping threshold. This is a small, selection-biased experiment, not an exhaustive game dataset; meteor/item labels remain unfinished.

YOLO26n fine-tuning with a frozen backbone, 320-pixel input, CPU, and nominal batch size 4 stopped at epoch 29; validation selected epoch 14. The separate 12-frame test scored precision 0.718, recall 0.848, mAP50 0.920, and mAP50–95 0.674. The exported native OpenVINO CPU backend measured 17.77 ms median / 21.48 ms p95 over 20 calls after three warmups on one real image. It produced two player predictions in a reviewed frame containing one player; this confirms a material false-positive problem. Precision is insufficient for promotion, so it was not deployed. See `player-detector-candidate.json`, `data/player-reviewed/`, and the reproducible `tools/train_player_candidate.py` experiment.

The 100,000-point objective is unverified. A setup/inspection run displayed 3,660 points and 01:28, potentially including human input, and is not an autonomous agent result. Human-managed reset detection is implemented. Score OCR, detailed power-up state, true hitboxes, robust natural game-over detection and transfer of learned policies remain unfinished.

## Training laptop transfer

The project now supports explicit NVIDIA/CPU device selection for detector training and evaluation, portable dataset resolution, and multiple PPO/DQN arenas with DummyVecEnv or Windows-safe spawned subprocesses. Numerical PPO stays on CPU by default. A real CPU training smoke test completed 128 transitions in two subprocess arenas and saved a candidate; a separate regression test loaded a two-arena candidate and took an environment step. These checks verify the new training path, not improved gameplay. The full local suite passed **51 tests**. CUDA selection has software checks, but the friend's NVIDIA hardware is not connected to this PC and actual GPU training remains unverified.

Essential templates and 15 regression image fixtures are included in Git. Modified ready/lobby fixtures keep only required UI regions, omit account/friend information and are never training data. Full recordings, weights, environments and the downloaded scrcpy executable remain outside Git. The transfer tool produced a separate local data archive with 1,376 files (approximately 106 MB) and per-file checksums; it does not upload the data.

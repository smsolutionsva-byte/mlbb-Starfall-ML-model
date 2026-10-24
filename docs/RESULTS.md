# Private model benchmark

## Maintainer-reported private model result

The maintainer reports that their **private, unreleased model and movement/control setup achieved 282,793 points** in Starfall Breakout. The score is published as a benchmark to demonstrate what was possible with that private setup and give future ML experiments a reference point.

The supplied screenshot shows **282,793 points** and **rank #1 in the displayed weekly ranking list**. Both appearances of the maintainer's username are redacted; the ranking and score remain visible.

| Model / setup | Release status | Game-score benchmark |
| --- | --- | --- |
| Earlier YOLO26n detector trained from 1,988 reviewed images | Public; available before the benchmark update | Public perception model; no separate game-score result assigned |
| Private model and movement/control setup | Unreleased | **282,793 points**, maintainer-reported run |

![Ranking screenshot with both username labels redacted](media/weekly-rank-282793-redacted.png)

The associated phone recording ends on a Starfall page showing **282793 Highest Points** and **15:58 Longest Time**. Those are displayed account records; the one-minute clip is not a recording of a complete 15:58 run.

## Annotated recording

[Download the MP4](https://github.com/smsolutionsva-byte/mlbb-Starfall-ML-model/releases/download/demo-2026-10-24/starfall-282793-preview.mp4) or visit the [demo release](https://github.com/smsolutionsva-byte/mlbb-Starfall-ML-model/releases/tag/demo-2026-10-24).

The recording is shared alongside the private model's reported benchmark. A visualization layer was added **after capture** using the earlier public detector and experimental planner: square boxes identify the player, meteors, and pickups, and yellow lines show preview suggestions. The visualization pipeline and the private model that controlled the reported run are separate components.

| Media property | Checked result |
| --- | --- |
| Resolution | 1280 × 576 |
| Source duration | 61.62 seconds |
| Source and output video frames | 3,066 each |
| Maximum frame presentation-time difference | 0 seconds |
| Original audio packets copied | 2,654 |
| Encoded audio packet bytes | Identical to the source |
| Maximum audio presentation-time difference | Less than 0.000012 seconds |
| Output encoding | H.264 video, original AAC audio |
| Model observations | 925, requested at 15 per video second |
| Observations with a confirmed player and suggested path | 720 |
| Phone input sent by the replay | None |

Observation counts measure pipeline activity, not labelled precision/recall. Detection can be interrupted by bright effects or overlap. Offline processing speed is not live reaction latency. Display boxes use sprite extents and extra padding, rather than physical collision hitboxes.

## Private artifacts and public baseline

The model and movement/control setup credited with the benchmark remain private. The maintainer is sharing the result to show what is possible, while withholding the additional artifacts to respect game guidelines and discourage automated farming. The earlier public detector and existing experimental tracking, planning, and synthetic-training utilities remain available as a foundation for further ML research.

Use the public components for recording analysis, labelled-data experiments, further detector training, tracking research, and offline evaluation. Read MLBB's [Terms of Service](https://www.mobilelegends.com/news/articleldetail?newsid=2728920) before interacting with its services. This project is not affiliated with or endorsed by Moonton.

Earlier capture measurements, movement experiments, and synthetic-policy results remain in [the historical evidence report](../reports/EVIDENCE.md). Their results and limitations describe the specific experiments recorded there.

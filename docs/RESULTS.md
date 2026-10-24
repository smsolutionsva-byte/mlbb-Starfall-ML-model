# Recorded result and demonstration

## Maintainer-reported score

The maintainer supplied a screenshot showing **282,793 points** and **rank #1 in the displayed weekly ranking list**. The same screenshot repeats the maintainer's row at the bottom. Both username labels are redacted in the public copy; the score and ranking remain visible. This is evidence of the result displayed on that screen, rather than an independently verified global ranking.

![Ranking screenshot with both username labels redacted](media/weekly-rank-282793-redacted.png)

The associated phone recording ends on a Starfall page showing **282793 Highest Points** and **15:58 Longest Time**. Those are displayed account records; the one-minute clip is not a recording of a complete 15:58 run.

## Annotated recording

[Download the MP4](https://github.com/smsolutionsva-byte/mlbb-Starfall-ML-model/releases/download/demo-2026-10-24/starfall-282793-preview.mp4) or visit the [demo release](https://github.com/smsolutionsva-byte/mlbb-Starfall-ML-model/releases/tag/demo-2026-10-24).

The public three-class detector and experimental planner were run over a previously recorded phone video. Square boxes identify the player, meteors, and pickups; yellow lines show suggested paths. These overlays were added **after capture**. They do not show that the public checkpoint generated the actions in the recording, nor do they establish an independently reproducible autonomous score.

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

The maintainer states that additional movement/control work and another model exist privately and will not be uploaded, to respect game guidelines and discourage automated farming. The repository already includes experimental baseline movement, planning, and synthetic-training code; that existing code is distinct from the additional private artifacts. The public detector alone cannot reproduce a private controller's behavior.

Use the public components for recording analysis, labelled-data experiments, further detector training, tracking research, and offline evaluation. Read MLBB's [Terms of Service](https://www.mobilelegends.com/news/articleldetail?newsid=2728920) before interacting with its services. This project is not affiliated with or endorsed by Moonton.

Earlier capture measurements, movement experiments, and synthetic-policy results remain in [the historical evidence report](../reports/EVIDENCE.md). Their results and limitations describe the specific experiments recorded there.

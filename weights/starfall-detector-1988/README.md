# Starfall detector from 1,988 reviewed images

`best.pt` contains a YOLO26n detector fine-tuned from the earlier 728-image
annotation assistant. Its three classes are `player`, `meteor`, and `pickup`
(displayed as Collectable in the local annotation UI).

The checkpoint is stored directly in Git. `model.json` contains portable paths,
the exact framework versions, training details, class roles and evaluation
results. `SHA256SUMS` verifies the checkpoint bytes.

## Training

| Setting | Value |
| --- | --- |
| Human-reviewed images | 1,988 from 23 capture groups |
| Train / validation / test | 1,557 / 228 / 203 images |
| Reviewed boxes | 1,783 player; 10,821 meteor; 684 pickup |
| Input size / batch | 416 / 16 |
| Optimizer / initial learning rate | AdamW / 0.0003 |
| Completed epochs / selected epoch | 12 / 2 |
| Training hardware | NVIDIA RTX 3050 Laptop, 6 GB |

Complete capture groups and near-duplicate frames were kept together when
splitting. The 0.35 suggestion confidence cutoff was selected on the 228
validation images by minimizing extra boxes plus missed objects.

## Use on an image

Install the project's learning dependencies, then run this from the repository
root with your own image path:

```python
from ultralytics import YOLO

model = YOLO("weights/starfall-detector-1988/best.pt")
results = model.predict("frame.jpg", imgsz=416, conf=0.35)
```

Class IDs are 0 = player, 1 = meteor, 2 = pickup. The checkpoint works with
Ultralytics on CPU or CUDA; CUDA availability depends on the local PyTorch setup.

## Evaluation and status

On the 203-image test split, mAP50 was 0.9931 and mAP50-95 was 0.9285.
The comparison with the previous model used 167 identical test frames from two
capture groups unseen during training of either model. See `comparison.json`:

| Shared test result | Previous 728-image model | This model |
| --- | --- | --- |
| mAP50-95 | 0.9277 | 0.9394 |
| Suggestion confidence cutoff | 0.20 | 0.35 |
| Correct matched boxes | 873 | 859 |
| Extra boxes against saved labels | 0 | 10 |
| Missed boxes against saved labels | 0 | 14 |

Suggestion matching requires the correct class and at least 0.5 intersection
over union. Each model uses its configured suggestion cutoff; mAP measures
performance across confidence thresholds.

The shared labels were human-reviewed after suggestions from the earlier model,
which can favor that model in this comparison. Some earlier reference labels
also contain mistakes. Fresh manually boxed gameplay frames are needed for an
independent accuracy check. This candidate was saved for further observation;
the local annotation assistant remained on the 728-image model. These results
do not measure physical collision hitboxes, pickup effects or autonomous score
improvement.

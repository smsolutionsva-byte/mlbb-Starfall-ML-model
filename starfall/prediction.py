"""Run a saved detector on an image and save a visible, annotated result."""

from pathlib import Path
import hashlib
import json
import time
import cv2
import torch
from ultralytics import YOLO
from .training import select_device


def predict_image(
    image,
    model="weights/starfall-detector-1988/best.pt",
    device="auto",
    output="reports/trained-model-test.png",
    confidence=0.35,
    imgsz=416,
):
    image_path, checkpoint = Path(image), Path(model)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint}. Download the latest repository."
        )
    metadata = checkpoint.parent / "model.json"
    if metadata.exists():
        expected = json.loads(metadata.read_text(encoding="utf-8")).get("sha256")
        if expected and hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected:
            raise ValueError("Checkpoint checksum does not match its metadata")
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Cannot read image: {image_path}")
    selected = select_device(device)
    torch.set_num_threads(2)
    detector = YOLO(str(checkpoint))
    start = time.perf_counter()
    result = detector.predict(
        frame, device=selected, conf=confidence, imgsz=imgsz, verbose=False
    )[0]
    elapsed = time.perf_counter() - start
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(output))
    rows = [
        {
            "class": result.names[int(cls)],
            "confidence": float(conf),
            "xyxy": box.tolist(),
        }
        for box, cls, conf in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        )
    ]
    report = {
        "image": str(image_path),
        "checkpoint": str(checkpoint),
        "device": selected,
        "annotated_image": str(output.resolve()),
        "first_prediction_s": elapsed,
        "speed_ms": result.speed,
        "confidence_cutoff": confidence,
        "detections": rows,
        "note": "Object detection only; this does not send phone input or measure gameplay accuracy.",
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report

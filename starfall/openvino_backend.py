"""Bounded-thread OpenVINO detection backend for exported YOLO detect models."""

from pathlib import Path
import cv2
import numpy as np
import yaml
from .types import Detection


class OpenvinoDetector:
    def __init__(self, folder, roles=None, imgsz=416, device="CPU", confidence=0.35):
        from openvino import Core

        folder = Path(folder)
        xml = list(folder.glob("*.xml"))
        if len(xml) != 1:
            raise ValueError("Expected one exported detection model XML")
        metadata = yaml.safe_load((folder / "metadata.yaml").read_text())
        if metadata.get("task") != "detect":
            raise ValueError("Only detection exports supported")
        self.names = metadata["names"]
        self.roles = roles
        self.imgsz = imgsz
        self.confidence = confidence
        core = Core()
        model = core.read_model(xml[0])
        model.reshape([1, 3, imgsz, imgsz])
        config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}
        if device == "CPU":
            config["INFERENCE_NUM_THREADS"] = 2
        self.compiled = core.compile_model(model, device, config)

    def boxes(self, image):
        h, w = image.shape[:2]
        size = self.imgsz
        ratio = min(size / w, size / h)
        nw, nh = round(w * ratio), round(h * ratio)
        resized = cv2.resize(image, (nw, nh))
        left = (size - nw) // 2
        top = (size - nh) // 2
        padded = np.full((size, size, 3), 114, np.uint8)
        padded[top : top + nh, left : left + nw] = resized
        tensor = (
            np.ascontiguousarray(
                padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32
            )
            / 255
        )
        raw = self.compiled(tensor)[self.compiled.output(0)][0]
        if raw.shape[-1] == 6:
            # End-to-end exported detections: xyxy, confidence, class.
            picked = raw[raw[:, 4] >= self.confidence]
            xyxy = picked[:, :4].copy()
            conf = picked[:, 4]
            cls = picked[:, 5].astype(int)
        else:
            # Conventional detection head: [4+classes, anchors], xywh pixels.
            expected = 4 + len(self.names)
            if raw.shape[0] != expected:
                raise ValueError(
                    f"Unsupported YOLO output shape {raw.shape}; expected {expected} channels"
                )
            rows = raw.T
            cls = rows[:, 4:].argmax(axis=1)
            conf = rows[np.arange(len(rows)), cls + 4]
            keep = conf >= self.confidence
            rows = rows[keep]
            cls = cls[keep]
            conf = conf[keep]
            boxes = rows[:, :4].copy()
            boxes[:, :2] -= boxes[:, 2:] / 2
            # Offset coordinates by class for class-aware NMS.
            nms_boxes = boxes.copy()
            nms_boxes[:, :2] += cls[:, None] * (size * 4)
            indices = (
                np.array(
                    cv2.dnn.NMSBoxes(
                        nms_boxes.tolist(), conf.tolist(), self.confidence, 0.45
                    )
                )
                .flatten()
                .astype(int)
            )
            boxes = boxes[indices]
            cls = cls[indices]
            conf = conf[indices]
            xyxy = boxes.copy()
            xyxy[:, 2:] += xyxy[:, :2]
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) / ratio
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) / ratio
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h)
        return xyxy, conf, cls

    def detect(self, image):
        if not self.roles:
            raise ValueError("Live detection requires validated game-class roles")
        boxes, confs, classes = self.boxes(image)
        scale = max(image.shape[:2])
        result = []
        for (x0, y0, x1, y1), confidence, cls in zip(boxes, confs, classes):
            name = self.names[int(cls)]
            role = self.roles.get(name)
            if role is None:
                continue
            result.append(
                Detection(
                    role["kind"],
                    np.array([(x0 + x1) / 2, (y0 + y1) / 2]) / scale,
                    role.get("radius_factor", 0.5) * max(x1 - x0, y1 - y0) / scale,
                    float(confidence),
                    role.get("label", name),
                    (float(x1 - x0) / scale, float(y1 - y0) / scale),
                )
            )
        return result

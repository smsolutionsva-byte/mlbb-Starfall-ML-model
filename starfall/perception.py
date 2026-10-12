"""Replaceable detectors; game roles must be configured from observed frames."""

from pathlib import Path
import cv2
import numpy as np
from .types import Detection


class TemplateDetector:
    def __init__(self, specs, bounds=None, search_scale=1.0):
        self.specs, self.templates = specs, []
        self.bounds = bounds
        if not 0.25 <= search_scale <= 1:
            raise ValueError("Template search scale must be between .25 and 1")
        self.search_scale = search_scale
        for spec in specs:
            image = cv2.imread(str(Path(spec["path"])), cv2.IMREAD_GRAYSCALE)
            if image is None or image.std() < 2:
                raise ValueError(f"Invalid or textureless template: {spec['path']}")
            if search_scale != 1:
                image = cv2.resize(
                    image,
                    None,
                    fx=search_scale,
                    fy=search_scale,
                    interpolation=cv2.INTER_AREA,
                )
            self.templates.append(image)

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.search_scale != 1:
            gray = cv2.resize(
                gray,
                None,
                fx=self.search_scale,
                fy=self.search_scale,
                interpolation=cv2.INTER_AREA,
            )
        scale = max(gray.shape[:2])
        x0, y0, x1, y1 = (0, 0, gray.shape[1], gray.shape[0])
        if self.bounds:
            x0, y0, x1, y1 = [int(v * scale) for v in self.bounds]
            # Bounds constrain the player's center, not the entire visible sprite.
            # Search beyond them so hair/wings at an arena edge stay detectable.
            padding = max(max(template.shape) for template in self.templates)
            x0, y0 = max(0, x0 - padding), max(0, y0 - padding)
            x1, y1 = min(gray.shape[1], x1 + padding), min(gray.shape[0], y1 + padding)
        crop = gray[y0:y1, x0:x1]
        result = []
        for spec, template in zip(self.specs, self.templates):
            h, w = template.shape
            if crop.shape[0] < h or crop.shape[1] < w:
                continue
            scores = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
            for _ in range(spec.get("max_objects", 30)):
                _, score, _, point = cv2.minMaxLoc(scores)
                if score < spec.get("threshold", 0.8):
                    break
                x, y = point
                result.append(
                    Detection(
                        spec["kind"],
                        (
                            np.array([x + x0 + w / 2, y + y0 + h / 2])
                            + np.asarray(spec.get("center_offset_pixels", [0, 0]))
                            * self.search_scale
                        )
                        / scale,
                        spec.get("radius_factor", 0.5) * max(w, h) / scale,
                        float(score),
                        spec["label"],
                    )
                )
                scores[
                    max(0, y - h // 2) : y + h // 2 + 1,
                    max(0, x - w // 2) : x + w // 2 + 1,
                ] = -1
        return suppress_duplicates(result)


def suppress_duplicates(detections):
    kept = []
    for d in sorted(detections, key=lambda x: x.confidence, reverse=True):
        if not any(
            k.kind == d.kind
            and (k.label == d.label or k.kind == "item")
            and np.linalg.norm(k.position - d.position) < 0.8 * min(k.radius, d.radius)
            for k in kept
        ):
            kept.append(d)
    return kept


class YoloDetector:
    def __init__(self, model, role_map, imgsz=416, confidence=0.35, device="cpu"):
        from ultralytics import YOLO
        from .training import select_device

        if not role_map:
            raise ValueError("Provide a manually validated class-name-to-role map")
        self.model = YOLO(model)
        self.role_map, self.imgsz, self.confidence, self.device = (
            role_map,
            imgsz,
            confidence,
            select_device(device),
        )

    def detect(self, image):
        results = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]
        scale = max(image.shape[:2])
        detections = []
        for box, cls, conf in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.cls.cpu().numpy(),
            results.boxes.conf.cpu().numpy(),
        ):
            name = results.names[int(cls)]
            role = self.role_map.get(name)
            if role is None:
                continue
            x0, y0, x1, y1 = box
            detections.append(
                Detection(
                    role["kind"],
                    np.array([(x0 + x1) / 2, (y0 + y1) / 2]) / scale,
                    role.get("radius_factor", 0.5) * max(x1 - x0, y1 - y0) / scale,
                    float(conf),
                    role.get("label", name),
                )
            )
        return suppress_duplicates(detections)


class CompositeDetector:
    def __init__(self, components):
        self.components = components

    def detect(self, image):
        return suppress_duplicates(
            [d for component in self.components for d in component.detect(image)]
        )


def create_detector(config):
    detector = config["detector"]
    if detector["type"] == "hybrid":
        return CompositeDetector(
            [
                create_detector(dict(config, detector=component))
                for component in detector["components"]
            ]
        )
    if detector["type"] == "templates":
        return TemplateDetector(
            detector["templates"],
            config.get("bounds"),
            detector.get("search_scale", 1.0),
        )
    if detector["type"] == "yolo":
        return YoloDetector(
            detector["model"],
            detector["roles"],
            detector.get("imgsz", 416),
            detector.get("confidence", 0.35),
            detector.get("device", "cpu"),
        )
    if detector["type"] == "openvino":
        from .openvino_backend import OpenvinoDetector

        return OpenvinoDetector(
            detector["model"],
            detector["roles"],
            detector.get("imgsz", 416),
            detector.get("device", "CPU"),
            detector.get("confidence", 0.35),
        )
    raise ValueError("Unknown detector type")

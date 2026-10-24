"""High-contrast observation overlays shared by live and recorded previews."""

import cv2
import numpy as np


COLORS = {
    "player": (80, 255, 100),
    "hazard": (60, 80, 255),
    "item": (255, 230, 0),
    "target": (240, 80, 255),
    "wall": (240, 80, 255),
}


def _label(image, text, point, color, scale):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48 * scale
    thickness = max(1, round(scale))
    (width, height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(4, min(int(point[0]), image.shape[1] - width - 12))
    y = max(height + 8, min(int(point[1]), image.shape[0] - baseline - 5))
    cv2.rectangle(
        image,
        (x - 4, y - height - 5),
        (x + width + 5, y + baseline + 4),
        (12, 12, 12),
        -1,
    )
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_preview(
    image,
    tracks,
    decision=None,
    *,
    status="",
    settings=None,
    coordinate_size=None,
    wall=None,
):
    """Square display boxes surround sprites; collision radii stay unchanged."""
    settings = settings or {}
    canvas = image.copy()
    height, width = canvas.shape[:2]
    ref_width, ref_height = coordinate_size or (width, height)
    factors = np.array([width / ref_width, height / ref_height]) * max(
        ref_width, ref_height
    )
    display_scale = max(width, height) / 960
    thickness = max(2, round(settings.get("box_thickness", 3) * display_scale))
    padding = settings.get("box_padding", 5) / 960
    minimum = settings.get("min_box_size", 28) / 960
    latest_seen = max((track.last_seen for track in tracks), default=0.0)
    for track in sorted(tracks, key=lambda t: t.kind != "hazard"):
        # Association can leave old meteor tracks alive briefly. Keep those for
        # prediction, but do not draw duplicate boxes from older observations.
        if track.last_seen < latest_seen - 1e-6 or track.time - track.last_seen > 0.12:
            continue
        size = max(track.box_size) if track.box_size else 2 * float(track.radius)
        half_size = max(size, minimum) / 2 + padding
        lower = np.rint((track.position - half_size) * factors).astype(int)
        upper = np.rint((track.position + half_size) * factors).astype(int)
        lower = np.clip(lower, [0, 0], [width - 1, height - 1])
        upper = np.clip(upper, [0, 0], [width - 1, height - 1])
        if np.any(upper <= lower):
            continue
        color = COLORS.get(track.kind, (255, 255, 255))
        cv2.rectangle(
            canvas, tuple(lower), tuple(upper), (8, 8, 8), thickness + 4, cv2.LINE_AA
        )
        cv2.rectangle(canvas, tuple(lower), tuple(upper), color, thickness, cv2.LINE_AA)
        # Repeating labels on every meteor hides the crowded arena.
        if track.kind != "hazard":
            text = {
                "player": "PLAYER",
                "item": "PICKUP",
                "target": "BOSS CORE",
                "wall": "BOSS WALL",
            }.get(track.kind, track.label.upper())
            label_y = (
                lower[1] - 8
                if lower[1] > 40 * display_scale
                else upper[1] + 20 * display_scale
            )
            _label(canvas, text, (lower[0], label_y), color, display_scale)
    if wall:
        points = np.rint(np.array(wall.bounds).reshape(2, 2) * factors).astype(int)
        cv2.rectangle(
            canvas,
            tuple(points[0]),
            tuple(points[1]),
            COLORS["wall"],
            thickness,
            cv2.LINE_AA,
        )
    if decision is not None:
        path = np.rint(decision.path * factors).astype(np.int32)
        cv2.polylines(canvas, [path], False, (8, 8, 8), thickness + 5, cv2.LINE_AA)
        cv2.polylines(canvas, [path], False, (0, 255, 255), thickness + 1, cv2.LINE_AA)
    if status:
        _label(canvas, status, (12, 22 * display_scale), (255, 255, 255), display_scale)
    x = 12
    for title, kind in (
        ("PLAYER", "player"),
        ("METEOR", "hazard"),
        ("PICKUP", "item"),
        ("SUGGESTED PATH", None),
    ):
        color = COLORS[kind] if kind else (0, 255, 255)
        _label(canvas, title, (x, height - 13), color, display_scale)
        x += round((len(title) * 9 + 26) * display_scale)
    return canvas

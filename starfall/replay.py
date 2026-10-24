"""Analyze a phone recording without connecting to or controlling a phone."""

from fractions import Fraction
from pathlib import Path
import json
import time

import av
import cv2
import numpy as np

from .combat import CombatReader, CombatReading
from .perception import create_detector, suppress_duplicates
from .planning import PredictivePlanner
from .preview import draw_preview
from .runtime import load_config
from .session import PageReader
from .tracking import Tracker
from .types import World


def preview_video(
    video,
    config_path="configs/phone-detector-1988-cpu.yaml",
    output=None,
    preview=True,
    detection_fps=15,
    seconds=0,
):
    if detection_fps <= 0 or seconds < 0:
        raise ValueError(
            "Detection rate must be positive and seconds cannot be negative"
        )
    source = Path(video).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output = Path(output or Path("reports") / f"{source.stem}-preview.mp4").resolve()
    if source == output:
        raise ValueError("Preview output must differ from the original recording")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial.mp4")
    cv2.setNumThreads(2)
    config = load_config(config_path)
    detector = create_detector(config)
    combat = (
        CombatReader(config["combat"])
        if config.get("combat", {}).get("enabled")
        else None
    )
    pages, tracker = PageReader(config["session"]), Tracker()
    motion = config["motion"]
    planner = PredictivePlanner(
        speed=motion.get("speed") or 0.35,
        delay=motion.get("delay") or 0.05,
        **config["planner"],
    )
    reference = tuple(config["session"]["frame_size"])
    decision, scene, tracks = None, CombatReading(), []
    next_detection = 0.0
    frames = samples = with_player = with_path = 0
    timestamp = 0.0
    processing_ms = []
    stopped = False
    started = last_progress = time.perf_counter()
    if preview:
        cv2.namedWindow("Starfall recorded preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Starfall recorded preview", 1280, 576)
    try:
        with (
            av.open(str(source)) as recording,
            av.open(str(partial), "w", options={"movflags": "+faststart"}) as rendered,
            output.with_suffix(".jsonl").open("w", encoding="utf-8") as events,
        ):
            input_video = recording.streams.video[0]
            origin = float((input_video.start_time or 0) * input_video.time_base)
            rate = input_video.average_rate or Fraction(30, 1)
            duration = (
                float(input_video.duration * input_video.time_base)
                if input_video.duration
                else None
            )
            output_video = rendered.add_stream("libx264", rate=rate)
            output_video.width = input_video.codec_context.width
            output_video.height = input_video.codec_context.height
            output_video.pix_fmt = "yuv420p"
            output_video.time_base = Fraction(1, 90000)
            output_video.codec_context.time_base = Fraction(1, 90000)
            output_video.codec_context.thread_count = 2
            output_video.options = {"preset": "veryfast", "crf": "20", "bf": "0"}
            audio = {
                s.index: rendered.add_stream_from_template(s)
                for s in recording.streams.audio
            }
            for packet in recording.demux():
                if packet.stream.type == "audio":
                    if packet.dts is not None:
                        if origin:
                            shift = round(origin / packet.time_base)
                            packet.pts = (
                                packet.pts - shift if packet.pts is not None else None
                            )
                            packet.dts -= shift
                        packet.stream = audio[packet.stream.index]
                        rendered.mux(packet)
                    continue
                if packet.stream.index != input_video.index:
                    continue
                for decoded in packet.decode():
                    timestamp = (
                        float(decoded.time) - origin
                        if decoded.time is not None
                        else frames / float(rate)
                    )
                    if seconds and timestamp >= seconds:
                        stopped = True
                        break
                    image = decoded.to_ndarray(format="bgr24")
                    sampled = timestamp + 1e-6 >= next_detection
                    detections = []
                    if sampled:
                        tick = time.perf_counter()
                        analysis = cv2.resize(
                            image, reference, interpolation=cv2.INTER_AREA
                        )
                        page = pages.read(analysis).page
                        observing = page in ("gameplay", "unknown")
                        scene = (
                            combat.read(analysis)
                            if observing and combat
                            else CombatReading()
                        )
                        detections = suppress_duplicates(
                            (detector.detect(analysis) if observing else [])
                            + scene.detections
                        )
                        if not observing:
                            tracker.reset()
                        next_detection = (
                            int(timestamp * detection_fps) + 1
                        ) / detection_fps
                    tracks = tracker.update(detections, timestamp)
                    if sampled:
                        players = [
                            t
                            for t in tracks
                            if t.kind == "player"
                            and timestamp - t.last_seen < 0.12
                            and t.hits >= 2
                        ]
                        decision = None
                        if len(players) == 1:
                            wall_tracks = [t for t in tracks if t.kind == "wall"]
                            if scene.wall and len(wall_tracks) == 1:
                                scene.wall.velocity[0] = wall_tracks[0].velocity[0]
                            world = World(
                                players[0],
                                [t for t in tracks if t.kind == "hazard"],
                                [t for t in tracks if t.kind == "item"],
                                tuple(
                                    config["bounds"]
                                    or (0, 0, 1, reference[1] / reference[0])
                                ),
                                timestamp,
                                targets=[t for t in tracks if t.kind == "target"],
                                walls=[scene.wall] if scene.wall else [],
                            )
                            decision = planner.choose(world)
                        elapsed_ms = (time.perf_counter() - tick) * 1000
                        processing_ms.append(elapsed_ms)
                        samples += 1
                        with_player += len(players) == 1
                        with_path += decision is not None
                        events.write(
                            json.dumps(
                                {
                                    "frame": frames,
                                    "video_s": timestamp,
                                    "page": page,
                                    "processing_ms": elapsed_ms,
                                    "suggested_action": decision.action
                                    if decision
                                    else None,
                                    "detections": [
                                        {
                                            "kind": d.kind,
                                            "label": d.label,
                                            "position": d.position.tolist(),
                                            "confidence": float(d.confidence),
                                            "box_size": d.box_size,
                                        }
                                        for d in detections
                                    ],
                                }
                            )
                            + "\n"
                        )
                    status = (
                        f"REPLAY {timestamp:05.1f}s"
                        + (f" / {duration:.1f}s" if duration else "")
                        + " | model + suggested path"
                    )
                    canvas = draw_preview(
                        image,
                        tracks,
                        decision,
                        status=status,
                        settings=config.get("preview"),
                        coordinate_size=reference,
                        wall=scene.wall,
                    )
                    frame = av.VideoFrame.from_ndarray(canvas, format="bgr24")
                    frame.pts = round(timestamp * 90000)
                    frame.time_base = Fraction(1, 90000)
                    for encoded in output_video.encode(frame):
                        rendered.mux(encoded)
                    if frames == 0 or (sampled and frames % 100 < 4):
                        cv2.imwrite(str(output.with_suffix(".jpg")), canvas)
                    frames += 1
                    if preview:
                        cv2.imshow("Starfall recorded preview", canvas)
                        if cv2.waitKey(1) & 255 == ord("q"):
                            stopped = True
                            break
                    if time.perf_counter() - last_progress >= 10:
                        print(
                            f"Processed {timestamp:.1f}s"
                            + (f" / {duration:.1f}s" if duration else "")
                            + f" ({samples} model observations)",
                            flush=True,
                        )
                        last_progress = time.perf_counter()
                if stopped:
                    break
            for encoded in output_video.encode():
                rendered.mux(encoded)
        partial.replace(output)
    finally:
        if preview:
            cv2.destroyWindow("Starfall recorded preview")
    report = {
        "input": str(source),
        "output": str(output),
        "config": str(config_path),
        "completed": not stopped,
        "frames_written": frames,
        "last_video_s": timestamp,
        "model_observations": samples,
        "requested_detection_fps": detection_fps,
        "observations_with_player": int(with_player),
        "observations_with_path": int(with_path),
        "processing_seconds": time.perf_counter() - started,
        "observation_processing_median_ms": float(np.median(processing_ms))
        if samples
        else None,
        "audio_preserved": bool(audio),
        "phone_inputs_sent": 0,
        "note": "Offline replay of recorded gameplay. Boxes include display padding; path is a suggestion. No training or live reaction-speed measurement.",
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report

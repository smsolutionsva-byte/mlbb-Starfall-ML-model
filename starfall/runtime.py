from pathlib import Path
import json
import time
import cv2
import numpy as np
import yaml
from .android import Adb, ScrcpyStream, Joystick, touch_packet
from .dataset import Collector
from .logging import RunLog, percentiles
from .perception import create_detector, suppress_duplicates
from .planning import PredictivePlanner, limited_drag
from .tracking import Tracker
from .types import World, DIRECTIONS
from .safety import HealthSupervisor, HealthAbort
from .session import PageReader, PageReading, EpisodeLoop
from .combat import CombatReader, CombatReading, UltimatePolicy
from .features import encode
from .control_lease import ControlLease
from .preview import draw_preview


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def capture_benchmark(seconds=15, serial=None, max_size=960):
    stream = ScrcpyStream(Adb(serial), max_size=max_size)
    rows = []
    seq = 0
    skipped = 0
    start = time.perf_counter()
    try:
        stream.start()
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            frame = stream.get_frame(seq, timeout=3)
            skipped += max(0, frame.sequence - seq - 1) if seq else 0
            seq = frame.sequence
            rows.append(
                {
                    "decode_ms": (frame.decoded_ns - frame.received_ns) / 1e6,
                    "receive_to_consume_ms": (
                        time.perf_counter_ns() - frame.received_ns
                    )
                    / 1e6,
                    "pts_us": frame.device_pts_us,
                    "received_ns": frame.received_ns,
                }
            )
        elapsed = time.perf_counter() - start
        result = {
            "device": stream.adb.serial,
            "scrcpy_version": "5.0.1",
            "frame_size": [stream.width, stream.height],
            "frames": len(rows),
            "seconds": elapsed,
            "fps": len(rows) / elapsed,
            "skipped_frames": skipped,
            "decode": percentiles([r["decode_ms"] for r in rows]),
            "receive_to_consume": percentiles(
                [r["receive_to_consume_ms"] for r in rows]
            ),
            "end_to_end_latency_ms": None,
            "note": "Receive-to-consume excludes phone capture, encoding, USB and device-visible action response.",
            "rows": rows,
        }
        Path("reports/capture-benchmark.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        stream.close()


def run_live(
    config_path,
    serial=None,
    seconds=60,
    control=False,
    collect=False,
    preview=True,
    model=None,
    algorithm="ppo",
    guard=False,
    provided_stream=None,
    repeat=False,
    max_episodes=None,
    stop_file="runs/stop-request",
):
    cv2.setNumThreads(2)
    config = load_config(config_path)
    if control:
        if not repeat and config.get("control_limits", {}).get("max_seconds"):
            seconds = min(seconds, float(config["control_limits"]["max_seconds"]))
        if not (
            config["joystick"]["calibrated"]
            and config["motion"]["measured"]
            and config["detector_validated"]
        ):
            raise ValueError(
                "Control needs measured joystick/movement and manually validated detector; run calibration first"
            )
    detector = create_detector(config)
    combat_config = config.get("combat", {})
    combat_reader = (
        CombatReader(combat_config) if combat_config.get("enabled") else None
    )
    ultimate = UltimatePolicy(config["planner"].get("muzzle_offset", 0.015))
    tracker = Tracker()
    motion = config["motion"]
    planner = PredictivePlanner(
        speed=motion.get("speed") or 0.35,
        delay=motion.get("delay") or 0.05,
        **config["planner"],
    )
    response_model = None
    if repeat and config["session"].get("learn_response", True):
        from .real_learning import response_signature

        saved = Path("models/real-response/latest.json")
        if saved.exists():
            report = json.loads(saved.read_text())
            if report.get("input_signature") == response_signature(
                config
            ) and report.get("promoted_actions"):
                values = np.asarray(report["validated_displacements"], dtype=float)
                if values.shape == (9, 2) and np.isfinite(values).all():
                    planner.action_velocities = values / motion["action_period_s"]
                    response_model = str(saved)
    policy = None
    if model:
        from stable_baselines3 import PPO, DQN

        policy = (PPO if algorithm == "ppo" else DQN).load(model, device="cpu")
        if control:
            raise ValueError(
                "Sim-trained policy has not passed real-game comparisons; use shadow mode"
            )
    lease = ControlLease().acquire() if control else None
    if repeat:
        Path(stop_file).unlink(missing_ok=True)
    log = RunLog(
        dict(
            config,
            policy_model=model,
            algorithm=algorithm if model else None,
            policy_kind=algorithm if model else "mpc",
            auto_restart=repeat,
            heart_scanning=guard,
            response_model=response_model,
        ),
        mode="learning_loop" if repeat else "live_control" if control else "shadow",
    )
    page_reader = PageReader(config["session"])
    episodes = EpisodeLoop(auto_start=repeat)
    lifecycle = (log.path / "episodes.jsonl").open("w", encoding="utf-8")
    print(
        f"Recording attempts in {log.path}. Heart scanning is {'on' if guard else 'off'}.",
        flush=True,
    )
    collector = Collector(session=log.id) if collect else None
    stream = provided_stream
    joystick = supervisor = None
    setup_needed = True
    seq = 0
    reconnects = 0
    previous_pts = None
    reason = "duration_limit"
    start = time.monotonic()
    last_action_at = -float("inf")
    attempt_id = None
    attempt_verified = False
    try:
        while seconds <= 0 or time.monotonic() - start < seconds:
            if repeat and Path(stop_file).exists():
                reason = "manual_stop"
                break
            if preview and cv2.waitKey(1) & 255 == ord("q"):
                reason = "manual_stop"
                break
            try:
                if stream is None or setup_needed:
                    if stream is None:
                        stream = ScrcpyStream(
                            Adb(serial), control=control or guard, **config["capture"]
                        ).start()
                    if control:
                        if config["joystick"]["frame_size"] != [
                            stream.width,
                            stream.height,
                        ]:
                            raise RuntimeError(
                                "Phone orientation/resolution changed: recalibrate"
                            )
                        joystick = Joystick(
                            stream,
                            config["joystick"]["base"],
                            config["joystick"]["radius"],
                        )
                    if guard:
                        supervisor = HealthSupervisor(
                            stream, config["health_guard"], joystick
                        ).start()
                    tracker.reset()
                    previous_pts = None
                    seq = 0
                    setup_needed = False
                frame = stream.get_frame(seq, timeout=1)
                if supervisor:
                    supervisor.check()
            except HealthAbort:
                raise
            except (RuntimeError, TimeoutError, OSError) as exc:
                if supervisor:
                    supervisor.close()
                    supervisor = None
                if joystick:
                    try:
                        joystick.close()
                    except OSError:
                        pass
                    joystick = None
                if stream:
                    stream.close()
                    stream = None
                setup_needed = True
                for event in episodes.interrupt("capture_disconnect"):
                    lifecycle.write(json.dumps(event) + "\n")
                lifecycle.flush()
                reconnects += 1
                if reconnects > 3 and not repeat:
                    raise RuntimeError(
                        "Capture failed after 3 reconnect attempts"
                    ) from exc
                if repeat and reconnects in (1, 4):
                    print(
                        "Phone capture or game view unavailable; waiting to reconnect.",
                        flush=True,
                    )
                time.sleep(min(5, 0.3 * reconnects))
                continue
            if previous_pts is not None and frame.device_pts_us <= previous_pts:
                continue
            previous_pts = frame.device_pts_us
            begin = time.perf_counter_ns()
            dropped = max(0, frame.sequence - seq - 1) if seq else 0
            seq = frame.sequence
            t = begin
            correct_geometry = (frame.image.shape[1], frame.image.shape[0]) == tuple(
                config["session"]["frame_size"]
            )
            if not correct_geometry and not repeat:
                raise ValueError(
                    "Phone orientation/resolution changed; control stopped"
                )
            page = (
                page_reader.read(frame.image)
                if correct_geometry
                else PageReading("orientation_changed")
            )
            in_game, start_game, page_events = episodes.update(page, time.monotonic())
            observe_scene = in_game or (
                preview
                and not control
                and not guard
                and config.get("preview", {}).get("observe_unknown_pages", False)
                and page.page == "unknown"
            )
            for event in page_events:
                if event["type"] == "challenge_tap":
                    attempt_id = f"{log.id}-attempt{episodes.starts}"
                    attempt_verified = True
                event["attempt_id"] = attempt_id
                event.update(
                    sequence=seq,
                    received_ns=frame.received_ns,
                    device_pts_us=frame.device_pts_us,
                )
                lifecycle.write(json.dumps(event) + "\n")
                lifecycle.flush()
                print(json.dumps(event), flush=True)
                if event["type"] == "episode_start" and collect:
                    collector = Collector(
                        session=attempt_id or f"{log.id}-unverified{episodes.episode}"
                    )
                if event["type"] == "episode_start":
                    ultimate.reset()
                if (
                    repeat
                    and event["type"] == "episode_end"
                    and config["session"].get("learn_response", True)
                ):
                    from .real_learning import train_real_response

                    log.file.flush()
                    learned = train_real_response(config)
                    planner.action_velocities = (
                        np.asarray(learned["validated_displacements"])
                        / motion["action_period_s"]
                    )
                    update = {
                        "type": "response_model_update",
                        "episode": episodes.episode,
                        "samples": learned["samples"],
                        "promoted_actions": learned["promoted_actions"],
                        "model": "models/real-response/latest.json",
                    }
                    lifecycle.write(json.dumps(update) + "\n")
                    lifecycle.flush()
                    print(json.dumps(update), flush=True)
                if event["type"] == "episode_end":
                    attempt_id = None
                    attempt_verified = False
            if not observe_scene:
                if joystick:
                    joystick.release()
                tracker.reset()
            if max_episodes and episodes.ended >= max_episodes:
                reason = "episodes_limit"
                break
            if start_game:
                x, y = map(int, page.challenge)
                stream.control.sendall(
                    touch_packet(0, x, y, stream.width, stream.height, pointer=1)
                )
                time.sleep(0.035)
                stream.control.sendall(
                    touch_packet(1, x, y, stream.width, stream.height, pointer=1)
                )
            scene = (
                combat_reader.read(frame.image)
                if observe_scene and combat_reader
                else CombatReading()
            )
            detections = suppress_duplicates(
                (detector.detect(frame.image) if observe_scene else [])
                + scene.detections
            )
            if supervisor:
                supervisor.check()
            dtime = time.perf_counter_ns()
            tracks = tracker.update(detections, frame.time)
            trtime = time.perf_counter_ns()
            players = [
                p
                for p in tracks
                if p.kind == "player"
                and frame.time - p.last_seen < 0.12
                and p.hits >= 2
            ]
            action = 0
            world = None
            gesture_sent = False
            gesture_fraction = 0.0
            gesture_started_ns = None
            decision = None
            override = False
            ultimate_sent = False
            ultimate_reason = None
            receive_age = (trtime - frame.received_ns) / 1e9
            if len(players) == 1 and receive_age < 0.15:
                scale = max(frame.image.shape[:2])
                bounds = tuple(
                    config["bounds"]
                    or (0, 0, stream.width / scale, stream.height / scale)
                )
                world = World(
                    players[0],
                    [h for h in tracks if h.kind == "hazard"],
                    [i for i in tracks if i.kind == "item"],
                    bounds,
                    frame.time,
                    health=supervisor.latest.hearts / 3
                    if supervisor
                    and supervisor.latest
                    and supervisor.latest.hearts is not None
                    else None,
                    targets=[
                        t
                        for t in tracks
                        if t.kind == "target" and frame.time - t.last_seen < 0.15
                    ],
                    walls=[scene.wall] if scene.wall else [],
                )
                walls = [
                    t
                    for t in tracks
                    if t.kind == "wall" and frame.time - t.last_seen < 0.15
                ]
                if scene.wall and len(walls) == 1:
                    scene.wall.velocity = np.array([walls[0].velocity[0], 0.0])
                decision = planner.choose(world, receive_age)
                action = decision.action
                if policy:
                    proposal = int(policy.predict(encode(world), deterministic=True)[0])
                    action, override = planner.shield(world, proposal, receive_age)
                if combat_config.get("ultimate_enabled"):
                    ultimate_reason = ultimate.choose(world, scene, time.monotonic())
            else:
                ultimate.previous, ultimate.stable = None, 0
            ptime = time.perf_counter_ns()
            if supervisor and (
                not supervisor.latest
                or supervisor.latest.hearts is None
                or supervisor.latest.hearts <= 1
            ):
                action = 0
            if joystick and correct_geometry:
                if supervisor:
                    supervisor.check()
                if (
                    frame.image.shape[1] != config["joystick"]["frame_size"][0]
                    or frame.image.shape[0] != config["joystick"]["frame_size"][1]
                ):
                    joystick.release()
                    raise ValueError("Orientation changed; control stopped")
                try:
                    if ultimate_reason:
                        joystick.release()
                        x, y = map(int, scene.button)
                        stream.control.sendall(
                            touch_packet(
                                0, x, y, stream.width, stream.height, pointer=2
                            )
                        )
                        time.sleep(0.025)
                        stream.control.sendall(
                            touch_packet(
                                1, x, y, stream.width, stream.height, pointer=2
                            )
                        )
                        ultimate.fired(scene.charges, time.monotonic())
                        ultimate_sent = True
                        print(
                            json.dumps(
                                {
                                    "type": "ultimate_tap",
                                    "episode": episodes.episode,
                                    "charges_before": scene.charges,
                                    "reason": ultimate_reason,
                                }
                            ),
                            flush=True,
                        )
                    elif motion.get("kind") == "drag":
                        if action == 0:
                            joystick.release()
                        elif time.monotonic() - last_action_at >= motion.get(
                            "action_period_s", 0.22
                        ):
                            last_action_at = time.monotonic()
                            direction, gesture_fraction = limited_drag(
                                world,
                                action,
                                planner.action_velocities,
                                motion.get("action_period_s", 0.22),
                            )
                            if gesture_fraction >= 0.08:
                                gesture_started_ns = time.perf_counter_ns()
                                joystick.drag(
                                    direction,
                                    duration=motion.get("stroke_duration_s", 0.08),
                                    steps=motion.get("stroke_steps", 4),
                                )
                                gesture_sent = True
                    else:
                        joystick.move(DIRECTIONS[action])
                except (RuntimeError, OSError):
                    if supervisor and supervisor.exiting.is_set():
                        raise HealthAbort("Health guard disarmed joystick")
                    if repeat:
                        try:
                            joystick.close()
                        except OSError:
                            pass
                        joystick = None
                        stream.close()
                        stream = None
                        setup_needed = True
                        for event in episodes.interrupt("input_disconnect"):
                            lifecycle.write(json.dumps(event) + "\n")
                        lifecycle.flush()
                        reconnects += 1
                        continue
                    raise
            dispatch = time.perf_counter_ns()
            log.event(
                {
                    "sequence": seq,
                    "episode": episodes.episode if in_game else None,
                    "attempt_id": attempt_id,
                    "attempt_verified": attempt_verified,
                    "page": page.page,
                    "observation": encode(world).tolist() if world else None,
                    "player_detection": next(
                        (d.position.tolist() for d in detections if d.kind == "player"),
                        None,
                    ),
                    "device_pts_us": frame.device_pts_us,
                    "received_ns": frame.received_ns,
                    "decision_ns": ptime,
                    "dispatch_ns": dispatch,
                    "action": action,
                    "control_enabled": control,
                    "gesture_sent": gesture_sent,
                    "gesture_started_ns": gesture_started_ns,
                    "gesture_fraction": gesture_fraction,
                    "ultimate_sent": ultimate_sent,
                    "ultimate_charges": scene.charges,
                    "ultimate_reason": ultimate_reason,
                    "combat_objective": decision.objective if decision else None,
                    "boss_wall": list(scene.wall.bounds) if scene.wall else None,
                    "item_count": sum(t.kind == "item" for t in tracks),
                    "hearts": supervisor.latest.hearts
                    if supervisor and supervisor.latest
                    else None,
                    "shield_override": override,
                    "dropped": dropped,
                    "player_valid": len(players) == 1,
                    "player_track_id": players[0].id if len(players) == 1 else None,
                    "hazard_count": sum(t.kind == "hazard" for t in tracks),
                    "latency_ms": {
                        "decode": (frame.decoded_ns - frame.received_ns) / 1e6,
                        "detection": (dtime - t) / 1e6,
                        "tracking": (trtime - dtime) / 1e6,
                        "planning": (ptime - trtime) / 1e6,
                        "dispatch": (dispatch - ptime) / 1e6,
                        "receive_to_dispatch": (dispatch - frame.received_ns) / 1e6,
                    },
                    "tracks": [
                        {
                            "id": t.id,
                            "kind": t.kind,
                            "label": t.label,
                            "state": t.state.tolist(),
                            "radius": float(t.radius),
                            "last_seen": t.last_seen,
                            "confidence": t.confidence,
                            "covariance": t.covariance.tolist(),
                        }
                        for t in tracks
                    ],
                }
            )
            if collector and in_game:
                collector.offer(frame)
            if preview:
                image = draw_preview(
                    frame.image,
                    tracks,
                    decision,
                    status=f"{'CONTROL' if control else 'SHADOW'}  action {action}  age {receive_age * 1000:.0f} ms  q=stop",
                    settings=config.get("preview"),
                    wall=scene.wall,
                )
                cv2.imshow("Starfall agent", image)
                if cv2.waitKey(1) & 255 == ord("q"):
                    reason = "manual_stop"
                    break
    except HealthAbort:
        reason = "health_protection"
        if supervisor:
            supervisor.finished.wait(3.5)
    except KeyboardInterrupt:
        reason = "interrupted"
    except BaseException as exc:
        reason = f"error: {exc}"
        raise
    finally:
        try:
            if supervisor and not supervisor.exiting.is_set() and stream:
                last = stream.latest
                if last:
                    reading = supervisor.reader.read(last.image)
                    if reading.gameplay and reading.return_button:
                        supervisor.return_to_lobby(last, reading, "session_end")
        finally:
            try:
                if supervisor:
                    supervisor.close()
                if joystick:
                    joystick.close()
            finally:
                if stream:
                    stream.close()
                for event in episodes.interrupt(reason):
                    lifecycle.write(json.dumps(event) + "\n")
                lifecycle.close()
                cv2.destroyAllWindows()
                summary = log.finish(
                    reason,
                    reconnects=reconnects,
                    episodes_started=episodes.episode,
                    episodes_ended=episodes.ended,
                    challenge_taps=episodes.starts,
                    health_exit=supervisor.report if supervisor else None,
                )
                if lease:
                    lease.close()
    return log.path, summary

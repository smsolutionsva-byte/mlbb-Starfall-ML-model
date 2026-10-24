import cv2
import pytest
from pathlib import Path
from starfall.session import EpisodeLoop, PageReader, PageReading
from starfall.runtime import load_config

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_restart_once_then_repeat_after_human_exit():
    loop = EpisodeLoop(auto_start=True)
    ready = PageReading("ready", (820, 348))
    assert not loop.update(ready, 0)[1]
    assert loop.update(ready, 0.4)[1]
    assert not loop.update(ready, 4)[1]  # Failed starts never repeatedly spend tickets.
    active, _, events = loop.update(PageReading("gameplay"), 5)
    assert active and events[0]["episode"] == 1
    assert not loop.update(PageReading("main_lobby"), 6)[0]
    _, _, events = loop.update(PageReading("main_lobby"), 6.4)
    assert events[0]["reason"] == "human_reset"
    assert events[0]["truncated"] and not events[0]["terminated"]
    loop.update(ready, 7)
    assert loop.update(ready, 7.4)[1]
    assert loop.update(PageReading("gameplay"), 8)[2][0]["episode"] == 2


def test_unknown_page_suspends_but_brief_occlusion_does_not_end_episode():
    loop = EpisodeLoop(auto_start=True)
    loop.update(PageReading("gameplay"), 0)
    assert loop.update(PageReading("unknown"), 1) == (False, False, [])
    assert loop.update(PageReading("gameplay"), 1.3)[2] == []
    assert loop.episode == 1
    assert loop.update(PageReading("unknown"), 20) == (False, False, [])
    assert loop.update(PageReading("unknown"), 30) == (False, False, [])
    assert loop.update(PageReading("gameplay"), 31) == (True, False, [])
    assert not loop.update(PageReading("ready_unavailable"), 2)[1]
    assert not loop.update(PageReading("main_lobby"), 3)[1]


def test_actual_ready_frame_with_return_label_is_never_gameplay():
    reader = PageReader(load_config("configs/phone.yaml")["session"])
    for path in (FIXTURES / "ready.png", FIXTURES / "ready-with-return.png"):
        image = cv2.imread(str(path))
        reading = reader.read(image)
        assert reading.page == "ready"
        assert 800 < reading.challenge[0] < 900
    for path in (FIXTURES / "lobby").glob("*.png"):
        assert reader.read(cv2.imread(str(path))).page == "main_lobby"


def test_normal_control_does_not_construct_heart_reader(monkeypatch):
    from starfall import safety

    def forbidden(*args, **kwargs):
        raise AssertionError("Heart scanning is forbidden in the human-reset loop")

    monkeypatch.setattr(safety, "HeartReader", forbidden)
    reader = PageReader(load_config("configs/phone.yaml")["session"])
    assert reader.read(cv2.imread(str(FIXTURES / "ready.png"))).page == "ready"


def test_page_transition_with_return_button_does_not_start_false_episode():
    path = FIXTURES / "transition.jpg"
    reader = PageReader(load_config("configs/phone.yaml")["session"])
    assert reader.read(cv2.imread(str(path))).page != "gameplay"


def test_pause_suspends_controls_without_inferring_a_death_or_new_attempt():
    reader = PageReader(load_config("configs/phone.yaml")["session"])
    image = cv2.imread(str(FIXTURES / "paused.jpg"))
    assert reader.read(image).page == "paused"
    loop = EpisodeLoop(auto_start=True)
    loop.update(PageReading("gameplay"), 0)
    assert loop.update(PageReading("paused"), 1) == (False, False, [])
    assert loop.update(PageReading("paused"), 10) == (False, False, [])
    assert loop.update(PageReading("gameplay"), 11) == (True, False, [])
    assert loop.episode == 1


def test_candidate_preview_recognizes_gameplay_without_return_label():
    reader = PageReader(load_config("configs/phone-detector-1988.yaml")["session"])
    for name in ("gameplay.png", "gameplay-no-return.png"):
        assert reader.read(cv2.imread(str(FIXTURES / name))).page == "gameplay"
    for name in ("paused.jpg", "paused-no-return.png"):
        assert reader.read(cv2.imread(str(FIXTURES / name))).page == "paused"
    for name in ("ready.png", "ready-with-return.png"):
        assert reader.read(cv2.imread(str(FIXTURES / name))).page == "ready"
    for path in (FIXTURES / "lobby").glob("*.png"):
        assert reader.read(cv2.imread(str(path))).page == "main_lobby"
    assert reader.read(cv2.imread(str(FIXTURES / "transition.jpg"))).page != "gameplay"


@pytest.mark.parametrize(
    "frame_name, expected_page, preview",
    [
        ("gameplay-no-return.png", "gameplay", False),
        ("gameplay-hud-occluded.png", "unknown", True),
    ],
)
def test_preview_logs_numpy_detection_radii(
    monkeypatch, tmp_path, frame_name, expected_page, preview
):
    import json
    import time
    import numpy as np
    from starfall import runtime
    from starfall.android import Frame
    from starfall.logging import RunLog
    from starfall.types import Detection

    image = cv2.imread(str(FIXTURES / frame_name))

    class Detector:
        def detect(self, _):
            return [Detection("player", np.array([0.25, 0.25]), np.float32(0.03))]

    class Stream:
        width, height = 960, 430
        latest = None
        count = 0
        closed = False

        def get_frame(self, after=0, timeout=1):
            if self.count == 2:
                raise KeyboardInterrupt
            self.count += 1
            now = time.perf_counter_ns()
            return Frame(image, self.count, self.count * 17000, now, now)

        def close(self):
            self.closed = True

    monkeypatch.setattr(runtime, "create_detector", lambda _: Detector())
    monkeypatch.setattr(runtime.cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(runtime.cv2, "waitKey", lambda _: -1)
    monkeypatch.setattr(
        runtime, "RunLog", lambda *a, **kw: RunLog(*a, **kw, root=tmp_path)
    )
    stream = Stream()
    path, report = runtime.run_live(
        "configs/phone-detector-1988.yaml", preview=preview, provided_stream=stream
    )
    rows = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
    assert stream.closed and report["processed_frames"] == 2
    assert all(row["page"] == expected_page and row["tracks"] for row in rows)
    assert all(isinstance(row["tracks"][0]["radius"], float) for row in rows)
    assert all(not row["gesture_sent"] and not row["control_enabled"] for row in rows)


def test_control_loop_restarts_without_hearts_or_lobby_exit_taps(monkeypatch, tmp_path):
    import time
    from starfall.android import Frame
    from starfall import runtime
    from starfall.logging import RunLog
    from starfall.control_lease import ControlLease

    config = load_config("configs/phone.yaml")
    config["session"]["learn_response"] = False
    ready = cv2.imread(str(FIXTURES / "ready.png"))
    game = cv2.imread(str(FIXTURES / "gameplay.png"))
    lobby_path = FIXTURES / "lobby/main.png"
    lobby = cv2.imread(str(lobby_path))
    frames = [
        ready,
        ready,
        game,
        game,
        lobby,
        lobby,
        ready,
        ready,
        game,
        game,
        lobby,
        lobby,
    ]
    packets = []

    class Sink:
        def sendall(self, packet):
            packets.append(packet)

    class Stream:
        width, height = 960, 430
        control = Sink()
        latest = None
        closed = False
        i = 0

        def get_frame(self, after=0, timeout=1):
            self.i += 1
            image = frames[self.i - 1]
            now = time.perf_counter_ns()
            self.latest = Frame(image, self.i, self.i * 10000, now, now)
            return self.latest

        def close(self):
            self.closed = True

    def forbidden(*args, **kwargs):
        raise AssertionError("Human-reset control must never scan hearts")

    monkeypatch.setattr(runtime, "HealthSupervisor", forbidden)
    monkeypatch.setattr(
        runtime, "ControlLease", lambda: ControlLease(tmp_path / "fake-control.lock")
    )
    monkeypatch.setattr(runtime, "load_config", lambda _: config)
    monkeypatch.setattr(
        runtime,
        "RunLog",
        lambda *args, **kwargs: RunLog(*args, **kwargs, root=tmp_path),
    )
    monkeypatch.setattr(
        runtime, "EpisodeLoop", lambda auto_start: EpisodeLoop(auto_start, stable_s=0)
    )
    stream = Stream()
    _, report = runtime.run_live(
        "unused",
        seconds=5,
        control=True,
        preview=False,
        repeat=True,
        max_episodes=2,
        provided_stream=stream,
    )
    assert stream.closed
    assert report["episodes_started"] == 2 and report["challenge_taps"] == 2
    assert report["health_exit"] is None
    # All UI-pointer taps belong to Challenge; none target Return to Lobby.
    import struct

    ui = [p for p in packets if struct.unpack(">Q", p[2:10])[0] == 1]
    assert len(ui) == 4
    assert all(struct.unpack(">i", p[10:14])[0] > 800 for p in ui)

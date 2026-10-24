"""Recognize game pages and restart human-interrupted attempts; no health scans."""

from dataclasses import dataclass
from .safety import find_template, load_template


@dataclass
class PageReading:
    page: str
    challenge: tuple | None = None


class PageReader:
    def __init__(self, config):
        self.reference = tuple(config["frame_size"])
        self.title = load_template(config["ready_template"])
        self.challenge = load_template(config["challenge_template"])
        self.game = load_template(config["game_template"])
        self.game_roi = tuple(
            config.get(
                "game_template_roi",
                (self.reference[0] // 3, 0, self.reference[0] * 2 // 3, 50),
            )
        )
        self.points = load_template(config["points_template"])
        self.paused = load_template(config["paused_template"])
        self.main = load_template(config["main_lobby_template"])

    def read(self, image):
        h, w = image.shape[:2]
        if (w, h) != self.reference:
            raise ValueError("Phone orientation/resolution changed; control stopped")
        if find_template(
            image, self.paused, (w // 3, h // 3, w * 2 // 3, h * 2 // 3), 0.85
        ):
            return PageReading("paused")
        title = find_template(image, self.title, (w // 2, 30, w, 120), 0.78)
        challenge = find_template(
            image, self.challenge, (w * 3 // 4, h * 2 // 3, w, h), 0.82
        )
        # The ready page may itself contain Return to Lobby. Title + button win.
        if title and challenge:
            return PageReading("ready", tuple(challenge[0][:2]))
        if title:
            return PageReading("ready_unavailable")
        if find_template(image, self.main, (w // 5, h * 2 // 3, w * 3 // 5, h), 0.85):
            return PageReading("main_lobby")
        if find_template(image, self.game, self.game_roi, 0.78) and find_template(
            image, self.points, (w * 2 // 3, 10, w, 75), 0.80
        ):
            return PageReading("gameplay")
        return PageReading("unknown")


class EpisodeLoop:
    """Stable pages trigger one start tap; departures suspend movement immediately."""

    def __init__(self, auto_start=False, stable_s=0.35):
        self.auto_start, self.stable_s = auto_start, stable_s
        self.page = None
        self.page_since = None
        self.active = False
        self.episode = 0
        self.starts = 0
        self.ended = 0
        self.start_latched = False

    def interrupt(self, reason):
        events = []
        if self.active:
            self.ended += 1
            events.append(
                {
                    "type": "episode_end",
                    "episode": self.episode,
                    "reason": reason,
                    "terminated": False,
                    "truncated": True,
                    "score": None,
                }
            )
        self.active = False
        self.page = self.page_since = None
        self.start_latched = False
        return events

    def update(self, reading, now):
        if reading.page != self.page:
            self.page, self.page_since = reading.page, now
        stable = now - self.page_since >= self.stable_s
        events = []
        if reading.page == "gameplay":
            if not self.active:
                self.active = True
                self.episode += 1
                self.start_latched = False
                events.append({"type": "episode_start", "episode": self.episode})
            return True, False, events
        # Never send joystick movements on a loading, lobby or unknown page.
        if (
            self.active
            and stable
            and reading.page in ("ready", "ready_unavailable", "main_lobby")
        ):
            self.active = False
            self.ended += 1
            events.append(
                {
                    "type": "episode_end",
                    "episode": self.episode,
                    "reason": "human_reset"
                    if reading.page in ("ready", "main_lobby")
                    else "page_change",
                    "terminated": False,
                    "truncated": True,
                    "score": None,
                }
            )
        if reading.page == "main_lobby" and stable:
            self.start_latched = False
        tap = (
            self.auto_start
            and reading.page == "ready"
            and stable
            and not self.active
            and not self.start_latched
        )
        if tap:
            self.start_latched = True
            self.starts += 1
            events.append({"type": "challenge_tap", "episode": self.episode + 1})
        return False, tap, events

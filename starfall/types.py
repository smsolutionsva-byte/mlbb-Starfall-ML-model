from dataclasses import dataclass, field
import numpy as np

# Euclidean coordinates: divide BOTH axes by max(frame width, frame height).
DIRECTIONS = np.array(
    [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)],
    dtype=np.float64,
)
DIRECTIONS[5:] /= np.sqrt(2)


@dataclass
class Detection:
    kind: str
    position: np.ndarray
    radius: float
    confidence: float = 1.0
    label: str = ""
    box_size: tuple[float, float] | None = None


@dataclass
class Track:
    id: int
    kind: str
    state: np.ndarray
    covariance: np.ndarray
    radius: float
    confidence: float
    label: str
    last_seen: float
    time: float
    hits: int = 1
    box_size: tuple[float, float] | None = None

    @property
    def position(self):
        return self.state[:2]

    @property
    def velocity(self):
        return self.state[2:]


@dataclass
class Wall:
    bounds: tuple[float, float, float, float]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))


@dataclass
class World:
    player: Track
    hazards: list[Track]
    items: list[Track]
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    time: float = 0.0
    health: float | None = None
    score: float | None = None
    shield_until: float | None = None
    effects: dict = field(default_factory=dict)
    targets: list[Track] = field(default_factory=list)
    walls: list[Wall] = field(default_factory=list)


def make_track(identity, kind, x, y, vx=0, vy=0, radius=0.025, label="", time=0):
    return Track(
        identity,
        kind,
        np.array([x, y, vx, vy], dtype=float),
        np.eye(4) * 1e-6,
        radius,
        1.0,
        label or kind,
        time,
        time,
    )

"""Timestamped constant-velocity Kalman filter and gated Hungarian association."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from .types import Track


class Tracker:
    def __init__(
        self, ttl=0.25, gate=0.10, process_noise=0.25, measurement_noise=0.000025
    ):
        self.ttl, self.gate = ttl, gate
        self.q, self.r = process_noise, measurement_noise
        self.tracks = {}
        self.next_id = 1
        self.time = None

    def reset(self):
        self.tracks.clear()
        self.time = None

    def update(self, detections, timestamp):
        if self.time is not None and timestamp <= self.time:
            raise ValueError("Tracking timestamps must increase monotonically")
        self.time = timestamp
        self.tracks = {
            i: t for i, t in self.tracks.items() if timestamp - t.last_seen <= self.ttl
        }
        tracks = list(self.tracks.values())
        for t in tracks:
            dt = timestamp - t.time
            f = np.eye(4)
            f[0, 2] = f[1, 3] = dt
            g = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
            t.state = f @ t.state
            t.covariance = f @ t.covariance @ f.T + self.q * g @ g.T
            t.time = timestamp
        used = set()
        if tracks and detections:
            costs = np.full((len(tracks), len(detections)), 1e6)
            for i, t in enumerate(tracks):
                for j, d in enumerate(detections):
                    if t.kind != d.kind or (t.kind == "item" and t.label != d.label):
                        continue
                    error = d.position - t.position
                    uncertainty = np.sqrt(np.trace(t.covariance[:2, :2]))
                    if np.linalg.norm(error) <= self.gate + 3 * uncertainty:
                        s = t.covariance[:2, :2] + np.eye(2) * self.r
                        costs[i, j] = float(error @ np.linalg.solve(s, error))
            rows, cols = linear_sum_assignment(costs)
            for i, j in zip(rows, cols):
                if costs[i, j] >= 1e6:
                    continue
                t, d = tracks[i], detections[j]
                h = np.eye(4)[:2]
                s = h @ t.covariance @ h.T + np.eye(2) * self.r
                k = np.linalg.solve(s, h @ t.covariance).T
                t.state += k @ (d.position - h @ t.state)
                # Joseph form keeps covariance positive semidefinite.
                a = np.eye(4) - k @ h
                t.covariance = a @ t.covariance @ a.T + k @ (np.eye(2) * self.r) @ k.T
                t.radius, t.confidence, t.label = d.radius, d.confidence, d.label
                t.box_size = d.box_size
                t.last_seen = timestamp
                t.hits += 1
                used.add(j)
        for j, d in enumerate(detections):
            if j in used:
                continue
            t = Track(
                self.next_id,
                d.kind,
                np.r_[d.position, 0.0, 0.0],
                np.diag([self.r, self.r, 0.25, 0.25]),
                d.radius,
                d.confidence,
                d.label,
                timestamp,
                timestamp,
                box_size=d.box_size,
            )
            self.tracks[t.id] = t
            self.next_id += 1
        return list(self.tracks.values())

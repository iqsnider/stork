"""Ideal spherical range scans and an accumulated map in local world coordinates."""

import numpy as np
from scipy.spatial import cKDTree


class RangeLidar:
    """Simulator-side sensor. Points inside the radius are visible, even through walls.

    The full scene stays here; control/planning code should use ``known_points``.
    No angular sampling, measurement noise, or occlusion is simulated.
    """

    scan_period = 0.1  # 10 Hz in simulation time.

    def __init__(self, points: np.ndarray, radius: float = 20.0) -> None:
        self._points = np.asarray(points, dtype=float)
        self._tree = cKDTree(self._points)
        self.radius = radius
        self.reset()

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        if not np.isfinite(value) or value <= 0:
            raise ValueError("LiDAR radius must be positive and finite.")
        self._radius = float(value)

    def reset(self) -> None:
        self.known_mask = np.zeros(len(self._points), dtype=bool)
        self.known_points = np.empty((0, 3), dtype=float)
        self.in_range_count = 0
        self.scan_count = 0
        self.last_scan_time = 0.0
        self.revision = 0

    def update(self, time: float, position: np.ndarray) -> None:
        """Scan at t=0 and each 0.1 s, accumulating all points ever in range."""
        if time + 1e-10 < self.scan_count * self.scan_period:
            return
        indices = self._tree.query_ball_point(position, self.radius, return_sorted=False)
        indices = np.asarray(indices, dtype=np.intp)
        self.in_range_count = len(indices)
        newly_seen = indices[~self.known_mask[indices]]
        if len(newly_seen):
            self.known_mask[newly_seen] = True
            self.known_points = self._points[self.known_mask]
            self.revision += 1
        self.last_scan_time = time
        self.scan_count += 1

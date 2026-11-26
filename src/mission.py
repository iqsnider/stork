"""Terrain-relative endpoint selection, separate from the robot's sensed map."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .config import TETHER_LEN


# Raised area to excavation floor, in the loader's local XY frame [m].
DEFAULT_START_XY = (16.0, -12.0)
DEFAULT_GOAL_XY = (4.0, 12.0)


class TerrainSurface:
    """Use the highest scene sample within 1 m in XY as a surface reference.

    This is a placement heuristic, not ground segmentation or a collision map.
    It uses simulator ground truth; it is not passed to the drone controller.
    """

    def __init__(self, points: np.ndarray) -> None:
        self.points = np.asarray(points, dtype=float)
        self.tree = cKDTree(self.points[:, :2])

    def ground_at(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("Choose a finite XY location.")
        indices = self.tree.query_ball_point(xy, r=1.0)
        if not indices:
            raise ValueError("No terrain samples within 1 m of this XY location; choose a point on the map.")
        return np.array([*xy, self.points[indices, 2].max()])

    def pick(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Select the frontmost sample within 0.35 m of the forward click ray."""
        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("The picking ray is invalid.")
        direction = direction / norm
        relative = self.points - np.asarray(origin, dtype=float)
        along = relative @ direction
        distance_squared = np.einsum("ij,ij->i", relative, relative) - along**2
        candidates = np.flatnonzero((along >= 0) & (distance_squared <= 0.35**2))
        if not len(candidates):
            raise ValueError("No point under the cursor. Click the terrain, or use the top-down view.")
        index = candidates[np.argmin(along[candidates])]
        return self.ground_at(self.points[index, :2])


@dataclass
class MissionEndpoints:
    start_ground: np.ndarray
    goal_ground: np.ndarray
    clearance: float = 3.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.clearance) or self.clearance <= 0.4:
            raise ValueError("Payload-centre clearance must exceed 0.4 m.")

    @property
    def start(self) -> np.ndarray:
        return self.start_ground + [0, 0, TETHER_LEN + self.clearance]

    @property
    def goal(self) -> np.ndarray:
        return self.goal_ground + [0, 0, TETHER_LEN + self.clearance]

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.goal - self.start))

    def command(self) -> str:
        """CLI arguments that reproduce the selected XY locations and offset."""
        return (
            f"--start-xy {self.start_ground[0]:.6f} {self.start_ground[1]:.6f} "
            f"--goal-xy {self.goal_ground[0]:.6f} {self.goal_ground[1]:.6f} "
            f"--clearance {self.clearance:g}"
        )

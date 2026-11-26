"""Load PLY vertex x/y/z; place the simulation in an explicitly local frame."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData

from . import config


DEFAULT_CLOUD = Path(__file__).resolve().parents[1] / "data/excavation/excavation.ply"


@dataclass
class ExcavationScene:
    points: np.ndarray
    origin: np.ndarray
    target: np.ndarray
    source_count: int


def load_excavation(path=DEFAULT_CLOUD, max_points=500_000, clearance=3.0) -> ExcavationScene:
    """Assume metres and +Z up. Clearance is payload-centre height above terrain.

    XY origin is the full cloud's bounding-box midpoint. Z origin is the
    highest sample within 3 m horizontally of it (a local surface reference,
    not a ground segmentation). The viewer and range sensor share the subsampled
    scene; the origin is calculated from the full-resolution cloud.
    """
    if max_points < 1 or not np.isfinite(clearance) or clearance <= 0.4:
        raise ValueError("max_points must be positive; clearance must exceed 0.4 m.")
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing {path}. Run: uv run python download_data.py")
    vertices = PlyData.read(path, mmap="r")["vertex"].data
    if len(vertices) == 0:
        raise ValueError("The point cloud is empty.")
    bounds = np.array([[vertices[k].min(), vertices[k].max()] for k in ("x", "y", "z")])
    if not np.isfinite(bounds).all():
        raise ValueError("The point cloud contains non-finite coordinates.")
    center = bounds[:2].mean(axis=1)
    local = (vertices["x"] - center[0])**2 + (vertices["y"] - center[1])**2 <= 3.0**2
    if not local.any():
        raise ValueError("No terrain samples within 3 m of the cloud centre.")
    origin = np.array([*center, vertices["z"][local].max()], dtype=float)
    stride = max(1, int(np.ceil(len(vertices) / max_points)))
    points = np.column_stack([vertices[k][::stride] for k in ("x", "y", "z")])
    points = points.astype(np.float64) - origin
    target = np.array([0.0, 0.0, config.TETHER_LEN + clearance])
    return ExcavationScene(points, origin, target, len(vertices))

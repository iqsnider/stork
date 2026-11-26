import unittest

import numpy as np

from src.config import TETHER_LEN
from src.mission import MissionEndpoints, TerrainSurface


class MissionTests(unittest.TestCase):
    def test_surface_height_and_missing_terrain(self) -> None:
        surface = TerrainSurface(np.array([[0, 0, 2], [0.5, 0, 3], [4, 0, 8]]))
        np.testing.assert_allclose(surface.ground_at([0, 0]), [0, 0, 3])
        with self.assertRaises(ValueError):
            surface.ground_at([20, 20])
        with self.assertRaises(ValueError):
            surface.ground_at([float("nan"), 0])

    def test_click_selects_frontmost_surface_not_points_behind_camera(self) -> None:
        surface = TerrainSurface(np.array([[2, 0, 1], [5, 0, 1], [-2, 0, 1], [1, 2, 1]]))
        np.testing.assert_allclose(surface.pick([0, 0, 1], [2, 0, 0]), [2, 0, 1])
        with self.assertRaises(ValueError):
            surface.pick([0, 0, 1], [0, 0, 1])
        with self.assertRaises(ValueError):
            surface.pick([0, 0, 1], [0, 0, 0])

    def test_each_endpoint_uses_its_own_ground_height(self) -> None:
        mission = MissionEndpoints(np.array([16, -12, 1.2]), np.array([4, 12, -2.2]), clearance=3)
        np.testing.assert_allclose(mission.start, [16, -12, 13.8])
        np.testing.assert_allclose(mission.goal, [4, 12, 10.4])
        self.assertAlmostEqual(mission.start[2] - TETHER_LEN - mission.start_ground[2], 3)
        self.assertAlmostEqual(mission.goal[2] - TETHER_LEN - mission.goal_ground[2], 3)
        self.assertGreater(mission.distance, 26)
        self.assertIn("--start-xy 16.000000 -12.000000", mission.command())
        for clearance in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                MissionEndpoints(mission.start_ground, mission.goal_ground, clearance)


if __name__ == "__main__":
    unittest.main()

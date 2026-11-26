import unittest

import numpy as np

from src.hover_simulation import HoverSimulation
from src.perception import RangeLidar


class PerceptionTests(unittest.TestCase):
    def test_spherical_boundary_and_accumulated_knowledge(self) -> None:
        points = np.array([[0, 0, 0], [3, 4, 0], [0, 0, 5], [4, 4, 0], [20, 0, 0]])
        lidar = RangeLidar(points, radius=5)
        lidar.update(0, np.zeros(3))
        np.testing.assert_array_equal(lidar.known_mask, [True, True, True, False, False])
        self.assertEqual(lidar.in_range_count, 3)
        lidar.update(0.1, np.array([20, 0, 0]))
        self.assertEqual(lidar.in_range_count, 1)
        np.testing.assert_array_equal(lidar.known_points, points[[0, 1, 2, 4]])
        lidar.radius = 1
        lidar.update(0.2, np.array([100, 0, 0]))
        self.assertEqual(lidar.in_range_count, 0)
        self.assertEqual(len(lidar.known_points), 4)

    def test_scan_clock_and_reset_in_simulation(self) -> None:
        lidar = RangeLidar(np.array([[0, 0, 12.6], [3, 0, 12.6]]), radius=1)
        sim = HoverSimulation([0, 0, 12.6], lidar=lidar)
        self.assertEqual(lidar.scan_count, 1)  # Initial scan at t=0.
        self.assertEqual(len(sim.known_points), 1)
        lidar.radius = 5
        for _ in range(19):
            sim.step()
        self.assertEqual(len(sim.known_points), 1)
        self.assertEqual(lidar.scan_count, 1)
        sim.step()
        self.assertEqual(len(sim.known_points), 2)
        self.assertEqual(lidar.scan_count, 2)
        for _ in range(180):
            sim.step()
        self.assertEqual(lidar.scan_count, 11)  # t=0, 0.1, ..., 1.0.
        self.assertAlmostEqual(lidar.last_scan_time, 1.0)
        lidar.radius = 1
        sim.reset()
        self.assertEqual(lidar.scan_count, 1)
        self.assertEqual(len(sim.known_points), 1)
        np.testing.assert_array_equal(lidar.known_mask, [True, False])

    def test_empty_scene_and_invalid_radius(self) -> None:
        lidar = RangeLidar(np.empty((0, 3)), radius=2)
        lidar.update(0, np.zeros(3))
        self.assertEqual(lidar.known_points.shape, (0, 3))
        for radius in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                lidar.radius = radius


if __name__ == "__main__":
    unittest.main()

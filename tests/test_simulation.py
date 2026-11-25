import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from src import config
from src.excavation_scene import load_excavation
from src.hover_simulation import HoverSimulation
from src.nonlinear_model import nonlinear_ode, payload_state


class DynamicsTests(unittest.TestCase):
    def test_total_weight_is_hover_equilibrium(self) -> None:
        sim = HoverSimulation([0, 0, 12.6])
        self.assertAlmostEqual(sim.control[0], config.MASS_TOTAL * config.GRAVITY)
        np.testing.assert_allclose(nonlinear_ode(sim.state, sim.control), 0, atol=1e-12)
        for _ in range(2000):
            sim.step()
        self.assertLess(sim.position_error, 1e-10)

    def test_small_disturbance_returns_to_hover(self) -> None:
        sim = HoverSimulation([0, 0, 12.6])
        sim.nudge()
        max_error = 0
        for _ in range(6000):
            sim.step()
            max_error = max(max_error, sim.position_error)
        self.assertGreater(max_error, 0.1)
        self.assertLess(max_error, 0.5)
        self.assertLess(sim.position_error, 0.02)
        self.assertLess(np.linalg.norm(sim.state[3:6]), 0.02)
        self.assertAlmostEqual(np.linalg.norm(sim.payload_position - sim.state[:3]), config.TETHER_LEN)
        sim.reset()
        self.assertEqual(sim.time, 0)
        np.testing.assert_allclose(sim.state[:3], sim.target)
        np.testing.assert_allclose(sim.state[3:], 0)

    def test_payload_kinematics_with_swing(self) -> None:
        state = np.zeros((1, 16))
        state[0, :6] = [1, 2, 12, 0.2, -0.1, 0.3]
        state[0, 12:] = [0.15, -0.12, 0.07, 0.04]
        position, velocity = payload_state(state)
        self.assertAlmostEqual(np.linalg.norm(position[0] - state[0, :3]), config.TETHER_LEN)
        future = state.copy()
        dt = 1e-6
        future[:, :3] += state[:, 3:6] * dt
        future[:, 12:14] += state[:, 14:16] * dt
        numerical_velocity = (payload_state(future)[0] - position) / dt
        np.testing.assert_allclose(numerical_velocity, velocity, atol=1e-6)


class SceneTests(unittest.TestCase):
    def test_local_origin_and_display_limit(self) -> None:
        vertices = np.array([
            (6, 16, 2), (14, 16, 2), (6, 24, 2), (14, 24, 2),
            (10, 20, 4), (11, 20, 5),
        ], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.ply"
            PlyData([PlyElement.describe(vertices, "vertex")]).write(path)
            scene = load_excavation(path, max_points=3, clearance=3)
        np.testing.assert_allclose(scene.origin, [10, 20, 5])
        np.testing.assert_allclose(scene.target, [0, 0, config.TETHER_LEN + 3])
        self.assertEqual(scene.source_count, 6)
        self.assertEqual(len(scene.points), 3)
        np.testing.assert_allclose(scene.points[0], [-4, -4, -3])


if __name__ == "__main__":
    unittest.main()

"""Fixed-step integration of the existing model with a simple position hold."""

import numpy as np

from .inner_loop_controller import ArduPilotFlightController
from .nonlinear_model import nonlinear_ode, payload_state


class HoverSimulation:
    def __init__(self, target, dt=0.005) -> None:
        self.target = np.asarray(target, dtype=float).copy()
        self.dt = dt
        self.controller = ArduPilotFlightController()
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros(16)
        self.state[:3] = self.target
        self.time = 0.0
        self.control = self.compute_control()

    def compute_control(self) -> np.ndarray:
        # Naive outer-loop PD; the existing controller handles attitude/rates.
        acceleration = (
            np.array([0.6, 0.6, 1.2]) * (self.target - self.state[:3])
            - np.array([1.3, 1.3, 1.8]) * self.state[3:6]
        )
        acceleration = np.clip(acceleration, [-2, -2, -3], [2, 2, 3])
        return np.asarray(self.controller.compute_u(self.state, acceleration, yaw_s=0.0))

    def step(self) -> None:
        # Forward Euler with the command held for one control interval.
        self.control = self.compute_control()
        self.state = self.state + self.dt * nonlinear_ode(self.state, self.control)
        self.time += self.dt
        if not np.isfinite(self.state).all():
            raise FloatingPointError("Simulation state became non-finite.")

    def nudge(self) -> None:
        """Apply a small velocity impulse to the drone to exercise position hold."""
        self.state[3:6] += [0.5, -0.3, 0.2]

    @property
    def payload_position(self) -> np.ndarray:
        return payload_state(self.state[None, :])[0][0]

    @property
    def position_error(self) -> float:
        return float(np.linalg.norm(self.state[:3] - self.target))

"""Run a hover simulation in the excavation point cloud."""

import argparse
import json

import numpy as np

from src.excavation_scene import DEFAULT_CLOUD, load_excavation
from src.hover_simulation import HoverSimulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", type=str, default=str(DEFAULT_CLOUD))
    parser.add_argument("--max-points", type=int, default=500_000, help="Maximum displayed cloud points")
    parser.add_argument("--clearance", type=float, default=3.0, help="Payload-centre height above local surface [m]")
    parser.add_argument("--headless", action="store_true", help="Run dynamics without opening a viewer")
    parser.add_argument("--duration", type=float, help="Simulation seconds; headless default is 30")
    parser.add_argument("--nudge", action="store_true", help="Start with a small drone velocity disturbance")
    args = parser.parse_args()
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be positive and finite")
    try:
        environment = load_excavation(args.cloud, args.max_points, args.clearance)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    simulation = HoverSimulation(environment.target)
    if args.nudge:
        simulation.nudge()
    print(f"Excavation: {environment.source_count:,} vertices; displaying {len(environment.points):,}.")
    print(f"Local origin in PLY coordinates: {environment.origin}")
    print(f"Hover target [m]: {simulation.target}; assuming PLY units are metres and +Z is up.")
    if args.headless:
        max_error = simulation.position_error
        for _ in range(int(np.ceil((args.duration or 30.0) / simulation.dt))):
            simulation.step()
            max_error = max(max_error, simulation.position_error)
        print(json.dumps({
            "time_s": simulation.time,
            "position_m": simulation.state[:3].tolist(),
            "payload_position_m": simulation.payload_position.tolist(),
            "final_error_m": simulation.position_error,
            "max_error_m": max_error,
            "speed_m_s": float(np.linalg.norm(simulation.state[3:6])),
        }, indent=2))
    else:
        from src.viewer import run_viewer
        run_viewer(environment, simulation, args.duration)


if __name__ == "__main__":
    main()

"""Select terrain-relative endpoints and hover at the start in the excavation."""

import argparse
import json

import numpy as np

from src.excavation_scene import DEFAULT_CLOUD, load_excavation
from src.hover_simulation import HoverSimulation
from src.mission import DEFAULT_GOAL_XY, DEFAULT_START_XY, MissionEndpoints, TerrainSurface
from src.perception import RangeLidar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", type=str, default=str(DEFAULT_CLOUD))
    parser.add_argument("--max-points", type=int, default=500_000, help="Maximum scene points for the viewer and sensor")
    parser.add_argument("--clearance", type=float, default=3.0, help="Payload-centre height above local surface [m]")
    parser.add_argument("--start-xy", type=float, nargs=2, default=DEFAULT_START_XY, metavar=("X", "Y"), help="Start in local map coordinates [m]")
    parser.add_argument("--goal-xy", type=float, nargs=2, default=DEFAULT_GOAL_XY, metavar=("X", "Y"), help="Destination in local map coordinates [m]")
    parser.add_argument("--headless", action="store_true", help="Run dynamics without opening a viewer")
    parser.add_argument("--duration", type=float, help="Simulation seconds; headless default is 30")
    parser.add_argument("--nudge", action="store_true", help="Start with a small drone velocity disturbance")
    parser.add_argument("--viewer", choices=("viser", "open3d"), default="viser")
    parser.add_argument("--lidar-radius", type=float, default=20.0, help="Spherical LiDAR range [m], updated at 10 Hz")
    parser.add_argument("--port", type=int, default=8080, help="Local Viser server port")
    args = parser.parse_args()
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be positive and finite")
    try:
        environment = load_excavation(args.cloud, args.max_points, args.clearance)
        terrain = TerrainSurface(environment.points)
        mission = MissionEndpoints(terrain.ground_at(args.start_xy), terrain.ground_at(args.goal_xy), args.clearance)
        environment.target = mission.start
        lidar = RangeLidar(environment.points, args.lidar_radius)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    simulation = HoverSimulation(environment.target, lidar=lidar)
    if args.nudge:
        simulation.nudge()
    print(f"Excavation: {environment.source_count:,} vertices; scene/sensor uses {len(environment.points):,}.")
    print(f"Local origin in PLY coordinates: {environment.origin}")
    print(f"Hover target [m]: {simulation.target}; assuming PLY units are metres and +Z is up.")
    print(f"Destination [m]: {mission.goal}; endpoint separation {mission.distance:.2f} m.")
    print(f"Ground Z [m]: start {mission.start_ground[2]:.3f}, destination {mission.goal_ground[2]:.3f}.")
    print(f"Payload offset: {mission.clearance:g} m. Endpoints only; no trajectory is executed.")
    print(f"LiDAR: {lidar.radius:g} m radius, 10 Hz, accumulated knowledge; no occlusion.")
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
            "known_points": len(simulation.known_points),
            "in_range_points": lidar.in_range_count,
            "lidar_scans": lidar.scan_count,
            "last_scan_time_s": lidar.last_scan_time,
            "start_m": mission.start.tolist(),
            "goal_m": mission.goal.tolist(),
            "start_ground_m": mission.start_ground.tolist(),
            "goal_ground_m": mission.goal_ground.tolist(),
        }, indent=2))
    elif args.viewer == "open3d":
        from src.viewer import run_viewer
        run_viewer(environment, simulation, args.duration)
    else:
        from src.viser_viewer import run_viewer
        run_viewer(environment, simulation, args.duration, args.port, mission=mission, terrain=terrain)


if __name__ == "__main__":
    main()

"""Browser viewer for ground truth, accumulated range scans, and hover dynamics."""

from dataclasses import replace
from queue import SimpleQueue
import time

import numpy as np
import viser
from viser.transforms import SO3

from .excavation_scene import ExcavationScene
from .hover_simulation import HoverSimulation
from .mission import MissionEndpoints, TerrainSurface


def circle_segments(radius: float, plane: tuple[int, int]) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, 97)
    points = np.zeros((len(angles), 3))
    points[:, plane[0]] = radius * np.cos(angles)
    points[:, plane[1]] = radius * np.sin(angles)
    return np.stack((points[:-1], points[1:]), axis=1)


class ViserHoverViewer:
    def __init__(self, environment: ExcavationScene, simulation: HoverSimulation, port: int,
                 mission: MissionEndpoints, terrain: TerrainSurface) -> None:
        self.environment = environment
        self.simulation = simulation
        self.mission = mission
        self.terrain = terrain
        assert simulation.lidar is not None
        self.lidar = simulation.lidar
        self.server = viser.ViserServer(host="127.0.0.1", port=port, label="Excavation · range scans")
        self.actions: SimpleQueue[tuple[str, object]] = SimpleQueue()
        self.paused = False
        self.rendered_revision = -1
        self.server.scene.set_up_direction("+z")
        self.server.gui.configure_theme(dark_mode=True, show_share_button=False, brand_color=(30, 190, 220))
        self.build_scene()
        self.build_controls()
        self.update_mission_scene()
        self.server.on_client_connect(self.connect_client)
        self.update_scene()

    def build_scene(self) -> None:
        scene = self.server.scene
        empty = np.empty((0, 3), dtype=np.float32)
        self.unseen = scene.add_point_cloud(
            "/map/unseen", empty, (115, 125, 140), point_size=0.10, precision="float32",
        )
        self.known = scene.add_point_cloud(
            "/map/known", empty, (30, 210, 235), point_size=0.10, precision="float32",
        )
        scene.add_frame("/origin", axes_length=2, axes_radius=0.035)
        scene.add_label("/origin/label", "Local origin · XYZ [m]", position=(0, 0, -0.4))
        self.drone = scene.add_frame("/drone", show_axes=False)
        scene.add_box("/drone/body", (245, 145, 45), (0.7, 0.5, 0.25))
        for index, yaw in enumerate((np.pi / 4, -np.pi / 4)):
            scene.add_box(
                f"/drone/arm_{index}", (200, 205, 215), (2.2, 0.10, 0.10),
                wxyz=SO3.from_rpy_radians(0.0, 0.0, yaw).wxyz,
            )
        for index, (x, y) in enumerate(((-0.78, -0.78), (-0.78, 0.78), (0.78, -0.78), (0.78, 0.78))):
            scene.add_icosphere(
                f"/drone/rotor_{index}", color=(60, 105, 200), subdivisions=2,
                scale=(0.34, 0.34, 0.035), position=(x, y, 0.1),
            )
        scene.add_label("/drone/label", "Drone", position=(0, 0, 0.8))
        self.payload = scene.add_icosphere("/payload", radius=0.35, color=(245, 190, 55))
        scene.add_label("/payload/label", "Payload", position=(0, 0, -0.7))
        self.tether = scene.add_line_segments(
            "/tether", np.zeros((1, 2, 3)), (230, 230, 230), thickness=0.05,
        )
        self.target = scene.add_line_segments(
            "/target", circle_segments(1.5, (0, 1)), (80, 230, 120), thickness=0.05,
        )
        self.range_sphere = scene.add_line_segments(
            "/lidar_range", np.concatenate([circle_segments(1, plane) for plane in ((0, 1), (0, 2), (1, 2))]),
            (200, 155, 70), thickness=1, thickness_units="screen", scale=self.lidar.radius,
        )
        self.endpoint_markers = {}
        self.ground_markers = {}
        self.offset_lines = {}
        for name, color in (("start", (80, 230, 120)), ("goal", (245, 105, 190))):
            self.endpoint_markers[name] = scene.add_frame(f"/mission/{name}", show_axes=False)
            scene.add_line_segments(
                f"/mission/{name}/ring", circle_segments(1, (0, 1)), color, thickness=0.08,
            )
            scene.add_label(f"/mission/{name}/label", name.title(), position=(0, 0, 1))
            self.ground_markers[name] = scene.add_icosphere(
                f"/mission/{name}_ground", radius=0.25, color=color,
            )
            self.offset_lines[name] = scene.add_line_segments(
                f"/mission/{name}_offset", np.zeros((1, 2, 3)), color,
                thickness=1.5, thickness_units="screen",
            )
        self.route_preview = scene.add_line_segments(
            "/mission/endpoint_preview", np.zeros((1, 2, 3)), (235, 235, 235),
            thickness=2, thickness_units="screen",
        )

    def build_mission_controls(self) -> None:
        gui = self.server.gui
        with gui.add_folder("Start and destination"):
            gui.add_markdown(
                "**Green: start · Pink: destination.** Click a terrain point or edit XY. "
                "The offset applies to the **payload centre**; the drone is another 9.6 m above it."
            )
            self.start_xy_input = gui.add_vector2("Start XY [m]", tuple(self.mission.start_ground[:2]), step=0.5)
            self.goal_xy_input = gui.add_vector2("Destination XY [m]", tuple(self.mission.goal_ground[:2]), step=0.5)
            self.clearance_input = gui.add_number("Payload ground offset [m]", self.mission.clearance, min=0.5, step=0.5)
            pick_start = gui.add_button("Pick start on terrain")
            pick_goal = gui.add_button("Pick destination on terrain")
            cancel = gui.add_button("Cancel picking")
            top_down = gui.add_button("Top-down map")
            spawn = gui.add_button("Spawn at start / reset knowledge")
            self.mission_status = gui.add_markdown("")
            self.mission_message = gui.add_markdown("Edit endpoints, then **Spawn at start** to apply the start and height.")
            gui.add_markdown(
                "The dashed connector is an **unplanned preview**, not a flight path. "
                "Ground is estimated from the highest sample within 1 m in XY."
            )
            with gui.add_folder("Reproduce these endpoints", expand_by_default=False):
                self.mission_command = gui.add_markdown("")

        @self.start_xy_input.on_update
        def start_changed(event: viser.GuiEvent) -> None:
            self.actions.put(("mission_inputs", None))

        @self.goal_xy_input.on_update
        def goal_changed(event: viser.GuiEvent) -> None:
            self.actions.put(("mission_inputs", None))

        @self.clearance_input.on_update
        def clearance_changed(event: viser.GuiEvent) -> None:
            self.actions.put(("mission_inputs", None))

        @pick_start.on_click
        def start_clicked(event: viser.GuiEvent) -> None:
            if event.client is not None:
                self.begin_pick(event.client, "start")

        @pick_goal.on_click
        def goal_clicked(event: viser.GuiEvent) -> None:
            if event.client is not None:
                self.begin_pick(event.client, "goal")

        @cancel.on_click
        def cancel_clicked(event: viser.GuiEvent) -> None:
            if event.client is not None:
                event.client.scene.remove_click_callback()
                self.mission_message.content = "Picking cancelled."

        @top_down.on_click
        def top_down_clicked(event: viser.GuiEvent) -> None:
            if event.client is not None:
                center = self.environment.points.min(axis=0) / 2 + self.environment.points.max(axis=0) / 2
                event.client.camera.up_direction = (0, 1, 0)
                event.client.camera.position = center + [0, 0, 90]
                event.client.camera.look_at = center

        @spawn.on_click
        def spawn_clicked(event: viser.GuiEvent) -> None:
            self.actions.put(("reset", None))

    def begin_pick(self, client: viser.ClientHandle, endpoint: str) -> None:
        client.scene.remove_click_callback()
        self.mission_message.content = f"Click the terrain for **{endpoint}**, or cancel picking."

        @client.scene.on_click()
        def point_clicked(event: viser.SceneClickEvent) -> None:
            try:
                ground = self.terrain.pick(event.ray_origin, event.ray_direction)
            except ValueError as error:
                self.mission_message.content = str(error)
                return
            client.scene.remove_click_callback(point_clicked)
            self.actions.put(("picked_endpoint", (endpoint, ground)))

    def read_mission_inputs(self) -> MissionEndpoints:
        return MissionEndpoints(
            self.terrain.ground_at(self.start_xy_input.value),
            self.terrain.ground_at(self.goal_xy_input.value),
            self.clearance_input.value,
        )

    def update_mission_scene(self) -> None:
        mission = self.mission
        with self.server.atomic():
            for name, ground, endpoint in (("start", mission.start_ground, mission.start),
                                            ("goal", mission.goal_ground, mission.goal)):
                self.endpoint_markers[name].position = endpoint
                self.ground_markers[name].position = ground
                self.offset_lines[name].points = np.array([[ground, endpoint]])
            samples = np.linspace(mission.start, mission.goal, 33)
            self.route_preview.points = np.stack((samples[:-1:2], samples[1::2]), axis=1)
            self.mission_status.content = (
                "| Local height [m] | Start | Destination |\n|---|---:|---:|\n"
                f"| Ground reference | {mission.start_ground[2]:.2f} | {mission.goal_ground[2]:.2f} |\n"
                f"| Payload centre | {mission.start_ground[2] + mission.clearance:.2f} | {mission.goal_ground[2] + mission.clearance:.2f} |\n"
                f"| Drone centre | {mission.start[2]:.2f} | {mission.goal[2]:.2f} |\n\n"
                f"Endpoint separation: **{mission.distance:.1f} m**."
            )
            arguments = mission.command().replace(" --", " \\\n  --")
            self.mission_command.content = f"```bash\nuv run python main.py {arguments}\n```"

    def build_controls(self) -> None:
        gui = self.server.gui
        gui.add_markdown(
            "**Excavation · simulated LiDAR**\n\n"
            "**Cyan:** points the robot has discovered.  \n"
            "**Gray:** unseen world points (for your reference).  \n"
            "**Gold sphere:** scanner range. **Green ring:** target.\n\n"
            "Scans accumulate at **10 Hz** in simulation time. Range only; walls do not block scans."
        )
        self.build_mission_controls()
        self.pause_button = gui.add_button("Pause")
        nudge = gui.add_button("Nudge drone")
        self.target_input = gui.add_vector3("Manual hover target [m]", tuple(self.simulation.target), step=0.5)
        self.radius_input = gui.add_number("LiDAR radius [m]", self.lidar.radius, min=0.01, step=0.5)
        show_unseen = gui.add_checkbox("Show unseen world points", initial_value=True)
        show_range = gui.add_checkbox("Show range sphere", initial_value=True)
        focus = gui.add_button("Focus drone")
        overview = gui.add_button("View whole site")
        self.status = gui.add_markdown("")
        gui.add_markdown("Positions are simulated truth in local metres. Hover control has no obstacle avoidance.")

        @self.pause_button.on_click
        def pause_clicked(event: viser.GuiEvent) -> None:
            self.actions.put(("pause", None))

        @nudge.on_click
        def nudge_clicked(event: viser.GuiEvent) -> None:
            self.actions.put(("nudge", None))

        @self.target_input.on_update
        def target_changed(event: viser.GuiEvent) -> None:
            self.actions.put(("target", event.target.value))

        @self.radius_input.on_update
        def radius_changed(event: viser.GuiEvent) -> None:
            self.actions.put(("radius", event.target.value))

        @show_unseen.on_update
        def unseen_changed(event: viser.GuiEvent) -> None:
            self.unseen.visible = event.target.value

        @show_range.on_update
        def range_changed(event: viser.GuiEvent) -> None:
            self.range_sphere.visible = event.target.value

        @focus.on_click
        def focus_clicked(event: viser.GuiEvent) -> None:
            if event.client is not None:
                self.set_camera(event.client, overview=False)

        @overview.on_click
        def overview_clicked(event: viser.GuiEvent) -> None:
            if event.client is not None:
                self.set_camera(event.client, overview=True)

    def connect_client(self, client: viser.ClientHandle) -> None:
        self.set_camera(client, overview=True)

    def set_camera(self, client: viser.ClientHandle, overview: bool) -> None:
        if overview:
            center = (self.environment.points.min(axis=0) + self.environment.points.max(axis=0)) / 2
        else:
            center = self.simulation.state[:3].copy()
            center[2] -= 5
        offset = np.array([42, -52, 35]) if overview else np.array([15, -20, 12])
        client.camera.up_direction = (0, 0, 1)
        client.camera.position = center + offset
        client.camera.look_at = center

    def process_actions(self) -> bool:
        """Apply GUI actions on the physics thread; return whether its clock resets."""
        clock_reset = False
        while not self.actions.empty():
            action, value = self.actions.get_nowait()
            if action == "pause":
                self.paused = not self.paused
                self.pause_button.label = "Resume" if self.paused else "Pause"
                clock_reset = True
            elif action == "reset":
                try:
                    self.mission = self.read_mission_inputs()
                except ValueError as error:
                    self.mission_message.content = str(error)
                    continue
                self.environment.target = self.mission.start
                self.simulation.target[:] = self.mission.start
                self.target_input.value = tuple(self.mission.start)
                self.simulation.reset()
                self.rendered_revision = -1
                clock_reset = True
                self.update_mission_scene()
                self.mission_message.content = "Spawned at the selected start; sensor knowledge reset."
                print(f"Selected endpoints: uv run python main.py {self.mission.command()}")
            elif action == "mission_inputs":
                try:
                    self.mission = self.read_mission_inputs()
                except ValueError as error:
                    self.mission_message.content = str(error)
                    continue
                self.update_mission_scene()
                self.mission_message.content = "Preview updated. **Spawn at start** applies the start and height."
            elif action == "picked_endpoint":
                endpoint, ground = value
                if endpoint == "start":
                    self.mission = replace(self.mission, start_ground=ground)
                    self.start_xy_input.value = tuple(ground[:2])
                else:
                    self.mission = replace(self.mission, goal_ground=ground)
                    self.goal_xy_input.value = tuple(ground[:2])
                self.update_mission_scene()
                self.mission_message.content = f"Selected {endpoint}. **Spawn at start** applies the start and height."
            elif action == "nudge":
                self.simulation.nudge()
            elif action == "target":
                target = np.asarray(value, dtype=float)
                if np.isfinite(target).all():
                    self.simulation.target[:] = target
            elif action == "radius":
                self.lidar.radius = float(value)
        return clock_reset

    def update_scene(self) -> None:
        simulation = self.simulation
        with self.server.atomic():
            self.drone.position = simulation.state[:3]
            self.drone.wxyz = SO3.from_rpy_radians(*simulation.state[6:9]).wxyz
            self.payload.position = simulation.payload_position
            self.tether.points = np.array([[simulation.state[:3], simulation.payload_position]])
            self.target.position = simulation.target
            self.range_sphere.position = simulation.state[:3]
            self.range_sphere.scale = self.lidar.radius
            # Point buffers change only when knowledge changes, never at the physics rate.
            if self.rendered_revision != self.lidar.revision:
                self.known.points = simulation.known_points.astype(np.float32)
                self.unseen.points = self.environment.points[~self.lidar.known_mask].astype(np.float32)
                self.rendered_revision = self.lidar.revision
            position = simulation.state[:3]
            self.status.content = (
                f"**Time:** {simulation.time:.1f} s · {'paused' if self.paused else 'running'}  \n"
                f"**Known:** {len(simulation.known_points):,} / {len(self.environment.points):,}  \n"
                f"**In latest scan:** {self.lidar.in_range_count:,}  \n"
                f"**Scans:** {self.lidar.scan_count:,} · last at {self.lidar.last_scan_time:.1f} s  \n"
                f"**Drone XYZ [m]:** {position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}  \n"
                f"**Target error:** {simulation.position_error:.3f} m"
            )

    def run(self, duration: float | None) -> None:
        previous = time.perf_counter()
        accumulator = 0.0
        try:
            while duration is None or self.simulation.time < duration:
                started = time.perf_counter()
                elapsed = min(started - previous, 0.1)
                previous = started
                if self.process_actions():
                    accumulator = 0.0
                    elapsed = 0.0
                if not self.paused:
                    accumulator += elapsed
                    while accumulator >= self.simulation.dt:
                        self.simulation.step()
                        accumulator -= self.simulation.dt
                        if duration is not None and self.simulation.time >= duration:
                            break
                self.update_scene()
                time.sleep(max(0, 1 / 30 - (time.perf_counter() - started)))
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()


def run_viewer(environment: ExcavationScene, simulation: HoverSimulation, duration: float | None = None,
               port: int = 8080, *, mission: MissionEndpoints, terrain: TerrainSurface) -> None:
    ViserHoverViewer(environment, simulation, port, mission, terrain).run(duration)

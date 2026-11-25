"""Open3D desktop viewer. Geometry is ground truth, not simulated sensor output."""

from collections import deque
import os
import sys
import time

import numpy as np

# Use XWayland for Open3D's native Filament viewer when available. This changes
# only this process's environment, not the user's desktop session.
if sys.platform.startswith("linux") and os.environ.get("DISPLAY") and os.environ.get("WAYLAND_DISPLAY"):
    os.environ.pop("WAYLAND_DISPLAY")
    os.environ["XDG_SESSION_TYPE"] = "x11"

import open3d as o3d
from open3d.visualization import gui, rendering

from . import config
from .transformations import T_IB


def material(color=None, shader="defaultLit", size=3.0) -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    mat.shader = shader
    if color is not None:
        mat.base_color = [*color, 1.0]
    mat.point_size = size
    mat.line_width = size
    return mat


def box(size, color) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_box(*size)
    mesh.translate(-np.asarray(size) / 2)
    mesh.paint_uniform_color(color)
    return mesh


def drone_mesh() -> o3d.geometry.TriangleMesh:
    mesh = box((0.7, 0.5, 0.25), (1.0, 0.35, 0.10))
    for angle in (np.pi/4, -np.pi/4):
        arm = box((2.2, 0.10, 0.10), (0.3, 0.36, 0.43))
        arm.rotate(T_IB(0, 0, angle), center=(0, 0, 0))
        mesh += arm
    for x in (-0.78, 0.78):
        for y in (-0.78, 0.78):
            rotor = o3d.geometry.TriangleMesh.create_cylinder(0.32, 0.035, resolution=24)
            rotor.translate((x, y, 0.10))
            rotor.paint_uniform_color((0.15, 0.65, 0.95))
            mesh += rotor
    mesh.compute_vertex_normals()
    return mesh


def body_transforms(state, payload) -> dict[str, np.ndarray]:
    drone = np.eye(4)
    drone[:3, :3] = T_IB(*state[6:9])
    drone[:3, 3] = state[:3]
    load = np.eye(4)
    load[:3, 3] = payload
    # The cylinder is centred at zero, with its long axis along local Z.
    z = (payload - state[:3]) / config.TETHER_LEN
    reference = np.array([0, 1, 0] if abs(z[1]) < 0.9 else [1, 0, 0])
    x = np.cross(reference, z)
    x /= np.linalg.norm(x)
    tether = np.eye(4)
    tether[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    tether[:3, 3] = (payload + state[:3]) / 2
    return {"Drone": drone, "Payload": load, "Tether": tether}


class HoverViewer:
    def __init__(self, environment, simulation, duration=None) -> None:
        self.environment = environment
        self.sim = simulation
        self.duration = duration
        self.paused = False
        self.closing = False
        self.accumulator = 0.0
        self.trail = deque(maxlen=600)
        self.window = gui.Application.instance.create_window("Stork | Excavation hover", 1280, 800)
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene = self.widget.scene
        self.scene.set_background([0.035, 0.05, 0.075, 1.0])
        self.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS, (0.5, -0.5, -1))
        self.window.add_child(self.widget)

        em = self.window.theme.font_size
        self.panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        self.panel.add_child(gui.Label("EXCAVATION / HOVER"))
        self.panel.add_child(gui.Label("Naive position hold + inner loop\n16-state drone / suspended load"))
        self.status = gui.Label("")
        self.panel.add_child(self.status)
        self.pause_button = gui.Button("Pause")
        self.pause_button.set_on_clicked(self.toggle_pause)
        self.panel.add_child(self.pause_button)
        for label, callback in (("Reset", self.reset), ("Nudge drone", self.sim.nudge),
                                ("Focus drone", self.focus_drone), ("View whole site", self.overview)):
            button = gui.Button(label)
            button.set_on_clicked(callback)
            self.panel.add_child(button)
        self.panel.add_child(gui.Label("Orange: drone / Blue: rotors\nGold: payload / White: tether\nGreen: target / Cyan: flight trail\nCloud colour: relative height"))
        self.panel.add_child(gui.Label("Frame: local XYZ [m], +Z up\nUnits and up direction assumed.\nOrigin: site XY centre, local surface.\nFull map view; no collision or sensor model."))
        self.panel.add_child(gui.Label(f"Display: {len(environment.points):,} points\nSource: {environment.source_count:,} points\nTether: {config.TETHER_LEN:g} m\nDrone: {config.MASS_DRONE:g} kg\nPayload: {config.MASS_PAYLOAD:g} kg"))
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(environment.points))
        heights = environment.points[:, 2]
        low, high = np.percentile(heights, [2, 98])
        blend = np.clip((heights - low) / max(high - low, 1e-6), 0, 1)[:, None]
        colors = (1-blend)*np.array([0.22, 0.37, 0.40]) + blend*np.array([0.87, 0.77, 0.56])
        cloud.colors = o3d.utility.Vector3dVector(colors)
        self.scene.add_geometry("Excavation", cloud, material(shader="defaultUnlit", size=2.0))
        self.scene.add_geometry("Drone", drone_mesh(), material())
        load = o3d.geometry.TriangleMesh.create_sphere(radius=0.35, resolution=16)
        load.compute_vertex_normals()
        self.scene.add_geometry("Payload", load, material((1.0, 0.72, 0.12)))
        cable = o3d.geometry.TriangleMesh.create_cylinder(0.04, config.TETHER_LEN, resolution=12)
        cable.compute_vertex_normals()
        self.scene.add_geometry("Tether", cable, material((0.9, 0.94, 1.0), "defaultUnlit"))
        angles = np.linspace(0, 2*np.pi, 48, endpoint=False)
        ring = self.sim.target + np.column_stack([1.5*np.cos(angles), 1.5*np.sin(angles), np.zeros(48)])
        target = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(ring),
            o3d.utility.Vector2iVector(np.column_stack([np.arange(48), np.roll(np.arange(48), -1)])))
        target.paint_uniform_color((0.35, 1.0, 0.55))
        self.scene.add_geometry("Hover target", target, material(shader="unlitLine", size=1.0))
        self.scene.add_geometry("Local axes", o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0), material())
        self.widget.add_3d_label([2.2, 0, 0], "X [m]")
        self.widget.add_3d_label([0, 2.2, 0], "Y [m]")
        self.widget.add_3d_label([0, 0, 2.2], "Z [m]")
        self.drone_label = self.widget.add_3d_label(self.sim.target + [1.3, 0, 0.5], "Drone")
        self.payload_label = self.widget.add_3d_label(self.sim.payload_position + [0.6, 0, 0], "Payload")
        self._draw_state()
        self.focus_drone()
        self.last_tick = time.perf_counter()
        self.last_draw = self.last_tick
        self.window.set_on_tick_event(self._on_tick)
        self.window.set_on_close(self._on_close)

    def _on_close(self) -> bool:
        # Release Python's scene references before Open3D destroys the renderer.
        self.closing = True
        self.scene = None
        self.widget = None
        self.drone_label = None
        self.payload_label = None
        self.panel = None
        self.status = None
        self.pause_button = None
        self.window = None
        return True

    def _on_layout(self, context) -> None:
        if self.closing:
            return
        rect = self.window.content_rect
        width = min(310, rect.width // 3)
        self.panel.frame = gui.Rect(rect.x, rect.y, width, rect.height)
        self.widget.frame = gui.Rect(rect.x + width, rect.y, rect.width - width, rect.height)

    def focus_drone(self) -> None:
        center = (self.sim.state[:3] + self.sim.payload_position) / 2
        self.widget.setup_camera(55, self.scene.bounding_box, center)
        self.scene.camera.look_at(center, center + [16, -22, 14], [0, 0, 1])
        self.widget.force_redraw()

    def overview(self) -> None:
        bounds = self.scene.bounding_box
        center = bounds.get_center()
        span = max(bounds.get_extent())
        self.widget.setup_camera(60, bounds, center)
        self.scene.camera.look_at(center, center + span*np.array([0.65, -0.8, 0.65]), [0, 0, 1])
        self.widget.force_redraw()

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_button.text = "Resume" if self.paused else "Pause"
        self.accumulator = 0.0

    def reset(self) -> None:
        self.sim.reset()
        self.accumulator = 0.0
        self.trail.clear()
        if self.scene.has_geometry("Flight trail"):
            self.scene.remove_geometry("Flight trail")
        self._draw_state()

    def _draw_state(self) -> None:
        payload = self.sim.payload_position
        for name, transform in body_transforms(self.sim.state, payload).items():
            self.scene.set_geometry_transform(name, transform)
        self.drone_label.position = self.sim.state[:3] + [1.3, 0, 0.5]
        self.payload_label.position = payload + [0.6, 0, 0]
        position = self.sim.state[:3]
        self.status.text = (
            f"Time: {self.sim.time:7.2f} s\n"
            f"Position [m]:\n  {position[0]: .2f}, {position[1]: .2f}, {position[2]: .2f}\n"
            f"Target error: {self.sim.position_error:.3f} m\n"
            f"Speed: {np.linalg.norm(self.sim.state[3:6]):.3f} m/s\n"
            f"Thrust: {self.sim.control[0]:.1f} N"
        )
        if not self.paused and (not self.trail or np.linalg.norm(position - self.trail[-1]) > 1e-4):
            self.trail.append(position.copy())
        if len(self.trail) > 1:
            if self.scene.has_geometry("Flight trail"):
                self.scene.remove_geometry("Flight trail")
            trail = o3d.geometry.LineSet()
            trail.points = o3d.utility.Vector3dVector(np.array(self.trail))
            indices = np.arange(len(self.trail) - 1)
            trail.lines = o3d.utility.Vector2iVector(np.column_stack([indices, indices + 1]))
            trail.paint_uniform_color((0.1, 0.85, 1.0))
            self.scene.add_geometry("Flight trail", trail, material(shader="unlitLine", size=2.0))
        self.widget.force_redraw()

    def _on_tick(self) -> bool:
        if self.closing:
            return False
        now = time.perf_counter()
        elapsed = min(now - self.last_tick, 0.1)
        self.last_tick = now
        if not self.paused:
            self.accumulator += elapsed
            while self.accumulator >= self.sim.dt:
                self.sim.step()
                self.accumulator -= self.sim.dt
            if self.duration is not None and self.sim.time >= self.duration:
                # Closing here would invalidate Open3D's active window iterator.
                self.closing = True
                gui.Application.instance.post_to_main_thread(self.window, self.window.close)
                return False
        if now - self.last_draw < 1/30:
            return False
        self._draw_state()
        self.last_draw = now
        return True


def run_viewer(environment, simulation, duration=None) -> None:
    app = gui.Application.instance
    app.initialize()
    viewer = HoverViewer(environment, simulation, duration)
    app.run()

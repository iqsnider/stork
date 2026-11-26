# stork
MPC simulation study for multicopter with tether-suspended load. Acronym TBD.

## Note to friends

When programming in python, my preferred package manager is `uv`. It's the best. Just clone the project then run `uv sync` in the root directory. If you haven't installed `uv`, follow the instructions here: [uv installation](https://docs.astral.sh/uv/getting-started/installation/).

## Running `uv` projects

Instead of doing python this python that, run python programs by doing: `uv run <program_name>.py` e.g.
```
uv run main.py
```

## Adding packages

run `uv add <package_name>` e.g.
```
uv add scipy
```

## Downloading data

I added the point cloud data to a hugging face repository since it's too large to add here ~500 MB. Get the data by running:

```
uv run download_data.py
```

## Excavation mission setup and hover simulation

Run the simulation in Viser's browser viewer:

```bash
uv run python main.py --lidar-radius 20
```

Open the printed local URL (normally `http://127.0.0.1:8080`). Use `--port` to
choose another port. This loads `data/excavation/excavation.ply` and displays the
point cloud, a drone, the 9.6 m tether, and its payload. Drag to orbit and scroll
to zoom. The green ring is the hover target. On-screen positions are simulated
truth in metres. The drone now spawns away from the map centre, above a suggested
start on the raised area at local XY `(16, -12)`. The suggested destination is
inside the lower excavation at `(4, 12)`, about 27 m away.

### Choose start and destination

In **Start and destination**, use **Pick start on terrain** or **Pick destination
on terrain**, then click the cloud. **Top-down map** helps with selection;
**Cancel picking** returns to normal navigation. You can also enter XY directly.
The green marker is the start, pink is the destination, and small spheres show
their surface references. The vertical lines show the ground-to-drone offsets.

**Payload ground offset [m]** defaults to 3 m. At each endpoint independently:

```text
payload_z = local_surface_z + offset
drone_z   = local_surface_z + offset + 9.6 m tether
```

The surface reference is the **highest sampled point within 1 m horizontally**
of the selected XY. This uses the simulator's full sampled scene for human
mission setup, separately from the robot's sensed knowledge. It is a placement
heuristic, not ground segmentation; it can select an object's top. Locations
without nearby samples are rejected. The UI displays the resulting heights.

Editing these controls updates the endpoint preview. **Spawn at start / reset
knowledge** applies the selected start and offset, zeros the drone's velocities,
and resets simulation time and accumulated scans. The drone then holds there.
Changing the destination does not command a flight.

The dashed line is only an endpoint connector: **no trajectory planning or
collision checking is performed**. The suggested pair crosses raised terrain;
endpoint clearance alone does not maintain clearance along that connector.
Planning and trajectory tracking can be added after choosing the endpoints.

For repeatable selections, copy the command in **Reproduce these endpoints**:

```bash
uv run python main.py --start-xy 16 -12 --goal-xy 4 12 --clearance 3
```

XY coordinates are local, not original PLY coordinates. The defaults are chosen
for the excavation dataset; set appropriate XY values when using another cloud.
The same CLI start works in headless mode and Open3D; interactive endpoint
selection and destination annotations are currently specific to Viser.

### Simulated perception

- **Cyan points:** the robot's accumulated knowledge, retained after leaving range.
- **Gray points:** unseen world points, shown only as a reference for the viewer.
- **Gold wire sphere:** the scanner's radius, centred on the drone.

The ideal range sensor takes an initial scan at t=0, then scans every **0.1 s
(10 Hz of simulation time)**. Each scan finds all scene points whose 3D Euclidean
distance to the drone is at most the radius. A SciPy KD-tree indexes the scene;
the viewer updates cloud buffers only when the known set changes.

Change **LiDAR radius [m]** or **Manual hover target [m]** in the sidebar to explore.
Radius changes take effect on the next scan; reducing it does not erase past
knowledge. Uncheck **Show unseen world points** to view only the robot's map.
**Pause** freezes dynamics and scanning. **Spawn at start / reset knowledge** returns
to the selected start and clears the map before taking a new initial scan.
**Nudge drone**, **Focus drone**, and **View whole site** are also available.

`simulation.known_points` is an `(N, 3)` NumPy array in the local world frame for
an outer-loop/MPC controller to consume. The full scene stays on the simulation
side in `RangeLidar`; unseen points are not included in this array. The current
naive hover controller does not use these points for avoidance.

The simulation uses `src/nonlinear_model.py` and the existing
`ArduPilotFlightController`, with an outer position/velocity PD loop. The plant
uses fixed 5 ms forward Euler steps, holding each control command for one step. The drone
starts at equilibrium; **Nudge drone** applies a small velocity disturbance to
demonstrate recovery. This is a hover prototype with a taut, fixed-length tether;
collision response, sensor occlusion, and MPC are not implemented yet. This
range-only model sees points **through walls** inside the ball; it does not model
beam directions, occlusion, measurement noise, or an actual LiDAR scan pattern.

The PLY's `x`, `y`, and `z` fields supply geometry. Viser colours indicate
knowledge; intensity is unused. We **assume metres and +Z up**, since the PLY
does not declare units or a coordinate system. The local origin uses the full
cloud's horizontal bounding-box centre and the highest point within a 3 m
horizontal radius as its vertical reference. This is a placement heuristic,
not ground segmentation. This defines the coordinate origin only. The selected
start and destination use their own local surface heights as described above.
The original file is unchanged, and its origin offset is printed at startup.

The viewer and sensor both use a uniformly subsampled scene, with at most
500,000 points by default. The local origin still uses the full source cloud.
Change the scene limit or payload clearance with:

```bash
uv run python main.py --max-points 1000000 --clearance 5
```

Viser meshes, line segments, coordinate frames, and labels provide space for
future paths, constraints, and annotations in `src/viser_viewer.py`. Physics and
perception remain independent of either viewer.

The original Open3D viewer is still available, with height colours and the drone
trajectory (knowledge colouring is currently specific to Viser):

```bash
uv run python main.py --viewer open3d
```

Open3D requires a desktop session with graphics support. On Linux Wayland
desktops it uses XWayland when `DISPLAY` is available.

For numerical verification without a window:

```bash
uv run python main.py --headless --duration 30 --nudge
uv run python -m unittest discover -s tests -v
```

Headless output includes known-point counts and scan timing. `--duration` also
stops either viewer after that many simulated seconds. Without it, stop Viser
with Ctrl+C in the terminal; closing the browser alone does not stop the server.

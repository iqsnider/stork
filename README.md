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

## Excavation hover simulation

Run the simulation in Open3D's native desktop viewer:

```bash
uv run python main.py
```

This loads `data/excavation/excavation.ply` and displays the point cloud, a drone,
the 9.6 m tether, and its payload. **Pause**, **Reset**, **Nudge drone**, **Focus
drone**, and **View whole site** are available in the sidebar. Drag to orbit and
scroll to zoom. The green marker is the hover target; cyan is the recent drone
trajectory. On-screen positions are simulated truth in metres.

The simulation uses `src/nonlinear_model.py` and the existing
`ArduPilotFlightController`, with an outer position/velocity PD loop. The plant
uses fixed 5 ms forward Euler steps, holding each control command for one step. The drone
starts at equilibrium; **Nudge drone** applies a small velocity disturbance to
demonstrate recovery. This is a hover prototype with a taut, fixed-length tether;
collision response, sensor occlusion, and MPC are not implemented yet.

The PLY's `x`, `y`, and `z` fields supply geometry. Display colours are derived
from height; intensity is unused. We **assume metres and +Z up**, since the PLY
does not declare units or a coordinate system. The local origin uses the full
cloud's horizontal bounding-box centre and the highest point within a 3 m
horizontal radius as its vertical reference. This is a placement heuristic,
not ground segmentation. The payload centre starts 3 m above that reference;
the drone starts another tether length above it. The original file is unchanged,
and its origin offset is printed at startup.

Only the displayed cloud is uniformly subsampled, to at most 500,000 points.
Change the display limit or payload clearance with:

```bash
uv run python main.py --max-points 1000000 --clearance 5
```

Open3D's `SceneWidget` supports meshes, line sets, coordinate frames, and 3D text
labels for future paths, constraints, and sensor annotations. These are managed
in `src/viewer.py`; physics remains independent of the viewer. A desktop session
with graphics support for Open3D is required.

On Linux Wayland desktops, the viewer uses XWayland when `DISPLAY` is available
for compatibility with Open3D's native renderer.

For numerical verification without a window:

```bash
uv run python main.py --headless --duration 30 --nudge
uv run python -m unittest discover -s tests -v
```

`--duration` also closes the viewer after that many simulated seconds. Without
it, the desktop simulation continues until the window is closed.

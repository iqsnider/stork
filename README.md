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

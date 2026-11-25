from pathlib import Path
from huggingface_hub import snapshot_download


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    print("Downloading point cloud datasets from Hugging Face...")

    snapshot_download(
        repo_id="iqsnider/stork_data",
        repo_type="dataset",
        allow_patterns="data/**/*.ply",
        local_dir=repo_root,
    )
    print("Done! Files saved to data/excavation/ and data/parking_lot/.")


if __name__ == "__main__":
    main()

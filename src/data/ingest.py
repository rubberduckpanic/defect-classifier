"""Data ingestion module.

Downloads and organizes the Casting Product Image dataset from Kaggle.
"""


def download_dataset(output_dir: str = "data/raw") -> None:
    """Download dataset from Kaggle and extract to output_dir."""
    # TODO: Implement Kaggle API download or manual download instructions
    raise NotImplementedError


def organize_splits(raw_dir: str = "data/raw", seed: int = 42) -> None:
    """Organize raw images into train/val/test splits."""
    # TODO: Implement train/val/test splitting logic
    raise NotImplementedError


if __name__ == "__main__":
    download_dataset()
    organize_splits()

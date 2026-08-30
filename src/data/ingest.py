"""Data ingestion module.

Downloads and organizes the Casting Product Image dataset from Kaggle.
"""

import os
import zipfile
import argparse


def download_dataset(output_dir: str = "data/raw") -> None:
    """Download dataset from Kaggle and extract to output_dir."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    os.makedirs(output_dir, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    dataset = "ravirajsinh45/real-life-industrial-dataset-of-casting-product"
    print(f"Downloading dataset: {dataset}")
    api.dataset_download_files(dataset, path=output_dir, unzip=True)
    print(f"Dataset downloaded and extracted to: {output_dir}")

    # The dataset extracts with nested folders, flatten if needed
    # Expected structure: data/raw/casting_data/def_front/ and data/raw/casting_data/ok_front/
    # or data/raw/def_front/ and data/raw/ok_front/
    _flatten_structure(output_dir)


def _flatten_structure(output_dir: str) -> None:
    """Ensure images are directly under output_dir/def_front and output_dir/ok_front."""
    import shutil

    # Check if there's a nested folder (e.g., casting_data or casting_512x512)
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item not in ("def_front", "ok_front"):
            # Check if this subfolder contains def_front/ok_front
            sub_def = os.path.join(item_path, "def_front")
            sub_ok = os.path.join(item_path, "ok_front")
            if os.path.isdir(sub_def) or os.path.isdir(sub_ok):
                # Move contents up
                for sub_item in os.listdir(item_path):
                    src = os.path.join(item_path, sub_item)
                    dst = os.path.join(output_dir, sub_item)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                # Remove the now-empty nested folder
                shutil.rmtree(item_path, ignore_errors=True)
                print(f"Flattened nested folder: {item}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download casting product dataset")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    args = parser.parse_args()
    download_dataset(args.output)

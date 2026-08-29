"""Stage an extracted Kaggle casting dataset for the pipeline."""

import argparse
import os
import shutil
from pathlib import Path


CLASS_DIRECTORIES = {
    "defective": ("def_front", "test_def_front"),
    "non_defective": ("ok_front", "test_ok_front"),
}


def _find_dataset_root(source_dir: Path) -> Path:
    """Find the extracted directory containing train and test folders."""
    candidates = [source_dir] + [path for path in source_dir.rglob("*") if path.is_dir()]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find train/test folders below {source_dir}")


def download_dataset(output_dir: str = "data/raw", source_dir: str = "") -> dict:
    """Copy an extracted Kaggle archive into normalized class directories."""
    configured_source = source_dir or os.environ.get("CASTING_DATASET_DIR")
    if not configured_source:
        raise ValueError("Set CASTING_DATASET_DIR or pass --source")
    source = Path(configured_source).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Dataset source not found: {source}")

    dataset_root = _find_dataset_root(source)
    destination = Path(output_dir)
    counts = {class_name: 0 for class_name in CLASS_DIRECTORIES}
    for class_name, source_names in CLASS_DIRECTORIES.items():
        class_destination = destination / class_name
        class_destination.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "test"):
            for source_name in source_names:
                source_class = dataset_root / split_name / source_name
                if not source_class.is_dir():
                    continue
                for image_path in source_class.iterdir():
                    if image_path.is_file():
                        shutil.copy2(image_path, class_destination / image_path.name)
                        counts[class_name] += 1
    return counts


def organize_splits(raw_dir: str = "data/raw", seed: int = 42) -> None:
    """Validate that staged data exists; preprocessing creates final splits."""
    if not Path(raw_dir).exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage the casting dataset")
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--source", default="")
    args = parser.parse_args()
    print(download_dataset(args.output, args.source))
    organize_splits(args.output)

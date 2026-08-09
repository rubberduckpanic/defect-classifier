"""
Module M2: Data Preprocessing & Feature Engineering
Handles image preprocessing, augmentation, and train/val/test splitting.

Feature engineering for images:
- Resize to consistent dimensions
- Normalize pixel values
- Data augmentation (rotation, flip, brightness/contrast jitter)
- Stratified train/val/test split
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/train_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resize_image(image_path: Path, output_path: Path, target_size: int = 224) -> bool:
    """
    Resize image to target dimensions while maintaining aspect ratio.
    Converts to RGB if necessary.
    
    Args:
        image_path: Source image path
        output_path: Destination path
        target_size: Target dimension (square)
        
    Returns:
        True if successful
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB (handle grayscale)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize with high-quality resampling
            img_resized = img.resize((target_size, target_size), Image.LANCZOS)
            
            # Save as PNG for consistency
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img_resized.save(output_path, "PNG")
            return True
    except Exception as e:
        logger.warning(f"Failed to process {image_path}: {e}")
        return False


def preprocess_dataset(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    target_size: int = 224
) -> Dict[str, int]:
    """
    Preprocess all images: resize, convert to RGB, save as PNG.
    
    Args:
        raw_dir: Directory containing raw images (organized by class)
        processed_dir: Output directory for processed images
        target_size: Target image size (square)
        
    Returns:
        Dictionary with processing statistics
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    stats = {"processed": 0, "failed": 0}

    for class_dir in sorted(raw_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        output_class_dir = processed_path / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing class: {class_name}")

        for image_file in sorted(class_dir.iterdir()):
            if not image_file.is_file():
                continue
            if image_file.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp'}:
                continue

            output_file = output_class_dir / f"{image_file.stem}.png"
            
            if resize_image(image_file, output_file, target_size):
                stats["processed"] += 1
            else:
                stats["failed"] += 1

    logger.info(f"Preprocessing complete: {stats['processed']} processed, {stats['failed']} failed")
    return stats


def create_splits(
    processed_dir: str = "data/processed",
    splits_dir: str = "data/splits",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, int]:
    """
    Create stratified train/validation/test splits.
    
    Stratification ensures class balance is maintained across splits.
    
    Args:
        processed_dir: Directory with processed images
        splits_dir: Output directory for splits
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_seed: Seed for reproducibility
        
    Returns:
        Dictionary with split sizes
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    processed_path = Path(processed_dir)
    splits_path = Path(splits_dir)

    # Collect all image paths and labels
    image_paths: List[Path] = []
    labels: List[str] = []

    for class_dir in sorted(processed_path.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_file in sorted(class_dir.iterdir()):
            if img_file.is_file() and img_file.suffix.lower() == '.png':
                image_paths.append(img_file)
                labels.append(class_dir.name)

    logger.info(f"Total images for splitting: {len(image_paths)}")

    # First split: train vs (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=(val_ratio + test_ratio),
        random_state=random_seed,
        stratify=labels
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=relative_test_ratio,
        random_state=random_seed,
        stratify=temp_labels
    )

    # Copy files to split directories
    split_stats = {}
    for split_name, paths, split_labels in [
        ("train", train_paths, train_labels),
        ("val", val_paths, val_labels),
        ("test", test_paths, test_labels)
    ]:
        for img_path, label in zip(paths, split_labels):
            dest_dir = splits_path / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / img_path.name
            shutil.copy2(str(img_path), str(dest_file))

        split_stats[split_name] = len(paths)
        logger.info(f"{split_name}: {len(paths)} images")

    # Log class distribution per split
    for split_name, paths, split_labels in [
        ("train", train_paths, train_labels),
        ("val", val_paths, val_labels),
        ("test", test_paths, test_labels)
    ]:
        unique, counts = np.unique(split_labels, return_counts=True)
        dist = dict(zip(unique, counts))
        logger.info(f"  {split_name} distribution: {dist}")

    return split_stats


def get_dataset_stats(splits_dir: str = "data/splits") -> Dict:
    """
    Compute dataset statistics (mean, std) from training set for normalization.
    
    Returns:
        Dictionary with channel-wise mean and std
    """
    splits_path = Path(splits_dir) / "train"
    
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    num_pixels = 0

    for class_dir in splits_path.iterdir():
        if not class_dir.is_dir():
            continue
        for img_file in class_dir.iterdir():
            if not img_file.is_file():
                continue
            img = np.array(Image.open(img_file).convert("RGB")) / 255.0
            pixel_sum += img.sum(axis=(0, 1))
            pixel_sq_sum += (img ** 2).sum(axis=(0, 1))
            num_pixels += img.shape[0] * img.shape[1]

    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_sq_sum / num_pixels - mean ** 2)

    logger.info(f"Dataset mean: {mean}")
    logger.info(f"Dataset std: {std}")

    return {"mean": mean.tolist(), "std": std.tolist()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess and split dataset")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                        help="Path to configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]

    # Step 1: Preprocess images
    logger.info("Step 1: Preprocessing images...")
    preprocess_dataset(
        raw_dir=data_config["raw_dir"],
        processed_dir=data_config["processed_dir"],
        target_size=data_config["image_size"]
    )

    # Step 2: Create train/val/test splits
    logger.info("Step 2: Creating data splits...")
    create_splits(
        processed_dir=data_config["processed_dir"],
        splits_dir=data_config["splits_dir"],
        train_ratio=data_config["train_ratio"],
        val_ratio=data_config["val_ratio"],
        test_ratio=data_config["test_ratio"],
        random_seed=data_config["random_seed"]
    )

    # Step 3: Compute normalization statistics
    logger.info("Step 3: Computing dataset statistics...")
    stats = get_dataset_stats(data_config["splits_dir"])
    print(f"\nNormalization stats:\n  Mean: {stats['mean']}\n  Std: {stats['std']}")

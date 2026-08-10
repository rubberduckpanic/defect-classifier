"""
Module M2: Data Validation

Validation checks:
1. File format verification (valid image files)
2. Image corruption detection (can be decoded)
3. Dimension consistency checks
4. Class balance reporting
5. Duplicate detection via hashing
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}
EXPECTED_MIN_SIZE = 50  # Minimum image dimension in pixels
EXPECTED_MAX_SIZE = 1000  # Maximum image dimension in pixels


class ValidationReport:
    """Stores validation results for the dataset."""

    def __init__(self):
        self.total_images = 0
        self.valid_images = 0
        self.corrupt_images: List[str] = []
        self.invalid_format: List[str] = []
        self.size_anomalies: List[Tuple[str, Tuple[int, int]]] = []
        self.duplicates: List[Tuple[str, str]] = []
        self.class_distribution: Dict[str, int] = {}

    def summary(self) -> str:
        report = [
            "=" * 60,
            "DATA VALIDATION REPORT",
            "=" * 60,
            f"Total images scanned: {self.total_images}",
            f"Valid images: {self.valid_images}",
            f"Corrupt images: {len(self.corrupt_images)}",
            f"Invalid format: {len(self.invalid_format)}",
            f"Size anomalies: {len(self.size_anomalies)}",
            f"Duplicates found: {len(self.duplicates)}",
            "",
            "Class Distribution:",
        ]
        for cls, count in self.class_distribution.items():
            report.append(f"  {cls}: {count}")

        if self.corrupt_images:
            report.append(f"\nCorrupt files (first 10): {self.corrupt_images[:10]}")
        if self.size_anomalies:
            report.append(f"\nSize anomalies (first 10): {self.size_anomalies[:10]}")

        report.append("=" * 60)
        return "\n".join(report)


def compute_image_hash(image_path: Path) -> str:
    """Compute MD5 hash of image file for duplicate detection."""
    hasher = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_single_image(image_path: Path) -> Tuple[bool, str]:
    """
    Validate a single image file.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file extension
    if image_path.suffix.lower() not in VALID_EXTENSIONS:
        return False, f"Invalid extension: {image_path.suffix}"

    # Try to open and verify the image
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify image integrity
    except Exception as e:
        return False, f"Corrupt image: {str(e)}"

    # Re-open to check dimensions (verify() invalidates the image object)
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            if w < EXPECTED_MIN_SIZE or h < EXPECTED_MIN_SIZE:
                return False, f"Image too small: {w}x{h}"
            if w > EXPECTED_MAX_SIZE or h > EXPECTED_MAX_SIZE:
                return False, f"Image too large: {w}x{h}"
    except Exception as e:
        return False, f"Cannot read dimensions: {str(e)}"

    return True, ""


def validate_dataset(data_dir: str = "data/raw") -> ValidationReport:
    """
    Run full validation on the dataset.
    
    Args:
        data_dir: Path to the raw data directory
        
    Returns:
        ValidationReport with all findings
    """
    data_path = Path(data_dir)
    report = ValidationReport()

    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Collect all image files
    hash_map: Dict[str, str] = {}  # hash -> first file path

    for class_dir in sorted(data_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        class_count = 0

        for image_file in sorted(class_dir.iterdir()):
            if not image_file.is_file():
                continue

            report.total_images += 1

            # Validate format and integrity
            is_valid, error_msg = validate_single_image(image_file)

            if not is_valid:
                if "Corrupt" in error_msg:
                    report.corrupt_images.append(str(image_file))
                elif "Invalid extension" in error_msg:
                    report.invalid_format.append(str(image_file))
                elif "too small" in error_msg or "too large" in error_msg:
                    w, h = Image.open(image_file).size
                    report.size_anomalies.append((str(image_file), (w, h)))
                continue

            # Check for duplicates
            file_hash = compute_image_hash(image_file)
            if file_hash in hash_map:
                report.duplicates.append((str(image_file), hash_map[file_hash]))
            else:
                hash_map[file_hash] = str(image_file)

            report.valid_images += 1
            class_count += 1

        report.class_distribution[class_name] = class_count

    logger.info(report.summary())
    return report


def remove_invalid_images(data_dir: str = "data/raw", dry_run: bool = True) -> List[str]:
    """
    Remove corrupt or invalid images from the dataset.
    
    Args:
        data_dir: Path to data directory
        dry_run: If True, only report what would be removed
        
    Returns:
        List of removed (or would-be-removed) file paths
    """
    report = validate_dataset(data_dir)
    to_remove = report.corrupt_images + report.invalid_format

    if dry_run:
        logger.info(f"DRY RUN: Would remove {len(to_remove)} files")
    else:
        for filepath in to_remove:
            Path(filepath).unlink()
            logger.info(f"Removed: {filepath}")
        logger.info(f"Removed {len(to_remove)} invalid files")

    return to_remove


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate dataset images")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Path to raw data directory")
    parser.add_argument("--remove-invalid", action="store_true",
                        help="Remove invalid images (default: dry run)")
    args = parser.parse_args()

    if args.remove_invalid:
        remove_invalid_images(args.data_dir, dry_run=False)
    else:
        report = validate_dataset(args.data_dir)
        print(report.summary())

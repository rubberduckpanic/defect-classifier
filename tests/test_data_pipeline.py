"""
Unit tests for the data pipeline (M2).
Tests validation logic and preprocessing functions.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.validate import ValidationReport, compute_image_hash, validate_single_image
from src.data.preprocess import resize_image


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def valid_image(temp_dir):
    """Create a valid test image."""
    img = Image.new("RGB", (300, 300), color=(100, 150, 200))
    path = temp_dir / "valid.png"
    img.save(path)
    return path


@pytest.fixture
def corrupt_file(temp_dir):
    """Create a corrupt 'image' file."""
    path = temp_dir / "corrupt.png"
    path.write_bytes(b"this is not a valid image file")
    return path


@pytest.fixture
def small_image(temp_dir):
    """Create an image below minimum size."""
    img = Image.new("RGB", (20, 20), color=(100, 100, 100))
    path = temp_dir / "small.png"
    img.save(path)
    return path


class TestValidateSingleImage:
    def test_valid_image_passes(self, valid_image):
        is_valid, error = validate_single_image(valid_image)
        assert is_valid is True
        assert error == ""

    def test_corrupt_file_fails(self, corrupt_file):
        is_valid, error = validate_single_image(corrupt_file)
        assert is_valid is False
        assert "Corrupt" in error

    def test_small_image_fails(self, small_image):
        is_valid, error = validate_single_image(small_image)
        assert is_valid is False
        assert "too small" in error

    def test_invalid_extension(self, temp_dir):
        path = temp_dir / "file.txt"
        path.write_text("not an image")
        is_valid, error = validate_single_image(path)
        assert is_valid is False
        assert "Invalid extension" in error


class TestImageHash:
    def test_same_image_same_hash(self, valid_image):
        hash1 = compute_image_hash(valid_image)
        hash2 = compute_image_hash(valid_image)
        assert hash1 == hash2

    def test_different_images_different_hash(self, temp_dir):
        img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img2 = Image.new("RGB", (100, 100), color=(0, 255, 0))
        path1 = temp_dir / "img1.png"
        path2 = temp_dir / "img2.png"
        img1.save(path1)
        img2.save(path2)

        assert compute_image_hash(path1) != compute_image_hash(path2)


class TestResizeImage:
    def test_resize_to_target(self, valid_image, temp_dir):
        output = temp_dir / "resized.png"
        success = resize_image(valid_image, output, target_size=224)
        assert success is True
        assert output.exists()

        resized = Image.open(output)
        assert resized.size == (224, 224)

    def test_converts_grayscale_to_rgb(self, temp_dir):
        gray_img = Image.new("L", (300, 300), color=128)
        gray_path = temp_dir / "gray.png"
        gray_img.save(gray_path)

        output = temp_dir / "rgb_output.png"
        success = resize_image(gray_path, output, target_size=224)
        assert success is True

        result = Image.open(output)
        assert result.mode == "RGB"
        assert result.size == (224, 224)


class TestValidationReport:
    def test_empty_report(self):
        report = ValidationReport()
        assert report.total_images == 0
        assert report.valid_images == 0
        assert len(report.corrupt_images) == 0

    def test_summary_format(self):
        report = ValidationReport()
        report.total_images = 100
        report.valid_images = 95
        report.class_distribution = {"defective": 45, "non_defective": 50}
        summary = report.summary()
        assert "100" in summary
        assert "95" in summary

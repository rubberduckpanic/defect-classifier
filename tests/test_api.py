"""
Integration tests for the FastAPI prediction endpoint.
Tests input validation, error handling, and response format.
"""

import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client():
    """Create test client."""
    from src.serving.app import app
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a valid test image."""
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


@pytest.fixture
def tiny_image():
    """Create an image that's too small."""
    img = Image.new("RGB", (10, 10), color=(128, 128, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data

    def test_health_reports_model_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["model_loaded"], bool)


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "architecture" in data
        assert "num_classes" in data
        assert data["num_classes"] == 2


class TestPredictEndpoint:
    def test_rejects_non_image_file(self, client):
        """API should reject non-image uploads."""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400

    def test_rejects_too_small_image(self, client, tiny_image):
        """API should reject images below minimum dimensions."""
        response = client.post(
            "/predict",
            files={"file": ("tiny.png", tiny_image, "image/png")}
        )
        assert response.status_code == 400

    def test_rejects_corrupt_image(self, client):
        """API should reject corrupt image data."""
        response = client.post(
            "/predict",
            files={"file": ("corrupt.png", b"not valid png data", "image/png")}
        )
        assert response.status_code == 400

    def test_predict_returns_503_when_model_not_loaded(self, client, sample_image):
        """API returns 503 if model isn't loaded."""
        from src.serving.app import model_service
        original_model = model_service.model
        model_service.model = None

        response = client.post(
            "/predict",
            files={"file": ("test.png", sample_image, "image/png")}
        )
        assert response.status_code == 503

        model_service.model = original_model


class TestPredictResponseFormat:
    """Test that successful predictions have correct format."""

    def test_response_has_required_fields(self, client, sample_image):
        """When model is loaded, response should have all fields."""
        from src.serving.app import model_service
        if model_service.model is None:
            pytest.skip("Model not loaded for this test")

        response = client.post(
            "/predict",
            files={"file": ("test.png", sample_image, "image/png")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction_id" in data
        assert "label" in data
        assert "confidence" in data
        assert "defective" in data
        assert "inference_time_ms" in data
        assert "timestamp" in data
        assert data["label"] in ["defective", "non_defective"]
        assert 0.0 <= data["confidence"] <= 1.0

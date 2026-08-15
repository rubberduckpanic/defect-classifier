"""
Unit tests for model architectures (M3).
Tests model creation, forward pass, and parameter counts.
"""

import pytest
import torch

from src.training.models import (
    CNNBaseline,
    EfficientNetB0Classifier,
    ResNet18Classifier,
    count_parameters,
    get_model,
)


@pytest.fixture
def sample_input():
    """Create sample input tensor (batch of 2, 3-channel, 224x224)."""
    return torch.randn(2, 3, 224, 224)


class TestCNNBaseline:
    def test_forward_shape(self, sample_input):
        model = CNNBaseline(num_classes=2)
        output = model(sample_input)
        assert output.shape == (2, 2)

    def test_different_num_classes(self, sample_input):
        model = CNNBaseline(num_classes=5)
        output = model(sample_input)
        assert output.shape == (2, 5)


class TestResNet18:
    def test_forward_shape(self, sample_input):
        model = ResNet18Classifier(num_classes=2, pretrained=False)
        output = model(sample_input)
        assert output.shape == (2, 2)

    def test_frozen_backbone_has_fewer_trainable_params(self):
        model_frozen = ResNet18Classifier(pretrained=False, freeze_backbone=True)
        model_full = ResNet18Classifier(pretrained=False, freeze_backbone=False)

        frozen_params = count_parameters(model_frozen)
        full_params = count_parameters(model_full)

        assert frozen_params["trainable"] < full_params["trainable"]


class TestEfficientNet:
    def test_forward_shape(self, sample_input):
        model = EfficientNetB0Classifier(num_classes=2, pretrained=False)
        output = model(sample_input)
        assert output.shape == (2, 2)


class TestModelFactory:
    def test_get_model_cnn_baseline(self):
        model = get_model("cnn_baseline", num_classes=2)
        assert isinstance(model, CNNBaseline)

    def test_get_model_resnet18(self):
        model = get_model("resnet18", num_classes=2, pretrained=False)
        assert isinstance(model, ResNet18Classifier)

    def test_get_model_efficientnet(self):
        model = get_model("efficientnet_b0", num_classes=2, pretrained=False)
        assert isinstance(model, EfficientNetB0Classifier)

    def test_get_model_invalid_raises(self):
        with pytest.raises(ValueError):
            get_model("invalid_model", num_classes=2)


class TestParameterCounts:
    def test_count_parameters(self):
        model = CNNBaseline(num_classes=2)
        counts = count_parameters(model)
        assert counts["total"] > 0
        assert counts["trainable"] > 0
        assert counts["total"] == counts["trainable"] + counts["frozen"]

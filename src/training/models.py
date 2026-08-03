"""Model architecture definitions.

Contains baseline CNN and transfer learning models for defect classification.
"""

import torch.nn as nn


class BaselineCNN(nn.Module):
    """Simple CNN baseline for binary defect classification."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # TODO: Define conv layers, pooling, fully-connected head
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def get_resnet_model(num_classes: int = 2, pretrained: bool = True):
    """Return a fine-tunable ResNet-18 with modified classification head."""
    # TODO: Load pretrained ResNet-18, freeze backbone, replace fc layer
    raise NotImplementedError


def get_efficientnet_model(num_classes: int = 2, pretrained: bool = True):
    """Return a fine-tunable EfficientNet-B0 with modified classification head."""
    # TODO: Load pretrained EfficientNet-B0, replace classifier
    raise NotImplementedError
